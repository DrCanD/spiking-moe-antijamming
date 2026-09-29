"""Current-block routing and received-only bounded ALE rescue.

Every output block is buffered until that block's end. This is block-causal,
not sample-by-sample zero-latency inference. No packet positions or truth enter.
"""
from collections import Counter
import time
import numpy as np
import common_v5 as io
core=io.core;legacy=io.streaming_v3;v4=io.receiver_v4

def risk_stats(x):
    n=len(x);ss=float(np.dot(x,x));lag=float(np.dot(x[1:],x[:-1]))
    denom=max(ss-0.5*(float(x[0])**2+float(x[-1])**2),1e-12)
    return max(ss/n,1e-12),lag/denom

def risk_event(now,previous,settings):
    if previous is None:return False
    ratio=now[0]/previous[0]
    return bool(ratio>=settings['power_ratio_trigger'] or ratio<=1/settings['power_ratio_trigger'] or abs(now[1]-previous[1])>=settings['lag1_correlation_change_trigger'])

def process_one(rx,cfg,bank,feats,kind,policy,rescue):
    if policy in ('lag_periodic','lag_guard8') or kind in ('raw','rule'):
        base={'lag_periodic':'periodic','lag_guard8':'guard8'}.get(policy,policy)
        result=v4.process_one(rx,cfg,bank,feats,kind,base)
        result['timing_mode']='previous_block_decision' if kind not in ('raw','rule') else 'current_buffered_block'
        result['risk_log']=[]
        result['counts']['ale_tap_products']=cfg.ale_taps*(2*result['counts']['ale_predictions']+result['counts']['ale_weight_updates'])
        return result
    if kind not in ('spike','analog_delta') or policy not in ('current_periodic','current_guard8','refresh_guard8','rescue_guard8','rescue_periodic'):
        raise ValueError((kind,policy))
    periodic=policy.endswith('periodic');use_risk=policy.startswith(('refresh','rescue'));use_rescue=policy.startswith('rescue')
    s=bank.settings;b=legacy._validate(cfg,s);n=len(rx);ns=n//cfg.sps
    if n%b or len(feats)!=n//b:raise ValueError('Complete block/feature mismatch')
    soft=np.zeros(ns);old=np.zeros(ns,bool);refined=np.zeros(ns,bool)
    ale=legacy.StatefulALE(cfg);action='pass';anchor=None;armed=True;last_call=-10**9
    previous_stats=None;risk_remaining=0;previous_rejected=False
    decisions=[];actions=[];risk_log=[]
    counts=Counter(router_calls=0,ale_samples=0,ale_predictions=0,ale_weight_updates=0,pulse_tests=0,pulse_accepts=0,pulse_group_rms=0,
        forced_refreshes=0,event_triggers=0,blank_guard_rejections=0,guard_refreshes=0,confirmation_refreshes=0,
        history_sample_updates=n,bank_sample_updates=n*cfg.rf_bands,delta_sample_updates=n,fft_calls=0,quality_power_samples=n,
        detector_samples=0,detector_products=0,detector_events=0,risk_blocks=0,risk_refreshes=0,rescue_blocks=0,rescue_accepted=0,rescue_rejected=0,rescue_quality_products=0)
    started=time.perf_counter();router_wall=0.;expert_wall=0.;detector_wall=0.
    for bi,a in enumerate(range(0,n,b)):
        x=rx[a:a+b];risk=False;event_risk=False
        if use_risk:
            rt=time.perf_counter();now=risk_stats(x);event_risk=risk_event(now,previous_stats,rescue)
            if bi==0:risk_remaining=max(risk_remaining,rescue['startup_blocks'])
            if event_risk:risk_remaining=max(risk_remaining,rescue['hold_blocks'])
            risk=risk_remaining>0;risk_remaining=max(0,risk_remaining-1);previous_stats=now
            counts.update(detector_samples=b,detector_products=2*b+1,detector_events=int(event_risk),risk_blocks=int(risk))
            detector_wall+=time.perf_counter()-rt
        # Current feature refers only to the buffered x, whose end is now available.
        z=bank.standardized(kind,feats[bi][kind]);change=float(np.sqrt(np.mean((z-anchor)**2))) if anchor is not None else 0.
        if change<=s.change_low:armed=True
        feature_event=anchor is not None and armed and bi-last_call>=s.min_decision_blocks and change>=s.change_high
        force=anchor is not None and bi-last_call>=8
        due=periodic or anchor is None or force or feature_event or previous_rejected or risk
        if due:
            rt=time.perf_counter();label,confidence=bank.predict_features(kind,feats[bi][kind]);router_wall+=time.perf_counter()-rt
            action=legacy._label_to_action(label);last_call=bi;anchor=z.copy()
            counts.update(router_calls=1,forced_refreshes=int(force and not periodic),event_triggers=int(feature_event and not periodic),guard_refreshes=int(previous_rejected and not periodic),risk_refreshes=int(risk and not periodic))
            if not periodic:armed=False
            decisions.append(dict(block=bi,observed_end_sample=a+b,apply_sample=a+b,affected_start_sample=a,affected_end_sample=a+b,
                output_available_sample=a+b,predicted_class=label,confidence=confidence,action=action,change_score=change,risk=risk))
        expert_start=time.perf_counter();rejected=False;rho=None;accept=None
        if use_rescue and risk:
            raw_mask,_=legacy._pulse_mask(x,cfg,bank,False)
            residual,updates,pred=ale.process(x,raw_mask,rescue['fast_mu'],True)
            # One persistent ALE is advanced once. Rejected output still incurs work.
            rho=1-float(np.dot(residual,residual))/max(float(np.dot(x,x)),1e-12)
            accept=bool(rho>rescue['residual_power_reduction_threshold']);y=residual if accept else x
            erase,pa=legacy._pulse_mask(y,cfg,bank,False);used='rescue_fast' if accept else 'rescue_pass'
            counts.update(ale_samples=b,ale_predictions=pred,ale_weight_updates=updates,pulse_tests=2,pulse_accepts=int(pa),pulse_group_rms=2*b//(cfg.sps*cfg.blanker_group),rescue_blocks=1,rescue_accepted=int(accept),rescue_rejected=int(not accept),rescue_quality_products=2*b)
        else:
            active=action in ('slow','fast');raw_mask,raw_accept=legacy._pulse_mask(x,cfg,bank,action=='blank');used=action
            counts.update(pulse_tests=1,pulse_group_rms=b//(cfg.sps*cfg.blanker_group))
            rejected=bool(action=='blank' and np.mean(raw_mask)>=cfg.blank_max_fraction)
            if rejected:raw_mask[:]=False;raw_accept=False;used='pass';counts['blank_guard_rejections']+=1
            y,updates,pred=ale.process(x,raw_mask,cfg.ale_mu_nb if action=='slow' else cfg.ale_mu_sw,active)
            counts.update(ale_predictions=pred,ale_weight_updates=updates)
            if active:
                erase,pa=legacy._pulse_mask(y,cfg,bank,False)
                counts.update(ale_samples=b,pulse_tests=1,pulse_accepts=int(pa),pulse_group_rms=b//(cfg.sps*cfg.blanker_group))
            elif action=='blank':erase=raw_mask;counts['pulse_accepts']+=int(raw_accept)
            else:erase=np.zeros(b,bool)
        previous_rejected=rejected
        aa=a//cfg.sps;zz=(a+b)//cfg.sps;values=y.reshape(-1,cfg.sps)
        soft[aa:zz]=values.mean(axis=1);old[aa:zz]=erase.reshape(-1,cfg.sps).any(axis=1)
        rms=np.sqrt(np.mean(values**2,axis=1));refined[aa:zz]=old[aa:zz]&(rms>2*np.median(rms));actions.append(used)
        expert_wall+=time.perf_counter()-expert_start
        if use_risk:risk_log.append(dict(block=bi,observed_end_sample=a+b,power=now[0],lag1=now[1],trigger=event_risk,risk=risk,rescue=bool(use_rescue and risk),rho=rho,accepted=accept))
    counts['ale_tap_products']=cfg.ale_taps*(2*counts['ale_predictions']+counts['ale_weight_updates'])
    return dict(soft=soft,flags={'legacy_any':old,'block_refine2':refined},counts=dict(counts),decisions=decisions,actions_by_block=actions,
        risk_log=risk_log,block_buffer_ms=b/cfg.fs*1000,timing_mode='current_buffered_block',
        simulation_policy_wall_s=time.perf_counter()-started,simulation_router_predict_wall_s=router_wall,
        simulation_expert_and_mask_wall_s=expert_wall,simulation_detector_wall_s=detector_wall,
        timing_scope='serial policy; shared frontend excluded; sample-product counters are partial work, not energy or total MACs')
