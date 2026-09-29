"""Five matched feature sets, causal block receiver and independent rule control.

Only received samples enter inference. A complete 5 ms block is buffered;
router decisions affect the next block. ALE/front-end state persists across
all codewords. A precomputed feature cache is causal and prefix-tested.
"""
from collections import Counter
from dataclasses import replace,asdict
import time
import numpy as np
import numba
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import core
import streaming_v3 as legacy

KINDS=('spike','analog_delta','analog_raw','delta','fft')
POLICIES={'legacy8':dict(guard=False,hold=8,confirm=False,periodic=False),
          'guard8':dict(guard=True,hold=8,confirm=False,periodic=False),
          'guard4':dict(guard=True,hold=4,confirm=False,periodic=False),
          'guard2':dict(guard=True,hold=2,confirm=False,periodic=False),
          'confirm8':dict(guard=True,hold=8,confirm=True,periodic=False),
          'periodic':dict(guard=True,hold=8,confirm=False,periodic=True)}

@numba.njit(cache=True,fastmath=False)
def raw_power(x,rot,tau,q,n_subframes):
    out=np.zeros((n_subframes,len(q)));sizes=np.zeros(n_subframes,np.int64);step=max(1,len(x)//n_subframes)
    for t in range(len(x)):
        sf=min(t//step,n_subframes-1);sizes[sf]+=1
        for k in range(len(q)):
            q[k]=q[k]*rot[k]+x[t]/tau
            out[sf,k]+=q[k].real*q[k].real+q[k].imag*q[k].imag
    return out/np.maximum(sizes,1)[:,None]

class Features:
    def __init__(self,cfg):
        self.cfg=cfg;self.front=legacy.StatefulFrontEnd(cfg);self.raw_q=np.zeros(cfg.rf_bands,complex)
    def advance(self,x):
        c=self.cfg;spikes,counts,power,sizes=self.front.advance(x)
        first=list(core.spike_stats(spikes,c.n_subframes))+[core.isi_drift(spikes,c.n_subframes)]
        rates=counts/(np.maximum(sizes,1)[:,None]*self.front.om[None,:]/(2*np.pi))
        p=power/np.maximum(sizes,1)[:,None]
        rp=raw_power(np.ascontiguousarray(x,np.float64),self.front.rot,c.rf_tau,self.raw_q,c.n_subframes)
        return dict(spike=np.asarray(first+list(core.rf_features(rates))),
            analog_delta=np.asarray(first+list(core.rf_features(p))),
            analog_raw=np.asarray(first+list(core.rf_features(rp))),delta=np.asarray(first),
            fft=np.asarray(core.extract_conventional_features(x,c.n_subframes)))

def train(cfg,seed,model_seed,condition,n_train=20,trees=100):
    s=legacy.StreamSettings(n_train_streams_per_class=n_train,rf_trees=trees)
    b=legacy._validate(cfg,s);rng=np.random.default_rng(seed);start=time.perf_counter()
    rows={k:[] for k in KINDS};labels=[];plan=[];clean_rms=[]
    tc=replace(cfg,n_sym_test=(s.train_blocks+s.warmup_blocks)*b//cfg.sps)
    for ci,cls in enumerate(['none']+core.JT):
        for trial in range(n_train):
            sd=int(np.random.SeedSequence([seed,ci,trial,771]).generate_state(1,dtype=np.uint64)[0]);jsr=float(rng.choice(cfg.router_train_jsr))
            f=core.generate_frame(tc,condition,() if cls=='none' else (cls,),() if cls=='none' else (jsr,),sd)
            front=Features(cfg)
            for wi,a in enumerate(range(0,len(f['rx']),b)):
                x=f['rx'][a:a+b];feats=front.advance(x)
                if cls=='none':
                    g=cfg.sps*cfg.blanker_group;clean_rms.extend(np.sqrt(np.mean(x.reshape(-1,g)**2,axis=1)).tolist())
                if wi<s.warmup_blocks:continue
                for k in KINDS:rows[k].append(feats[k])
                labels.append(cls)
            plan.append(dict(seed=sd,label=cls,jsr=jsr))
    models={};scalers={};acc={}
    for k in KINDS:
        X=np.asarray(rows[k]);sc=StandardScaler().fit(X);m=RandomForestClassifier(n_estimators=trees,max_depth=10,random_state=model_seed,n_jobs=1)
        m.fit(sc.transform(X),labels);models[k]=m;scalers[k]=sc;acc[k]=float(np.mean(m.predict(sc.transform(X))==labels))
    metadata=dict(data_seed=seed,model_seed=model_seed,condition=condition,n_windows=len(labels),training_plan=plan,
        n_train_streams=len(plan),training_accuracy=acc,evaluation_data_used=False,training_mixtures=False,
        training_cfg=asdict(tc),feature_kinds=KINDS,elapsed_s=time.perf_counter()-start)
    return legacy.StreamBank(cfg,s,models,scalers,metadata,np.mean(clean_rms),max(np.std(clean_rms),1e-6))

def feature_cache(rx,cfg,block_samples):
    f=Features(cfg);start=time.perf_counter();feats=[]
    for a in range(0,len(rx),block_samples):feats.append(f.advance(rx[a:a+block_samples]))
    return feats,time.perf_counter()-start

def process_one(rx,cfg,bank,feats,kind,policy):
    """No truth, framing positions or channel labels accepted here."""
    control=kind in ('raw','rule');p=POLICIES[policy] if not control else None
    s=bank.settings;b=legacy._validate(cfg,s);ns=len(rx)//cfg.sps
    if len(rx)%b:raise ValueError('Complete decision blocks required')
    soft=np.zeros(ns);old=np.zeros(ns,bool);refined=np.zeros(ns,bool)
    ale=legacy.StatefulALE(cfg);action='pass';anchor=None;armed=True;previous_call=-10**9;burst=0
    decisions=[];actions=[];start=time.perf_counter();counts=Counter(router_calls=0,ale_samples=0,ale_predictions=0,ale_weight_updates=0,
        pulse_tests=0,pulse_accepts=0,pulse_group_rms=0,forced_refreshes=0,event_triggers=0,
        blank_guard_rejections=0,guard_refreshes=0,confirmation_refreshes=0,
        history_sample_updates=len(rx) if kind!='raw' else 0,
        bank_sample_updates=len(rx)*cfg.rf_bands if kind in ('spike','analog_delta','analog_raw') else 0,
        delta_sample_updates=len(rx) if kind in ('spike','analog_delta','analog_raw','delta') else 0,
        fft_calls=len(rx)//b if kind=='fft' else 0,quality_power_samples=len(rx) if kind!='raw' else 0)
    router_wall=0.;expert_wall=0.
    for bi,a in enumerate(range(0,len(rx),b)):
        x=rx[a:a+b];expert_start=time.perf_counter();rejected=False
        if kind=='raw':y=x;erase=np.zeros(b,bool);used='pass'
        elif kind=='rule':
            raw_mask,_=legacy._pulse_mask(x,cfg,bank,False)
            residual,updates,pred=ale.process(x,raw_mask,cfg.ale_mu_sw,True)
            rho=1-np.mean(residual**2)/(np.mean(x**2)+1e-12);accept=rho>cfg.rho_min;y=residual if accept else x
            erase,pa=legacy._pulse_mask(y,cfg,bank,False);used='rule_fast' if accept else 'rule_pass'
            counts.update(ale_samples=b,ale_predictions=pred,ale_weight_updates=updates,pulse_tests=2,pulse_accepts=int(pa),pulse_group_rms=2*b//(cfg.sps*cfg.blanker_group))
        else:
            active=action in ('slow','fast');raw_mask,raw_accept=legacy._pulse_mask(x,cfg,bank,action=='blank');used=action
            counts.update(pulse_tests=1,pulse_group_rms=b//(cfg.sps*cfg.blanker_group))
            rejected=bool(p['guard'] and action=='blank' and np.mean(raw_mask)>=cfg.blank_max_fraction)
            if rejected:raw_mask[:]=False;raw_accept=False;used='pass';counts['blank_guard_rejections']+=1
            y,updates,pred=ale.process(x,raw_mask,cfg.ale_mu_nb if action=='slow' else cfg.ale_mu_sw,active)
            counts.update(ale_predictions=pred,ale_weight_updates=updates)
            if active:
                erase,pa=legacy._pulse_mask(y,cfg,bank,False);counts.update(ale_samples=b,pulse_tests=1,pulse_accepts=int(pa),pulse_group_rms=b//(cfg.sps*cfg.blanker_group))
            elif action=='blank':erase=raw_mask;counts['pulse_accepts']+=int(raw_accept)
            else:erase=np.zeros(b,bool)
        aa=a//cfg.sps;zz=(a+b)//cfg.sps;values=y.reshape(-1,cfg.sps)
        soft[aa:zz]=values.mean(axis=1);old[aa:zz]=erase.reshape(-1,cfg.sps).any(axis=1)
        rms=np.sqrt(np.mean(values**2,axis=1));refined[aa:zz]=old[aa:zz]&(rms>2*np.median(rms));actions.append(used)
        expert_wall+=time.perf_counter()-expert_start
        if control:continue
        z=bank.standardized(kind,feats[bi][kind]);change=float(np.sqrt(np.mean((z-anchor)**2))) if anchor is not None else 0.
        if change<=s.change_low:armed=True
        event=anchor is not None and armed and bi-previous_call>=s.min_decision_blocks and change>=s.change_high
        force=anchor is not None and bi-previous_call>=p['hold'];confirm=bool(p['confirm'] and burst>0)
        due=p['periodic'] or anchor is None or force or event or rejected or confirm
        if due:
            rt=time.perf_counter();label,confidence=bank.predict_features(kind,feats[bi][kind]);router_wall+=time.perf_counter()-rt
            new_action=legacy._label_to_action(label);counts['router_calls']+=1
            counts['forced_refreshes']+=int(force and not p['periodic']);counts['event_triggers']+=int(event and not p['periodic'])
            counts['guard_refreshes']+=int(rejected and not p['periodic']);counts['confirmation_refreshes']+=int(confirm)
            burst=max(0,burst-1)
            if p['confirm'] and (new_action!=action or confidence<.75 or event):burst=max(burst,2)
            decisions.append(dict(block=bi,observed_end_sample=a+b,apply_sample=a+b,predicted_class=label,confidence=confidence,action=new_action,change_score=change,confirmation_remaining=burst))
            action=new_action;previous_call=bi;anchor=z.copy()
            if not p['periodic']:armed=False
    return dict(soft=soft,flags={'legacy_any':old,'block_refine2':refined},counts=dict(counts),decisions=decisions,
        actions_by_block=actions,block_buffer_ms=b/cfg.fs*1000,simulation_policy_wall_s=time.perf_counter()-start,
        simulation_router_predict_wall_s=router_wall,simulation_expert_and_mask_wall_s=expert_wall,
        timing_scope='serial policy loop; shared feature extraction excluded; not energy or deployment latency')
