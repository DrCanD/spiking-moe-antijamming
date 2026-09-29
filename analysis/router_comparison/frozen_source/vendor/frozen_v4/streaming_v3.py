"""New causal streaming candidate, distinct from historical frame receiver.

Every router observes one COMPLETE short block, updates its decision at block
end, and applies it to the NEXT block. Both front ends retain delta-reference,
refractory, resonator and threshold-crossing state. Expert time axes are never
compacted. Full-block pulse masking entails one block of buffering, explicitly
reported below. No receiver function accepts transmitted bits or jammer labels.

This is a preregisterable candidate, not a proven improvement or reproduction
of Ding/HybNet. Event thresholds are fixed by StreamSettings; tune on DEV only.
"""
from dataclasses import dataclass, asdict, replace
from collections import Counter
import time
import numpy as np
import numba
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
try:
    from . import core
except ImportError:
    import core

KINDS = ('spike', 'analog_delta', 'fft')


@dataclass
class StreamSettings:
    block_ms: float = 5.0
    dwell_ms: float = 30.0
    dwell_jitter: float = 0.25
    n_train_streams_per_class: int = 20
    train_blocks: int = 5
    warmup_blocks: int = 1
    rf_trees: int = 100
    rf_depth: int = 10
    change_high: float = 0.75
    change_low: float = 0.30
    min_decision_blocks: int = 1
    max_hold_blocks: int = 8
    periodic_blocks: int = 1
    transition_window_ms: float = 10.0
    jsr_db: float = 12.0
    relative_pulse_jsr_db: tuple = (-6.0, 0.0, 6.0)
    kinds: tuple = KINDS


def settings_from(value=None):
    if value is None:
        return StreamSettings()
    if isinstance(value, StreamSettings):
        return value
    if hasattr(value, '__dataclass_fields__'):
        value = asdict(value)
    allowed = set(StreamSettings.__dataclass_fields__)
    unknown = set(value) - allowed
    if unknown:
        raise ValueError('Unknown streaming settings: '+str(sorted(unknown)))
    return StreamSettings(**value)


def _validate(cfg, s):
    b = int(round(s.block_ms * cfg.fs / 1000 / cfg.sps)) * cfg.sps
    if b < max(cfg.n_subframes, cfg.sps * cfg.blanker_group * 2):
        raise ValueError('Streaming block too short for features and pulse groups')
    if not (0 <= s.change_low < s.change_high):
        raise ValueError('Need 0 <= change_low < change_high')
    if s.min_decision_blocks < 1 or s.max_hold_blocks < s.min_decision_blocks or s.periodic_blocks < 1:
        raise ValueError('Invalid decision interval')
    if s.n_train_streams_per_class < 1 or s.train_blocks < 2 or s.warmup_blocks < 0:
        raise ValueError('Invalid training size')
    if s.dwell_ms <= s.block_ms or not (0 <= s.dwell_jitter < .9):
        raise ValueError('Invalid dwell setting')
    if not s.relative_pulse_jsr_db:
        raise ValueError('Empty relative pulse power list')
    if not s.kinds or not set(s.kinds) <= set(KINDS):
        raise ValueError('Unsupported front end')
    return b


@numba.njit(cache=True, fastmath=False)
def _advance_frontend(x, rot, tau, threshold, enc_threshold, refractory,
                      q, prev, ref, last, sample_index, initialized, n_subframes):
    n = len(x); bands = len(q); sf_len = max(1, n // n_subframes)
    spikes = np.zeros(n, np.int8)
    counts = np.zeros((n_subframes, bands))
    power = np.zeros((n_subframes, bands))
    binsizes = np.zeros(n_subframes, np.int64)
    for t in range(n):
        idx = sample_index + t
        event = 0
        if not initialized:
            ref = x[t]; initialized = True
        elif idx-last > refractory:
            d = x[t]-ref
            if d > enc_threshold:
                event = 1; ref += enc_threshold; last = idx
            elif d < -enc_threshold:
                event = -1; ref -= enc_threshold; last = idx
        spikes[t] = event
        sf = min(t//sf_len, n_subframes-1); binsizes[sf] += 1
        for k in range(bands):
            q[k] = q[k]*rot[k]+event/tau
            active = q[k].imag >= threshold
            if active and not prev[k]:
                counts[sf,k] += 1
            prev[k] = active
            power[sf,k] += q[k].real*q[k].real+q[k].imag*q[k].imag
    return spikes, counts, power, binsizes, ref, last, sample_index+n, initialized


class StatefulFrontEnd:
    """Shared recurrence, two independent readouts; no q reset at crossings."""
    def __init__(self, cfg):
        self.cfg = cfg
        self.om = 2*np.pi*np.linspace(cfg.rf_f_lo_hz,cfg.rf_f_hi_hz,cfg.rf_bands)/cfg.fs
        self.rot = (1.-1./cfg.rf_tau)*np.exp(1j*self.om)
        self.q = np.zeros(cfg.rf_bands, complex)
        self.prev = np.zeros(cfg.rf_bands, bool)
        self.ref = 0.; self.last = -cfg.enc_refractory-1
        self.sample_index = 0; self.initialized = False

    def advance(self, x):
        c = self.cfg
        r = _advance_frontend(np.ascontiguousarray(x,dtype=np.float64),self.rot,
             c.rf_tau,c.rf_th,c.router_theta,c.enc_refractory,self.q,self.prev,
             self.ref,self.last,self.sample_index,self.initialized,c.n_subframes)
        spikes,counts,power,binsizes,self.ref,self.last,self.sample_index,self.initialized = r
        return spikes, counts, power, binsizes

    def features(self, x):
        spikes, counts, power, sizes = self.advance(x)
        c = self.cfg
        first = list(core.spike_stats(spikes,c.n_subframes))+[core.isi_drift(spikes,c.n_subframes)]
        rates = counts/(np.maximum(sizes,1)[:,None]*self.om[None,:]/(2*np.pi))
        pmean = power/np.maximum(sizes,1)[:,None]
        return {'spike': np.asarray(first+list(core.rf_features(rates))),
                'analog_delta': np.asarray(first+list(core.rf_features(pmean))),
                'fft': np.asarray(core.extract_conventional_features(x,c.n_subframes))}


class StreamBank:
    def __init__(self,cfg,settings,models,scalers,metadata,blank_mean,blank_std):
        self.cfg=cfg; self.settings=settings; self.models=models; self.scalers=scalers
        self.metadata=metadata; self.blank_mean=float(blank_mean); self.blank_std=float(blank_std)

    def predict_features(self,kind,features):
        X = np.asarray(features,dtype=float).reshape(1,-1)
        p = self.models[kind].predict_proba(self.scalers[kind].transform(X))[0]
        j = int(np.argmax(p))
        return str(self.models[kind].classes_[j]), float(p[j])

    def standardized(self,kind,x):
        return self.scalers[kind].transform(np.asarray(x).reshape(1,-1))[0]


def train_stream_bank(cfg,seed,settings=None,conditions='random',model_seed=None):
    """Single-jammer TRAIN streams; mixture and switching tests are unseen.

    Windows within one training stream are correlated; metadata preserves
    stream IDs. No window-level train/test split is performed here.
    """
    s=settings_from(settings); b=_validate(cfg,s)
    model_seed=int(cfg.rf_seed if model_seed is None else model_seed)
    if not (0 <= model_seed < 2**32):
        raise ValueError('model_seed must fit sklearn uint32 random_state')
    rows={k:[] for k in s.kinds}; labels=[]; stream_ids=[]; plan=[]; clean_rms=[]
    rng=np.random.default_rng(seed); start=time.perf_counter()
    localcfg=replace(cfg,n_sym_test=(s.train_blocks+s.warmup_blocks)*b//cfg.sps)
    classes=['none']+core.JT
    for ci,cls in enumerate(classes):
        for j in range(s.n_train_streams_per_class):
            panel_seed=int(np.random.SeedSequence([int(seed),ci,j,771]).generate_state(1,dtype=np.uint64)[0])
            jsr=float(rng.choice(cfg.router_train_jsr))
            f=core.generate_frame(localcfg,conditions,() if cls=='none' else (cls,),
                                 () if cls=='none' else (jsr,),panel_seed)
            front=StatefulFrontEnd(cfg)
            for wi,a in enumerate(range(0,len(f['rx']),b)):
                x=f['rx'][a:a+b]; feats=front.features(x)
                if cls=='none':
                    g=cfg.sps*cfg.blanker_group; n=len(x)//g
                    clean_rms.extend(np.sqrt(np.mean(x[:n*g].reshape(n,g)**2,axis=1)).tolist())
                if wi<s.warmup_blocks:
                    continue
                for k in rows:
                    rows[k].append(feats[k])
                labels.append(cls); stream_ids.append(panel_seed)
            plan.append(dict(seed=panel_seed,label=cls,jsr_db=jsr))
    models={}; scalers={}; train_acc={}
    for k in s.kinds:
        X=np.asarray(rows[k]); sc=StandardScaler().fit(X)
        model=RandomForestClassifier(n_estimators=s.rf_trees,max_depth=s.rf_depth,
              random_state=model_seed,n_jobs=1)
        model.fit(sc.transform(X),labels)
        models[k]=model; scalers[k]=sc
        train_acc[k]=float(np.mean(model.predict(sc.transform(X))==labels))
    meta=dict(seed=int(seed),model_seed=model_seed,conditions=conditions,settings=asdict(s),training_plan=plan,
              window_stream_ids=stream_ids,n_windows=len(labels),n_streams=len(plan),
              train_accuracy=train_acc,training_mixtures=False,
              elapsed_s=time.perf_counter()-start,
              features='shared delta 5 + crossing summary 4 / resonator power summary 4; fft legacy 7',
              train_cfg=asdict(cfg),threshold_selection='predeclared; no DEV or TEST tuning inside module')
    return StreamBank(cfg,s,models,scalers,meta,np.mean(clean_rms),max(np.std(clean_rms),1e-6))


def generate_stream(cfg,seed,settings=None,conditions='random'):
    """Independent test stream. Truth is returned for evaluator only."""
    s=settings_from(settings); b=_validate(cfg,s); rng=np.random.default_rng(seed)
    if conditions not in ('fixed','random'):
        raise ValueError(conditions)
    names=['none','narrowband','narrowband+pulse','narrowband','sweep','none']
    lengths=[]
    for _ in names:
        d=s.dwell_ms*(rng.uniform(1-s.dwell_jitter,1+s.dwell_jitter) if conditions=='random' else 1)
        n=max(2*b,int(round(d*cfg.fs/1000/cfg.sps))*cfg.sps)
        lengths.append(n)
    # Avoid placing transitions on decision boundaries, also in fixed panel.
    edges=[0]
    for n in lengths:
        end=edges[-1]+n
        if end%b==0:
            end+=cfg.sps
        edges.append(end)
    # Last stream end alone is padded to complete the final decision block.
    edges[-1]=int(np.ceil(edges[-1]/b))*b
    n=edges[-1]; bits=rng.integers(0,2,n//cfg.sps)
    tx=np.repeat(2*bits-1,cfg.sps).astype(float)
    noisy=tx+rng.normal(0,10**(-cfg.snr_db/20),n)
    signal_power=1.+10**(-cfg.snr_db/10) # ensemble power; never estimated using future samples
    jsr=s.jsr_db+(rng.uniform(-4,4) if conditions=='random' else 0)
    ratio=float(rng.choice(s.relative_pulse_jsr_db))
    fa=(rng.uniform(*cfg.nb_f_range) if conditions=='random' else .3)*cfg.fs/2
    fb=(rng.uniform(*cfg.sweep_f_range) if conditions=='random' else .95)*cfg.fs/2
    phase0=rng.uniform(0,2*np.pi); idx=np.arange(n)
    # Preserve tone phase through pulse insertion/removal and into sweep.
    phase=phase0+2*np.pi*fa*idx/cfg.fs
    sw0,sw1=edges[4],edges[5]
    u=np.arange(sw1-sw0)/cfg.fs; duration=(sw1-sw0)/cfg.fs
    phase[sw0:sw1]=phase0+2*np.pi*fa*sw0/cfg.fs+2*np.pi*(fa*u+.5*(fb-fa)*u*u/duration)
    tone=np.sqrt(2*signal_power*10**(jsr/10))*np.sin(phase)
    rx=noisy.copy(); rx[edges[1]:edges[5]]+=tone[edges[1]:edges[5]]
    duty=(rng.uniform(*cfg.pulse_duty_range) if conditions=='random' else .2)
    period=int(rng.integers(cfg.pulse_period_range[0],cfg.pulse_period_range[1]+1)) if conditions=='random' else 500
    offset=int(rng.integers(0,period)); env=((idx+offset)%period)<max(1,int(period*duty))
    pulse=rng.normal(0,np.sqrt(signal_power*10**((jsr+ratio)/10)/duty),n)*env
    rx[edges[2]:edges[3]]+=pulse[edges[2]:edges[3]]
    transitions=[dict(sample=int(edges[i]),symbol=int(edges[i]//cfg.sps),
                     before=names[i-1],after=names[i],end_sample=int(edges[i+1]))
                 for i in range(1,len(names))]
    return dict(rx=rx,bits=bits,transitions=transitions,
                segments=[dict(name=nm,start_sample=int(a),end_sample=int(z)) for nm,a,z in zip(names,edges[:-1],edges[1:])],
                metadata=dict(seed=int(seed),conditions=conditions,n_samples=n,jsr_db=float(jsr),
                pulse_relative_jsr_db=ratio,tone_hz=float(fa),sweep_end_hz=float(fb),
                pulse_duty=float(duty),pulse_period=period,block_samples=b,
                tone_phase_continuous=True,transitions_on_decision_boundary=False))


@numba.njit(cache=True, fastmath=False)
def _ale_block(x,mask,history,history_mask,w,taps,delay,mu,active,seen):
    h=len(history); xx=np.concatenate((history,x)); mm=np.concatenate((history_mask,mask))
    y=x.copy(); n_updates=0; n_predictions=0
    for t in range(len(x)):
        if seen+t<delay+taps or not active:
            continue
        idx=h+t; estimate=0.; norm=0.; clean=not mm[idx]
        for k in range(taps):
            j=idx-delay-k; v=xx[j]; estimate+=w[k]*v; norm+=v*v
            if mm[j]:
                clean=False
        err=x[t]-estimate; y[t]=err; n_predictions+=1
        if clean:
            gain=mu*err/(norm+1e-6)
            for k in range(taps):
                w[k]+=gain*xx[idx-delay-k]
            n_updates+=1
    history[:]=xx[-h:]; history_mask[:]=mm[-h:]
    return y,n_updates,n_predictions


class StatefulALE:
    def __init__(self,cfg):
        self.cfg=cfg; self.w=np.zeros(cfg.ale_taps)
        self.history=np.zeros(cfg.ale_taps+cfg.ale_delay)
        self.history_mask=np.zeros(len(self.history),bool); self.seen=0
    def process(self,x,mask,mu,active):
        y,updates,predictions=_ale_block(np.asarray(x,float),np.asarray(mask,bool),self.history,
           self.history_mask,self.w,self.cfg.ale_taps,self.cfg.ale_delay,mu,active,self.seen)
        self.seen+=len(x)
        return y,int(updates),int(predictions)


def _pulse_mask(x,cfg,bank,calibrated=False):
    g=cfg.blanker_group*cfg.sps; ng=len(x)//g; mask=np.zeros(len(x),bool)
    if ng==0:
        return mask,False
    rms=np.sqrt(np.mean(x[:ng*g].reshape(ng,g)**2,axis=1))
    threshold=bank.blank_mean+cfg.blanker_z*bank.blank_std if calibrated else cfg.pulse_k*np.median(rms)
    flag=rms>threshold; fraction=float(np.mean(flag))
    accept=calibrated or cfg.blank_min_fraction<fraction<cfg.blank_max_fraction
    if accept:
        mask[:ng*g]=np.repeat(flag,g)
    return mask,bool(accept and np.any(flag))


def _label_to_action(label):
    return {'narrowband':'slow','sweep':'fast','pulse':'blank'}.get(label,'pass')


def process_stream(rx,cfg,bank,settings=None,blank_guard=False):
    """Receiver-only entry point. No bits, condition names or transition labels.

    Output arrays retain every symbol position. Feature computations shared in
    this software are charged independently to each hypothetical receiver.
    CPU wall time is simulation wall time and is never an energy estimate.
    """
    s=settings_from(settings if settings is not None else bank.settings); b=_validate(cfg,s)
    rx=np.asarray(rx,dtype=np.float64)
    if len(rx)%b:
        raise ValueError('Provide complete blocks; padding is generator responsibility')
    front=StatefulFrontEnd(cfg); all_features=[]; feature_start=time.perf_counter()
    for a in range(0,len(rx),b):
        all_features.append(front.features(rx[a:a+b]))
    feature_wall=time.perf_counter()-feature_start
    ns=len(rx)//cfg.sps; out={}
    for kind in s.kinds:
        for policy in ('periodic','event'):
            name=kind+'_'+policy; start=time.perf_counter(); ale=StatefulALE(cfg)
            action='pass'; previous_call=-10**9; anchor=None; armed=True
            soft=np.zeros(ns); erased=np.zeros(ns,bool); decisions=[]; changes=[]; actions=[]
            counts=Counter(router_calls=0,ale_samples=0,ale_predictions=0,ale_weight_updates=0,
                           pulse_tests=0,pulse_accepts=0,pulse_group_rms=0,forced_refreshes=0,event_triggers=0,history_sample_updates=len(rx),
                           bank_sample_updates=(len(rx)*cfg.rf_bands if kind!='fft' else 0),
                           delta_sample_updates=(len(rx) if kind!='fft' else 0),
                           fft_calls=(len(rx)//b if kind=='fft' else 0))
            for bi,a in enumerate(range(0,len(rx),b)):
                x=rx[a:a+b]; active=action in ('slow','fast'); actions.append(action)
                # One block buffered: group medians may look ahead INSIDE this block.
                raw_mask,raw_accept=_pulse_mask(x,cfg,bank,calibrated=(action=='blank'))
                counts['pulse_tests']+=1; counts['pulse_group_rms']+=len(x)//(cfg.blanker_group*cfg.sps)
                rejected=bool(blank_guard and action=='blank' and np.mean(raw_mask)>=cfg.blank_max_fraction)
                if rejected:
                    raw_mask[:]=False; raw_accept=False; actions[-1]='pass'
                    counts['blank_guard_rejections']+=1
                y,updates,predictions=ale.process(x,raw_mask,cfg.ale_mu_nb if action=='slow' else cfg.ale_mu_sw,active)
                counts['ale_weight_updates']+=updates; counts['ale_predictions']+=predictions
                if active:
                    counts['ale_samples']+=len(x)
                    erase_mask,accepted=_pulse_mask(y,cfg,bank,False); counts['pulse_tests']+=1
                    counts['pulse_group_rms']+=len(x)//(cfg.blanker_group*cfg.sps)
                    counts['pulse_accepts']+=int(accepted)
                elif action=='blank':
                    erase_mask=raw_mask; counts['pulse_accepts']+=int(raw_accept)
                else:
                    erase_mask=np.zeros(b,bool)
                aa=a//cfg.sps; zz=(a+b)//cfg.sps
                soft[aa:zz]=y.reshape(-1,cfg.sps).mean(axis=1)
                erased[aa:zz]=erase_mask.reshape(-1,cfg.sps).any(axis=1)
                f=all_features[bi][kind]; z=bank.standardized(kind,f)
                change=float(np.sqrt(np.mean((z-anchor)**2))) if anchor is not None else 0.
                if change<=s.change_low:
                    armed=True
                due=(bi%s.periodic_blocks==0) if policy=='periodic' else (
                    anchor is None or bi-previous_call>=s.max_hold_blocks or
                    (armed and bi-previous_call>=s.min_decision_blocks and change>=s.change_high))
                if rejected and policy=='event':
                    due=True; counts['guard_refreshes']+=1
                if due:
                    forced=anchor is not None and bi-previous_call>=s.max_hold_blocks
                    trigger=anchor is not None and armed and change>=s.change_high
                    label,confidence=bank.predict_features(kind,f); new_action=_label_to_action(label)
                    counts['router_calls']+=1
                    counts['forced_refreshes']+=int(policy=='event' and forced)
                    counts['event_triggers']+=int(policy=='event' and trigger)
                    apply_sample=a+b
                    decisions.append(dict(observed_end_sample=apply_sample,apply_sample=apply_sample,
                         predicted_class=label,confidence=confidence,action=new_action,change_score=change))
                    if new_action!=action:
                        changes.append(dict(sample=apply_sample,before=action,after=new_action))
                    action=new_action; previous_call=bi; anchor=z.copy()
                    if policy=='event':
                        armed=False
            out[name]=dict(soft=soft,erasure_mask=erased,decisions=decisions,action_changes=changes,
                         actions_by_block=actions,counts=dict(counts),simulation_policy_wall_s=time.perf_counter()-start,
                         shared_feature_simulation_wall_s=feature_wall,block_buffer_delay_s=b/cfg.fs,
                         feature_window_s=b/cfg.fs,decision_applies_to_next_block=True,
                         time_axis_preserved=True)
    return out


def _metric(bits,soft,er):
    error=(np.asarray(soft)>0)!=bits; er=np.asarray(er,bool); kept=~er
    n=len(bits); ne=int(np.count_nonzero(error&kept)); nr=int(kept.sum()); erased=int(er.sum())
    return dict(n_bits=n,retained_bits=nr,erased_bits=erased,retained_bit_errors=ne,
                retained_ber=(ne/nr if nr else None),retention=nr/n if n else None,
                correct_retained_fraction=(nr-ne)/n if n else None,
                delivery_loss_count=ne+erased,delivery_loss_fraction=(ne+erased)/n if n else None)


def run_stream_trial(cfg,bank,seed,settings=None,conditions='random'):
    """Evaluator; JSON-safe records only. Truth never crosses process_stream API."""
    s=settings_from(settings if settings is not None else bank.settings)
    f=generate_stream(cfg,seed,s,conditions); start=time.perf_counter()
    outputs=process_stream(f['rx'],cfg,bank,s); bits=f['bits']; methods={}
    raw_soft=f['rx'].reshape(-1,cfg.sps).mean(axis=1); raw_err=(raw_soft>0)!=bits
    window=int(round(s.transition_window_ms*cfg.fs/1000/cfg.sps))
    for name,o in outputs.items():
        m=_metric(bits,o['soft'],o['erasure_mask']); m.update(counts=o['counts'],
             simulation_policy_wall_s=o['simulation_policy_wall_s'],
             block_buffer_delay_s=o['block_buffer_delay_s'],feature_window_s=o['feature_window_s'],
             decisions=o['decisions'],action_changes=o['action_changes'],actions_by_block=o['actions_by_block'])
        m['segments']=[]
        b=f['metadata']['block_samples']
        applied_actions=np.repeat(np.asarray(o['actions_by_block']),b//cfg.sps)
        for seg in f['segments']:
            a=seg['start_sample']//cfg.sps; z=seg['end_sample']//cfg.sps
            mm=_metric(bits[a:z],o['soft'][a:z],o['erasure_mask'][a:z]); mm.update(seg)
            mm['intervention_symbols']=int(np.count_nonzero(applied_actions[a:z]!='pass'))
            mm['intervention_fraction']=mm['intervention_symbols']/max(1,z-a)
            mm['raw_bit_errors']=int(raw_err[a:z].sum())
            mm['excess_delivery_loss_vs_raw']=mm['delivery_loss_count']-mm['raw_bit_errors']
            m['segments'].append(mm)
        m['transitions']=[]
        for tr in f['transitions']:
            a=tr['symbol']; end=tr['end_sample']//cfg.sps; z=min(end,a+window)
            tm=_metric(bits[a:z],o['soft'][a:z],o['erasure_mask'][a:z]); tm.update(tr)
            calls=[d['apply_sample'] for d in o['decisions'] if tr['sample']<=d['apply_sample']<tr['end_sample']]
            changes=[d['sample'] for d in o['action_changes'] if tr['sample']<=d['sample']<tr['end_sample']]
            tm['first_post_transition_decision_delay_ms']=(1000*(calls[0]-tr['sample'])/cfg.fs if calls else None)
            tm['first_post_transition_action_change_delay_ms']=(1000*(changes[0]-tr['sample'])/cfg.fs if changes else None)
            tm['decision_censored']=not bool(calls); tm['action_change_censored']=not bool(changes)
            tm['censor_time_ms']=1000*(tr['end_sample']-tr['sample'])/cfg.fs
            tm['window_raw_bit_errors']=int(raw_err[a:z].sum())
            tm['window_excess_delivery_loss_vs_raw']=tm['delivery_loss_count']-tm['window_raw_bit_errors']
            tm['delay_interpretation']='scheduling/action change only; not recovery or correct-expert latency'
            m['transitions'].append(tm)
        methods[name]=m
    methods['raw_pass']=_metric(bits,raw_soft,np.zeros(len(bits),bool))
    return dict(seed=int(seed),conditions=conditions,settings=asdict(s),metadata=f['metadata'],
                transitions=f['transitions'],methods=methods,elapsed_s=time.perf_counter()-start,
                claim_status='new streaming candidate; no superiority assumed',
                latency_note='full block buffer + measured software time; decision from preceding block',
                expert_rule='persistent common ALE weights; inactive weights frozen; raw time history always advances; update only if target and all delayed taps unmasked; relative residual pulse mask after ALE',
                protocol_note='single-jammer training; compound transition sequence unseen in training; no transmitted-bit access by receiver')


def self_test():
    """Integrity gates: partition state equality, prefix causality, no compaction."""
    cfg=core.Config(rf_trees=8,ale_taps=16,ale_delay=10)
    rng=np.random.default_rng(9031); x=rng.normal(0,3,4700)
    whole=StatefulFrontEnd(cfg); sw,cw,pw,_=whole.advance(x)
    split=StatefulFrontEnd(cfg); sp=[]; ct=np.zeros(cfg.rf_bands); pt=np.zeros(cfg.rf_bands)
    for a,z in [(0,197),(197,1203),(1203,2501),(2501,len(x))]:
        ss,cc,pp,_=split.advance(x[a:z]); sp.append(ss); ct+=cc.sum(axis=0); pt+=pp.sum(axis=0)
    assert np.array_equal(sw,np.concatenate(sp))
    assert np.array_equal(whole.q,split.q) and np.array_equal(whole.prev,split.prev)
    assert whole.ref==split.ref and whole.last==split.last
    assert np.array_equal(cw.sum(axis=0),ct)
    assert np.allclose(pw.sum(axis=0),pt,rtol=1e-13,atol=1e-13)
    mask=np.zeros(len(x),bool); mask[1100:1400]=True
    a1=StatefulALE(cfg); y1,_,_=a1.process(x,mask,.02,True)
    a2=StatefulALE(cfg); ya,_,_=a2.process(x[:1777],mask[:1777],.02,True)
    yb,_,_=a2.process(x[1777:],mask[1777:],.02,True)
    assert len(y1)==len(x) and np.array_equal(y1,np.concatenate([ya,yb]))
    assert np.array_equal(a1.w,a2.w) and np.array_equal(a1.history,x[-len(a1.history):])
    s=StreamSettings(block_ms=1,dwell_ms=3,n_train_streams_per_class=1,train_blocks=2,
                     warmup_blocks=0,rf_trees=4,kinds=('spike','analog_delta'))
    bank=train_stream_bank(cfg,90821,s)
    f=generate_stream(cfg,90822,s); b=f['metadata']['block_samples']
    p=process_stream(f['rx'][:4*b],cfg,bank,s)
    extended=f['rx'].copy(); extended[4*b:]+=123
    q=process_stream(extended,cfg,bank,s)
    for key in p:
        n=len(p[key]['soft'])
        assert np.array_equal(p[key]['soft'],q[key]['soft'][:n])
        assert np.array_equal(p[key]['erasure_mask'],q[key]['erasure_mask'][:n])
        assert p[key]['decisions']==[d for d in q[key]['decisions'] if d['apply_sample']<=4*b]
    record=run_stream_trial(cfg,bank,90823,s)
    import json, inspect
    json.dumps(record,allow_nan=False)
    assert set(inspect.signature(process_stream).parameters)=={'rx','cfg','bank','settings'}
    return dict(partition_state_equal=True,ale_partition_equal=True,prefix_causal_at_block_boundaries=True,
                time_axis_preserved=True,no_truth_receiver_api=True,json_safe=True,
                methods=list(record['methods']),elapsed_s=record['elapsed_s'])


if __name__=='__main__':
    import json
    print(json.dumps(self_test(),indent=2))
