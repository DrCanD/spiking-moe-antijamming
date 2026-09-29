"""CPU diagnostic reproduction of the final anti-jamming receiver.

All legacy formulas originate in reference_core.py. This module accelerates the
sample loops without fastmath and adds isolated experimental comparisons.
Receivers never accept transmitted bits or jammer identities. `conv_full` uses
conventional features in BOTH routing stages; `rule_only` uses no classifier.
Legacy pulse-first cascading concatenates surviving samples and retains this
known time-axis defect deliberately. Outputs restore the original symbol axis.
"""
from dataclasses import dataclass, field, asdict
from collections import Counter
import time
import numpy as np
import numba
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
try:
    from . import reference_core as ref
except ImportError:
    import reference_core as ref


@dataclass
class Config:
    fs: float = 1e6
    symbol_rate: float = 1e5
    n_sym_test: int = 10000
    snr_db: float = 10.0
    rho_min: float = 0.25
    blank_min_fraction: float = 0.01
    blank_max_fraction: float = 0.6
    pulse_k: float = 2.0
    ale_taps: int = 128
    ale_delay: int = 20
    ale_mu_nb: float = 0.02
    ale_mu_sw: float = 0.05
    blanker_z: float = 3.0
    blanker_group: int = 5
    blanker_cal_seed_base: int = 600
    blanker_cal_n: int = 30
    blanker_cal_samples: int = 10000
    router_train_jsr: list = field(default_factory=lambda: list(range(0,21,2)))
    router_n_train_clean: int = 50
    router_n_train_per: int = 5
    router_seed_base: int = 60000
    router_theta: float = 2.0
    enc_refractory: int = 2
    rf_bands: int = 16
    rf_f_lo_hz: float = 25e3
    rf_f_hi_hz: float = 475e3
    rf_tau: float = 15.0
    rf_th: float = 0.12
    n_subframes: int = 20
    rf_trees: int = 100
    rf_depth: int = 10
    rf_seed: int = 42
    n_jobs: int = 1
    nb_f_range: tuple = (0.05, 0.95)
    sweep_f_range: tuple = (0.05, 0.95)
    sweep_dur_range: tuple = (0.5, 2.0)
    pulse_duty_range: tuple = (0.1, 0.4)
    pulse_period_range: tuple = (300, 800)

    @property
    def sps(self):
        return int(self.fs/self.symbol_rate)


JT = list(ref.JT)
SNN_FEATURES = ['fin','isi','fano','sfr','isi_drift','rf_sharp','rf_hf','rf_drift','rf_adrift']
CONV_FEATURES = ['rms','variance','excess_kurtosis','spectral_flatness','peak_ratio','subframe_energy_variance','zero_crossing_rate']
METHODS = ('none','ale02','ale05','hard','override','cascade','full','conv_full','rule_only')
spike_stats = ref.spike_stats
isi_drift = ref.isi_drift
rf_features = ref.rf_features
extract_conventional_features = ref.extract_conventional_features
soft_symbols = ref.soft_symbols


@numba.njit(cache=True, fastmath=False)
def delta_encode(signal, threshold=2.0, refractory=2):
    n=len(signal); spikes=np.zeros(n,np.int8)
    state=signal[0]; last=-refractory-1
    for i in range(1,n):
        if i-last <= refractory:
            continue
        d=signal[i]-state
        if d > threshold:
            spikes[i]=1; state += threshold; last=i
        elif d < -threshold:
            spikes[i]=-1; state -= threshold; last=i
    return spikes


@numba.njit(cache=True, fastmath=False)
def _rf_counts(spikes, rot, tau, threshold, nsf):
    n=len(spikes); bands=len(rot); sf_len=n//nsf
    counts=np.zeros((nsf,bands),np.float64)
    # Each band is independent; its sample recurrence is evaluated in order.
    for k in range(bands):
        z=0.0+0.0j; prev=False
        for t in range(n):
            z=z*rot[k]+float(spikes[t])/tau
            active=z.imag >= threshold
            if active and not prev:
                counts[min(t//sf_len,nsf-1),k] += 1.0
            prev=active
    return counts


def rf_bank(spikes, cfg):
    om=2*np.pi*np.linspace(cfg.rf_f_lo_hz,cfg.rf_f_hi_hz,cfg.rf_bands)/cfg.fs
    rot=(1.0-1.0/cfg.rf_tau)*np.exp(1j*om)
    counts=_rf_counts(np.asarray(spikes,np.int8),rot,cfg.rf_tau,cfg.rf_th,cfg.n_subframes)
    return counts/((len(spikes)//cfg.n_subframes)*om/(2*np.pi))[None,:]


_ale_fast = numba.njit(cache=True, fastmath=False)(ref._ale_py)


def apply_ale(signal,cfg,mu):
    return _ale_fast(np.ascontiguousarray(signal,dtype=np.float64),cfg.ale_taps,cfg.ale_delay,mu,1e-6)


def snn_features(signal,cfg):
    x=np.ascontiguousarray(signal,np.float64)
    if len(x) < cfg.n_subframes:
        raise ValueError('signal shorter than number of feature subframes')
    spikes=delta_encode(x,cfg.router_theta,cfg.enc_refractory)
    return np.asarray(list(spike_stats(spikes,cfg.n_subframes))+[isi_drift(spikes,cfg.n_subframes)]+list(rf_features(rf_bank(spikes,cfg))),np.float64)


def features(signal,cfg,kind):
    if kind=='snn':
        return snn_features(signal,cfg)
    if kind=='conv':
        return extract_conventional_features(signal,cfg.n_subframes)
    raise ValueError(kind)


def generate_frame(cfg,cond,components=(),jsrs=(),seed=1,bits=None):
    """Generate legacy channel; component JSRs reference SAME noisy signal.

    Supplied bits support new coded experiments. In that case the random-bit
    draw is skipped and the RNG begins at AWGN. This is a new waveform panel,
    not historical-bit reproduction. Clean frames use components=().
    """
    if cond not in ('fixed','random'):
        raise ValueError(cond)
    components=tuple(components)
    if np.isscalar(jsrs):
        jsrs=(float(jsrs),)*len(components)
    else:
        jsrs=tuple(jsrs)
    if len(components)!=len(jsrs):
        raise ValueError('one JSR is required for every component')
    rng=np.random.default_rng(seed)
    cc=ref.ChannelConfig(fs=cfg.fs,symbol_rate=cfg.symbol_rate,snr_db=cfg.snr_db,n_symbols=cfg.n_sym_test,seed=seed)
    if bits is None:
        bits,symbols,tx,sps=ref.generate_bpsk(cc,rng)
    else:
        bits=np.asarray(bits,np.int64).copy()
        if bits.ndim != 1 or not np.all((bits==0)|(bits==1)):
            raise ValueError('bits must be a one-dimensional binary array')
        sps=cfg.sps; symbols=2*bits-1
        tx=np.repeat(symbols,sps).astype(np.float64)
        tx/=np.sqrt(np.mean(tx**2))
    noisy=ref.add_awgn(tx,cfg.snr_db,rng)
    jams=[ref.jam_component(vars(ref),cfg,cond,noisy,jt,jsr,cfg.fs,rng) for jt,jsr in zip(components,jsrs)]
    rx=noisy.copy()
    # Matches original left-associative noisy+j1+j2 arithmetic.
    for jam in jams:
        rx=rx+jam
    return dict(bits=bits,symbols=symbols,tx=tx,rx=rx,noisy=noisy,jams=jams,sps=sps,seed=int(seed),components=components,jsrs=jsrs,cond=cond)


class RouterSet:
    def __init__(self,cfg,cond,models,scalers,training_meta):
        self.cfg=cfg; self.cond=cond; self.models=models; self.scalers=scalers; self.training_meta=training_meta

    def predict_features(self,kind,X):
        X=np.asarray(X).reshape(-1,len(SNN_FEATURES if kind=='snn' else CONV_FEATURES))
        p=self.models[kind].predict_proba(self.scalers[kind].transform(X))
        inds=np.argmax(p,axis=1)
        return self.models[kind].classes_[inds],p[np.arange(len(inds)),inds]

    def predict_signal(self,kind,signal):
        label,p=self.predict_features(kind,features(signal,self.cfg,kind)[None,:])
        return str(label[0]),float(p[0])


def fit_routers(cfg,cond,progress=None):
    """Fit matched forests on identical legacy 270-frame training panel."""
    started=time.perf_counter(); rows={'snn':[],'conv':[]}; labels=[]; seeds=[]
    plan=[('none',0,cfg.router_seed_base+i) for i in range(cfg.router_n_train_clean)]
    for jt in JT:
        for ji,jsr in enumerate(cfg.router_train_jsr):
            for i in range(cfg.router_n_train_per):
                plan.append((jt,jsr,cfg.router_seed_base+1000+i+JT.index(jt)*700+ji*50))
    for idx,(jt,jsr,seed) in enumerate(plan):
        f=generate_frame(cfg,cond,() if jt=='none' else (jt,),() if jt=='none' else (jsr,),seed)
        for kind in rows:
            rows[kind].append(features(f['rx'],cfg,kind))
        labels.append(jt); seeds.append(seed)
        if progress and ((idx+1)%25==0 or idx+1==len(plan)):
            progress(dict(cond=cond,done=idx+1,total=len(plan),elapsed_s=time.perf_counter()-started))
    models={};scalers={};acc={}
    for kind in rows:
        X=np.asarray(rows[kind]);sc=StandardScaler().fit(X)
        model=RandomForestClassifier(n_estimators=cfg.rf_trees,max_depth=cfg.rf_depth,random_state=cfg.rf_seed,n_jobs=1)
        model.fit(sc.transform(X),labels)
        acc[kind]=float(np.mean(model.predict(sc.transform(X))==np.asarray(labels)))
        models[kind]=model;scalers[kind]=sc
    meta=dict(cond=cond,n_train=len(plan),train_accuracy=acc,seeds=seeds,training_plan=plan,features={'snn':SNN_FEATURES,'conv':CONV_FEATURES},elapsed_s=time.perf_counter()-started,n_jobs=1)
    return RouterSet(cfg,cond,models,scalers,meta)


class Receiver:
    def __init__(self,cfg,routers):
        self.cfg=cfg; self.routers=routers
        self.blanker=ref.PulseBlankerExpert(blank_threshold_z=cfg.blanker_z,group_size=cfg.blanker_group)
        cal=[]
        for i in range(cfg.blanker_cal_n):
            cc=ref.ChannelConfig(fs=cfg.fs,symbol_rate=cfg.symbol_rate,snr_db=cfg.snr_db,n_symbols=cfg.blanker_cal_samples//cfg.sps,seed=cfg.blanker_cal_seed_base+i)
            cal.append(ref.simulate_channel(cc,'none')['rx'][:cfg.blanker_cal_samples])
        # Same operations as original calibrate; suppress its print.
        all_rms=np.concatenate([self.blanker._compute_group_rms(x,cfg.sps) for x in cal])
        self.blanker.baseline_rms_mean=np.mean(all_rms)
        self.blanker.baseline_rms_std=max(np.std(all_rms),1e-6)

    def pulse_check(self,y):
        cfg=self.cfg;sps=cfg.sps;soft=soft_symbols(y,sps)
        win=cfg.blanker_group*sps;ng=len(y)//win
        rms=np.sqrt(np.mean(y[:ng*win].reshape(ng,win)**2,axis=1))
        flag=rms>cfg.pulse_k*np.median(rms);frac=float(np.mean(flag)) if ng else 0.0
        er=np.zeros(len(soft),bool)
        for g in np.where(flag)[0]:
            er[g*cfg.blanker_group:min((g+1)*cfg.blanker_group,len(soft))]=True
        return soft,er,frac

    def evaluate_all(self,rx,methods=METHODS,predictions=None):
        """Cache common numerical work; charge every method logically in counts.

        predictions optional {'snn':(label,pmax),'conv':(...)} supports batch
        evaluation. It accepts only outputs of routers, never ground truth.
        """
        cfg=self.cfg;sps=cfg.sps;rx=np.asarray(rx,np.float64)
        n_sym=len(rx)//sps
        if len(rx)%sps:
            raise ValueError('frame must contain complete symbols')
        cache={};preds={} if predictions is None else dict(predictions)
        def ale(x,mu,tag):
            key=('ale',tag,float(mu))
            if key not in cache:
                cache[key]=apply_ale(x,cfg,mu)
            return cache[key]
        def predict(kind,x,tag):
            key=(kind,tag)
            if tag=='full' and kind in preds:
                return preds[kind]
            if key not in cache:
                cache[key]=self.routers.predict_signal(kind,x)
            return cache[key]
        def blank():
            if 'blank' not in cache:
                cache['blank']=self.blanker.correct_symbols(rx,sps)
            return cache['blank']
        out={}
        for method in methods:
            if method not in METHODS:
                raise ValueError(method)
            counts=Counter(ale_calls=0,ale_samples=0,router_calls=0,router_samples=0,pulse_tests=0,blanker_calls=0,override_accepted=0,cascade_accepted=0)
            er=np.zeros(n_sym,bool);soft=None;rho=None;pred1=None;pred2=None;pmax=None;gap=False
            def charged_ale(x,mu,tag):
                counts['ale_calls']+=1;counts['ale_samples']+=len(x)
                return ale(x,mu,tag)
            if method=='none':
                soft=soft_symbols(rx,sps);path='pass'
            elif method in ('ale02','ale05'):
                mu=cfg.ale_mu_nb if method=='ale02' else cfg.ale_mu_sw
                soft=soft_symbols(charged_ale(rx,mu,'full'),sps);path=method
            elif method=='rule_only':
                # Predefined classifier-free control: probe fast ALE; accept
                # only rho > legacy threshold; test pulses on accepted residual
                # or on raw input otherwise. Uses original within-frame rule.
                e=charged_ale(rx,cfg.ale_mu_sw,'full')
                rho=float(1-np.mean(e**2)/(np.mean(rx**2)+1e-12))
                y=e if rho>cfg.rho_min else rx
                counts['override_accepted']=int(rho>cfg.rho_min)
                path='rule->ALE' if rho>cfg.rho_min else 'rule->pass'
                soft,candidate,frac=self.pulse_check(y);counts['pulse_tests']+=1
                if cfg.blank_min_fraction<frac<cfg.blank_max_fraction:
                    er=candidate;counts['cascade_accepted']+=1;path+='->blank'
            else:
                kind='conv' if method=='conv_full' else 'snn'
                pred1,pmax=predict(kind,rx,'full');cls=pred1;path=pred1[:2]
                counts['router_calls']+=1;counts['router_samples']+=len(rx)
                use_override=method in ('override','full','conv_full')
                use_cascade=method in ('cascade','full','conv_full')
                if use_override and cls in ('none','broadband'):
                    e=charged_ale(rx,cfg.ale_mu_sw,'full')
                    rho=float(1-np.mean(e**2)/(np.mean(rx**2)+1e-12))
                    if rho>cfg.rho_min:
                        cls='sweep';path+=f'->override(rho={rho:.2f})';counts['override_accepted']+=1
                if cls in ('narrowband','sweep'):
                    mu=cfg.ale_mu_nb if cls=='narrowband' else cfg.ale_mu_sw
                    y=charged_ale(rx,mu,'full');path+='->ALE'
                    soft=soft_symbols(y,sps)
                    if use_cascade:
                        soft,candidate,frac=self.pulse_check(y);counts['pulse_tests']+=1
                        if cfg.blank_min_fraction<frac<cfg.blank_max_fraction:
                            er=candidate;path+='->blank';counts['cascade_accepted']+=1
                elif cls=='pulse':
                    pr=blank();counts['blanker_calls']+=1
                    er=pr['erasure_mask'].copy();soft=pr['soft'].copy();path+='->blank'
                    idx=pr['surviving_indices']
                    if use_cascade and len(idx)>=200:
                        sample_mask=np.repeat(~er,sps)
                        survivors=rx[sample_mask]
                        pred2,_=predict(kind,survivors,'survivors')
                        counts['router_calls']+=1;counts['router_samples']+=len(survivors)
                        if pred2 in ('narrowband','sweep'):
                            mu=cfg.ale_mu_nb if pred2=='narrowband' else cfg.ale_mu_sw
                            y=charged_ale(survivors,mu,'survivors')
                            soft[idx]=soft_symbols(y,sps)[:len(idx)]
                            path+=f'->{pred2[:2]}->ALE';gap=True;counts['cascade_accepted']+=1
                else:
                    soft=soft_symbols(rx,sps);path+='->pass'
            out[method]=dict(soft=soft,erasure_mask=er,path=path,counts=dict(counts),pred1=pred1,pred2=pred2,pmax=pmax,rho=rho,legacy_gap_concatenation=gap)
        return out
