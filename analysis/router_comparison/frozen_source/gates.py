"""Causal software gate candidates. Counters describe work, never measured energy."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[k]='1'
import sys
from pathlib import Path
from collections import Counter
import numpy as np
import numba
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor'))
import common_v5 as io
core=io.core; legacy=io.streaming_v3

@numba.njit(cache=True,fastmath=False)
def encode(x,threshold=2.,refractory=2):
    out=np.zeros(len(x),np.int8);ref=x[0];last=-refractory-1
    for t in range(1,len(x)):
        if t-last<=refractory:continue
        d=x[t]-ref
        if d>threshold:out[t]=1;ref+=threshold;last=t
        elif d<-threshold:out[t]=-1;ref-=threshold;last=t
    return out

@numba.njit(cache=True,fastmath=False)
def dense(sp,rot,q,prev,tau=15.,threshold=.12,nsf=20):
    n=len(sp);step=max(1,n//nsf);counts=np.zeros((nsf,len(q)));power=np.zeros_like(counts)
    for t in range(n):
        sf=min(t//step,nsf-1)
        for k in range(len(q)):
            q[k]=q[k]*rot[k]+float(sp[t])/tau
            active=q[k].imag>=threshold
            if active and not prev[k]:counts[sf,k]+=1
            prev[k]=active
            power[sf,k]+=q[k].real*q[k].real+q[k].imag*q[k].imag
    return counts,power

@numba.njit(cache=True,fastmath=False)
def lazy(sp,q,prev,stamp,safe,small,big,offset,tau=15.,threshold=.12,nsf=20):
    n=len(sp);step=max(1,n//nsf);counts=np.zeros((nsf,len(q)))
    updates=0;cmults=0;skips=0;reads=0;long_gaps=0
    for t in range(n):
        idx=offset+t;sf=min(t//step,nsf-1)
        for k in range(len(q)):
            if sp[t]==0 and safe[k]:
                skips+=1
                continue
            delta=idx-stamp[k];z=q[k];lo=delta&63;hi=delta>>6
            if lo:
                z=z*small[k,lo];cmults+=1;reads+=1
            if hi:long_gaps+=1
            j=0
            while hi:
                if hi&1:z=z*big[k,j];cmults+=1;reads+=1
                hi>>=1;j+=1
            z=z+float(sp[t])/tau;q[k]=z;stamp[k]=idx;updates+=1
            active=z.imag>=threshold
            if active and not prev[k]:counts[sf,k]+=1
            prev[k]=active
            # Conservative L1 envelope: no square/square-root in gate.
            safe[k]=(abs(z.real)+abs(z.imag)<threshold-1e-12)
    return counts,np.array([updates,cmults,skips,reads,long_gaps],np.int64)

class Resonators:
    def __init__(self,cfg,lazy_mode=False,start=0):
        self.cfg=cfg;self.om=2*np.pi*np.linspace(cfg.rf_f_lo_hz,cfg.rf_f_hi_hz,cfg.rf_bands)/cfg.fs
        self.rot=(1.-1./cfg.rf_tau)*np.exp(1j*self.om)
        self.q=np.zeros(cfg.rf_bands,complex);self.prev=np.zeros(cfg.rf_bands,bool)
        self.stamp=np.full(cfg.rf_bands,start-1,np.int64);self.safe=np.ones(cfg.rf_bands,bool)
        self.offset=start;self.lazy_mode=lazy_mode;self.work=Counter()
        self.small=self.rot[:,None]**np.arange(64)[None,:]
        self.big=self.rot[:,None]**(2.**np.arange(6,31))[None,:]
    def advance(self,sp):
        c=self.cfg;n=len(sp)
        if self.lazy_mode:
            cnt,w=lazy(sp,self.q,self.prev,self.stamp,self.safe,self.small,self.big,self.offset,c.rf_tau,c.rf_th,c.n_subframes)
            self.work.update(bank_updates=int(w[0]),rotation_real_products=4*int(w[1]),skipped_updates=int(w[2]),
                coefficient_reads=int(w[3]),long_gap_updates=int(w[4]),envelope_abs=2*int(w[0]),envelope_adds=int(w[0]),envelope_compares=int(w[0]),
                gate_sample_band_checks=n*c.rf_bands,threshold_checks=int(w[0]))
            power=None
        else:
            cnt,power=dense(sp,self.rot,self.q,self.prev,c.rf_tau,c.rf_th,c.n_subframes)
            self.work.update(bank_updates=n*c.rf_bands,rotation_real_products=4*n*c.rf_bands,threshold_checks=n*c.rf_bands)
        self.offset+=n
        sizes=np.full(c.n_subframes,max(1,n//c.n_subframes));sizes[-1]=n-sum(sizes[:-1])
        rates=cnt/(sizes[:,None]*self.om[None,:]/(2*np.pi))
        return rates,None if power is None else power/sizes[:,None]

def first_features(sp,cfg):
    return list(core.spike_stats(sp,cfg.n_subframes))+[core.isi_drift(sp,cfg.n_subframes)]

def fft1024(x,cfg):
    windows=x[:len(x)//1024*1024].reshape(-1,1024)*np.hanning(1024)
    mag=np.abs(np.fft.rfft(windows,axis=1)).mean(axis=0)
    mu=x.mean();v=x.var();sf=np.mean(x.reshape(cfg.n_subframes,-1)**2,axis=1)
    return np.array([np.sqrt(np.mean(x*x)),v,np.mean((x-mu)**4)/(v*v+1e-10)-3,
        np.exp(np.mean(np.log(mag[1:]+1e-10)))/(np.mean(mag[1:])+1e-10),
        mag[1:].max()/(mag[1:].mean()+1e-10),sf.var()/(sf.mean()+1e-10),np.sum(np.diff(np.sign(x))!=0)/len(x)])

def schedule(rx,sp,b,hold):
    """Only current buffered samples and past sentinel anchors enter decisions."""
    awake=[];reasons=[];last=-1000000;remaining=2;anchor=None;stats=[]
    for bi,a in enumerate(range(0,len(rx),b)):
        x=rx[a:a+b];ss=float(np.dot(x,x));den=max(ss-.5*(x[0]**2+x[-1]**2),1e-12)
        now=(max(ss/b,1e-12),float(np.dot(x[1:],x[:-1]))/den,float(np.count_nonzero(sp[a:a+b]))/b)
        event=anchor is not None and (now[0]/anchor[0]>=2 or now[0]/anchor[0]<=.5 or abs(now[1]-anchor[1])>=.2 or abs(now[2]-anchor[2])>=.03)
        if event:remaining=max(remaining,2)
        forced=bi-last>=hold
        run=remaining>0 or forced or anchor is None
        awake.append(run);reasons.append('event' if event else ('hold' if remaining>0 else ('watchdog' if forced else 'sleep')))
        if run:anchor=now;last=bi
        remaining=max(remaining-1,0);stats.append(now)
    return np.array(awake,bool),reasons,np.asarray(stats)

def extract(rx,cfg,awake=None,lazy_mode=False):
    """Features absent on sleeping blocks. Warm replay is explicitly charged."""
    b=5000;sp=encode(rx,cfg.router_theta,cfg.enc_refractory);nb=len(rx)//b
    if awake is None:awake=np.ones(nb,bool)
    reson=None;rows={k:[] for k in ('spike','analog_delta','fft1024')};work=Counter(delta_samples=len(rx),warm_samples=0,active_blocks=int(sum(awake)),total_blocks=nb)
    last_active=False;counts_all=[]
    for bi,a in enumerate(range(0,len(rx),b)):
        if not awake[bi]:
            for k in rows:rows[k].append(None)
            last_active=False;continue
        if not last_active:
            start=max(0,a-256);reson=Resonators(cfg,lazy_mode,start)
            if a>start:reson.advance(sp[start:a]);work['warm_samples']+=a-start
        before=reson.work.copy();rates,power=reson.advance(sp[a:a+b]);base=first_features(sp[a:a+b],cfg)
        rows['spike'].append(np.array(base+list(core.rf_features(rates))))
        rows['analog_delta'].append(None if lazy_mode else np.array(base+list(core.rf_features(power))))
        rows['fft1024'].append(fft1024(rx[a:a+b],cfg))
        # Sum each continuous active run once; include warm replay on creation.
        if not last_active:work.update(reson.work)
        else:work.update({k:reson.work[k]-before[k] for k in reson.work})
        last_active=True
    return rows,dict(work),sp

def predictions(bank,kind,feats):
    ids=[i for i,v in enumerate(feats) if v is not None]
    X=bank.scalers[kind].transform(np.array([feats[i] for i in ids]));clf=bank.models[kind]
    p=clf.predict_proba(X);j=p.argmax(axis=1);labels=[None]*len(feats);conf=[None]*len(feats)
    for a,i in enumerate(ids):labels[i]=str(clf.classes_[j[a]]);conf[i]=float(p[a,j[a]])
    visits=int(clf.decision_path(X)[0].nnz-len(ids)*len(clf.estimators_))
    return labels,conf,visits

def receive(rx,cfg,bank,labels):
    """Same v5 current-block expert/mask path, with externally causal wake plan."""
    b=5000;soft=np.zeros(len(rx)//cfg.sps);flags=np.zeros(len(soft),bool);ale=legacy.StatefulALE(cfg)
    action='pass';actions=[];work=Counter(router_calls=sum(v is not None for v in labels),ale_predictions=0,ale_weight_updates=0,ale_samples=0,pulse_tests=0,pulse_group_rms=0)
    for bi,a in enumerate(range(0,len(rx),b)):
        x=rx[a:a+b]
        if labels[bi] is not None:action=legacy._label_to_action(labels[bi])
        active=action in ('slow','fast');raw_mask,raw_accept=legacy._pulse_mask(x,cfg,bank,action=='blank');used=action
        work.update(pulse_tests=1,pulse_group_rms=b//(cfg.sps*cfg.blanker_group))
        if action=='blank' and np.mean(raw_mask)>=cfg.blank_max_fraction:raw_mask[:]=False;raw_accept=False;used='pass'
        y,updates,pred=ale.process(x,raw_mask,cfg.ale_mu_nb if action=='slow' else cfg.ale_mu_sw,active)
        work.update(ale_predictions=pred,ale_weight_updates=updates)
        if active:
            erase,pa=legacy._pulse_mask(y,cfg,bank,False);work.update(ale_samples=b,pulse_tests=1,pulse_group_rms=b//(cfg.sps*cfg.blanker_group))
        elif action=='blank':erase=raw_mask
        else:erase=np.zeros(b,bool)
        values=y.reshape(-1,cfg.sps);rms=np.sqrt(np.mean(values**2,axis=1));aa=a//cfg.sps;zz=(a+b)//cfg.sps
        soft[aa:zz]=values.mean(axis=1);flags[aa:zz]=erase.reshape(-1,cfg.sps).any(axis=1)&(rms>2*np.median(rms));actions.append(used)
    work['ale_tap_products']=cfg.ale_taps*(2*work['ale_predictions']+work['ale_weight_updates'])
    return soft,flags,actions,dict(work)
