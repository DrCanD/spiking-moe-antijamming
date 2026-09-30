#!/usr/bin/env python3
"""Common-workload anti-jamming frontend audit. Does NOT estimate unmeasured joules.
Run: python analysis/energy/run_front_end_comparison.py [--smoke] [--only-hardware]
Shared receiver definitions are selected by AST.
"""
import os
for _k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ.setdefault(_k,'1')
import argparse, ast, csv, hashlib, json, platform, sys, time
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import numba, scipy, sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'simulation/streaming/frozen_v4'))
import core
METHODS=['legacy_lif64_fft','legacy_delta_fft_no_lif','res16_spike',
         'res16_analog_delta','conv_fullfft','conv_windowfft1024']
CLASSES=['none','broadband','narrowband','sweep','pulse']
SEEDS=[42,2026,9407]

def dump(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')

def csv_write(path,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with open(path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def load_legacy():
    p=ROOT/'simulation/frame_receiver/reference_receiver.py';source=p.read_text()
    names={'SpikeEncoderConfig','WatchdogConfig','DeltaEncoder','LIFLayer','SNNWatchdogClassifier','extract_conventional_features'}
    nodes=[n for n in ast.parse(source).body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names]
    assert {n.name for n in nodes}==names
    env={'np':np,'dataclass':dataclass,'StandardScaler':StandardScaler,'RandomForestClassifier':RandomForestClassifier}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p),'exec'),env)
    return env

LEGACY=load_legacy()

@numba.njit(cache=True,fastmath=False)
def lif_fast(sp,w,decay,threshold,refractory):
    n=len(sp);m=w.shape[1];v=np.zeros(m);last=np.full(m,-refractory-1,np.int64);total=0
    for t in range(n):
        for k in range(m):
            v[k]*=decay
            if sp[t]>0:v[k]+=w[0,k]
            elif sp[t]<0:v[k]+=w[1,k]
            if v[k]>=threshold and t-last[k]>refractory:
                total+=1;v[k]=0.;last[k]=t
    return total

@numba.njit(cache=True,fastmath=False)
def bank_power(sp,rot,tau,nsf):
    n=len(sp);k=len(rot);sf_len=n//nsf;p=np.zeros((nsf,k))
    for b in range(k):
        z=0.+0.j
        for t in range(n):
            z=z*rot[b]+float(sp[t])/tau
            p[min(t//sf_len,nsf-1),b]+=z.real*z.real+z.imag*z.imag
    return p/sf_len

def old_features(x,sp,total,fullmag):
    n=len(x);ids=np.flatnonzero(sp);isi=np.diff(ids).astype(float)
    cv=float(np.std(isi)/(np.mean(isi)+1e-10)) if len(ids)>2 else 0.
    counts=np.array([np.sum((ids>=k*(n//20))&(ids<(k+1)*(n//20))) for k in range(20)])
    fano=counts.var()/(counts.mean()+1e-10) if len(ids) else 0.
    sfr=counts.max()/(counts.min()+1.) if len(ids) else 1.
    hf=np.sum(fullmag[len(fullmag)//2:]**2)/(np.sum(fullmag**2)+1e-10)
    peak=np.max(fullmag[1:])/(np.mean(fullmag[1:])+1e-10)
    return np.array([len(ids)/n,total/(n*64),cv,fano,hf,sfr,peak])

def conv_features_from_mag(x,mag):
    n=len(x);rms=np.sqrt(np.mean(x*x));v=np.var(x);mu=np.mean(x)
    kurt=np.mean((x-mu)**4)/(v*v+1e-10)-3
    flat=np.exp(np.mean(np.log(mag[1:]+1e-10)))/(np.mean(mag[1:])+1e-10)
    peak=np.max(mag[1:])/(np.mean(mag[1:])+1e-10)
    se=np.mean(x.reshape(20,-1)**2,axis=1)
    sfvar=se.var()/(se.mean()+1e-10)
    zc=np.sum(np.diff(np.sign(x))!=0)/n
    return np.array([rms,v,kurt,flat,peak,sfvar,zc])

def get_features(x,cfg):
    n=len(x);sp0=core.delta_encode(x,.04,2);sp=core.delta_encode(x,2.,2)
    w=np.random.default_rng(42).normal(0,.3,(2,64))
    total=int(lif_fast(sp0,w,np.exp(-.1),1.,3))
    mag=np.abs(np.fft.rfft(x));old=old_features(x,sp0,total,mag)
    first=list(core.spike_stats(sp,20))+[core.isi_drift(sp,20)]
    rates=core.rf_bank(sp,cfg)
    om=2*np.pi*np.linspace(25e3,475e3,16)/cfg.fs;rot=(1.-1./15)*np.exp(1j*om)
    p=bank_power(sp,rot,15.,20)
    windows=x[:(n//1024)*1024].reshape(-1,1024)*np.hanning(1024)
    wm=np.abs(np.fft.rfft(windows,axis=1)).mean(axis=0)
    rows={
        'legacy_lif64_fft':old,
        'legacy_delta_fft_no_lif':np.delete(old,1),
        'res16_spike':np.array(first+list(core.rf_features(rates))),
        'res16_analog_delta':np.array(first+list(core.rf_features(p))),
        'conv_fullfft':conv_features_from_mag(x,mag),
        'conv_windowfft1024':conv_features_from_mag(x,wm),
    }
    meta=dict(n_samples=n,old_input_events=int(np.count_nonzero(sp0)),old_output_events=total,
              current_input_events=int(np.count_nonzero(sp)),
              current_output_events=int(np.rint((rates*((n//20)*om/(2*np.pi))[None,:]).sum())),
              waveform_sha256=hashlib.sha256(np.asarray(x,dtype='<f8').tobytes()).hexdigest())
    return rows,meta

def verify():
    cfg=core.Config();raw=LEGACY['SNNWatchdogClassifier']();tests=[]
    for jt in CLASSES:
        f=core.generate_frame(cfg,'fixed',() if jt=='none' else (jt,),() if jt=='none' else (10,),seed=892000+CLASSES.index(jt))
        x=f['rx'][:4000]
        rows,meta=get_features(x,cfg)
        a=raw.extract_features(x);b=rows['legacy_lif64_fft']
        np.testing.assert_allclose(a,b,rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(LEGACY['extract_conventional_features'](x),rows['conv_fullfft'],rtol=1e-12,atol=1e-12)
        np.testing.assert_array_equal(raw.encoder.encode(x)[0],core.delta_encode(x,.04,2))
        np.testing.assert_allclose(core.snn_features(x,cfg),rows['res16_spike'],rtol=1e-12,atol=1e-12)
        # Independent array recurrence, including analog readout.
        sp=core.delta_encode(x,2.,2);om=2*np.pi*np.linspace(25e3,475e3,16)/1e6
        rot=(1.-1./15)*np.exp(1j*om);z=np.zeros(16,complex);p=np.zeros((20,16))
        for t in range(len(x)):
            z=z*rot+float(sp[t])/15.;p[t//200]+=z.real*z.real+z.imag*z.imag
        np.testing.assert_allclose(bank_power(sp,rot,15.,20),p/200,rtol=1e-12,atol=1e-12)
        tests.append(dict(jammer=jt,n=len(x),legacy_max_abs_error=float(np.max(np.abs(a-b)))))
    return tests

def hardware_audit():
    import subprocess
    subprocess.run([sys.executable,str(ROOT/'analysis/energy/reproduce.py')],check=True)
    subprocess.run([sys.executable,str(ROOT/'analysis/energy/classifier_tail.py')],check=True)
    data=json.loads((ROOT/'analysis/energy/reference/independent_raw_audit.json').read_text())
    blocks=[dict(vector=name,mode=mode,mean_nJ_per_sample_replica=v['mean'],std_nJ=v['sd'])
            for name,row in data['campaigns']['matched']['vectors'].items() for mode,v in row['modes'].items()]
    tail=json.loads((ROOT/'results/hardware/classifier_tail/power_records.json').read_text())
    classifiers=[dict(mode=name,rate_per_replica=v['rate_per_replica'],nJ_per_inference=v['nJ_per_inference'])
                 for name,v in tail['designs'].items()]
    return data,blocks,classifiers,dict(paired_block_audit=True,classifier_tail_audit=True)


def plan(cond,split,smoke=False):
    ci=['fixed','random'].index(cond)
    base=202609260000+ci*100000+(0 if split=='train' else 50000)
    rows=[]
    jsrs=list(range(0,21,2));per=5 if split=='train' else 10;clean=50 if split=='train' else 110
    if smoke:jsrs=[0,10,20];per=1;clean=3
    for c,jt in enumerate(CLASSES):
        if jt=='none':
            for k in range(clean):rows.append(dict(cond=cond,split=split,jammer=jt,jsr_db=0,seed=base+k))
        else:
            for j,jsr in enumerate(jsrs):
                for k in range(per):rows.append(dict(cond=cond,split=split,jammer=jt,jsr_db=jsr,seed=base+c*5000+j*100+k))
    return rows

def prepare_data(cond,split,cfg,out,smoke):
    path=out/f'{cond}_{split}_features.npz';mp=out/f'{cond}_{split}_frames.json'
    if path.exists() and mp.exists():
        d=np.load(path);return {m:d[m] for m in METHODS},json.loads(mp.read_text())
    rows=plan(cond,split,smoke);feats={m:[] for m in METHODS};done=[];start=time.time()
    for i,row in enumerate(rows):
        jt=row['jammer'];f=core.generate_frame(cfg,cond,() if jt=='none' else (jt,),() if jt=='none' else (row['jsr_db'],),seed=row['seed'])
        features,meta=get_features(f['rx'],cfg)
        for m in METHODS:feats[m].append(features[m])
        done.append({**row,**meta})
        if (i+1)%50==0 or i+1==len(rows):print(f'{cond} {split}: {i+1}/{len(rows)} frames ({time.time()-start:.1f}s)',flush=True)
    arrays={m:np.array(v) for m,v in feats.items()};np.savez_compressed(path,**arrays);dump(mp,done)
    return arrays,done

def classify(cond,train,test,tr,te,out):
    y=np.array([r['jammer'] for r in tr]);truth=np.array([r['jammer'] for r in te]);rows=[];preds={};node_rows=[]
    models=out/'models';models.mkdir(exist_ok=True)
    for m in METHODS:
        sc=StandardScaler().fit(train[m]);a=sc.transform(train[m]);b=sc.transform(test[m]);preds[m]=[]
        for seed in SEEDS:
            clf=RandomForestClassifier(n_estimators=100,max_depth=10,random_state=seed,n_jobs=1)
            clf.fit(a,y);pred=clf.predict(b);preds[m].append(pred)
            path=clf.decision_path(b)[0];visits=np.diff(path.indptr)-100
            joblib.dump({'scaler':sc,'classifier':clf},models/f'{cond}_{m}_{seed}.joblib',compress=3)
            for i,p in enumerate(pred):
                rows.append(dict(cond=cond,method=m,forest_seed=seed,frame=i,waveform_seed=te[i]['seed'],
                            jammer=truth[i],jsr_db=te[i]['jsr_db'],prediction=p,correct=int(p==truth[i]),
                            tree_comparisons=int(visits[i])))
            node_rows.append(dict(cond=cond,method=m,forest_seed=seed,features=a.shape[1],
                tree_nodes=sum(t.tree_.node_count for t in clf.estimators_),
                mean_tree_comparisons=float(visits.mean()),min_tree_comparisons=int(visits.min()),max_tree_comparisons=int(visits.max())))
        preds[m]=np.array(preds[m])
    summaries=[];cells=[];rng=np.random.default_rng(61725001)
    # Paired frame bootstrap averages the same three forest seeds; not 3x independent frames.
    ids=np.stack([np.flatnonzero(truth==c) for c in CLASSES])
    for m in METHODS:
        correct=preds[m]==truth;ref=preds['conv_fullfft']==truth
        delta=ref.mean(axis=0)-correct.mean(axis=0) # positive => more errors than reference
        bs=[]
        for _ in range(5000):
            pick=np.concatenate([rng.choice(ix,len(ix),replace=True) for ix in ids]);bs.append(float(delta[pick].mean()))
        acc=correct.mean(axis=1)
        summaries.append(dict(cond=cond,method=m,n_unique_test_frames=len(te),n_forest_seeds=len(SEEDS),
            accuracy_mean=float(acc.mean()),accuracy_min=float(acc.min()),accuracy_max=float(acc.max()),
            extra_error_vs_conv_fullfft=float(delta.mean()),paired_CI95_low=float(np.quantile(bs,.025)),
            paired_CI95_high=float(np.quantile(bs,.975)),
            descriptive_within_1pp_of_conv_fullfft=bool(delta.mean()<=.01),
            matched_energy_comparison_eligible=False,
            eligibility_reason='No complete same-scope energy measurements for these fitted float64 pipelines'))
        for jt in CLASSES:
            ix=truth==jt;cells.append(dict(cond=cond,method=m,jammer=jt,n_unique_frames=int(ix.sum()),accuracy=float(correct[:,ix].mean())))
    return rows,summaries,node_rows,cells

def operation_ledger(frame_rows):
    rows=[]
    for cond in ['fixed','random']:
        for jt in CLASSES:
            a=[r for r in frame_rows if r['cond']==cond and r['jammer']==jt];n=a[0]['n_samples']
            old=float(np.mean([r['old_input_events'] for r in a]));new=float(np.mean([r['current_input_events'] for r in a]));out=float(np.mean([r['old_output_events'] for r in a]))
            for m in METHODS:
                r=dict(cond=cond,jammer=jt,method=m,n_samples=n,n_unique_frames=len(a),
                    state_mults=0,state_adds=0,threshold_comparisons=0,
                    event_weight_adds=0,source_dense_input_mults=0,
                    ternary_input_scalings=0,readout_squares=0,full_real_FFT_calls=0,
                    full_real_FFT_length=0,window_real_FFT_calls=0,window_real_FFT_length=0,
                    hann_mults=0,assumed_pJ_per_synaptic_event=None,synaptic_only_proxy_mW=None,
                    estimate_status='partial_operation_ledger_not_energy')
                if m.startswith('legacy'):
                    r.update(input_events=old,full_real_FFT_calls=1,full_real_FFT_length=n)
                    if m=='legacy_lif64_fft':
                        r.update(state_mults=64*n,threshold_comparisons=64*n,
                           event_weight_adds=64*old,source_dense_input_mults=128*old,output_spikes=out,
                           assumed_pJ_per_synaptic_event=20,
                           synaptic_only_proxy_mW=64*old/(n/1e6)*20e-12*1e3)
                elif m.startswith('res16'):
                    r.update(input_events=new,state_mults=64*n,state_adds=48*n,ternary_input_scalings=16*n)
                    if m=='res16_spike':r['threshold_comparisons']=16*n
                    else:r.update(readout_squares=32*n,readout_accumulation_adds=32*n)
                elif m=='conv_fullfft':r.update(full_real_FFT_calls=1,full_real_FFT_length=n)
                else:r.update(window_real_FFT_calls=n//1024,window_real_FFT_length=1024,hann_mults=(n//1024)*1024)
                rows.append(r)
    return rows

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');ap.add_argument('--only-hardware',action='store_true');ap.add_argument('--output',type=Path,default=Path('runs/front_end_workload'));a=ap.parse_args()
    out=ROOT/a.output;out.mkdir(parents=True,exist_ok=True)
    raw,hw,rfs,checks=hardware_audit();csv_write(out/'measured_blocks_recomputed.csv',hw);csv_write(out/'measured_finalization_recomputed.csv',rfs)
    original_proxy=.05*128*1e5*20e-12*1e3+.001
    dump(out/'historical_budget.json',dict(original_code_power_mW=original_proxy,manuscript_rounded_mW=.014,
        Cortex_M4_assumed_mW=39.6*(504608/48e6)/.1,
        original_cortex_ratio=(39.6*(504608/48e6)/.1)/original_proxy,
        original_budget_frequency_Hz=1e5,actual_input_sampling_Hz=1e6,
        original_claimed_FFT_length=512,original_actual_FFT_length=100000,
        input_vs_output_activity_confused=True,hardware_raw_recalculation_checks=checks))
    if a.only_hardware:return
    protocol=dict(version=1,task='five-class full-frame frontend+100-tree RF classification',
        fs_Hz=1e6,symbol_rate_Hz=1e5,n_samples=100000,n_symbols=10000,frame_ms=100,
        snr_db=10,jsr_db=list(range(0,21,2)),protocols=['fixed','random'],
        train_frames_per_protocol=270,test_frames_per_protocol=550,forest_seeds=SEEDS,
        methods=METHODS,quality_tolerance_accuracy_pp=1.,
        scope='New matched software diagnostic, not v5 streaming validation or a hardware-power experiment',
        fixed_point_hardware_equivalence='Not established for the new common panel',
        early_stopping=False,hyperparameter_selection=False,smoke=a.smoke)
    dump(out/'protocol.json',protocol)
    dump(out/'environment.json',dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,numba=numba.__version__,platform=platform.platform()))
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for folder in ['simulation/frame_receiver','simulation/streaming/frozen_v4'] for p in (ROOT/folder).glob('*') if p.is_file()}
    hashes['analysis/energy/run_front_end_comparison.py']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();dump(out/'source_manifest.json',hashes)
    dump(out/'reference_verification.json',verify());print('Reference checks passed',flush=True)
    summaries=[];allframes=[];allpred=[];allnodes=[];allcells=[]
    cfg=core.Config()
    for cond in ['fixed','random']:
        train,tr=prepare_data(cond,'train',cfg,out,a.smoke);test,te=prepare_data(cond,'test',cfg,out,a.smoke)
        pred,summ,nodes,cells=classify(cond,train,test,tr,te,out)
        summaries+=summ;allframes+=te;allpred+=pred;allnodes+=nodes;allcells+=cells
        print(cond,[(s['method'],round(s['accuracy_mean'],4)) for s in summ],flush=True)
    csv_write(out/'quality_summary.csv',summaries);csv_write(out/'quality_by_class.csv',allcells)
    csv_write(out/'paired_predictions.csv',allpred);csv_write(out/'forest_work.csv',allnodes)
    csv_write(out/'operation_ledger.csv',operation_ledger(allframes));csv_write(out/'event_activity.csv',allframes)
    dump(out/'completion.json',dict(status='completed',unique_test_frames=len(allframes),methods=len(METHODS),new_hardware_energy_measured=False))
    print('COMPLETE',out,flush=True)

if __name__=='__main__':main()
