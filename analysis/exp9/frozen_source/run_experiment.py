#!/usr/bin/env python3
"""Reproducible gate pilot. No pJ constants or inferred hardware energy."""
from gates import *
import argparse,json,hashlib,time,pickle,gzip,platform,csv
from dataclasses import replace,asdict
from concurrent.futures import ProcessPoolExecutor,as_completed
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import sklearn
import receiver_v5

def dump(path,x):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False))
def digest(x):return hashlib.sha256(x).hexdigest()
def derive(*args):return int(np.random.SeedSequence([926202617,*args]).generate_state(1,dtype=np.uint64)[0])

BASE=json.loads((ROOT/'vendor/protocol_v5.json').read_text())
CASES=BASE['cases']+[dict(name='long_dwell',dwell_ms=120),dict(name='equal_power_hop',dwell_ms=30,hop=True)]
PROTOCOL=dict(id='UAV_resonator_gate_pilot_20260926',namespace=926202617,model_seeds=[811,821,823],conditions=['fixed','random'],
    trials_per_cell=4,cases=CASES,train_streams_per_class=20,trees=100,block_samples=5000,fs=1000000,
    candidates_max_hold_blocks=[2,4,8],sentinel=dict(power_ratio=2.,lag1_change=.2,spike_rate_change=.03,startup_blocks=2,event_hold_blocks=2,reference='last awake block'),
    warm_replay_samples=256,lazy_envelope='abs(real)+abs(imag) < .12-1e-12',
    quality_tolerances=dict(bler_pp=1.,first_word_pp=2.,crossing_word_pp=2.,extra_undetected_wrong=0),
    selection='All three candidates are run once on the same new panel. Descriptive per-cell gates; no tuning, no confirmatory noninferiority claim.',
    energy='Not measured. Rotation products, full bank updates, sentinel, LUT, warm replay and ALE work reported separately.',
    timing='Current buffered 5 ms block only. No packet feedback or ground truth enters receiver. Wake latency is frontend refresh latency, not guaranteed correct classification.',
    state='Encoder continuous; lazy bank timestamps continuous. Whole-bank sleep drops bank state; next wake replays preceding 256 encoded samples before current block.',
    limitations=['Float64 functional prototype, not fixed-point equivalence or HLS implementation','Same simulator family as v5; new seeds are not a new physical domain',
        'Whole-bank policy is a new sentinel/watchdog policy; v5 feature-change and blank-rejection refresh logic are not retained while asleep',
        'No gating-energy superiority inferred from counters; clocks, lookups, controller, ADC/I/O, static power and decoder require physical accounting'])

def generate(cfg,cond,case,ch,pay):
    f=io.generator_v4.generate(cfg,ch,pay,case,cond)
    if case.get('hop'):
        # Replace the clean middle of the second narrowband segment with an
        # equal-amplitude phase-continuous frequency hop. No pulse there.
        a=f['segments'][3]['start_symbol']*cfg.sps;z=f['segments'][3]['end_symbol']*cfg.sps
        mid=((a+z)//2//cfg.sps)*cfg.sps
        # Recover original tone by regenerating the same channel RNG prefix.
        rng=np.random.default_rng(ch)
        for _ in range(6):
            if cond=='random':rng.uniform(.75,1.25)
        rng.normal(0,10**(-case.get('snr_db',10)/20),len(f['rx']))
        if cond=='random':rng.uniform(-4,4)
        rng.choice(case.get('pulse_relative_db',[-6.,0.,6.]))
        if cond=='random':rng.uniform(*cfg.nb_f_range);rng.uniform(*cfg.sweep_f_range)
        phase0=rng.uniform(0,2*np.pi)
        fa=f['metadata']['tone_hz'];fb=(.44 if fa<250000 else .12)*cfg.fs
        idx=np.arange(mid,z);amp=np.sqrt(2*(1+10**(-case.get('snr_db',10)/10))*10**(f['metadata']['jsr_db']/10))
        old=amp*np.sin(phase0+2*np.pi*fa*idx/cfg.fs)
        new=amp*np.sin(phase0+2*np.pi*fa*mid/cfg.fs+2*np.pi*fb*(idx-mid)/cfg.fs)
        f['rx'][mid:z]+=new-old
        f['transitions']=sorted(f['transitions']+[mid//cfg.sps]);f['metadata']['hop']=dict(sample=mid,old_hz=fa,new_hz=fb,same_amplitude=True)
    return f

def train_one(args):
    cond,ms,out=args;out=Path(out);cfg=core.Config();ci=PROTOCOL['conditions'].index(cond);seed=derive(10,ci,ms)
    settings=legacy.StreamSettings();rows={k:[] for k in ('spike','analog_delta','fft1024')};labels=[];clean=[];plan=[]
    rng=np.random.default_rng(seed);tc=replace(cfg,n_sym_test=3000)
    for ci2,cls in enumerate(['none']+core.JT):
        for trial in range(PROTOCOL['train_streams_per_class']):
            sd=derive(11,ci,ms,ci2,trial);jsr=float(rng.choice(cfg.router_train_jsr))
            f=core.generate_frame(tc,cond,() if cls=='none' else (cls,),() if cls=='none' else (jsr,),sd)
            feat,_,_=extract(f['rx'],cfg)
            for bi in range(6):
                x=f['rx'][bi*5000:(bi+1)*5000]
                if cls=='none':clean.extend(np.sqrt(np.mean(x.reshape(-1,50)**2,axis=1)).tolist())
                if bi==0:continue
                for k in rows:rows[k].append(feat[k][bi])
                labels.append(cls)
            plan.append(dict(seed=sd,label=cls,jsr_db=jsr))
    models={};scalers={}
    for k in rows:
        X=np.array(rows[k]);sc=StandardScaler().fit(X);model=RandomForestClassifier(n_estimators=100,max_depth=10,n_jobs=1,random_state=ms)
        model.fit(sc.transform(X),labels);models[k]=model;scalers[k]=sc
    bank=legacy.StreamBank(cfg,settings,models,scalers,dict(seed=seed,plan=plan),np.mean(clean),max(np.std(clean),1e-6))
    path=out/'models'/f'{cond}_{ms}.pkl';path.parent.mkdir(exist_ok=True);path.write_bytes(pickle.dumps(bank,protocol=5))
    return dict(condition=cond,model_seed=ms,model_sha256=digest(path.read_bytes()),training=plan)

def verify(out):
    cfg=core.Config();records=[]
    # Impulses, ringing without further input, long silence, and random events.
    patterns=[]
    s=np.zeros(15000,np.int8);s[[0,1,2,5,10,7000,14000]]=[1,1,1,-1,1,1,-1];patterns.append(s)
    rng=np.random.default_rng(92601);patterns.extend(rng.choice(np.array([-1,0,1],np.int8),15000,p=[p/2,1-p,p/2]) for p in [.01,.2,.6])
    for i,sp in enumerate(patterns):
        d=Resonators(cfg);g=Resonators(cfg,True)
        for a in range(0,len(sp),5000):
            da,_=d.advance(sp[a:a+5000]);ga,_=g.advance(sp[a:a+5000]);np.testing.assert_array_equal(da,ga)
        records.append(dict(pattern=i,sample_count=len(sp),count_equal=True,work=dict(g.work)))
    # Match the frozen continuous encoder/features and v5 expert path.
    f=generate(cfg,'random',CASES[1],derive(90,1),derive(91,1));rx=f['rx'];sp=encode(rx)
    ref=legacy.StatefulFrontEnd(cfg);feats,_,_=extract(rx,cfg);lazyfeats,_,_=extract(rx,cfg,lazy_mode=True)
    maxerr=0.
    for bi,a in enumerate(range(0,len(rx),5000)):
        spikes,counts,power,sizes=ref.advance(rx[a:a+5000]);np.testing.assert_array_equal(sp[a:a+5000],spikes)
        z=np.array(first_features(spikes,cfg)+list(core.rf_features(counts/(sizes[:,None]*ref.om[None,:]/(2*np.pi)))))
        np.testing.assert_array_equal(z,feats['spike'][bi]);np.testing.assert_array_equal(z,lazyfeats['spike'][bi])
        z2=np.array(first_features(spikes,cfg)+list(core.rf_features(power/sizes[:,None])))
        np.testing.assert_array_equal(z2,feats['analog_delta'][bi])
    # Prefix causality, including gates and encoder state.
    awake,_,_=schedule(rx,sp,5000,8);cut=20*5000
    assert cut<len(rx)
    a2,_,_=schedule(rx[:cut],encode(rx[:cut]),5000,8);np.testing.assert_array_equal(a2,awake[:20])
    fg,_,_=extract(rx,cfg,awake,True);fp,_,_=extract(rx[:cut],cfg,a2,True)
    for i in range(20):
        if awake[i]:np.testing.assert_array_equal(fg['spike'][i],fp['spike'][i])
    # Confirm the hop replacement uses precisely the original tone phase.
    for cond in ('fixed','random'):
        ch=derive(92,0 if cond=='fixed' else 1);pay=derive(93,0 if cond=='fixed' else 1)
        case=CASES[-1];h=generate(cfg,cond,case,ch,pay)
        baseline=io.generator_v4.generate(cfg,ch,pay,case,cond)
        clean=io.generator_v4.generate(cfg,ch,pay,{**case,'clean':True},cond)
        a=baseline['segments'][3]['start_symbol']*cfg.sps;z=baseline['segments'][3]['end_symbol']*cfg.sps
        mid=h['metadata']['hop']['sample']
        np.testing.assert_array_equal(h['rx'][:mid],baseline['rx'][:mid])
        np.testing.assert_array_equal(h['rx'][z:],baseline['rx'][z:])
        # Equal amplitude is a generation contract, not equal finite-window power.
        amp=np.sqrt(2*1.1*10**(baseline['metadata']['jsr_db']/10))
        assert np.max(np.abs(h['rx'][mid:z]-clean['rx'][mid:z]))<=amp+1e-10
    dump(out/'verification.json',dict(patterns=records,frozen_features_equal=True,encoder_equal=True,lazy_spike_features_equal=True,prefix_causality=True,
        hop_unchanged_outside_target=True,warm_state_error_bound=float((14/15)**256),fixed_point_tested=False))

def evaluate_task(t):
    cond,ms,caseid,trial,out=t;out=Path(out);cfg=core.Config();ci=PROTOCOL['conditions'].index(cond);case=CASES[caseid]
    ch=derive(20,ci,ms,caseid,trial);pay=derive(30,ci,ms,caseid,trial)
    bank=pickle.loads((out/'models'/f'{cond}_{ms}.pkl').read_bytes());f=generate(cfg,cond,case,ch,pay);rx=f['rx'];nb=len(rx)//5000
    full,wfull,sp=extract(rx,cfg);lazyfull,wlazy,_=extract(rx,cfg,lazy_mode=True)
    np.testing.assert_array_equal(np.array(full['spike']),np.array(lazyfull['spike']))
    plans={};featwork={};awakeplans={};sentinel_records={}
    for kind in full:
        name=kind+'_dense';plans[name]=(kind,full[kind]);featwork[name]=wfull.copy();awakeplans[name]=np.ones(nb,bool)
    plans['spike_lazy']=('spike',lazyfull['spike']);featwork['spike_lazy']=wlazy;awakeplans['spike_lazy']=np.ones(nb,bool)
    for h in [2,4,8]:
        awake,reasons,stats=schedule(rx,sp,5000,h);fd,wd,_=extract(rx,cfg,awake,False);fl,wl,_=extract(rx,cfg,awake,True)
        for i in np.flatnonzero(awake):np.testing.assert_array_equal(fd['spike'][i],fl['spike'][i])
        sentinel_records[str(h)]=dict(awake=awake.astype(int).tolist(),reasons=reasons,stats=stats.tolist())
        for kind in full:
            name=kind+'_gate'+str(h);plans[name]=(kind,fl[kind] if kind=='spike' else fd[kind]);featwork[name]=(wl if kind=='spike' else wd).copy();awakeplans[name]=awake
    rows={};cache={};route_pred={}
    for name,(kind,feats) in plans.items():
        labels,conf,comp=predictions(bank,kind,feats);soft,flags,actions,work=receive(rx,cfg,bank,labels)
        score,eff=io.link_v4.evaluate(soft,flags,'block_refine2',f['bits'],f['payloads'],f['words'],f['coding'],500,cfg.symbol_rate,f['transitions'],cache)
        fw=featwork[name].copy();active=fw['active_blocks'];gated='_gate' in name
        if kind=='fft1024':
            fw={k:fw[k] for k in ('active_blocks','total_blocks')};fw.update(delta_samples=len(rx) if gated else 0,fft1024_calls=4*active,fft_input_window_products=4096*active,fft_tail_time_only_samples=904*active)
        elif kind=='analog_delta':fw['analog_power_products']=2*fw['bank_updates']
        if gated:fw.update(sentinel_power_products=len(rx),sentinel_lag_products=len(rx)-nb,sentinel_boundary_products=2*nb,sentinel_rate_tests=len(rx),sentinel_scalar_divisions=3*nb)
        work.update(fw);work['rf_tree_comparisons']=comp
        # At each true transition: first completed active block that contains
        # post-transition samples; refresh latency, not decision correctness.
        delays=[]
        for ts in f['transitions']:
            sample=ts*cfg.sps;ids=np.flatnonzero(awakeplans[name] & ((np.arange(nb)+1)*5000>sample))
            if len(ids):delays.append(((int(ids[0])+1)*5000-sample)/cfg.fs*1000)
        crossing=[v for v in score['words'] if v['crosses_transition']]
        score.update(counts=work,crossing_words=len(crossing),crossing_failed=sum(not v['success'] for v in crossing),
            transition_refresh_ms=delays,actions=actions,predictions=labels,confidences=conf)
        rows[name]=score;route_pred[name]=labels
        if name=='spike_dense':
            reference_feats=[{k:full[k][i] for k in ('spike','analog_delta')} for i in range(nb)]
            v=receiver_v5.process_one(rx,cfg,bank,reference_feats,'spike','current_periodic',BASE['rescue'])
            np.testing.assert_array_equal(soft,v['soft']);np.testing.assert_array_equal(flags,v['flags']['block_refine2'])
    # Existing v5 sparse RF control, still dense frontend.
    v=receiver_v5.process_one(rx,cfg,bank,reference_feats,'spike','refresh_guard8',BASE['rescue'])
    score,_=io.link_v4.evaluate(v['soft'],v['flags']['block_refine2'],'block_refine2',f['bits'],f['payloads'],f['words'],f['coding'],500,cfg.symbol_rate,f['transitions'],cache)
    crossing=[q for q in score['words'] if q['crosses_transition']]
    score.update(counts=v['counts'],crossing_words=len(crossing),crossing_failed=sum(not q['success'] for q in crossing),actions=v['actions_by_block'])
    rows['spike_v5_refresh']=score
    result=dict(condition=cond,model_seed=ms,case=case['name'],trial=trial,channel_seed=ch,payload_seed=pay,n_samples=len(rx),
        waveform_sha256=digest(np.ascontiguousarray(rx,dtype='<f8').tobytes()),coding=f['coding'],metadata=f['metadata'],transitions=f['transitions'],
        lazy_features_equal=True,v5_periodic_output_equal=True,schedules=sentinel_records,methods=rows)
    path=out/'shards'/f'{cond}_{ms}_{caseid}_{trial}.json.gz';path.parent.mkdir(exist_ok=True)
    path.write_bytes(gzip.compress(json.dumps(result,allow_nan=False).encode(),mtime=0))
    return dict(condition=cond,model_seed=ms,case=case['name'],trial=trial,n_samples=len(rx))

def writecsv(path,rows):
    with open(path,'w',newline='') as f:
        keys=list(dict.fromkeys(k for row in rows for k in row));w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def summarize(out):
    data=[json.loads(gzip.decompress(p.read_bytes())) for p in sorted((out/'shards').glob('*.gz'))]
    summaries=[];cells=[];paired=[];workrows=[]
    for condition in PROTOCOL['conditions']:
        ds=[d for d in data if d['condition']==condition]
        for name in ds[0]['methods']:
            groups=[('ALL',ds)]+[(case['name'],[d for d in ds if d['case']==case['name']]) for case in CASES]
            for case,group in groups:
                vals=[d['methods'][name] for d in group];nw=sum(v['n_words'] for v in vals);cw=sum(v['crossing_words'] for v in vals)
                work=Counter()
                for v in vals:work.update(v['counts'])
                n=sum(d['n_samples'] for d in group);refresh=[q for v in vals for q in v.get('transition_refresh_ms',[])]
                r=dict(condition=condition,case=case,method=name,streams=len(vals),words=nw,failed_words=sum(v['failed_words'] for v in vals),
                    bler_pct=100*sum(v['failed_words'] for v in vals)/nw,first_fail_pct=100*np.mean([v['first_word_failure'] for v in vals]),
                    crossing_bler_pct=100*sum(v['crossing_failed'] for v in vals)/cw if cw else 0.,undetected_wrong=sum(v['undetected_wrong'] for v in vals),
                    bank_update_saving_pct=100*(1-work['bank_updates']/(16*n)) if 'bank_updates' in work else None,
                    rotation_product_saving_pct=100*(1-work['rotation_real_products']/(64*n)) if 'rotation_real_products' in work else None,
                    active_block_pct=100*work['active_blocks']/work['total_blocks'] if work['total_blocks'] else None,
                    mean_rf_calls=work['router_calls']/len(vals),refresh_p95_ms=float(np.percentile(refresh,95)) if refresh else None,
                    refresh_max_ms=max(refresh) if refresh else None,ale_products=work['ale_tap_products'])
                (summaries if case=='ALL' else cells).append(r)
                if case=='ALL':workrows.append(dict(condition=condition,method=name,n_samples=n,**work))
            if '_gate' in name or name=='spike_lazy':
                kind=name.split('_gate')[0] if '_gate' in name else 'spike';ref=kind+'_dense'
                diffs=np.array([d['methods'][name]['bler']-d['methods'][ref]['bler'] for d in ds])
                rng=np.random.default_rng(926001);boots=[]
                strata=[[i for i,d in enumerate(ds) if d['case']==c['name'] and d['model_seed']==ms] for c in CASES for ms in PROTOCOL['model_seeds']]
                strata=[s for s in strata if s]
                for _ in range(4000):boots.append(np.mean(diffs[np.concatenate([rng.choice(s,len(s),replace=True) for s in strata])]))
                lo,hi=np.percentile(boots,[2.5,97.5])
                paired.append(dict(condition=condition,method=name,reference=ref,stream_mean_bler_delta_pp=100*float(diffs.mean()),ci95_lo_pp=100*lo,ci95_hi_pp=100*hi,
                    bootstrap='paired stream, stratified case/model; conditional on these trained models; no multiplicity correction'))
    writecsv(out/'summary.csv',summaries);writecsv(out/'cells.csv',cells);writecsv(out/'paired.csv',paired);writecsv(out/'work.csv',workrows)
    gates=[]
    for kind in ('spike','analog_delta','fft1024'):
        for h in [2,4,8]:
            name=kind+'_gate'+str(h);failed=[]
            for row in [c for c in cells if c['method']==name]:
                ref=next(c for c in cells if c['condition']==row['condition'] and c['case']==row['case'] and c['method']==kind+'_dense')
                reasons=[key for key,margin in [('bler_pct',1.),('first_fail_pct',2.),('crossing_bler_pct',2.),('undetected_wrong',0)] if row[key]-ref[key]>margin+1e-12]
                if reasons:failed.append(dict(condition=row['condition'],case=row['case'],reasons=reasons,bler_delta_pp=row['bler_pct']-ref['bler_pct']))
            gates.append(dict(method=name,passed=not failed,failed_cells=failed))
    dump(out/'quality_gates.json',gates)
    dump(out/'completion.json',dict(streams=len(data),total_samples=sum(d['n_samples'] for d in data),
        unique_waveform_hashes=len({d['waveform_sha256'] for d in data}),lazy_feature_mismatches=0,v5_periodic_mismatches=0,
        all_words=sum(d['methods']['spike_dense']['n_words'] for d in data),hardware_energy_measured=False))
    print(json.dumps(dict(completion=json.loads((out/'completion.json').read_text()),gates=gates),indent=2),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=ROOT/'results');ap.add_argument('--workers',type=int,default=4);ap.add_argument('--smoke',action='store_true');ap.add_argument('--summarize',action='store_true');args=ap.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if args.summarize:summarize(out);return
    if args.smoke:PROTOCOL['model_seeds']=[811];PROTOCOL['trials_per_cell']=1;PROTOCOL['train_streams_per_class']=3
    spec=dict(protocol=PROTOCOL,config=asdict(core.Config()),python=platform.python_version(),numpy=np.__version__,numba=numba.__version__,sklearn=sklearn.__version__,
        sources={str(p.relative_to(ROOT)):digest(p.read_bytes()) for p in sorted([ROOT/'gates.py',ROOT/'run_experiment.py',*(ROOT/'vendor').rglob('*.py')]) if '__pycache__' not in str(p)})
    path=out/'lock.json'
    if path.exists():assert json.loads(path.read_text())==json.loads(json.dumps(spec)),'Run lock differs; use a fresh output directory.'
    else:dump(path,spec)
    print('Verifying recurrences and prefix causality...',flush=True);verify(out)
    print('Training matched models...',flush=True)
    jobs=[(c,ms,str(out)) for c in PROTOCOL['conditions'] for ms in PROTOCOL['model_seeds']]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for r in pool.map(train_one,jobs):
            dump(out/'models'/f"{r['condition']}_{r['model_seed']}.json",r);print('Trained',r['condition'],r['model_seed'],flush=True)
    tasks=[(c,ms,ci,tr,str(out)) for c in PROTOCOL['conditions'] for ms in PROTOCOL['model_seeds'] for ci in range(len(CASES)) for tr in range(PROTOCOL['trials_per_cell'])]
    if args.smoke:tasks=[t for t in tasks if t[2] in (0,1,7)]
    started=time.time();done=0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(evaluate_task,t) for t in tasks if not (out/'shards'/f'{t[0]}_{t[1]}_{t[2]}_{t[3]}.json.gz').exists()]
        for fut in as_completed(futures):
            r=fut.result();done+=1;print(f"{done}/{len(futures)} {r['condition']} {r['case']} {time.time()-started:.1f}s",flush=True)
    if not args.smoke:summarize(out)
    print('COMPLETE',out,flush=True)
if __name__=='__main__':main()
