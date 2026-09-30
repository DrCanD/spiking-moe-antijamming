"""Streaming router, resonator-band and matched-feature comparisons."""
from pathlib import Path
from dataclasses import asdict, replace
import argparse
import copy
import hashlib
import json
import pickle
import sys
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'hardware/kv260/software'))
import run_experiment as reference
import receiver_v5
from gates import core,io,legacy,extract,fft1024,first_features


def fft_bands(x,cfg,subframes=False):
    size=len(x)//cfg.n_subframes if subframes else 1024
    windows=x[:len(x)//size*size].reshape(-1,size)*np.hanning(size)
    spectrum=np.abs(np.fft.rfft(windows,axis=1))
    frequencies=np.fft.rfftfreq(size,1/cfg.fs)
    centres=np.linspace(cfg.rf_f_lo_hz,cfg.rf_f_hi_hz,cfg.rf_bands)
    edges=np.r_[0,(centres[:-1]+centres[1:])/2,cfg.fs/2]
    powers=np.column_stack([np.sqrt(np.sum(spectrum[:,(frequencies>=edges[i])&(frequencies<edges[i+1])]**2,axis=1)) for i in range(cfg.rf_bands)])
    return np.array(core.rf_features(powers))


def feature_rows(rx,cfg):
    raw,_,_=extract(rx,cfg)
    output=[]
    for i,a in enumerate(range(0,len(rx),5000)):
        x=rx[a:a+5000];fft=raw['fft1024'][i]
        band=fft_bands(x,cfg)
        output.append(dict(spike9=raw['spike'][i],fft7=fft,spike_rf4=raw['spike'][i][-4:],
                           fft_rf4=band,fft_rf4_sf20=fft_bands(x,cfg,True),
                           fft_spec=np.r_[fft[3:5],fft[6],band]))
    return output


def train(cfg,condition,model_seed,n_train,trees,kinds,seed_offset=0):
    ci=['fixed','random'].index(condition)
    rng=np.random.default_rng(reference.derive(10,ci,model_seed)+seed_offset)
    rows={k:[] for k in kinds};labels=[];clean=[];plan=[]
    tc=replace(cfg,n_sym_test=3000)
    for j,label in enumerate(['none']+core.JT):
        for trial in range(n_train):
            seed=reference.derive(11,ci,model_seed,j,trial)+seed_offset
            jsr=float(rng.choice(cfg.router_train_jsr))
            frame=core.generate_frame(tc,condition,() if label=='none' else (label,),() if label=='none' else (jsr,),seed)
            features=feature_rows(frame['rx'],cfg)
            for i,a in enumerate(range(0,len(frame['rx']),5000)):
                if label=='none':clean.extend(np.sqrt(np.mean(frame['rx'][a:a+5000].reshape(-1,50)**2,axis=1)))
                if i==0:continue
                for k in kinds:rows[k].append(features[i][k])
                labels.append(label)
            plan.append(dict(seed=seed,label=label,jsr_db=jsr))
    models={};scalers={}
    for kind in kinds:
        X=np.asarray(rows[kind]);sc=StandardScaler().fit(X)
        rf=RandomForestClassifier(n_estimators=trees,max_depth=10,random_state=model_seed,n_jobs=1)
        rf.fit(sc.transform(X),labels);models[kind]=rf;scalers[kind]=sc
    return legacy.StreamBank(cfg,legacy.StreamSettings(),models,scalers,
                             dict(training_plan=plan,model_seed=model_seed,condition=condition,training_seed_offset=seed_offset),
                             float(np.mean(clean)),max(float(np.std(clean)),1e-6))


def adapted(bank,kind):
    b=copy.copy(bank);b.models={'spike':bank.models[kind]};b.scalers={'spike':bank.scalers[kind]}
    return b


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False));temp.replace(path)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--study',choices=['router','bands','features'],default='router')
    ap.add_argument('--config',type=Path)
    ap.add_argument('--output',type=Path)
    ap.add_argument('--smoke',action='store_true')
    a=ap.parse_args()
    default='band_count' if a.study=='bands' else 'stream_router_comparison'
    settings=json.loads((ROOT/(a.config or Path('configs/'+default+'.json'))).read_text())
    if a.smoke:settings.update(conditions=['random'],model_seeds=[811],case_ids=[0,1],trials_per_cell=1,
                                k_list=[16,8],tau_modes=['fixed'],train_streams_per_class=1,trees=10)
    variants=[(16,'fixed',0)]
    if a.study=='bands':
        variants=[(k,tau,0) for k in settings['k_list'] for tau in settings['tau_modes'] if k!=16 or tau=='fixed']
        if settings.get('retrain_control',True):variants.append((16,'fixed',settings.get('reseed_offset',100000)))
    kinds=['spike9','fft7'] if a.study!='features' else ['spike9','fft7','spike_rf4','fft_spec','fft_rf4','fft_rf4_sf20']
    out=ROOT/(a.output or Path('runs/streaming_'+a.study));out.mkdir(parents=True,exist_ok=True)
    lock=dict(study=a.study,settings=settings,smoke=a.smoke,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if (out/'protocol.json').exists() and json.loads((out/'protocol.json').read_text())!=lock:
        raise RuntimeError('Output belongs to a different protocol; choose a new --output.')
    write(out/'protocol.json',lock)
    rows=json.loads((out/'records.json').read_text()) if (out/'records.json').exists() else []
    done={r['id'] for r in rows}
    for bands,tau_mode,seed_offset in variants:
        tau=15.0 if tau_mode=='fixed' else float(bands-1)
        cfg=core.Config(rf_bands=bands,rf_tau=tau)
        for condition in settings['conditions']:
            ci=['fixed','random'].index(condition)
            for model_seed in settings['model_seeds']:
                mp=out/'models'/f'{condition}_{model_seed}_bands{bands}_{tau_mode}_seedoffset{seed_offset}.pkl'
                if mp.exists():bank=pickle.loads(mp.read_bytes())
                else:
                    bank=train(cfg,condition,model_seed,settings.get('train_streams_per_class',20),settings.get('trees',100),kinds,seed_offset)
                    mp.parent.mkdir(exist_ok=True);mp.write_bytes(pickle.dumps(bank,protocol=5))
                for case_id in settings['case_ids']:
                    case=reference.CASES[case_id]
                    for trial in range(settings['trials_per_cell']):
                        uid=f'{condition}|{model_seed}|{case_id}|{trial}|{bands}|{tau_mode}|{seed_offset}'
                        if uid in done:continue
                        ch=reference.derive(20,ci,model_seed,case_id,trial);pay=reference.derive(30,ci,model_seed,case_id,trial)
                        frame=reference.generate(cfg,condition,case,ch,pay);rx=frame['rx']
                        features=feature_rows(rx,cfg);outputs={};decoder_cache={}
                        for kind in kinds:
                            b=adapted(bank,kind);f=[{'spike':v[kind]} for v in features]
                            for policy in settings.get('policies',['current_periodic','refresh_guard8']):
                                v=receiver_v5.process_one(rx,cfg,b,f,'spike',policy,reference.BASE['rescue'])
                                score,_=io.link_v4.evaluate(v['soft'],v['flags']['block_refine2'],'block_refine2',frame['bits'],frame['payloads'],frame['words'],frame['coding'],500,cfg.symbol_rate,frame['transitions'],decoder_cache)
                                counts=v['counts'].copy()
                                if kind.startswith('fft'):
                                    counts.update(delta_sample_updates=0,bank_sample_updates=0,
                                                  fft_calls=len(features)*(20 if kind=='fft_rf4_sf20' else 4),
                                                  fft_length=250 if kind=='fft_rf4_sf20' else 1024)
                                score['counts']=counts;score['actions']=v['actions_by_block'];outputs[kind+'__'+policy]=score
                        for name,mu in [('rule_slow',.02),('rule_fast',.05),('raw',.05)]:
                            c=replace(cfg,ale_mu_sw=mu)
                            v=receiver_v5.process_one(rx,c,bank,[], 'raw' if name=='raw' else 'rule','periodic',reference.BASE['rescue'])
                            score,_=io.link_v4.evaluate(v['soft'],v['flags']['block_refine2'],'block_refine2',frame['bits'],frame['payloads'],frame['words'],frame['coding'],500,cfg.symbol_rate,frame['transitions'],decoder_cache)
                            score['counts']=v['counts'];score['actions']=v['actions_by_block'];outputs[name]=score
                        rows.append(dict(id=uid,condition=condition,model_seed=model_seed,case=case['name'],trial=trial,
                                         bands=bands,tau=tau,training_seed_offset=seed_offset,channel_seed=ch,payload_seed=pay,rx_sha256=hashlib.sha256(rx.tobytes()).hexdigest(),methods=outputs))
                        write(out/'records.json',rows);print(len(rows),uid,flush=True)
    aggregate={}
    for r in rows:
        for name,v in r['methods'].items():
            key=f"bands{r['bands']}|tau{r['tau']}|training_seed_offset{r['training_seed_offset']}|{r['condition']}|{name}"
            d=aggregate.setdefault(key,dict(streams=0,words=0,failed=0,ale_samples=0,router_calls=0))
            d['streams']+=1;d['words']+=len(v['words']);d['failed']+=sum(not w['success'] for w in v['words'])
            for k in ['ale_samples','router_calls']:d[k]+=v['counts'].get(k,0)
    for d in aggregate.values():d['bler']=d['failed']/d['words']
    write(out/'summary.json',dict(smoke=a.smoke,results=aggregate))


if __name__=='__main__':
    main()
