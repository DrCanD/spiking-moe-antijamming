"""v3 fixed-point frontend (Q1.17 rotations, Q20 state; dense and gate paths) against the frozen dense Q2.30/Q20 frontend,
with the frozen v5 classifier/expert. Existing 192 streams are regression, 96 additional streams are fresh checks.
Summary keys: dense_* = frozen reference, lazy_* = v3 gate path (the measured hardware), v3dense_* = v3 dense path.
new_failures / different_action_streams count against the reference and are the maximum over both v3 paths.
"""
from pathlib import Path
import os,sys,json,pickle,gzip,hashlib,argparse
from concurrent.futures import ProcessPoolExecutor
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'software'))
import run_experiment as sim
import receiver_v5
from fixed_reference import run,run_v3,quantize
OUT=ROOT/'evidence'

def one(t):
    panel,cond,ms,ci,tr=t;cfg=sim.core.Config();cidx=['fixed','random'].index(cond)
    ch=sim.derive(20 if panel=='regression' else 220,cidx,ms,ci,tr)
    pay=sim.derive(30 if panel=='regression' else 230,cidx,ms,ci,tr)
    f=sim.generate(cfg,cond,sim.CASES[ci],ch,pay);x=quantize(f['rx']);rx=x.astype(float)/1024
    sp,c_ref,_,_=run(x);sp3,c3,w,final=run_v3(x);assert (sp3==sp).all();nb=len(rx)//5000;om=2*np.pi*np.linspace(25e3,475e3,16)/1e6
    c=np.stack([c_ref[0],c3[1],c3[0]])      # reference dense, v3 gate (lazy), v3 dense
    bank=pickle.loads((ROOT/'software/models'/f'{cond}_{ms}.pkl').read_bytes());methods={};cache={}
    for mode,name in enumerate(['dense_q20','lazy_q20','v3dense_q20']):
        features=[]
        for b in range(nb):
            rates=c[mode,b*20:(b+1)*20]/(250*om[None,:]/(2*np.pi))
            features.append({'spike':np.array(sim.first_features(sp[b*5000:(b+1)*5000],cfg)+list(sim.core.rf_features(rates)))})
        result=receiver_v5.process_one(rx,cfg,bank,features,'spike','refresh_guard8',sim.BASE['rescue'])
        score,_=sim.io.link_v4.evaluate(result['soft'],result['flags']['block_refine2'],'block_refine2',f['bits'],f['payloads'],f['words'],f['coding'],500,1e5,f['transitions'],cache)
        score.update(actions=result['actions_by_block'],router_calls=result['counts']['router_calls']);methods[name]=score
    row=dict(panel=panel,condition=cond,model_seed=ms,case=sim.CASES[ci]['name'],trial=tr,channel_seed=ch,payload_seed=pay,n_samples=len(x),
        q10_sha256=hashlib.sha256(x.tobytes()).hexdigest(),input_clipped_samples=int(np.count_nonzero((f['rx']*1024>32767)|(f['rx']*1024<-32768))),
        different_count_cells=int(np.count_nonzero(c[0]!=c[1])),different_count_cells_v3dense=int(np.count_nonzero(c[0]!=c[2])),
        work=dict(zip(['updates','skips','long_gaps','spike_time_differences','max_state_error_LSB','bank_skips'],map(int,w))),methods=methods)
    p=OUT/'quality_shards'/f'{panel}_{cond}_{ms}_{ci}_{tr}.json.gz';p.parent.mkdir(exist_ok=True)
    p.write_bytes(gzip.compress(json.dumps(row).encode(),mtime=0));return row

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=4);a=ap.parse_args()
    protocol=dict(reference='frozen dense Q20 state, Q2.30 coefficients',candidate='v3: Q1.17 coefficients (18 bit), Q20 state; dense path and lazy path (Q1.17 jump table; 64 LSB sleeping margin; max jump 256)',
        source_sha256={str(p.relative_to(ROOT)):sim.digest(p.read_bytes()) for p in [ROOT/'fixed_reference.py',ROOT/'validate_fixed.py']},
        quantization='round-to-nearest-even Q6.10 then saturate int16',regression_streams=192,fresh_streams=96,
        quality='paired BLER, CRC, actions; no parameter tuning after this run',scope='Fixed-point frontend only; experts/decoder simulated in existing CPU reference, not a full fixed-point receiver')
    sim.dump(OUT/'fixed_quality_protocol.json',protocol)
    tasks=[(panel,cond,ms,ci,tr) for panel,nt in [('regression',4),('fresh',2)] for cond in ['fixed','random'] for ms in [811,821,823] for ci in range(8) for tr in range(nt)]
    rows=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for i,r in enumerate(pool.map(one,tasks)):
            rows.append(r)
            if (i+1)%24==0:print(i+1,'/',len(tasks),flush=True)
    summary=[]
    for panel in ['regression','fresh']:
        for cond in ['fixed','random']:
            rs=[r for r in rows if r['panel']==panel and r['condition']==cond];wordpairs=[];v3pairs=[]
            for r in rs:wordpairs.extend(zip(r['methods']['dense_q20']['words'],r['methods']['lazy_q20']['words']))
            for r in rs:v3pairs.extend(zip(r['methods']['dense_q20']['words'],r['methods']['v3dense_q20']['words']))
            nw=len(wordpairs);a0=sum(not a['success'] for a,b in wordpairs);b0=sum(not b['success'] for a,b in wordpairs)
            v3new=sum(a['success'] and not b['success'] for a,b in v3pairs)
            v3act=sum(r['methods']['dense_q20']['actions']!=r['methods']['v3dense_q20']['actions'] for r in rs)
            summary.append(dict(panel=panel,condition=cond,streams=len(rs),words=nw,dense_failed=a0,lazy_failed=b0,
                dense_bler_pct=100*a0/nw,lazy_bler_pct=100*b0/nw,
                new_failures=max(v3new,sum(a['success'] and not b['success'] for a,b in wordpairs)),v3dense_new_failures=v3new,
                recoveries=sum(not a['success'] and b['success'] for a,b in wordpairs),
                different_action_streams=max(v3act,sum(r['methods']['dense_q20']['actions']!=r['methods']['lazy_q20']['actions'] for r in rs)),
                v3dense_different_action_streams=v3act,v3dense_failed=sum(not b['success'] for a,b in v3pairs),
                v3dense_undetected=sum(r['methods']['v3dense_q20']['undetected_wrong'] for r in rs),
                different_count_cells_v3dense=sum(r['different_count_cells_v3dense'] for r in rs),
                bank_skip_fraction=sum(r['work']['bank_skips'] for r in rs)/sum(r['n_samples'] for r in rs),
                different_count_cells=sum(r['different_count_cells'] for r in rs),spike_time_differences=sum(r['work']['spike_time_differences'] for r in rs),
                max_state_error_LSB=max(r['work']['max_state_error_LSB'] for r in rs),
                rotation_update_saving=1-sum(r['work']['updates'] for r in rs)/(16*sum(r['n_samples'] for r in rs)),
                dense_undetected=sum(r['methods']['dense_q20']['undetected_wrong'] for r in rs),lazy_undetected=sum(r['methods']['lazy_q20']['undetected_wrong'] for r in rs)))
    sim.dump(OUT/'fixed_quality_summary.json',summary);print(json.dumps(summary,indent=2),flush=True)
