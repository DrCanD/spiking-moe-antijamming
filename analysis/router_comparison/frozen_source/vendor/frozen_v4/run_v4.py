"""Fail-closed, checksum-locked, process-based resumable v4 experiments."""
import os
for _k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[_k]='1'
import argparse,fcntl,gzip,hashlib,json,multiprocessing,pickle,platform,time,traceback
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
import numpy as np
import core,receiver_v4,link_v4,generator_v4

ROOT=Path(__file__).resolve().parent;STATE={}
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def sha(x):return hashlib.sha256(x).hexdigest()
def file_sha(p):return sha(Path(p).read_bytes())
def read(p):return json.loads(Path(p).read_text())
def atomic(p,b):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp.'+str(os.getpid()))
    with tmp.open('wb') as f:f.write(b);f.flush();os.fsync(f.fileno())
    tmp.replace(p)
def write(p,x):atomic(p,canonical(x)+b'\n')
def sources():return {p.name:file_sha(p) for p in sorted(ROOT.iterdir()) if p.is_file() and p.suffix in ('.py','.json','.txt','.md')}
def dependencies():
    wanted=dict(line.strip().split('==') for line in (ROOT/'requirements.txt').read_text().splitlines() if line.strip() and not line.startswith('#'))
    got={k:version(k) for k in wanted}
    if got!=wanted:raise RuntimeError('Exact dependencies required: '+str(got))
    return got
def derive(p,profile,stage,ms,condition,case=0,trial=0):
    seq=[p['seed_namespace'],p['profile_ids'][profile],p['stage_ids'][stage],ms,p['conditions'].index(condition),case,trial]
    return int(np.random.SeedSequence(seq).generate_state(1,dtype=np.uint64)[0])
def model_key(ms,c):return f'm{ms}_{c}'
def lock_for(out,p,profile,selection):
    spec=dict(protocol=p,profile=profile,selection=selection,sources=sources(),dependencies=dependencies(),python=platform.python_version(),config=asdict(core.Config()))
    lock=dict(fingerprint=sha(canonical(spec)),spec=spec)
    path=out/'lock.json'
    if path.exists():
        old=read(path)
        if old.get('fingerprint')!=sha(canonical(old.get('spec'))):raise RuntimeError('Existing lock is corrupted')
        if old['fingerprint']!=lock['fingerprint']:raise RuntimeError('Source/protocol/environment/selection differs. Never mix with existing output; restore exact build/runtime.')
    else:
        if list(out.glob('shards/*.gz')) or list(out.glob('models/*')):raise RuntimeError('Evidence without lock; refusing adoption')
        write(path,lock);write(out/'environment.json',dict(python=platform.python_version(),platform=platform.platform(),dependencies=spec['dependencies'],thread_limits={k:os.environ[k] for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS')}))
    return lock
def load_model(out,key,fingerprint,bytes_required=True):
    path=out/'models'/(key+'.pkl');mp=out/'models'/(key+'.meta.json')
    if not mp.exists():return None
    m=read(mp)
    if m['lock_fingerprint']!=fingerprint or m['key']!=key:raise RuntimeError('Model identity mismatch')
    if bytes_required and (not path.exists() or file_sha(path)!=m['sha256']):raise RuntimeError('Model bytes corrupted/missing')
    return m
def make_tasks(lock):
    p=lock['spec']['protocol'];profile=lock['spec']['profile'];f=p['profiles'][profile];tasks=[]
    for ms in f['model_seeds']:
        for c in p['conditions']:
            for ci,case in enumerate(p['cases']):
                if case['name'] not in f['cases']:continue
                for start in range(0,f['trials'],f['chunk']):
                    stop=min(start+f['chunk'],f['trials']);key=model_key(ms,c)
                    tasks.append(dict(id=f'{key}_{case["name"]}_{start:04d}_{stop:04d}',model_key=key,model_seed=ms,condition=c,case=case,case_id=ci,start=start,stop=stop))
    return tasks
def shard_path(out,t):return out/'shards'/(t['id']+'.json.gz')
def load_shard(out,t,lock,m):
    path=shard_path(out,t);side=Path(str(path)+'.sha256')
    if not side.exists():return None
    if not path.exists() or file_sha(path)!=side.read_text().strip():raise RuntimeError('Shard checksum mismatch: '+t['id'])
    v=json.loads(gzip.decompress(path.read_bytes()))
    if v['task']!=t or v['lock_fingerprint']!=lock['fingerprint'] or v['model_sha256']!=m['sha256'] or len(v['rows'])!=t['stop']-t['start']:raise RuntimeError('Shard identity mismatch: '+t['id'])
    if [r['trial'] for r in v['rows']]!=list(range(t['start'],t['stop'])):raise RuntimeError('Shard trial identity mismatch')
    return v
def init_worker(out,lock):
    if sources()!=lock['spec']['sources']:raise RuntimeError('Source changed after lock')
    STATE.update(out=Path(out),lock=lock,banks={})
def method_plan(lock):
    p=lock['spec']['protocol'];selected=lock['spec']['selection']
    if lock['spec']['profile'] in ('smoke','development'):
        plans=[(k,pol,mask) for k in p['frontends'] for pol in ['legacy8']+p['policy_candidates'] for mask in p['masks']]
    else:
        mask=selected['mask'];pol=selected['policy']
        combos=list(dict.fromkeys([('legacy8','legacy_any'),('guard8','legacy_any'),('guard8',mask),('periodic',mask),(pol,mask)]))
        plans=[(k,po,ma) for k in p['frontends'] for po,ma in combos]
    return plans+[('rule','rule',ma) for ma in p['masks']]+[('raw','raw','legacy_any')]
def metrics_for_segments(bits,soft,mask,segs):
    hard=soft>0;res=[]
    for i,s in enumerate(segs):
        a=s['start_symbol'];z=s['end_symbol'];kept=~mask[a:z];err=int(np.count_nonzero((hard[a:z]!=bits[a:z])&kept));ret=int(kept.sum())
        res.append(dict(index=i,**s,n_bits=z-a,retained_bits=ret,retained_errors=err,correct_retained_fraction=(ret-err)/(z-a)))
    return res
def task_worker(t):
    out=STATE['out'];lock=STATE['lock'];p=lock['spec']['protocol'];profile=lock['spec']['profile'];cfg=core.Config(**lock['spec']['config']);key=t['model_key'];start=time.perf_counter()
    if key not in STATE['banks']:
        meta=load_model(out,key,lock['fingerprint']);bank=pickle.loads((out/'models'/(key+'.pkl')).read_bytes());STATE['banks'][key]=(bank,meta)
    bank,meta=STATE['banks'][key];rows=[];plan=method_plan(lock)
    for tr in range(t['start'],t['stop']):
        ch=derive(p,profile,'channel',t['model_seed'],t['condition'],t['case_id'],tr);pay=derive(p,profile,'payload',t['model_seed'],t['condition'],t['case_id'],tr)
        f=generator_v4.generate(cfg,ch,pay,t['case'],t['condition']);b=f['metadata']['block_samples'];feats,feature_time=receiver_v4.feature_cache(f['rx'],cfg,b)
        outputs={};methods={};routes={};decoder_cache={}
        for kind,pol,mask in plan:
            route=kind+'__'+pol
            if route not in outputs:
                outputs[route]=receiver_v4.process_one(f['rx'],cfg,bank,feats,kind,pol)
                routes[route]={k:v for k,v in outputs[route].items() if k not in ('soft','flags')}
            v=outputs[route]
            score,eff=link_v4.evaluate(v['soft'],v['flags'][mask],mask,f['bits'],f['payloads'],f['words'],f['coding'],b//cfg.sps,cfg.symbol_rate,f['transitions'],decoder_cache)
            segs=metrics_for_segments(f['bits'],v['soft'],eff,f['segments']);score.update(segments=segs,first_nb_correct=segs[1]['correct_retained_fraction'] if segs[1]['name']=='narrowband' else None,counts=v['counts'])
            methods[route+'__'+mask]=score
        rows.append(dict(model_seed=t['model_seed'],condition=t['condition'],case=t['case']['name'],trial=tr,channel_seed=ch,payload_seed=pay,
            rx_sha256=sha(np.ascontiguousarray(f['rx']).tobytes()),coding=f['coding'],channel=f['metadata'],transitions=f['transitions'],
            shared_feature_simulation_wall_s=feature_time,unique_decoder_inputs=len(decoder_cache),methods=methods,routes=routes))
    return dict(task=t,lock_fingerprint=lock['fingerprint'],model_sha256=meta['sha256'],rows=rows,elapsed_s=time.perf_counter()-start)
def all_rows(out,lock,require_models=False):
    rows=[];units=set();models={}
    for t in read(out/'tasks.json'):
        key=t['model_key']
        if key not in models:models[key]=load_model(out,key,lock['fingerprint'],require_models)
        m=models[key];s=load_shard(out,t,lock,m)
        if s is None:raise RuntimeError('Incomplete: '+t['id'])
        for r in s['rows']:
            u=(r['model_seed'],r['condition'],r['case'],r['trial'])
            if u in units:raise RuntimeError('Duplicate experimental unit')
            units.add(u);rows.append(r)
    return rows
def run(args):
    out=args.output;out.mkdir(parents=True,exist_ok=True);p=read(ROOT/'protocol.json');sel=None
    if args.profile=='overnight':
        if args.selection is None:raise RuntimeError('Locked DEV selection required')
        sel=read(args.selection);proof=sel.copy();checksum=proof.pop('selection_sha256',None)
        if checksum!=sha(canonical(proof)) or sel['sources']!=sources() or sel['profile']!='development':raise RuntimeError('Selection provenance mismatch')
    with (out/'.writer.lock').open('a') as writer:
        fcntl.flock(writer,fcntl.LOCK_EX|fcntl.LOCK_NB);lock=lock_for(out,p,args.profile,sel);started=time.perf_counter();f=p['profiles'][args.profile]
        tasks=make_tasks(lock);write(out/'tasks.json',tasks)
        def progress(stage,done,total,message):
            elapsed=time.perf_counter()-started;record=dict(stage=stage,done=done,total=total,message=message,elapsed_session_s=elapsed,lock_fingerprint=lock['fingerprint'])
            write(out/'progress.json',record);print(f'[{stage}] {done}/{total} | {elapsed/60:.1f} min | {message}',flush=True)
        try:
            keys=[(ms,c) for ms in f['model_seeds'] for c in p['conditions']]
            for i,(ms,c) in enumerate(keys):
                key=model_key(ms,c)
                if load_model(out,key,lock['fingerprint']) is not None:progress('train',i+1,len(keys),'resumed '+key);continue
                progress('train',i,len(keys),key);sd=derive(p,args.profile,'training',ms,c)
                bank=receiver_v4.train(core.Config(**lock['spec']['config']),sd,ms,c,f['train_streams_per_class'],f['trees'])
                raw=pickle.dumps(bank,protocol=5);atomic(out/'models'/(key+'.pkl'),raw)
                write(out/'models'/(key+'.meta.json'),dict(key=key,lock_fingerprint=lock['fingerprint'],sha256=sha(raw),metadata=bank.metadata))
            pending=[];validated_models={}
            for t in tasks:
                key=t['model_key']
                if key not in validated_models:validated_models[key]=load_model(out,key,lock['fingerprint'])
                m=validated_models[key]
                if load_shard(out,t,lock,m) is None:pending.append(t)
            done=len(tasks)-len(pending);progress('evaluate',done,len(tasks),'validated committed results')
            if pending:
                ctx=multiprocessing.get_context('spawn')
                with ProcessPoolExecutor(max_workers=args.workers,mp_context=ctx,initializer=init_worker,initargs=(str(out),lock)) as pool:
                    it=iter(pending);futures={}
                    def submit():
                        t=next(it,None)
                        if t is not None:futures[pool.submit(task_worker,t)]=t
                    for _ in range(args.workers*2):submit()
                    while futures:
                        ready,_=wait(futures,timeout=30,return_when=FIRST_COMPLETED)
                        if not ready:progress('evaluate',done,len(tasks),'workers active');continue
                        for fut in ready:
                            t=futures.pop(fut);v=fut.result();raw=gzip.compress(canonical(v),mtime=0,compresslevel=6);path=shard_path(out,t)
                            atomic(path,raw);atomic(Path(str(path)+'.sha256'),(sha(raw)+'\n').encode());done+=1;submit()
                        progress('evaluate',done,len(tasks),'committed')
            import summary_v4
            summary_v4.summarize(out,lock)
            progress('complete',len(tasks),len(tasks),'COMPLETE')
            (out/'failure.json').unlink(missing_ok=True)
        except BaseException as exc:
            write(out/'failure.json',dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()));raise

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--profile',choices=['smoke','development','overnight'],required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--selection',type=Path);ap.add_argument('--workers',type=int,default=2);args=ap.parse_args()
    if args.workers<1:raise ValueError('workers >=1')
    run(args)
