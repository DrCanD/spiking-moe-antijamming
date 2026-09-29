"""CPU, exact-source locked, resumable v5 development/validation runner."""
import common_v5 as io
import argparse,fcntl,gzip,json,multiprocessing,pickle,time,traceback
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from pathlib import Path
import numpy as np
import receiver_v5
STATE={}

def init_worker(out,lock):
    if io.sources()!=lock['spec']['sources']:raise RuntimeError('Source changed after lock')
    STATE.update(out=Path(out),lock=lock,banks={})

def task_worker(t):
    out=STATE['out'];lock=STATE['lock'];p=lock['spec']['protocol'];profile=lock['spec']['profile']
    cfg=io.core.Config(**lock['spec']['config']);key=t['model_key'];started=time.perf_counter()
    if key not in STATE['banks']:
        meta=io.load_model(out,key,lock['fingerprint'])
        STATE['banks'][key]=(pickle.loads((out/'models'/(key+'.pkl')).read_bytes()),meta)
    bank,meta=STATE['banks'][key];rows=[]
    for tr in range(t['start'],t['stop']):
        ch=io.derive(p,profile,'channel',t['model_seed'],t['condition'],t['case_id'],tr)
        pay=io.derive(p,profile,'payload',t['model_seed'],t['condition'],t['case_id'],tr)
        f=io.generator_v4.generate(cfg,ch,pay,t['case'],t['condition']);b=f['metadata']['block_samples']
        feats,feature_time=io.receiver_v4.feature_cache(f['rx'],cfg,b)
        methods={};routes={};decoder_cache={}
        for kind,pol in io.method_plan(p):
            v=receiver_v5.process_one(f['rx'],cfg,bank,feats,kind,pol,p['rescue'])
            score,eff=io.link_v4.evaluate(v['soft'],v['flags'][p['mask']],p['mask'],f['bits'],f['payloads'],f['words'],f['coding'],b//cfg.sps,cfg.symbol_rate,f['transitions'],decoder_cache)
            segs=io.run_v4.metrics_for_segments(f['bits'],v['soft'],eff,f['segments'])
            score.update(segments=segs,first_nb_correct=segs[1]['correct_retained_fraction'] if segs[1]['name']=='narrowband' else None,counts=v['counts'])
            methods[io.method_name(kind,pol)]=score
            routes[kind+'__'+pol]={k:value for k,value in v.items() if k not in ('soft','flags')}
        rows.append(dict(model_seed=t['model_seed'],condition=t['condition'],case=t['case']['name'],trial=tr,channel_seed=ch,payload_seed=pay,
            rx_sha256=io.sha(np.ascontiguousarray(f['rx']).tobytes()),coding=f['coding'],channel=f['metadata'],transitions=f['transitions'],
            shared_feature_simulation_wall_s=feature_time,unique_decoder_inputs=len(decoder_cache),methods=methods,routes=routes))
    return dict(task=t,lock_fingerprint=lock['fingerprint'],model_sha256=meta['sha256'],rows=rows,elapsed_s=time.perf_counter()-started)

def run(args):
    out=args.output;out.mkdir(parents=True,exist_ok=True);p=io.read(io.ROOT/'protocol_v5.json');sel=None
    if out.resolve().is_relative_to(io.ROOT.resolve()):raise ValueError('Output must be outside source directory')
    if args.profile=='validation':
        if args.selection is None:raise RuntimeError('Frozen development selection required')
        sel=io.read(args.selection);proof=sel.copy();checksum=proof.pop('selection_sha256',None)
        if checksum!=io.sha(io.canonical(proof)) or sel['sources']!=io.sources() or sel['profile']!='development':raise RuntimeError('Selection provenance mismatch')
        if not sel['sparse_candidate_passed']:raise RuntimeError('DEV found no qualifying sparse policy. Validation is intentionally not run.')
    with (out/'.writer.lock').open('a') as writer:
        fcntl.flock(writer,fcntl.LOCK_EX|fcntl.LOCK_NB)
        lock=io.lock_for(out,p,args.profile,sel);started=time.perf_counter();f=p['profiles'][args.profile]
        tasks=io.make_tasks(lock)
        if (out/'tasks.json').exists() and io.read(out/'tasks.json')!=tasks:raise RuntimeError('Task file changed')
        io.write(out/'tasks.json',tasks)
        def progress(stage,done,total,message):
            elapsed=time.perf_counter()-started
            io.write(out/'progress.json',dict(stage=stage,done=done,total=total,message=message,elapsed_session_s=elapsed,lock_fingerprint=lock['fingerprint']))
            print(f'[{stage}] {done}/{total} | {elapsed/60:.1f} min | {message}',flush=True)
        try:
            keys=[(ms,c) for ms in f['model_seeds'] for c in p['conditions']]
            for i,(ms,c) in enumerate(keys):
                key=io.model_key(ms,c)
                if io.load_model(out,key,lock['fingerprint']) is not None:progress('train',i+1,len(keys),'resumed '+key);continue
                progress('train',i,len(keys),key);sd=io.derive(p,args.profile,'training',ms,c)
                bank=io.receiver_v4.train(io.core.Config(**lock['spec']['config']),sd,ms,c,f['train_streams_per_class'],f['trees'])
                raw=pickle.dumps(bank,protocol=5);io.atomic(out/'models'/(key+'.pkl'),raw)
                io.write(out/'models'/(key+'.meta.json'),dict(key=key,lock_fingerprint=lock['fingerprint'],sha256=io.sha(raw),metadata=bank.metadata))
            pending=[];metas={}
            for t in tasks:
                key=t['model_key']
                if key not in metas:metas[key]=io.load_model(out,key,lock['fingerprint'])
                if io.load_shard(out,t,lock,metas[key]) is None:pending.append(t)
            done=len(tasks)-len(pending);progress('evaluate',done,len(tasks),'validated committed results')
            if pending:
                with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn'),initializer=init_worker,initargs=(str(out),lock)) as pool:
                    it=iter(pending);futures={}
                    def submit():
                        t=next(it,None)
                        if t is not None:futures[pool.submit(task_worker,t)]=t
                    for _ in range(args.workers*2):submit()
                    while futures:
                        ready,_=wait(futures,timeout=30,return_when=FIRST_COMPLETED)
                        if not ready:progress('evaluate',done,len(tasks),'workers active');continue
                        for fut in ready:
                            t=futures.pop(fut);v=fut.result();raw=gzip.compress(io.canonical(v),mtime=0,compresslevel=6);path=io.shard_path(out,t)
                            io.atomic(path,raw);io.atomic(Path(str(path)+'.sha256'),(io.sha(raw)+'\n').encode());done+=1;submit()
                        progress('evaluate',done,len(tasks),'committed')
            import summary_v5
            summary_v5.summarize(out,lock)
            progress('complete',len(tasks),len(tasks),'COMPLETE')
            (out/'failure.json').unlink(missing_ok=True)
        except BaseException as exc:
            io.write(out/'failure.json',dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()));raise

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--profile',choices=['smoke','local_pilot','development','validation'],required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--selection',type=Path);ap.add_argument('--workers',type=int,default=2)
    args=ap.parse_args()
    if args.workers<1:raise ValueError('workers >= 1')
    run(args)
