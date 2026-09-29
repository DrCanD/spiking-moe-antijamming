"""Frozen-v4 dependencies and common v5 evidence IO."""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[_key]='1'
import sys,json,platform
from pathlib import Path
from dataclasses import asdict
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'frozen_v4'))
import core,receiver_v4,streaming_v3,link_v4,generator_v4,run_v4
canonical=run_v4.canonical;sha=run_v4.sha;file_sha=run_v4.file_sha
read=run_v4.read;write=run_v4.write;atomic=run_v4.atomic
dependencies=run_v4.dependencies;derive=run_v4.derive;model_key=run_v4.model_key
make_tasks=run_v4.make_tasks;shard_path=run_v4.shard_path
load_model=run_v4.load_model;load_shard=run_v4.load_shard;all_rows=run_v4.all_rows

def sources():
    paths=[p for p in ROOT.iterdir() if p.is_file() and p.suffix in ('.py','.json','.txt','.md')]
    paths += [p for p in (ROOT/'frozen_v4').iterdir() if p.is_file()]
    return {str(p.relative_to(ROOT)):file_sha(p) for p in sorted(paths)}

def assert_frozen():
    manifest=read(ROOT/'frozen_v4_manifest.json')
    assert {p.name for p in (ROOT/'frozen_v4').iterdir() if p.is_file()}==set(manifest)
    for name,h in manifest.items():
        if file_sha(ROOT/'frozen_v4'/name)!=h:raise RuntimeError('Frozen v4 changed: '+name)

def lock_for(out,p,profile,selection):
    assert_frozen()
    spec=dict(protocol=p,profile=profile,selection=selection,sources=sources(),dependencies=dependencies(),python=platform.python_version(),config=asdict(core.Config()))
    lock=dict(fingerprint=sha(canonical(spec)),spec=spec);path=out/'lock.json'
    if path.exists():
        old=read(path)
        if old.get('fingerprint')!=sha(canonical(old.get('spec'))) or old['fingerprint']!=lock['fingerprint']:
            raise RuntimeError('Source/protocol/environment/selection differs. Refusing mixed resume.')
    else:
        if list(out.glob('shards/*')) or list(out.glob('models/*')):raise RuntimeError('Evidence without lock')
        write(path,lock)
        write(out/'environment.json',dict(python=platform.python_version(),platform=platform.platform(),dependencies=spec['dependencies'],thread_limits={k:os.environ[k] for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS')}))
    return lock

def method_plan(p):
    return [(k,pol) for k in p['frontends'] for pol in p['policies']]+[('rule','rule'),('raw','raw')]

def method_name(kind,policy):return kind+'__'+policy+'__block_refine2'
