"""Scientific invariants for the new block-causal rescue implementation."""
import inspect,subprocess,sys,tempfile,gzip,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import common_v5 as io
import receiver_v5 as rx
import summary_v5

def main():
    io.assert_frozen();p=io.read(io.ROOT/'protocol_v5.json');cfg=io.core.Config();checks={}
    # Run original verification against the unmodified original v4 tree.
    result=subprocess.run([sys.executable,str(io.ROOT/'frozen_v4/verify_v4.py')],capture_output=True,text=True,check=True)
    old=json.loads(result.stdout);assert old['status']=='PASS';checks['frozen_v4_and_original_13_checks']=True
    bank=io.receiver_v4.train(cfg,5251841,599,'random',n_train=2,trees=10)
    frame=io.generator_v4.generate(cfg,5259182,5259183,p['cases'][1],'random')
    x=frame['rx'];cut=65000;b=5000
    features,_=io.receiver_v4.feature_cache(x,cfg,b);prefix,_=io.receiver_v4.feature_cache(x[:cut],cfg,b)
    for ff,pp in zip(features,prefix):
        for k in p['frontends']:assert np.array_equal(ff[k],pp[k])
    for kind,policy in io.method_plan(p):
        full=rx.process_one(x,cfg,bank,features,kind,policy,p['rescue'])
        pre=rx.process_one(x[:cut],cfg,bank,prefix,kind,policy,p['rescue'])
        assert np.array_equal(full['soft'][:cut//cfg.sps],pre['soft']),(kind,policy)
        for mask in ['legacy_any','block_refine2']:assert np.array_equal(full['flags'][mask][:cut//cfg.sps],pre['flags'][mask])
        assert [d for d in full['decisions'] if d['observed_end_sample']<=cut]==pre['decisions']
        assert [d for d in full['risk_log'] if d['observed_end_sample']<=cut]==pre['risk_log']
        assert full['counts']['router_calls']==len(full['decisions'])
        if kind not in ('raw','rule'):
            active=sum(a in ('slow','fast','rescue_fast','rescue_pass') for a in full['actions_by_block'])
            assert full['counts']['ale_samples']==active*b
        for d in full['decisions']:
            assert d['apply_sample']==d['observed_end_sample']
            if 'affected_start_sample' in d:assert d['output_available_sample']==d['affected_start_sample']+b
        if policy in ('lag_periodic','lag_guard8'):
            legacy=io.receiver_v4.process_one(x,cfg,bank,features,kind,policy.removeprefix('lag_'))
            assert np.array_equal(full['soft'],legacy['soft'])
            for mask in ['legacy_any','block_refine2']:assert np.array_equal(full['flags'][mask],legacy['flags'][mask])
            assert all(full['counts'][k]==v for k,v in legacy['counts'].items())
    checks['all_16_methods_prefix_causal_at_block_boundaries']=True
    checks['original_lag_baselines_exact']=True
    checks['one_ALE_update_per_active_sample_all_work_charged']=True
    checks['current_block_observation_and_release_timestamps']=True
    assert set(inspect.signature(rx.process_one).parameters)=={'rx','cfg','bank','feats','kind','policy','rescue'}
    assert not {'bits','truth','payloads','words','transitions','condition','offset'}&set(inspect.signature(rx.process_one).parameters)
    checks['no_truth_or_packet_positions_in_receiver']=True
    # Constant received blocks isolate startup/hold semantics from event noise.
    constant=np.ones(6*b);fake_features=[{'spike':np.zeros(9),'analog_delta':np.zeros(9)} for _ in range(6)]
    class PassBank:
        settings=bank.settings;blank_mean=bank.blank_mean;blank_std=bank.blank_std
        def standardized(self,kind,f):return f
        def predict_features(self,kind,f):return 'none',1.
    fake=PassBank();r=rx.process_one(constant,cfg,fake,fake_features,'spike','rescue_guard8',p['rescue'])
    assert [d['risk'] for d in r['risk_log']]==[True,True,False,False,False,False]
    assert r['counts']['rescue_blocks']==2 and r['counts']['detector_events']==0
    jumped=constant.copy();jumped[3*b:]*=3
    r=rx.process_one(jumped,cfg,fake,fake_features,'spike','rescue_guard8',p['rescue'])
    assert [d['risk'] for d in r['risk_log']]==[True,True,False,True,True,False]
    # Force rejection: attempted adaptation must still count.
    reject=dict(p['rescue'],residual_power_reduction_threshold=2.)
    r=rx.process_one(constant,cfg,fake,fake_features,'spike','rescue_guard8',reject)
    assert r['counts']['rescue_rejected']==2 and r['counts']['ale_samples']==2*b
    assert np.array_equal(r['soft'][:2*b//10],np.ones(2*b//10))
    checks['startup_exact_two_blocks_and_event_exact_two_blocks']=True
    checks['rejected_rescue_still_charged_and_raw_output']=True
    # Selection must not hide a bad first packet behind aggregate improvements.
    synthetic=[]
    for kind in p['frontends']:
        for c in p['conditions']:
            for ca in p['profiles']['development']['cases']:
                for pol in p['selection']['candidates']+[p['selection']['reference']]:
                    synthetic.append(dict(condition=c,case=ca,method=io.method_name(kind,pol),bler=.1,first_word_bler=.1,crossing_word_bler=None if ca=='clean' else .1,mean_stream_correct_retained=.9,first_nb_correct=None if ca=='clean' else .9,undetected_wrong=0,mean_rf_calls=30 if pol=='current_periodic' else 5))
    assert summary_v5.choose(synthetic,p)['policy']=='current_guard8'
    for r in synthetic:
        if r['condition']=='fixed' and r['case']=='nominal' and '__current_periodic__' not in r['method']:r['first_word_bler']=.13
    assert not summary_v5.choose(synthetic,p)['sparse_candidate_passed']
    checks['first_word_gate_veto_and_periodic_fallback']=True
    vals=[io.derive(p,pr,st,601,'fixed') for pr in p['profiles'] for st in p['stage_ids']]
    assert len(vals)==len(set(vals));checks['new_namespace_and_profile_stage_separation']=True
    with tempfile.TemporaryDirectory() as temp:
        out=Path(temp);lock=io.lock_for(out,p,'smoke',None);assert io.lock_for(out,p,'smoke',None)['fingerprint']==lock['fingerprint']
        changed=json.loads(json.dumps(p));changed['rescue']['hold_blocks']+=1
        try:io.lock_for(out,changed,'smoke',None)
        except RuntimeError:pass
        else:raise AssertionError('Changed protocol allowed to resume')
        t=io.make_tasks(lock)[0];m={'sha256':'test-model'};v=dict(task=t,lock_fingerprint=lock['fingerprint'],model_sha256=m['sha256'],rows=[{'trial':0}])
        blob=gzip.compress(io.canonical(v));path=io.shard_path(out,t);io.atomic(path,blob)
        assert io.load_shard(out,t,lock,m) is None # missing commit sidecar is uncommitted
        io.atomic(Path(str(path)+'.sha256'),(io.sha(blob)+'\n').encode());assert io.load_shard(out,t,lock,m)==v
        io.atomic(path,blob+b'corrupt')
        try:io.load_shard(out,t,lock,m)
        except RuntimeError:pass
        else:raise AssertionError('Corrupt checkpoint accepted')
    checks['lock_roundtrip_changed_protocol_and_corrupt_shard_rejection']=True
    print(json.dumps(dict(status='PASS',checks=checks),indent=2),flush=True)
    return checks

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path);args=ap.parse_args()
    checks=main()
    if args.output:io.write(args.output,dict(status='PASS',checks=checks,sources=io.sources(),dependencies=io.dependencies()))
