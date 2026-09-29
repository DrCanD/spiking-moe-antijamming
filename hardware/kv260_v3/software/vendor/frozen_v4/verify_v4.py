"""Scientific invariants, causal-prefix checks and v3 baseline equivalence."""
import hashlib,inspect,json,tempfile,zlib
from dataclasses import replace
from pathlib import Path
import numpy as np
import core,streaming_v3 as legacy,receiver_v4 as rx,link_v4 as link,generator_v4,summary_v4,run_v4 as io

def main():
    checks={};cfg=core.Config();p=io.read(io.ROOT/'protocol.json')
    for name,h in io.read(io.ROOT/'baseline_sources.json')['files'].items():assert io.file_sha(io.ROOT/name)==h
    checks['frozen_v3_sources']=True
    assert zlib.crc32(b'123456789')==0xcbf43926
    bits,payloads,words,meta=link.make_bits(404,12000,offset=499);soft=2*bits.astype(float)-1;flags=np.zeros(len(bits),bool)
    score,eff=link.evaluate(soft,flags,'legacy_any',bits,payloads,words,meta)
    assert score['failed_words']==0 and score['useful_bits']==len(words)*155*8 and score['undetected_wrong']==0
    assert score['goodput_per_transmitted_symbol']==len(words)*1240/len(bits)
    checks['clean_crc_and_overhead']=True
    message=link.crc_message(payloads[0]);original=words[0]
    for e,v in [(48,0),(0,96),(20,56),(0,0)]:
        received=original.copy();erase=np.zeros(255,bool);erase[:v]=True;received[:v+e]^=123
        d=link.decode_word(received,erase);assert d['crc_pass'] and d['payload']==message
    er=np.zeros(255,bool);er[:97]=True;d=link.decode_word(original,er);assert not d['attempted'] and not d['returned']
    checks['rs_error_erasure_radius_and_overbudget']=True
    invalid=bytearray(message);invalid[0]^=1;cw=np.frombuffer(bytes(link._codec(159).encode(bytes(invalid))),np.uint8)
    d=link.decode_word(cw,np.zeros(255,bool));assert d['returned'] and not d['crc_pass']
    other=payloads[0].copy();other[0]^=1;cw=np.frombuffer(bytes(link._codec(159).encode(link.crc_message(other))),np.uint8)
    altered=soft.copy();a=meta['offset_symbols'];altered[a:a+2040]=2*np.unpackbits(cw).astype(float)-1
    s,_=link.evaluate(altered,flags,'legacy_any',bits,payloads,words,meta)
    assert s['undetected_wrong']==1 and s['failed_words']==1 and s['words'][0]['crc_pass'] and not s['words'][0]['success']
    checks['crc_rejection_and_accepted_wrong_scoring']=True
    crossing=np.zeros(len(bits),bool);crossing[499]=True;crossing[500]=True
    assert link.byte_mask(crossing,499,len(words),'block_refine2')[0,0]
    crossing[500]=False;assert not link.byte_mask(crossing,499,len(words),'block_refine2')[0,0]
    checks['byte_crosses_decision_block']=True
    # Offline vector scoring must match the chronological block arrival schedule.
    for w in score['words']:
        ready=next(end for end in range(500,len(bits)+1,500) if end>=w['end_symbol'])
        assert w['available_symbol']==ready and 20.4<=w['assembly_plus_buffer_ms']<25.4
    checks['packet_availability_and_latency']=True
    for case in p['cases']:
        for cond in p['conditions']:
            f=generator_v4.generate(cfg,117,118,case,cond)
            assert len(f['rx'])%5000==0 and len(f['bits'])*10==len(f['rx'])
            assert all(t%500!=0 for t in f['transitions'])
            if case.get('clean'):assert not f['transitions']
    checks['all_scenario_geometry']=True
    bank=rx.train(cfg,1841,401,'random',n_train=2,trees=10)
    s=legacy.StreamSettings();f=legacy.generate_stream(cfg,9182,s,'random');x=f['rx'][:80000];cut=65000
    feats,_=rx.feature_cache(x,cfg,5000);prefix,_=rx.feature_cache(x[:cut],cfg,5000)
    for a,b in zip(feats,prefix):
        for k in rx.KINDS:assert np.array_equal(a[k],b[k]),k
    oldfront=legacy.StatefulFrontEnd(cfg)
    for bi,a in enumerate(range(0,len(x),5000)):
        old=oldfront.features(x[a:a+5000])
        for k in legacy.KINDS:assert np.array_equal(old[k],feats[bi][k]),k
    for k in rx.KINDS:
        for pol in rx.POLICIES:
            full=rx.process_one(x,cfg,bank,feats,k,pol);pre=rx.process_one(x[:cut],cfg,bank,prefix,k,pol)
            assert np.array_equal(full['soft'][:cut//10],pre['soft'])
            for mask in p['masks']:assert np.array_equal(full['flags'][mask][:cut//10],pre['flags'][mask])
            assert [d for d in full['decisions'] if d['observed_end_sample']<=cut]==pre['decisions']
    for k in ['raw','rule']:
        full=rx.process_one(x,cfg,bank,feats,k,k);pre=rx.process_one(x[:cut],cfg,bank,prefix,k,k)
        assert np.array_equal(full['soft'][:cut//10],pre['soft'])
    checks['all_frontends_and_policies_causal_prefix']=True
    for guard in [False,True]:
        baseline=legacy.process_stream(x,cfg,bank,bank.settings,blank_guard=guard)
        for k in legacy.KINDS:
            pol='guard8' if guard else 'legacy8';new=rx.process_one(x,cfg,bank,feats,k,pol);old=baseline[k+'_event']
            assert np.array_equal(new['soft'],old['soft']) and np.array_equal(new['flags']['legacy_any'],old['erasure_mask'])
            for count in ['router_calls','ale_samples','ale_weight_updates']:assert new['counts'][count]==old['counts'][count]
            if guard:
                new=rx.process_one(x,cfg,bank,feats,k,'periodic');old=baseline[k+'_periodic']
                assert np.array_equal(new['soft'],old['soft']) and np.array_equal(new['flags']['legacy_any'],old['erasure_mask'])
    checks['v3_event_and_guarded_periodic_exact_outputs']=True
    for fun in [rx.process_one,link.decode_word]:
        assert not {'truth','bits','payloads','original','condition','transitions'}&set(inspect.signature(fun).parameters)
    checks['no_truth_in_inference_interfaces']=True
    # One failing frontend must veto a candidate even if all others improve.
    q=json.loads(json.dumps(p));q['policy_candidates']=['guard8','periodic'];synthetic=[]
    for k in q['frontends']:
        for c in q['conditions']:
            for case in q['profiles']['development']['cases']:
                for mask in q['masks']:
                    for pol in q['policy_candidates']:
                        synthetic.append(dict(condition=c,case=case,method=f'{k}__{pol}__{mask}',bler=.1,mean_stream_correct_retained=.9,first_nb_correct=None if case=='clean' else .9,mean_rf_calls=5. if pol=='guard8' else 30.,undetected_wrong=0))
    assert summary_v4.choose(synthetic,q)['policy']=='guard8'
    for r in synthetic:
        if r['condition']=='fixed' and r['case']=='nominal' and r['method'].startswith('spike__guard8'):r['first_nb_correct']=.87
    assert summary_v4.choose(synthetic,q)['policy']=='periodic'
    checks['per_frontend_gate_and_periodic_fallback']=True
    # Fresh profiles/stages are separate even when other indices coincide.
    values=[io.derive(p,profile,stage,401,'fixed',0,0) for profile in p['profiles'] for stage in p['stage_ids']]
    assert len(set(values))==len(values);checks['profile_stage_seed_separation']=True
    with tempfile.TemporaryDirectory() as temp:
        out=Path(temp);first=io.lock_for(out,p,'smoke',None);again=io.lock_for(out,p,'smoke',None)
        assert first['fingerprint']==again['fingerprint']
        changed=json.loads(json.dumps(p));changed['seed_namespace']+=1
        try:io.lock_for(out,changed,'smoke',None)
        except RuntimeError:pass
        else:raise AssertionError('Changed protocol accepted for resume')
    checks['json_lock_roundtrip_and_changed_protocol_rejection']=True
    print(json.dumps(dict(status='PASS',checks=checks),indent=2),flush=True)
    return checks

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path);args=ap.parse_args();checks=main()
    if args.output:io.write(args.output,dict(status='PASS',checks=checks,sources=io.sources(),dependencies=io.dependencies()))
