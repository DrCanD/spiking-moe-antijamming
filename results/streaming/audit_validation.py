"""Read-only audit of extracted UAV v5 RESULTS_REVIEW.zip; no experiment rerun.
Usage: python audit_validation.py /path/to/extracted/results
Requires only the Python standard library. Writes audit_metrics.json in that folder.
"""
from pathlib import Path
from collections import Counter, defaultdict
import csv, gzip, hashlib, json, math, sys

ROOT=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text())
canonical=lambda x:json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
sha=lambda x:hashlib.sha256(x).hexdigest()
name=lambda frontend,policy:frontend+'__'+policy+'__block_refine2'
mean=lambda vals:sum(vals)/len(vals)
run=read(ROOT/'RUN_SUMMARY.json');assert run['status']=='COMPLETE' and run['validation_status']=='COMPLETE'
devlock=read(ROOT/'development/lock.json');vallock=read(ROOT/'validation/lock.json')
assert devlock['fingerprint']=='fa73905ecf2ba67a6dcdd765bc0daf9009d430bddbb462f1e3b95cc03b086b0f'
sel=read(ROOT/'development/selection.json');proof=dict(sel);stored=proof.pop('selection_sha256')
assert stored==sha(canonical(proof))=='a045117be350d9776b65c1f21969891030fa25f18d8238754941d10258866f4a'
assert vallock['spec']['selection']==sel and sel['policy']=='refresh_guard8'
assert devlock['spec']['sources']==vallock['spec']['sources']
assert devlock['spec']['protocol']==vallock['spec']['protocol']
for rel,digest in vallock['spec']['sources'].items():assert sha((ROOT/'source'/rel).read_bytes())==digest,rel
verification=read(ROOT/'verification.json');assert verification['status']=='PASS' and all(verification['checks'].values())
assert verification['sources']==vallock['spec']['sources']
manifest=read(ROOT/'source/frozen_v4_manifest.json')
for rel,digest in manifest.items():assert sha((ROOT/'source/frozen_v4'/rel).read_bytes())==digest
p=vallock['spec']['protocol'];expected_methods={name(k,pol) for k in p['frontends'] for pol in p['policies']}|{name('rule','rule'),name('raw','raw')}
assert len(expected_methods)==16
all_rx={};seed_sets={};validation=[];integrity={}
for profile,lock in [('development',devlock),('validation',vallock)]:
    assert lock['fingerprint']==sha(canonical(lock['spec']))
    config=lock['spec']['protocol']['profiles'][profile];tasks=read(ROOT/profile/'tasks.json')
    rx=set();identities=set();channel_seeds=set();payload_seeds=set();words=0;row_count=0
    assert len(tasks)==len({t['id'] for t in tasks})
    for t in tasks:
        path=ROOT/profile/'shards'/(t['id']+'.json.gz');raw=path.read_bytes()
        assert sha(raw)==Path(str(path)+'.sha256').read_text().strip()
        s=json.loads(gzip.decompress(raw));assert s['task']==t and s['lock_fingerprint']==lock['fingerprint']
        meta=read(ROOT/profile/'models'/(t['model_key']+'.meta.json'))
        assert s['model_sha256']==meta['sha256'] and meta['lock_fingerprint']==lock['fingerprint']
        assert [r['trial'] for r in s['rows']]==list(range(t['start'],t['stop']))
        for r in s['rows']:
            row_count+=1;words+=r['coding']['n_words'];rx.add(r['rx_sha256'])
            identities.add((r['model_seed'],r['condition'],r['case'],r['trial']))
            channel_seeds.add(r['channel_seed']);payload_seeds.add(r['payload_seed'])
            assert set(r['methods'])==expected_methods
            for m in r['methods'].values():
                assert m['n_words']==r['coding']['n_words']
                assert m['failed_words']==m['decoder_failures']+m['crc_rejections']+m['undetected_wrong']
                assert m['useful_bits']==(m['n_words']-m['failed_words'])*1240
                assert m['first_word_failure']==int(not m['words'][0]['success'])
                for w in m['words']:
                    assert not(w['two_e_plus_v']<=96 and not w['success'])
                    assert w['available_symbol']==(w['end_symbol']+499)//500*500 and w['available_symbol']<=m['n_symbols']
            for route in r['routes'].values():
                assert route['counts']['router_calls']==len(route['decisions'])
                assert route['counts']['ale_samples']<=r['channel']['n_samples']
                for d in route['decisions']:
                    assert d['apply_sample']>=d['observed_end_sample']
                    if 'affected_start_sample' in d:assert d['affected_end_sample']==d['output_available_sample']==d['observed_end_sample']==d['affected_start_sample']+5000
            if profile=='validation':validation.append(r)
    expected={(sd,c,ca,tr) for sd in config['model_seeds'] for c in p['conditions'] for ca in config['cases'] for tr in range(config['trials'])}
    assert identities==expected and row_count==len(rx)==len(expected)
    expected_summary=read(ROOT/profile/'summary.json')
    assert row_count==expected_summary['n_streams'] and words==expected_summary['n_independent_transmitted_words']
    all_rx[profile]=rx;seed_sets[profile]={'channel':channel_seeds,'payload':payload_seeds,'model':set(config['model_seeds'])}
    integrity[profile]=dict(shards=len(tasks),streams=row_count,transmitted_words=words,lock_fingerprint=lock['fingerprint'])
assert not(all_rx['development']&all_rx['validation'])
for kind in ('channel','payload','model'):assert not(seed_sets['development'][kind]&seed_sets['validation'][kind])

groups=defaultdict(list);head=defaultdict(Counter);work=defaultdict(Counter)
for r in validation:
    for n,m in r['methods'].items():
        groups[(r['condition'],r['case'],n)].append(m)
        a=head[(r['condition'],n)]
        a.update({k:m[k] for k in ('n_words','failed_words','n_symbols','useful_bits','undetected_wrong','crc_rejections','wrong_returns')})
        a.update(n_streams=1,first_failures=m['first_word_failure'],router_calls=m['counts']['router_calls'],ale_samples=m['counts']['ale_samples'])
        w=work[(r['condition'],r['case'],n)];w.update(m['counts']);w.update(n_streams=1,n_input_samples=m['n_symbols']*10)
def table(fn):return list(csv.DictReader((ROOT/'validation/tables'/fn).open()))
for r in table('condition_totals.csv'):
    a=head[(r['condition'],r['method'])]
    for k in ('n_words','failed_words','n_symbols','useful_bits','undetected_wrong','n_streams','first_failures'):assert a[k]==int(r[k]),(r['method'],k)
    assert math.isclose(a['failed_words']/a['n_words'],float(r['bler']),abs_tol=1e-14)
for r in table('work_counts.csv'):
    a=work[(r['condition'],r['case'],r['method'])]
    for k,v in r.items():
        if k not in ('condition','case','method'):
            if v=='':assert k not in a,(r['method'],k,'unexpected missing counter')
            else:assert a[k]==int(v),(r['method'],k)
aggregate={}
for key,vals in groups.items():
    cross=[w for v in vals for w in v['words'] if w['crosses_transition']]
    nb=[v['first_nb_correct'] for v in vals if v['first_nb_correct'] is not None]
    aggregate[key]=dict(bler=sum(v['failed_words'] for v in vals)/sum(v['n_words'] for v in vals),first_word_bler=mean([v['first_word_failure'] for v in vals]),correct=mean([v['correct_retained_fraction'] for v in vals]),nb=mean(nb) if nb else None,cross=sum(not w['success'] for w in cross)/len(cross) if cross else None,undetected_wrong=sum(v['undetected_wrong'] for v in vals),mean_rf_calls=mean([v['counts']['router_calls'] for v in vals]))
bounds=p['selection'];gates=[];scores=[]
for policy in bounds['candidates']:
    passed=[];calls=[];refs=[]
    for frontend in p['frontends']:
        for condition in p['conditions']:
            for case in p['profiles']['validation']['cases']:
                a=aggregate[(condition,case,name(frontend,policy))];b=aggregate[(condition,case,name(frontend,bounds['reference']))]
                delta=dict(bler_difference=a['bler']-b['bler'],first_word_difference=a['first_word_bler']-b['first_word_bler'],crossing_word_difference=None if a['cross'] is None else a['cross']-b['cross'],correct_difference=a['correct']-b['correct'],first_nb_difference=None if a['nb'] is None else a['nb']-b['nb'])
                passed_cell=delta['bler_difference']<=bounds['bler_margin']+1e-12 and delta['first_word_difference']<=bounds['first_word_margin']+1e-12 and delta['correct_difference']>=-bounds['correct_retained_margin']-1e-12 and (delta['crossing_word_difference'] is None or delta['crossing_word_difference']<=bounds['crossing_word_margin']+1e-12) and (delta['first_nb_difference'] is None or delta['first_nb_difference']>=-bounds['first_nb_margin']-1e-12) and a['undetected_wrong']<=b['undetected_wrong']
                gates.append(dict(policy=policy,frontend=frontend,condition=condition,case=case,**delta,eligible=passed_cell))
                passed.append(passed_cell);calls.append(a['mean_rf_calls']);refs.append(b['mean_rf_calls'])
    saving=1-sum(calls)/sum(refs)
    scores.append(dict(policy=policy,n_failed_cells=len(passed)-sum(passed),rf_saving=saving,mean_rf_calls=mean(calls),passes_descriptive_bounds=all(passed) and saving>=bounds['minimum_mean_rf_saving'],used_for_selection=False))
lookup={(r['policy'],r['frontend'],r['condition'],r['case']):r for r in gates}
for r in table('quality_gates.csv'):
    expected=lookup[(r['policy'],r['frontend'],r['condition'],r['case'])]
    assert expected['eligible']==(r['eligible']=='True')
    for k in ('bler_difference','first_word_difference','crossing_word_difference','correct_difference','first_nb_difference'):
        assert (expected[k] is None and r[k]=='') or (expected[k] is not None and math.isclose(expected[k],float(r[k]),abs_tol=1e-12))
headlines=[]
for (condition,method),a in sorted(head.items()):
    headlines.append(dict(condition=condition,method=method,**a,bler=a['failed_words']/a['n_words'],first_word_bler=a['first_failures']/a['n_streams'],mean_rf_calls=a['router_calls']/a['n_streams'],ale_fraction=a['ale_samples']/(a['n_symbols']*10)))
savings=[]
for frontend in p['frontends']:
    for condition in p['conditions']+['both']:
        cc=p['conditions'] if condition=='both' else [condition]
        a=sum(head[(c,name(frontend,'refresh_guard8'))]['router_calls'] for c in cc)
        b=sum(head[(c,name(frontend,'current_periodic'))]['router_calls'] for c in cc)
        savings.append(dict(frontend=frontend,condition=condition,selected_rf_calls=a,reference_rf_calls=b,rf_saving=1-a/b))
contrasts=table('paired_contrasts.csv')
selected_ci=[r for r in contrasts if '__refresh_guard8__' in r['left'] and '__current_periodic__' in r['right'] and r['metric']=='mean_stream_bler']
for r in selected_ci:
    vals=[v['methods'][r['left']]['bler']-v['methods'][r['right']]['bler'] for v in validation if v['condition']==r['condition'] and v['case']==r['case']]
    assert len(vals)==100 and math.isclose(mean(vals),float(r['difference']),abs_tol=1e-14)
    assert int(r['n_model_seeds'])==5 and int(r['n_streams'])==100
per_seed=[]
for frontend in p['frontends']:
    for sd in p['profiles']['validation']['model_seeds']:
        for condition in p['conditions']:
            rr=[r for r in validation if r['model_seed']==sd and r['condition']==condition]
            counts={pol:dict(failed=sum(r['methods'][name(frontend,pol)]['failed_words'] for r in rr),words=sum(r['methods'][name(frontend,pol)]['n_words'] for r in rr)) for pol in ('current_periodic','refresh_guard8')}
            delta=counts['refresh_guard8']['failed']/counts['refresh_guard8']['words']-counts['current_periodic']['failed']/counts['current_periodic']['words']
            per_seed.append(dict(frontend=frontend,seed=sd,condition=condition,counts=counts,bler_difference=delta))
result=dict(integrity='PASS',profile_integrity=integrity,development_validation_disjoint=True,source_and_frozen_selection_unchanged=True,selection_sha256=stored,validation_scores_diagnostic=scores,headlines=headlines,rf_savings=savings,selected_gates=[g for g in gates if g['policy']=='refresh_guard8'],selected_bler_contrasts=selected_ci,spike_analog_contrasts=[r for r in contrasts if r['left']==name('spike','refresh_guard8') and r['right']==name('analog_delta','refresh_guard8') and r['metric']=='mean_stream_bler'],per_seed=per_seed,limitations=['Model pickle bytes are not present in the review ZIP; model SHA was checked against metadata only.','Existing bootstrap confidence intervals are descriptive and use mean-stream BLER; gate BLER is pooled by codeword.','No retuning or new selection was performed; additional candidate scores are diagnostics only.'])
(ROOT/'audit_metrics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print('INTEGRITY PASS',json.dumps(integrity))
print('VALIDATION SCORES',json.dumps(scores))
print('RF SAVINGS',json.dumps(savings))
print('SPIKE HEADLINES',json.dumps([r for r in headlines if r['method'] in (name('spike','lag_periodic'),name('spike','current_periodic'),name('spike','refresh_guard8'),name('rule','rule'))]))
print('WIDEST SELECTED BLER CI',max(selected_ci,key=lambda r:float(r['ci_high'])))
print('SELECTED BLER CIs with upper > 1pp',sum(float(r['ci_high'])>0.01 for r in selected_ci))
print('TOTAL UNDETECTED WRONG',sum(r['undetected_wrong'] for r in headlines))
