"""Paired effects, processing-work counters and DEV-only sparse-policy gate."""
from collections import defaultdict,Counter
import json,zipfile
import numpy as np
import common_v5 as io
import summary_v4
csvwrite=summary_v4.csvwrite

def choose(agg,p):
    lookup={(r['condition'],r['case'],r['method']):r for r in agg};gates=[];scores=[];bounds=p['selection']
    for policy in bounds['candidates']:
        calls=[];refs=[];passed=[]
        for k in p['frontends']:
            for c in p['conditions']:
                for case in p['profiles']['development']['cases']:
                    a=lookup[c,case,io.method_name(k,policy)];b=lookup[c,case,io.method_name(k,bounds['reference'])]
                    db=a['bler']-b['bler'];df=a['first_word_bler']-b['first_word_bler'];dc=a['mean_stream_correct_retained']-b['mean_stream_correct_retained']
                    dn=None if a['first_nb_correct'] is None else a['first_nb_correct']-b['first_nb_correct']
                    dx=None if a['crossing_word_bler'] is None else a['crossing_word_bler']-b['crossing_word_bler']
                    ok=(db<=bounds['bler_margin']+1e-12 and df<=bounds['first_word_margin']+1e-12 and dc>=-bounds['correct_retained_margin']-1e-12 and (dn is None or dn>=-bounds['first_nb_margin']-1e-12) and (dx is None or dx<=bounds['crossing_word_margin']+1e-12) and a['undetected_wrong']<=b['undetected_wrong'])
                    gates.append(dict(policy=policy,frontend=k,condition=c,case=case,bler_difference=db,first_word_difference=df,crossing_word_difference=dx,correct_difference=dc,first_nb_difference=dn,eligible=bool(ok)))
                    passed.append(ok);calls.append(a['mean_rf_calls']);refs.append(b['mean_rf_calls'])
        saving=1-float(np.sum(calls)/np.sum(refs));ok=bool(all(passed) and saving>=bounds['minimum_mean_rf_saving'])
        scores.append(dict(policy=policy,eligible=ok,n_failed_cells=int(len(passed)-sum(passed)),mean_rf_calls=float(np.mean(calls)),rf_saving=saving))
    candidates=[s for s in scores if s['eligible']]
    selected=min(candidates,key=lambda x:x['mean_rf_calls'])['policy'] if candidates else bounds['fallback']
    return dict(policy=selected,mask=p['mask'],sparse_candidate_passed=bool(candidates),candidate_scores=scores,gates=gates)

def summarize(out,lock):
    if io.sources()!=lock['spec']['sources']:raise RuntimeError('Source changed before summary')
    p=lock['spec']['protocol'];profile=lock['spec']['profile'];rows=io.all_rows(out,lock);f=p['profiles'][profile]
    expected={(sd,c,ca,tr) for sd in f['model_seeds'] for c in p['conditions'] for ca in f['cases'] for tr in range(f['trials'])}
    actual={(r['model_seed'],r['condition'],r['case'],r['trial']) for r in rows}
    if actual!=expected or len(rows)!=len(expected):raise RuntimeError('Incomplete or duplicate grid')
    if len({r['rx_sha256'] for r in rows})!=len(rows):raise RuntimeError('Duplicate received stream')
    methods={io.method_name(k,pol) for k,pol in io.method_plan(p)};scored=0
    costs=defaultdict(Counter);headlines=defaultdict(Counter)
    for r in rows:
        if set(r['methods'])!=methods:raise RuntimeError('Method set mismatch')
        for name,m in r['methods'].items():
            if m['failed_words']!=m['decoder_failures']+m['crc_rejections']+m['undetected_wrong']:raise RuntimeError('CRC partition mismatch')
            if m['useful_bits']!=(m['n_words']-m['failed_words'])*1240:raise RuntimeError('Goodput mismatch')
            if m['n_words']!=r['coding']['n_words']:raise RuntimeError('Missing words')
            for w in m['words']:
                scored+=1
                if w['two_e_plus_v']<=96 and not w['success']:raise RuntimeError('RS guarantee failed')
                if w['available_symbol']!=(w['end_symbol']+499)//500*500 or w['available_symbol']>m['n_symbols']:raise RuntimeError('Wrong block availability')
            g=(r['condition'],r['case'],name);costs[g].update(m['counts']);costs[g].update(n_streams=1,n_input_samples=m['n_symbols']*10)
            h=headlines[r['condition'],name];h.update(n_words=m['n_words'],failed_words=m['failed_words'],n_symbols=m['n_symbols'],useful_bits=m['useful_bits'],first_failures=m['first_word_failure'],n_streams=1,undetected_wrong=m['undetected_wrong'])
        for route in r['routes'].values():
            if route['counts']['router_calls']!=len(route['decisions']):raise RuntimeError('Uncharged RF call')
            for d in route['decisions']:
                if d['apply_sample']<d['observed_end_sample']:raise RuntimeError('Future observation in decision')
                if 'affected_start_sample' in d:
                    if not(d['affected_end_sample']==d['output_available_sample']==d['observed_end_sample']==d['affected_start_sample']+5000):raise RuntimeError('Current-block timing inconsistent')
            if route['counts']['ale_samples']>r['channel']['n_samples']:raise RuntimeError('Hidden extra ALE block')
    agg=summary_v4.aggregate(rows);csvwrite(out/'tables/links.csv',agg)
    csvwrite(out/'tables/per_seed.csv',[dict(model_seed=sd,**r) for sd in f['model_seeds'] for r in summary_v4.aggregate([rr for rr in rows if rr['model_seed']==sd])])
    csvwrite(out/'tables/work_counts.csv',[dict(condition=c,case=ca,method=m,**v) for (c,ca,m),v in sorted(costs.items())])
    top=[dict(condition=c,method=m,**v,bler=v['failed_words']/v['n_words'],goodput=v['useful_bits']/v['n_symbols'],first_word_bler=v['first_failures']/v['n_streams']) for (c,m),v in sorted(headlines.items())]
    csvwrite(out/'tables/condition_totals.csv',top)
    result=None
    if profile in ('local_pilot','development','validation'):
        result=choose(agg,p);gates=result.pop('gates');csvwrite(out/'tables/quality_gates.csv',gates)
        if profile=='development':
            selection=dict(profile=profile,development_fingerprint=lock['fingerprint'],sources=lock['spec']['sources'],selection_rule=p['selection'],**result)
            selection['selection_sha256']=io.sha(io.canonical(selection))
            if (out/'selection.json').exists() and io.read(out/'selection.json')!=selection:raise RuntimeError('Changed frozen DEV selection')
            io.write(out/'selection.json',selection)
        elif profile=='local_pilot':io.write(out/'pilot_diagnosis.json',dict(profile=profile,used_for_validation_selection=False,**result))
        else:
            # Keep DEV choice. Test gate calculations are diagnostic only.
            result=dict(policy=lock['spec']['selection']['policy'],mask=p['mask'],sparse_candidate_passed=lock['spec']['selection']['sparse_candidate_passed'],validation_used_for_selection=False)
        pairs=[]
        for k in p['frontends']:
            pairs.extend([(io.method_name(k,a),io.method_name(k,b)) for a,b in [('current_periodic','lag_periodic'),('current_guard8','lag_guard8'),('refresh_guard8','current_guard8'),('rescue_guard8','refresh_guard8'),('rescue_periodic','current_periodic'),('current_guard8','current_periodic'),('refresh_guard8','current_periodic'),('rescue_guard8','current_periodic')]])
            pairs.extend([(io.method_name(k,a),io.method_name('rule','rule')) for a in ('current_periodic','rescue_guard8','rescue_periodic')])
        pairs.extend([(io.method_name('spike',a),io.method_name('analog_delta',a)) for a in p['policies']])
        contrasts=[]
        for c in p['conditions']:
            for case in f['cases']:
                rr=[r for r in rows if r['condition']==c and r['case']==case]
                for left,right in pairs:
                    for metric in ('bler','goodput_per_transmitted_symbol','first_word_failure'):
                        contrasts.append(dict(condition=c,case=case,left=left,right=right,**summary_v4.bootstrap(rr,left,right,metric)))
        csvwrite(out/'tables/paired_contrasts.csv',contrasts)
    summary=dict(status='COMPLETE',profile=profile,lock_fingerprint=lock['fingerprint'],n_streams=len(rows),n_independent_transmitted_words=sum(r['coding']['n_words'] for r in rows),n_scored_method_words=scored,n_methods=len(methods),selection=result,
        interpretation='Development diagnosis unless profile=validation. Current-block receiver buffers 5 ms. Work counters are not hardware energy. No validation retuning. Method-word scores are paired duplicates.',statistics='Descriptive paired hierarchical bootstrap: model then whole stream, 2000 repetitions; no multiplicity correction; CIs use mean-stream metrics.')
    io.write(out/'summary.json',summary)
    with zipfile.ZipFile(out/'RESULTS_REVIEW.zip','w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name in ['lock.json','tasks.json','environment.json','summary.json','selection.json','pilot_diagnosis.json']:
            if (out/name).exists():z.write(out/name,name)
        for folder in ['tables','shards']:
            for file in sorted((out/folder).rglob('*')):
                if file.is_file():z.write(file,str(file.relative_to(out)))
        for file in sorted((out/'models').glob('*.meta.json')):z.write(file,'models/'+file.name)
        for name in lock['spec']['sources']:z.write(io.ROOT/name,'source/'+name)
    print(json.dumps(summary,indent=2),flush=True)
    return summary
