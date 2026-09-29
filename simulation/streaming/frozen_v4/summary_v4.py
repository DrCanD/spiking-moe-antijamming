"""Truth-scored summaries; DEV-only policy freeze; no TEST selection."""
from collections import defaultdict,Counter
from pathlib import Path
import csv,json,zipfile
import numpy as np
import run_v4 as io

def csvwrite(path,rows):
    if not rows:return
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)

def aggregate(rows):
    groups=defaultdict(list)
    for r in rows:
        for name,m in r['methods'].items():groups[(r['condition'],r['case'],name)].append(m)
    out=[]
    for (c,case,name),vals in sorted(groups.items()):
        n=len(vals);words=sum(v['n_words'] for v in vals);symbols=sum(v['n_symbols'] for v in vals);ret=sum(v['retained_bits'] for v in vals);err=sum(v['retained_bit_errors'] for v in vals)
        r=dict(condition=c,case=case,method=name,n_streams=n,n_words=words,n_symbols=symbols)
        for k in ['failed_words','useful_bits','decoder_failures','crc_rejections','wrong_returns','undetected_wrong','over_budget_words','crc_accepted_words']:r[k]=sum(v[k] for v in vals)
        r.update(bler=r['failed_words']/words,goodput=r['useful_bits']/symbols,mean_stream_bler=float(np.mean([v['bler'] for v in vals])),
            correct_retained_fraction=(ret-err)/symbols,mean_stream_correct_retained=float(np.mean([v['correct_retained_fraction'] for v in vals])),retained_ber=err/ret if ret else None,
            mean_rf_calls=float(np.mean([v['counts']['router_calls'] for v in vals])),mean_ale_samples=float(np.mean([v['counts']['ale_samples'] for v in vals])),
            first_word_bler=float(np.mean([v['first_word_failure'] for v in vals])))
        nb=[v['first_nb_correct'] for v in vals if v['first_nb_correct'] is not None];r['first_nb_correct']=float(np.mean(nb)) if nb else None
        cross=[w for v in vals for w in v['words'] if w['crosses_transition']];r['crossing_words']=len(cross);r['crossing_word_bler']=sum(not w['success'] for w in cross)/len(cross) if cross else None
        r['mean_buffer_extra_ms']=float(np.mean([w['buffer_extra_ms'] for v in vals for w in v['words']]))
        r['max_buffer_extra_ms']=max(w['buffer_extra_ms'] for v in vals for w in v['words'])
        out.append(r)
    return out

def choose(agg,p):
    """Deterministic DEV decision; periodic fallback; all cells checked."""
    lookup={(r['condition'],r['case'],r['method']):r for r in agg};devcases=p['profiles']['development']['cases'];bounds=p['selection']
    grid=[(k,c,case) for k in p['frontends'] for c in p['conditions'] for case in devcases]
    masks=[]
    for mask in p['masks']:
        chosen=[lookup[c,case,f'{k}__periodic__{mask}'] for k,c,case in grid]
        ok=all(lookup[c,'clean',f'{k}__periodic__{mask}']['bler']<=lookup[c,'clean',f'{k}__periodic__legacy_any']['bler']+bounds['bler_margin']+1e-12 for k in p['frontends'] for c in p['conditions'])
        masks.append(dict(mask=mask,eligible=bool(ok),mean_cell_bler=float(np.mean([r['bler'] for r in chosen]))))
    mask=min([r for r in masks if r['eligible']],key=lambda r:r['mean_cell_bler'])['mask']
    scores=[];gates=[]
    for policy in p['policy_candidates']:
        ok=True;calls=[]
        for k,c,case in grid:
            a=lookup[c,case,f'{k}__{policy}__{mask}'];b=lookup[c,case,f'{k}__periodic__{mask}'];calls.append(a['mean_rf_calls'])
            db=a['bler']-b['bler'];dc=a['mean_stream_correct_retained']-b['mean_stream_correct_retained']
            dn=None if a['first_nb_correct'] is None else a['first_nb_correct']-b['first_nb_correct']
            passed=db<=bounds['bler_margin']+1e-12 and dc>=-bounds['correct_retained_margin']-1e-12 and (dn is None or dn>=-bounds['first_nb_margin']-1e-12) and a['undetected_wrong']<=b['undetected_wrong']
            gates.append(dict(policy=policy,mask=mask,frontend=k,condition=c,case=case,bler_difference=db,correct_difference=dc,first_nb_difference=dn,eligible=bool(passed)))
            ok&=passed
        scores.append(dict(policy=policy,eligible=bool(ok),mean_rf_calls=float(np.mean(calls))))
    policy=min([r for r in scores if r['eligible']],key=lambda r:r['mean_rf_calls'])['policy']
    return dict(mask=mask,policy=policy,mask_scores=masks,policy_scores=scores,gates=gates)

def bootstrap(rows,left,right,metric):
    seeds=sorted({r['model_seed'] for r in rows});a=np.array([[r['methods'][left][metric]-r['methods'][right][metric] for r in sorted(rows,key=lambda r:r['trial']) if r['model_seed']==s] for s in seeds])
    ns,nt=a.shape;n=2000;rng=np.random.default_rng(925202605);draw=np.zeros(n)
    for _ in range(ns):
        si=rng.integers(ns,size=n);ii=rng.integers(nt,size=(n,nt));draw+=a[si[:,None],ii].mean(axis=1)/ns
    return dict(metric='mean_stream_'+metric,difference=float(a.mean()),ci_low=float(np.quantile(draw,.025)),ci_high=float(np.quantile(draw,.975)),n_model_seeds=ns,n_streams=ns*nt,n_boot=n)

def summarize(out,lock):
    if io.sources()!=lock['spec']['sources']:raise RuntimeError('Source changed before summary')
    p=lock['spec']['protocol'];profile=lock['spec']['profile'];rows=io.all_rows(out,lock);f=p['profiles'][profile]
    if len(rows)!=len(f['model_seeds'])*len(p['conditions'])*len(f['cases'])*f['trials']:raise RuntimeError('Incomplete grid')
    expected={'__'.join(x) for x in io.method_plan(lock)};seen=set();scored_words=0
    for r in rows:
        if set(r['methods'])!=expected:raise RuntimeError('Methods mismatch')
        if r['rx_sha256'] in seen:raise RuntimeError('Duplicate received stream')
        seen.add(r['rx_sha256'])
        for m in r['methods'].values():
            if m['failed_words']!=m['decoder_failures']+m['crc_rejections']+m['undetected_wrong']:raise RuntimeError('Failure partition mismatch')
            if m['useful_bits']!=(m['n_words']-m['failed_words'])*155*8:raise RuntimeError('Goodput mismatch')
            for w in m['words']:
                scored_words+=1
                if w['two_e_plus_v']<=96 and not w['success']:raise RuntimeError('RS radius violation')
                if w['available_symbol']<w['end_symbol'] or w['available_symbol']>m['n_symbols']:raise RuntimeError('Noncausal/out-of-stream delivery')
                if w['success'] and not w['crc_pass']:raise RuntimeError('Success without CRC')
        for route in r['routes'].values():
            if route['counts']['router_calls']!=len(route['decisions']):raise RuntimeError('RF count mismatch')
            if any(d['apply_sample']<d['observed_end_sample'] for d in route['decisions']):raise RuntimeError('Future lookahead')
    agg=aggregate(rows);csvwrite(out/'tables/links.csv',agg)
    seedrows=[]
    for sd in f['model_seeds']:
        for row in aggregate([r for r in rows if r['model_seed']==sd]):seedrows.append(dict(model_seed=sd,**row))
    csvwrite(out/'tables/per_seed.csv',seedrows)
    selection=lock['spec']['selection']
    if profile=='development':
        result=choose(agg,p);gates=result.pop('gates');csvwrite(out/'tables/selection_gates.csv',gates)
        selection=dict(profile=profile,development_fingerprint=lock['fingerprint'],sources=lock['spec']['sources'],selection_rule=p['selection'],**result)
        selection['selection_sha256']=io.sha(io.canonical(selection));dest=out/'selection.json'
        if dest.exists() and io.read(dest)!=selection:raise RuntimeError('Existing DEV selection differs')
        io.write(dest,selection)
    if profile=='overnight':
        contrasts=[];mask=selection['mask'];pol=selection['policy']
        gates=[];lookup={(r['condition'],r['case'],r['method']):r for r in agg};bound=p['selection']
        for k in p['frontends']:
            for c in p['conditions']:
                for case in f['cases']:
                    a=lookup[c,case,f'{k}__{pol}__{mask}'];b=lookup[c,case,f'{k}__periodic__{mask}']
                    db=a['bler']-b['bler'];dc=a['mean_stream_correct_retained']-b['mean_stream_correct_retained'];dn=None if a['first_nb_correct'] is None else a['first_nb_correct']-b['first_nb_correct']
                    ok=db<=bound['bler_margin']+1e-12 and dc>=-bound['correct_retained_margin']-1e-12 and (dn is None or dn>=-bound['first_nb_margin']-1e-12) and a['undetected_wrong']<=b['undetected_wrong']
                    gates.append(dict(frontend=k,condition=c,case=case,bler_difference=db,correct_difference=dc,first_nb_difference=dn,point_estimate_pass=bool(ok),used_for_selection=False))
        csvwrite(out/'tables/test_quality_gates.csv',gates)
        pairs=[(f'{k}__{pol}__{mask}',f'{k}__{bp}__{bm}') for k in p['frontends'] for bp,bm in [('periodic',mask),('legacy8','legacy_any')]]
        pairs += [(f'spike__{pol}__{mask}',f'{k}__{pol}__{mask}') for k in p['frontends'] if k!='spike']
        pairs += [(f'spike__{pol}__{mask}',f'rule__rule__{mask}')]
        for c in p['conditions']:
            for case in f['cases']:
                rr=[r for r in rows if r['condition']==c and r['case']==case]
                for left,right in pairs:
                    for metric in ['bler','goodput_per_transmitted_symbol']:
                        v=bootstrap(rr,left,right,metric);contrasts.append(dict(condition=c,case=case,left=left,right=right,**v))
        csvwrite(out/'tables/paired_contrasts.csv',contrasts)
    # Word positions and CRC partitions remain available without loading full shards.
    wp=defaultdict(Counter)
    for r in rows:
        for name,m in r['methods'].items():
            for w in m['words']:wp[r['condition'],r['case'],name,w['index']].update(n=1,fail=int(not w['success']),wrong=int(w['wrong_return']),crc_reject=int(w['crc_rejected']),undetected=int(w['undetected_wrong']),over=int(w['erased_bytes']>96))
    csvwrite(out/'tables/word_positions.csv',[dict(condition=c,case=ca,method=me,index=i,**v) for (c,ca,me,i),v in sorted(wp.items())])
    summary=dict(status='COMPLETE',profile=profile,lock_fingerprint=lock['fingerprint'],n_streams=len(rows),n_independent_transmitted_words=sum(r['coding']['n_words'] for r in rows),n_scored_method_words=scored_words,
        source_checks=True,shard_checks=True,rs_crc_partition_checks=True,selected=None if selection is None else {k:selection[k] for k in ('mask','policy')},
        statistics='Descriptive paired hierarchical bootstrap: model then whole stream, 2000 resamples; no multiplicity correction. Pooled aggregate BLER differs from mean-stream BLER.',
        interpretation='RF calls and software timings are not hardware energy. DEV tolerances are not a formal noninferiority test. No TEST retuning.')
    io.write(out/'summary.json',summary)
    with zipfile.ZipFile(out/'RESULTS_REVIEW.zip','w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name in ['lock.json','tasks.json','environment.json','summary.json','selection.json']:
            if (out/name).exists():z.write(out/name,name)
        for folder in ['tables','shards']:
            for file in sorted((out/folder).rglob('*')):
                if file.is_file():z.write(file,str(file.relative_to(out)))
        for file in sorted((out/'models').glob('*.meta.json')):z.write(file,'models/'+file.name)
        for name in lock['spec']['sources']:z.write(io.ROOT/name,'source/'+name)
    print(json.dumps(summary,indent=2),flush=True)
    return summary
