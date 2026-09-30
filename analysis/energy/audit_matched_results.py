from pathlib import Path
import json,math,statistics as st,hashlib,csv
ROOT=Path(__file__).resolve().parent
T=2.7765

def stats(x):
 x=list(map(float,x));m=st.mean(x);sd=st.stdev(x);h=T*sd/math.sqrt(len(x))
 return {'n':len(x),'mean':m,'sd':sd,'ci95':[m-h,m+h]}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
out={'unit':'nJ/input sample/replica','replication':4,'ci':'nominal paired Student t, n=5 repeats, df=4','archives':{},'campaigns':{}}
for tag,archive,analysis in [('matched','KV260_MATCHED_RESULTS.zip','analysis_matched.json'),('gate','KV260_GATE_RESULTS.zip','analysis.json')]:
 source=ROOT/tag;data=json.loads((source/'measurement.json').read_text());provided=json.loads((source/analysis).read_text());nr=data['n_repl'];assert nr==4
 grouped={k:[] for k in data['settings']['vectors']};raw_count=0;max_mean_error=0;sample_gaps=[];run_rates=[];all_temperatures=[];power_levels=set();sensitivity={k:[] for k in grouped}
 for rep in data['repeats']:
  assert rep['complete'];name=rep['vector'];runs=rep['runs'];idles=rep['idles'];assert len(idles)==len(runs)+1
  assert [r['mode'] for r in runs]==rep['order'];adj={};adj_win={};adj_interp={};raw={};power={};rates={}
  for block in idles+runs:
   samples=block['samples'];raw_count+=len(samples);assert len(samples)>=1080
   powers=[r['power_mW'] for r in samples];power_levels.update(powers);avg=st.mean(powers);max_mean_error=max(max_mean_error,abs(avg-block['mean_mW']));assert abs(avg-block['mean_mW'])<1e-8
   assert all(math.isfinite(p) and p>0 for p in powers)
   gaps=[b['t_s']-a['t_s'] for a,b in zip(samples,samples[1:])];assert min(gaps)>0 and max(gaps)<.5;sample_gaps.append(max(gaps))
   all_temperatures += [r['temperature_C'] for r in samples if r.get('temperature_C')]
  for i,r in enumerate(runs):
   idle=(idles[i]['mean_mW']+idles[i+1]['mean_mW'])/2;assert abs(idle-r['idle_bracket_mW'])<1e-8
   rate=r['rate_msps'];run_rates.append(rate);rates[r['mode']]=rate;assert abs(rate/data['settings']['target']-1)<=.01
   assert abs(rate-r['frames']*.1/r['elapsed_s'])<1e-10
   assert abs(r['window_rate_msps']-r['window_frames']*.1/r['duration_s'])<1e-10
   a,b=idles[i],idles[i+1];ta=a['started_monotonic_s']+a['duration_s']/2;tb=b['started_monotonic_s']+b['duration_s']/2;tr=r['started_monotonic_s']+r['duration_s']/2
   weight=(tr-ta)/(tb-ta);assert 0<weight<1;weighted=(1-weight)*a['mean_mW']+weight*b['mean_mW']
   mode=r['mode'];adj[mode]=(r['mean_mW']-idle)/rate;adj_win[mode]=(r['mean_mW']-idle)/r['window_rate_msps'];adj_interp[mode]=(r['mean_mW']-weighted)/rate
   raw[mode]=r['mean_mW']/rate;power[mode]=r['mean_mW']
  assert max(rates.values())/min(rates.values())-1<=.01
  e={k:(v-adj['D0_empty'])/nr for k,v in adj.items()};ew={k:(v-adj_win['D0_empty'])/nr for k,v in adj_win.items()};ei={k:(v-adj_interp['D0_empty'])/nr for k,v in adj_interp.items()}
  grouped[name].append({'repeat':rep['repeat'],'energy':e,'total_som_energy':raw,'power_mW':power,'rate_msps':rates,'window_energy':ew,'time_interpolated_idle_energy':ei})
 result={'raw_sample_count':raw_count,'active_blocks':sum(len(r['runs']) for r in data['repeats']),'idle_blocks':sum(len(r['idles']) for r in data['repeats']),'max_raw_mean_error_mW':max_mean_error,'maximum_sensor_gap_s':max(sample_gaps),'nonempty_temperature_observations':len(all_temperatures),'power_quantization_mW':min(b-a for a,b in zip(sorted(power_levels),sorted(power_levels)[1:])),'run_rate_range_msps':[min(run_rates),max(run_rates)],'vectors':{}}
 for name,reps in grouped.items():
  assert len(reps)==5; modes=reps[0]['energy']; v={'per_repeat':reps,'modes':{k:stats(r['energy'][k] for r in reps) for k in modes},'paired':{}}
  for k,s in v['modes'].items():
   target=provided['vectors'][name]['modes'][k]['D0_subtracted_nJ_per_replica'];assert max(abs(s[p]-target[p]) for p in ['mean','sd'])<1e-10
   assert max(abs(a-b) for a,b in zip(s['ci95'],target['ci95']))<1e-10
  pairs=[('FFT_minus_gated','FFT1024','FE_v3')]
  if tag=='matched':pairs+=[('FFT_ALE_minus_gated_ALE','FFT1024+ALE_NB','FE_v3+ALE_NB')]
  else:pairs+=[('dense_minus_gated','FE_dense','FE_v3')]
  for label,a,b in pairs:
   s=stats(r['energy'][a]-r['energy'][b] for r in reps)
   s.update(ratio_of_means=v['modes'][a]['mean']/v['modes'][b]['mean'],saving_fraction=1-v['modes'][b]['mean']/v['modes'][a]['mean'])
   s['window_rate_contrast']=stats(r['window_energy'][a]-r['window_energy'][b] for r in reps)
   s['interpolated_idle_contrast']=stats(r['time_interpolated_idle_energy'][a]-r['time_interpolated_idle_energy'][b] for r in reps)
   s['raw_SOM_power_difference_mW']=stats(r['power_mW'][a]-r['power_mW'][b] for r in reps)
   v['paired'][label]=s
  if tag=='matched':
   v['additivity']={}
   for combo,a in [('FE_v3+ALE_NB','FE_v3'),('FFT1024+ALE_NB','FFT1024')]:
    interaction=stats(r['energy'][combo]-r['energy'][a]-r['energy']['ALE_NB'] for r in reps);parts=st.mean(r['energy'][a]+r['energy']['ALE_NB'] for r in reps)
    v['additivity'][combo]={'interaction':interaction,'relative_mean':interaction['mean']/parts,'within_10pct_band':interaction['ci95'][0]>=-.1*parts and interaction['ci95'][1]<=.1*parts}
  result['vectors'][name]=v
 out['campaigns'][tag]=result
 out['archives'][archive]=sha(ROOT.parent/'records'/archive)
 # Validate recorded board evidence and immutable build provenance.
 for fn in ['board_verification.json']+(['board_matched_verification.json'] if tag=='matched' else []):
  b=json.loads((source/fn).read_text());assert b['completed'] and all(c['passed'] for c in b['checks'])
  result[fn]={'passed':len(b['checks']),'vectors':b['vectors'],'sha256':sha(source/fn)}
for p in ['provenance/build_identity.json','provenance/board/build_receipt.json']:
 assert (ROOT/'matched'/p).read_bytes()==(ROOT/'gate'/p).read_bytes()
activity=json.loads((ROOT.parent/'streaming_activity.json').read_text())
pooled_samples=sum(x['input_samples'] for x in activity['conditions'].values())
pooled_active=sum(x['receivers']['spike9_refresh']['ale_samples'] for x in activity['conditions'].values())
pooled_duty=pooled_active/pooled_samples
out['conditional_model']={'scope':'Illustration only. Not full receiver energy; no measured closed-loop duty-scheduled hardware or output-quality matching. Negative blanker estimates remain unresolved.','activity':activity['conditions'],'vectors':{}}
out['conditional_model']['pooled_activity']={'input_samples':pooled_samples,'ale_active_samples':pooled_active,'duty':pooled_duty}
out['conditional_model']['analysis_corrections']=['The generated archive composition uses ALE_SW for the rule; the strongest pooled rule_slow uses ALE_NB.','The duty is the ALE-active input fraction, not the ALE tap-product ratio.','Joint-block additivity is not established for all modes, residual inputs or closed-loop operation.','A negative blanker contrast is not negative physical energy.']
for name,v in out['campaigns']['matched']['vectors'].items():
 e={k:z['mean'] for k,z in v['modes'].items()};af=max(e['ALE_NB'],e['ALE_SW']);entry={}
 duties={k:v['receivers']['spike9_refresh']['ale_active_sample_fraction'] for k,v in activity['conditions'].items()}
 duties['pooled']=pooled_duty
 for regime,duty in duties.items():
  moe=e['FE_v3']+duty*af+(1+duty)*e['BLANK'];slow=e['ALE_NB']+2*e['BLANK']
  zero=1-(e['FE_v3']+duty*af)/e['ALE_NB']
  entry[regime]={'ale_sample_duty':duty,'conditional_spike_nJ':moe,'conditional_rule_slow_nJ':slow,'conditional_saving_fraction':1-moe/slow,'zero_blanker_sensitivity_saving_fraction':zero}
 entry['break_even_d_vs_slow']=(e['ALE_NB']+e['BLANK']-e['FE_v3'])/(af+e['BLANK'])
 out['conditional_model']['vectors'][name]=entry
out['analysis_sha256']=sha(Path(__file__))
(ROOT/'audited/independent_raw_audit.json').write_text(json.dumps(out,indent=2)+'\n')
with (ROOT/'audited/measured_joint_energy.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['vector','gated_ALE_nJ','FFT_ALE_nJ','difference_nJ','ci95_low','ci95_high','saving_percent'])
 for name,v in out['campaigns']['matched']['vectors'].items():
  s=v['paired']['FFT_ALE_minus_gated_ALE'];w.writerow([name,v['modes']['FE_v3+ALE_NB']['mean'],v['modes']['FFT1024+ALE_NB']['mean'],s['mean'],*s['ci95'],100*s['saving_fraction']])
for name,v in out['campaigns']['matched']['vectors'].items():
 print(name)
 for k,s in v['paired'].items():print(k,'mean',round(s['mean'],5),'CI',list(map(lambda x:round(x,5),s['ci95'])),'saving%',round(100*s['saving_fraction'],3),'window mean',round(s['window_rate_contrast']['mean'],5),'interp mean',round(s['interpolated_idle_contrast']['mean'],5))
 print('conditional',out['conditional_model']['vectors'][name])
print('ALL RAW MEANS, RATE AND CI CHECKS PASSED')
