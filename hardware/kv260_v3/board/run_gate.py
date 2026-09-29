#!/usr/bin/env python3
"""KV260 verification and paired, rate-matched SOM input-power characterization.
No simulator or invented power fallback. Run only after build_kv260.py and install_gate.sh.
"""
import argparse, glob, hashlib, json, math, os, platform, random, sys, time, zipfile
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from moe_ctl import MoE, R, DESIGNS, ALL_MODES, CMD_NOP, CMD_RUN, CAP_SPIKES, CAP_NONE

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VECTORS = ['fixed_none', 'fixed_narrowband', 'random_pulse', 'random_switching']
MODES = list(DESIGNS)

def utc(): return datetime.now(timezone.utc).isoformat()
try: sys.stdout.reconfigure(line_buffering=True)      # progress must reach nohup/tail logs line by line
except Exception: pass

# ── live progress (what is happening now, not only elapsed time) ──
CTX = {}          # current vector / repeat / mode / block kind, last idle mean, frame counter at block start
PROG = None       # overall plan: nominal seconds done / total -> percentage and remaining time
def fmt_dur(sec):
    sec=max(0,int(sec)); h,m_=divmod(sec,3600); m_,s_=divmod(m_,60)
    return f'{h} sa {m_:02d} dk' if h else f'{m_} dk {s_:02d} sn'
class Progress:
    CAL_S=90.0                                   # nominal calibration time per vector (5 modes)
    def __init__(self,a,n_vectors,n_modes):
        thermal=2 if a.quick else 60
        self.per_run=a.warmup+5+a.block          # measure_run: warm-up + power block + tail until the run finishes
        self.per_vector=self.CAL_S+thermal+a.repeats*(a.block+n_modes*(self.per_run+a.block))
        self.total=n_vectors*self.per_vector;self.done=0.0
    def add(self,sec):self.done+=sec
    def status(self,extra=0.0):
        d=min(self.total,self.done+extra);return f'genel %{100*d/self.total:4.1f}, kalan ~{fmt_dur(self.total-d)}'
def _tmax(samples):
    for row in reversed(samples):
        t=row.get('temperature_C')
        if t:return max(t.values())
    return None
def report_block(elapsed,seconds,samples,m,running):
    if not CTX:return
    p=[r['power_mW'] for r in samples];now_mw=p[-1];mean=float(np.mean(p))
    head=f"[{'KOSU' if running else 'BOS '}] {CTX.get('vector','?')} tekrar {CTX.get('rep','?')} | "
    head+=(f"{CTX.get('mode','?')} ({CTX.get('mode_i','?')})" if running else 'bos referans')
    line=f"{head} | blok {elapsed:5.0f}/{seconds:.0f} s | anlik {now_mw:7.1f} mW, blok ort. {mean:7.1f} mW"
    if running and CTX.get('last_idle_mW') is not None:
        line+=f", son bos'a gore {mean-CTX['last_idle_mW']:+6.1f} mW"
    if running and 'f0' in CTX:
        try:
            f=m.frames_done();r=(f-CTX['f0'])*CTX['n']/max(elapsed,1e-6)/1e6;line+=f" | hiz {r:6.4f} MS/s ({f} cerceve)"
        except Exception:pass
    t=_tmax(samples)
    if t is not None:line+=f" | sicaklik {t:4.1f} C"
    if PROG:line+=' | '+PROG.status(elapsed)
    print(line,flush=True)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def readjson(p): return json.loads(Path(p).read_text())
def save(p, obj):
    p = Path(p); tmp = p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False)); tmp.replace(p)
def require(ok, text):
    if not ok: raise RuntimeError(text)
def energy_nj(power_mw, rate_msps):
    require(rate_msps > 0, 'Nonpositive sample rate')
    # (10^-3 W)/(10^6 samples/s) = 10^-9 J/sample. NO extra /1000.
    return power_mw / rate_msps

def check_rate(rate, target, tolerance=.01):
    require(abs(rate/target-1) <= tolerance,
            f'RATE FAIL: achieved {rate:.6f}, target {target:.6f} MS/s. No silent rate reduction.')

def clock_hz(receipt):
    p = Path('/sys/kernel/debug/clk/pl0_ref/clk_rate')
    require(p.exists(), 'Actual PL clock unavailable. Run install_gate.sh (debugfs required).')
    value = int(p.read_text())
    require(0 < value <= receipt['clock_request_hz']*1.001+2000, 'Actual PL clock exceeds the timing-qualified limit')
    return value

def preflight():
    require(os.geteuid() == 0, 'Run with sudo on the KV260.')
    require(platform.machine() in ('aarch64', 'arm64'), 'Board runner requires the KV260 ARM Linux system.')
    receipt = readjson(ROOT/'board/build_receipt.json')
    identity = readjson(ROOT/'build_identity.json')
    require(receipt.get('completed') and 'TIMING MET' in receipt['timing_status'] and 'NOT MET' not in receipt['timing_status'], 'Build/timing gate failed')
    require(receipt['identity'] == identity, 'Source/build identity mismatch')
    for p,k in [('board/firmware/moe_gate.bit.bin','bitstream_sha256'), ('board/regmap.json','regmap_sha256'), ('board/firmware/pl.dtsi','overlay_source_sha256')]:
        require(sha(ROOT/p) == receipt[k], p+' hash mismatch')
    require(Path('/sys/class/fpga_manager/fpga0/state').read_text().strip() == 'operating', 'FPGA manager is not operating')
    q = readjson(ROOT/'evidence/fixed_quality_summary.json')
    require(len(q) == 4 and sum(r['streams'] for r in q)==288, 'Fixed-point quality evidence missing')
    require(all(r['new_failures']==0 and r['different_action_streams']==0 and r['lazy_undetected']==r['dense_undetected']==0 for r in q), 'Fixed-point quality gate failed')
    return receipt, identity, clock_hz(receipt)

def check_identity(r, identity):
    require(r[R['MAGIC']]==identity['magic'] and r[R['BUILD_ID']]==identity['build_id'] and
            r[R['GATE_PROTOCOL']]==identity['protocol'] and r[R['N_REPL']]==identity['default_replicas'],
            'Wrong IP, protocol, source build, or replication factor')

def verify_vectors(m, identity, manifest, out):
    m.call(CMD_NOP); check_identity(m.results(), identity)
    result = {'started_utc':utc(), 'completed':False, 'checks':[]}
    save(out/'board_verification.json', result)
    for item in manifest:
        name=item['name']; stem=ROOT/'vectors'/name
        require(sha(str(stem)+'.bin')==item['input_sha256'], name+' input hash mismatch')
        require(sha(str(stem)+'_spikes.bin')==item['spikes_sha256'], name+' spike hash mismatch')
        x=np.fromfile(str(stem)+'.bin',dtype='<i2'); require(len(x)==item['n_samples'], 'Vector length mismatch');m.load(x)
        for mode,label in [('FE_dense','dense'),('FE_dense_rom','dense'),('FE_gate','gate'),('FE_gate_iso','gate'),('FE_gate_iso_rom','gate'),
                           ('FE_v3','gate'),('FFT1024','fft')]:   # every power-saving variant must reproduce its base mode exactly
            expected_path=str(stem)+'_'+label+'_res.bin'
            require(sha(expected_path)==item[label+'_expected_sha256'], name+' expected hash mismatch')
            expected=np.fromfile(expected_path,dtype='<u4')
            m.verify(CAP_NONE if label=='fft' else CAP_SPIKES, ALL_MODES[mode]); got=np.asarray(m.results(),dtype=np.uint32)
            check_identity(got, identity)
            inds=list(range(394,958)) if label=='fft' else list(range(385))+[979,980,981]
            bad=[i for i in inds if got[i]!=expected[i]]
            require(not bad, f'{name} {mode}: first differing result words {bad[:8]}')
            if label!='fft':
                captured=m.read_capture(len(x)); spikes=np.fromfile(str(stem)+'_spikes.bin',dtype='<i2')
                require(np.array_equal(captured,spikes), name+' encoder capture mismatch')
            result['checks'].append({'vector':name,'mode':mode,'passed':True})
        save(out/'board_verification.json',result)
        print('[VERIFY]',name,'PASS',flush=True)
    result.update(completed=True,finished_utc=utc(),vectors=len(manifest))
    save(out/'board_verification.json',result)
    return result

def find_sensor():
    found=[]
    for d in Path('/sys/class/hwmon').glob('hwmon*'):
        try:
            if (d/'name').read_text().strip()=='ina260_u14' and (d/'power1_input').exists():found.append(d/'power1_input')
        except OSError: pass
    require(len(found)==1,'Expected exactly one ina260_u14 SOM input-power sensor; no fallback sensor is used')
    return found[0]

def temperatures():
    result={}
    for d in Path('/sys/class/thermal').glob('thermal_zone*'):
        try:result[d.name+':'+(d/'type').read_text().strip()]=float((d/'temp').read_text())/1000
        except (OSError,ValueError):pass
    return result

def read_governors():
    return {f:Path(f).read_text().strip() for f in glob.glob('/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor')}

def sample_power(sensor, seconds, m, running, hz=10):
    samples=[];start=time.perf_counter();k=0
    step=min(10.0,max(2.0,seconds/3));next_report=step
    if running and CTX:
        try:CTX['f0']=m.frames_done()
        except Exception:CTX.pop('f0',None)
    while True:
        now=time.perf_counter();deadline=start+k/hz
        if now < deadline:time.sleep(deadline-now)
        now=time.perf_counter()
        if now-start >= seconds:break
        require(m.idle()!=running,'IP stopped during a power block or was active during idle reference')
        power_mw=float(sensor.read_text())/1000
        require(math.isfinite(power_mw) and power_mw>0,'Invalid sensor power')
        row={'t_s':now-start,'power_mW':power_mw}
        if k%10==0:row['temperature_C']=temperatures()
        samples.append(row);k+=1
        if now-start>=next_report:
            next_report+=step
            try:report_block(now-start,seconds,samples,m,running)
            except Exception as e:print('[ILERLEME] durum satiri yazilamadi:',e,flush=True)   # never let reporting stop a measurement
    require(len(samples)>=int(seconds*hz*.9),'Insufficient power samples')
    require(max((b['t_s']-a['t_s'] for a,b in zip(samples,samples[1:])), default=0)<.5,'Power sampling stalled')
    return {'started_monotonic_s':start,'duration_s':time.perf_counter()-start,'samples':samples,
            'mean_mW':float(np.mean([s['power_mW'] for s in samples]))}

def calibrate(m, mode, n, target, clk, hist):
    pace=0;tried={};frames=max(20,math.ceil(2*target*1e6/n))
    for iteration in range(12):
        dt=m.call(CMD_RUN,arg0=frames,blk_en=ALL_MODES[mode],pace=pace,n=n)
        require(m.frames_done()==frames,'Calibration frame counter mismatch')
        rate=frames*n/dt/1e6
        hist.append({'pace':pace,'rate_msps':rate,'duration_s':dt,'frames':frames})
        print(f'[KAL ] {mode}: deneme {iteration+1}, pace={pace} -> {rate:.4f} MS/s (hedef {target:.4f})',flush=True)
        if iteration==0:require(rate>=target*.99, f'{mode} maximum {rate:.6f} MS/s < target {target}. Stop; optimize or explicitly run --target {rate*.9:.4f} as a lower-throughput characterization.')
        if abs(rate/target-1)<=.008:return pace,rate
        tried[pace]=rate
        fast=[p for p,r in tried.items() if r>target];slow=[p for p,r in tried.items() if r<target]
        if fast and slow:
            lo=max(fast);hi=min(slow)
            if hi-lo<=1:break
            new=(lo+hi)//2
        elif len(hist)>1 and hist[-1]['pace']!=hist[-2]['pace']:
            h0,h1=hist[-2:];slope=(1/h1['rate_msps']-1/h0['rate_msps'])/(h1['pace']-h0['pace'])
            new=round(pace+(1/target-1/rate)/slope) if slope>0 else pace+max(1,round(clk/1e6*(1/target-1/rate)))
        else:new=round(pace+clk/1e6*(1/target-1/rate))
        new=max(0,min(1000000,new))
        if new==pace:new=pace+1 if rate>target else max(0,pace-1)
        if new in tried:break
        pace=new
    raise RuntimeError(f'{mode}: cannot match target within 0.8%; inspect calibration.json. No silent rate reduction.')

def measure_run(m,sensor,mode,n,pace,rate,a):
    frames=math.ceil((a.block+a.warmup+5)*rate*1e6/n)
    m.start(CMD_RUN,arg0=frames,blk_en=ALL_MODES[mode],pace=pace,n=n)
    time.sleep(a.warmup)
    first=m.frames_done();block=sample_power(sensor,a.block,m,True);last=m.frames_done()
    require(last>first,'Hardware frame counter did not advance')
    dt=m.wait(timeout=a.block+a.warmup+60);require(m.frames_done()==frames,'Completed frame count mismatch')
    achieved=frames*n/dt/1e6;check_rate(achieved,a.target)
    window_rate=(last-first)*n/block['duration_s']/1e6
    # Counter quantization is one frame; short diagnostic blocks need a larger bound.
    require(abs(window_rate/achieved-1)<=.01+2*n/(a.block*achieved*1e6),'Measurement-window throughput disagrees with whole-run throughput')
    registers=m.results()
    block.update(work_last_frame_replica0={k:int(registers[R[k]]) for k in ['GATE_UPDATES','GATE_SKIPPED','GATE_LONG']},
                 mode=mode,pace=pace,frames=frames,elapsed_s=dt,rate_msps=achieved,
                 window_rate_msps=window_rate,window_frames=last-first)
    return block

# Conservative Student-t critical values; repeat means, not sensor samples, are the units.
T975=[None,12.7062,4.3027,3.1825,2.7765,2.5706,2.4469,2.3646,2.3061,2.2622,2.2282,
      2.2010,2.1788,2.1604,2.1448,2.1315,2.1199,2.1098,2.1009,2.0930,2.0860,
      2.0796,2.0739,2.0687,2.0639,2.0596,2.0556,2.0518,2.0484,2.0452,2.0423]

def interval(values):
    x=np.asarray(values,dtype=float);n=len(x);mean=float(x.mean())
    if n<2:return {'n':n,'mean':mean,'sd':None,'ci95':None}
    sd=float(x.std(ddof=1));half=T975[min(n-1,30)]*sd/math.sqrt(n)
    return {'n':n,'mean':mean,'sd':sd,'ci95':[mean-half,mean+half]}

# paired contrasts (positive = the second mode uses less energy); modes absent from a run are skipped
CONTRASTS=[('dense_minus_v3','FE_dense','FE_v3'),('dense_minus_gate_iso','FE_dense','FE_gate_iso'),
           ('gate_iso_minus_gate_iso_rom','FE_gate_iso','FE_gate_iso_rom'),('gate_iso_rom_minus_v3','FE_gate_iso_rom','FE_v3'),
           ('dense_minus_dense_rom','FE_dense','FE_dense_rom'),('fft_minus_v3','FFT1024','FE_v3'),
           ('dense_minus_gate','FE_dense','FE_gate'),('gate_minus_gate_iso','FE_gate','FE_gate_iso'),('fft_minus_gate_iso','FFT1024','FE_gate_iso'),
           ('fft_minus_gate','FFT1024','FE_gate')]
CONTRAST_LABELS={'dense_minus_v3':'dense-v3','dense_minus_gate_iso':'dense-gate_iso','gate_iso_minus_gate_iso_rom':'gate_iso-(+rom)',
                 'gate_iso_rom_minus_v3':'(+rom)-(+banka)','dense_minus_dense_rom':'dense-dense_rom','fft_minus_v3':'fft-v3',
                 'dense_minus_gate':'dense-gate','gate_minus_gate_iso':'gate-gate_iso','fft_minus_gate_iso':'fft-gate_iso','fft_minus_gate':'fft-gate'}

def analyze(data):
    result={'scope':'SOM input-power frontend characterization; excludes full receiver and RF/ADC chain',
      'units':'mW / MS/s = nJ/sample; per replica division is an allocation across the replicated measurement design',
      'inference':'Nominal paired t intervals across repeat means. FFT comparison is exploratory; no equal receiver-quality proof here.',
      'complete':data.get('complete',False),'diagnostic_only':data['settings']['quick'],
      'target_msps':data['settings']['target'],'realtime_1MHz_average_rate':data['settings']['target']>=1,
      'continuous_stream_realtime_proven':False,'vectors':{}}
    nr=data['n_repl']
    for name in data['settings']['vectors']:
        reps=[r for r in data['repeats'] if r['vector']==name and r.get('complete')]
        if not reps:continue
        modes={k:[] for k in MODES}
        contrasts=CONTRASTS
        paired={key:[] for key,_,_ in contrasts}
        for r in reps:
            by={x['mode']:x for x in r['runs']}
            for mode,row in by.items():
                row['idle_adjusted_nJ_all_replicas']=energy_nj(row['mean_mW']-row['idle_bracket_mW'],row['rate_msps'])
            d0=by['D0_empty']['idle_adjusted_nJ_all_replicas']
            for mode,row in by.items():
                modes.setdefault(mode,[]).append({'som_input_nJ_all_replicas':energy_nj(row['mean_mW'],row['rate_msps']),
                   'idle_adjusted_nJ_all_replicas':row['idle_adjusted_nJ_all_replicas'],
                   'D0_subtracted_nJ_per_replica':(row['idle_adjusted_nJ_all_replicas']-d0)/nr,
                   'rate_msps':row['rate_msps']})
            for key,base,cand in contrasts:
                if base in by and cand in by:
                    paired[key].append((by[base]['idle_adjusted_nJ_all_replicas']-by[cand]['idle_adjusted_nJ_all_replicas'])/nr)
        item={'modes':{mode:{k:interval([row[k] for row in rows]) for k in rows[0]} for mode,rows in modes.items() if rows},'paired_saving_nJ_per_replica':{}}
        for key,vals in paired.items():
            if not vals:continue
            stat=interval(vals);ci=stat['ci95'];stat['positive_saving_resolved_nominal95']=bool(data.get('complete',False) and not data['settings']['quick'] and ci and ci[0]>0)
            item['paired_saving_nJ_per_replica'][key]=stat
        result['vectors'][name]=item
    result['note']='Negative D0-subtracted estimates are retained as unresolved/noisy operational contrasts, never negative physical energy. D0 includes control/wait overhead; its subtraction is not a pure component energy measurement.'
    return result

def print_repeat_summary(an,name,done,total):
    v=an['vectors'].get(name)
    if not v:return
    parts=[]
    for mode,st in v['modes'].items():
        e=st.get('D0_subtracted_nJ_per_replica',{})
        if mode!='D0_empty' and e:parts.append(f"{mode} {e['mean']:.2f}")
    print(f"[OZET] {name}: {done}/{total} tekrar tamam | kopya basina, D0 cikarilmis (nJ/ornek): "+' | '.join(parts),flush=True)
    labels=CONTRAST_LABELS
    out=[]
    for key,lab in labels.items():
        st=v['paired_saving_nJ_per_replica'].get(key)
        if not st:continue
        ci=st.get('ci95');out.append(f"{lab} {st['mean']:+.2f}"+(f" [%95 GA {ci[0]:+.2f}, {ci[1]:+.2f}]" if ci else ''))
    if out:print('[OZET]   eslestirilmis fark (pozitif = sagdaki daha az enerji): '+' | '.join(out),flush=True)

def write_archive(out):
    with zipfile.ZipFile(out/'KV260_GATE_RESULTS.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file() and p.suffix not in ('.zip','.tmp'):z.write(p,p.relative_to(out))
        for rel in ['build_identity.json','board/build_receipt.json','evidence/fixed_quality_summary.json','evidence/native_verification.json']:
            if (ROOT/rel).exists():z.write(ROOT/rel,'provenance/'+rel)
        for p in (ROOT/'vivado/reports').glob('*'):
            if p.is_file():z.write(p,'implementation_reports/'+p.name)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    mode=ap.add_mutually_exclusive_group(required=True);mode.add_argument('--verify-only',action='store_true');mode.add_argument('--quick',action='store_true');mode.add_argument('--overnight',action='store_true')
    ap.add_argument('--target',type=float,default=1.0,help='MS/s per replica; lower values are explicitly lower-throughput characterization')
    ap.add_argument('--vectors',nargs='+',default=DEFAULT_VECTORS)
    ap.add_argument('--block',type=float);ap.add_argument('--repeats',type=int);ap.add_argument('--warmup',type=float,default=5)
    ap.add_argument('--out',type=Path);ap.add_argument('--order-seed',type=int,default=2602026)
    a=ap.parse_args();a.block=a.block or (5 if a.quick else 120);a.repeats=a.repeats or (1 if a.quick else 5)
    require(a.target>0 and a.block>=2 and a.repeats>=1 and a.warmup>=0,'Invalid measurement settings')
    require(not a.overnight or (a.block>=120 and a.repeats>=5),'Overnight requires >=120 s blocks and >=5 repeats')
    out=(a.out or ROOT/'board/results'/datetime.now().strftime('%Y%m%d_%H%M%S')).resolve();out.mkdir(parents=True,exist_ok=False)
    m=None;governors={};data=None
    try:
        receipt,identity,clk=preflight();manifest=readjson(ROOT/'vectors/manifest.json')
        names={r['name']:r for r in manifest};require(len(a.vectors)==len(set(a.vectors)) and all(n in names for n in a.vectors),'Unknown/duplicate vectors')
        m=MoE();verify_vectors(m,identity,manifest,out)
        if a.verify_only:print('BOARD VERIFICATION COMPLETE',out,flush=True);return
        sensor=find_sensor();governors=read_governors();require(governors,'CPU governor information unavailable')
        for p in governors:Path(p).write_text('performance')
        settings={k:v for k,v in vars(a).items() if k!='out'}
        data={'started_utc':utc(),'complete':False,'settings':settings,'n_repl':identity['default_replicas'],
              'environment':{'platform':platform.platform(),'clock_hz':clk,'sensor':str(sensor),'sensor_units':'microW converted to mW',
                  'governors_before':governors,'governors_during':read_governors(),'temperatures_start_C':temperatures(),
                  'bitstream_sha256':receipt['bitstream_sha256']},'calibration':{},'repeats':[]}
        save(out/'measurement.json',data)
        rng=random.Random(a.order_seed)
        global PROG
        PROG=Progress(a,len(a.vectors),len(MODES))
        print(f'[PLAN] {len(a.vectors)} vektor x {a.repeats} tekrar x {len(MODES)} mod ({", ".join(MODES)}), blok {a.block:.0f} s, '
              f'hedef {a.target} MS/s/kopya -> tahmini toplam {fmt_dur(PROG.total)}',flush=True)
        for vi,name in enumerate(a.vectors):
            x=np.fromfile(ROOT/'vectors'/f'{name}.bin',dtype='<i2');n=len(x);m.load(x)
            cal={};data['calibration'][name]=cal
            print(f'[VEKTOR {vi+1}/{len(a.vectors)}] {name}: {n} ornek yuklendi; {len(MODES)} mod ayni hiza kalibre ediliyor | {PROG.status()}',flush=True)
            for design in MODES:
                hist=[];cal[design]={'history':hist};save(out/'calibration.json',data['calibration'])
                try:pace,rate=calibrate(m,design,n,a.target,clk,hist)
                finally:save(out/'calibration.json',data['calibration'])
                cal[design].update(pace=pace,rate_msps=rate);print('[RATE]',name,design,f'{rate:.6f} MS/s, pace={pace}',flush=True)
            cr=[v['rate_msps'] for v in cal.values()]
            require(max(cr)/min(cr)-1<=.01,'Calibrated modes differ by >1% in rate; comparison blocked')
            # Thermal preparation before the first reference block on this vector.
            thermal_s=2 if a.quick else 60
            PROG.add(Progress.CAL_S)
            print(f'[ISINMA] {name}: {thermal_s} s FE_dense calistiriliyor (ilk referans oncesi sicaklik dengesi) | {PROG.status()}',flush=True)
            cc=cal['FE_dense'];m.call(CMD_RUN,arg0=max(1,math.ceil(thermal_s*cc['rate_msps']*1e6/n)),blk_en=DESIGNS['FE_dense'],pace=cc['pace'],n=n)
            PROG.add(thermal_s)
            for rep in range(a.repeats):
                order=MODES.copy();rng.shuffle(order)
                record={'vector':name,'repeat':rep,'order':order,'complete':False,'runs':[],'idles':[]};data['repeats'].append(record)
                print(f'[TEKRAR] {name} tekrar {rep+1}/{a.repeats}, mod sirasi: {" -> ".join(order)} | {PROG.status()}',flush=True)
                CTX.clear();CTX.update(vector=f'{name} ({vi+1}/{len(a.vectors)})',rep=f'{rep+1}/{a.repeats}',n=n,last_idle_mW=None)
                record['idles'].append(sample_power(sensor,a.block,m,False));save(out/'measurement.json',data)
                PROG.add(a.block);CTX['last_idle_mW']=record['idles'][-1]['mean_mW']
                for mi,design in enumerate(order):
                    CTX.update(mode=design,mode_i=f'{mi+1}/{len(order)}')
                    cc=cal[design];row=measure_run(m,sensor,design,n,cc['pace'],cc['rate_msps'],a);record['runs'].append(row)
                    PROG.add(PROG.per_run)
                    save(out/'measurement.json',data)
                    idle=sample_power(sensor,a.block,m,False);record['idles'].append(idle)
                    PROG.add(a.block);CTX['last_idle_mW']=idle['mean_mW']
                    row['idle_bracket_mW']=(record['idles'][-2]['mean_mW']+idle['mean_mW'])/2
                    dmw=row['mean_mW']-row['idle_bracket_mW'];nr=data['n_repl']
                    print('[POWER]',name,rep+1,design,f"{row['mean_mW']:.2f} mW; idle delta {dmw:+.2f} mW "
                          f"(kopya basina {dmw/nr:+.2f} mW = {energy_nj(dmw/nr,row['rate_msps']):+.2f} nJ/ornek, D0 cikarilmamis) | {PROG.status()}",flush=True)
                    save(out/'measurement.json',data)
                rates=[r['rate_msps'] for r in record['runs']]
                require(max(rates)/min(rates)-1<=.01,'Measured modes differ by >1% in rate; this repeat is invalid')
                require(clock_hz(receipt)==clk,'PL clock changed during measurement')
                record['complete']=True;save(out/'measurement.json',data);an=analyze(data);save(out/'analysis.json',an)
                try:print_repeat_summary(an,name,sum(1 for r in data['repeats'] if r['vector']==name and r.get('complete')),a.repeats)
                except Exception as e:print('[OZET] yazilamadi:',e,flush=True)
        data.update(complete=True,finished_utc=utc());save(out/'measurement.json',data);save(out/'analysis.json',analyze(data))
        print('COMPLETED:',out/'KV260_GATE_RESULTS.zip',flush=True)
    except BaseException as exc:
        save(out/'failure.json',{'utc':utc(),'type':type(exc).__name__,'error':str(exc)})
        print('STOPPED:',str(exc),'\nEvidence:',out,flush=True)
        raise
    finally:
        for p,value in governors.items():
            try:Path(p).write_text(value)
            except OSError:pass
        if m is not None:m.close()
        write_archive(out)

if __name__=='__main__':main()
