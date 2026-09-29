#!/usr/bin/env python3
"""Run on the Vivado/Vitis workstation, not Colab or the KV260 ARM processor."""
import sys,os,shutil,subprocess,json,hashlib,time,re,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def executable(name):
    p=shutil.which(name) or shutil.which(name+'.bat') or shutil.which(name+'.exe')
    if p:return p
    import glob
    roots=[r+':/' for r in 'CDEF'] if os.name=='nt' else ['/opt/','/tools/']
    dirs=[]
    for r in roots:
        for pat in ('AMDDesignTools/*/Vitis/bin','AMDDesignTools/*/Vivado/bin','Xilinx/Vitis/*/bin','Xilinx/Vivado/*/bin','Xilinx/*/Vitis/bin','Xilinx/*/Vivado/bin'):
            dirs+=glob.glob(r+pat)
    dirs.sort(key=lambda d:('2025.2' in d,d),reverse=True)
    for d in dirs:
        for ext in ('.bat','.exe',''):
            q=Path(d)/(name+ext)
            if q.exists():print(f'[ARAC] {name}: {q} (PATH disinda bulundu)',flush=True);return str(q)
    raise RuntimeError(name+' not found. Open the AMD/Vitis command prompt or source settings64.sh.')
EVENT=re.compile(r'(Pipelining result|Finished|Starting|Phase \d|CSIM|C-SIM|Generating|Implementing|Synthesizing|report_|write_bitstream|'
                  r'launch_runs|Waiting for|route_design|place_design|opt_design|INFO: \[HLS 200-(10|111|1470|789)\]|ERROR|CRITICAL WARNING|PASS )')
def last_event(log):
    '''Latest meaningful line of a tool log, for the progress display.'''
    try:
        with open(log,'rb') as f:
            f.seek(0,2);size=f.tell();f.seek(max(0,size-65536));lines=f.read().decode(errors='replace').splitlines()
        for ln in reversed(lines):
            if EVENT.search(ln):return ln.strip()[:120]
        return (lines[-1].strip()[:120] if lines else '(log bos)')
    except OSError:return '(log okunamadi)'
def execute(command,cwd,log):
    print('RUN',*command,flush=True)
    started=time.monotonic()
    with open(log,'w') as f:
        proc=subprocess.Popen(command,cwd=cwd,stdout=f,stderr=subprocess.STDOUT)
        try:
            while True:
                try:code=proc.wait(timeout=30);break
                except subprocess.TimeoutExpired:
                    print(f'  [{Path(log).stem}] {(time.monotonic()-started)/60:.1f} dk | son olay: {last_event(log)}',flush=True)
        except BaseException:
            proc.terminate()
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:proc.kill()
            raise
    if code:
        tail=Path(log).read_text(errors='replace')[-6000:]
        raise RuntimeError(f'Failed ({code}); log={log}\n{tail}')

def check_rf_bank(log):
    '''v3: the band loop must stay at II=1 with the registered DSP multipliers (DEPENDENCE pragmas); report what HLS achieved.'''
    txt=Path(log).read_text(errors='replace');rows=[ln.strip() for ln in txt.splitlines() if 'Pipelining result' in ln and 'RF_BANK' in ln]
    for ln in rows:print('[HLS] '+ln[:200],flush=True)
    iis=[int(m.group(1)) for ln in rows for m in [re.search(r'Final II = (\d+)',ln)] if m]
    if not rows:print('[HLS] UYARI: synth.log icinde RF_BANK pipeline satiri bulunamadi',flush=True)
    elif max(iis or [1])>1:print('[HLS] UYARI: RF_BANK II>1; orneklik cevrim sayisi artar (enerji yine olculebilir). Log ve csynth raporunu gonderin.',flush=True)
    else:print('[HLS] RF_BANK II=1 (bant dongusu her cevrimde bir bant)',flush=True)
    return rows

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--flow',choices=['unified','classic'],default='unified');ap.add_argument('--jobs',type=int,default=4)
    # --from package: resume after a finished csim + synth in the SAME gate_hls_work (their logs must exist and have passed).
    # --skip-cosim: RTL co-simulation is not run; recorded in the receipt. The board --verify-only run (14 vectors x 4 modes,
    #               bit-exact on silicon) then carries the RTL check. Default behaviour is unchanged.
    ap.add_argument('--from',dest='start',choices=['csim','package'],default='csim');ap.add_argument('--skip-cosim',action='store_true');args=ap.parse_args()
    identity=json.loads((ROOT/'build_identity.json').read_text())
    h=hashlib.sha256(b''.join(p.name.encode()+p.read_bytes() for p in sorted((ROOT/'hls/src').glob('*')) if p.name!='build_id.h')).hexdigest()
    if h!=identity['source_digest']:raise RuntimeError('Hardware source identity changed; regenerate and revalidate the package.')
    quality=json.loads((ROOT/'evidence/fixed_quality_summary.json').read_text())
    if any(r['new_failures'] or r['different_action_streams'] or r['lazy_undetected']>r['dense_undetected']
           or r.get('v3dense_new_failures',1) or r.get('v3dense_different_action_streams',1) or r.get('v3dense_undetected',1)>r['dense_undetected'] for r in quality):
        raise RuntimeError('Fixed-point quality check did not pass; build blocked.')
    protocol=json.loads((ROOT/'evidence/fixed_quality_protocol.json').read_text())
    for name,wanted in protocol['source_sha256'].items():
        if sha(ROOT/name)!=wanted:raise RuntimeError('Quality evidence is stale for '+name)
    native=json.loads((ROOT/'evidence/native_verification.json').read_text())
    if (native.get('source_digest')!=h or native.get('integer_oracle_sha256')!=sha(ROOT/'fixed_reference.py')
        or native.get('vector_manifest_sha256')!=sha(ROOT/'vectors/manifest.json') or not native['native_frontend_matches_python']):
        raise RuntimeError('Native verification is stale; rerun prepare_vectors.py.')
    (ROOT/'board/build_receipt.json').unlink(missing_ok=True)
    (ROOT/'board/firmware').mkdir(parents=True,exist_ok=True)
    logs=ROOT/'build_logs';logs.mkdir(exist_ok=True);receipt={'identity':identity,'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'steps':[]}
    if args.start=='package':
        if args.flow!='unified':raise RuntimeError('--from package is only supported for the unified flow')
        csim_log=(logs/'csim.log').read_text(errors='replace') if (logs/'csim.log').exists() else ''
        synth_log=(logs/'synth.log').read_text(errors='replace') if (logs/'synth.log').exists() else ''
        if 'CSim done with 0 errors' not in csim_log and 'CSIM finish' not in csim_log:raise RuntimeError('--from package: build_logs/csim.log does not show a finished, error-free csim')
        if 'ERROR' in csim_log.replace('0 errors',''):raise RuntimeError('--from package: csim.log contains ERROR lines')
        if 'Estimated Fmax' not in synth_log or 'ERROR:' in synth_log:raise RuntimeError('--from package: build_logs/synth.log does not show a finished, error-free synthesis')
        if identity['source_digest'] not in json.dumps(identity):raise RuntimeError('identity')
        receipt['steps']+=['csim (previous run, build_logs/csim.log)','synth (previous run, build_logs/synth.log)']
    if args.skip_cosim:receipt['cosim_skipped']='RTL co-simulation skipped by --skip-cosim; silicon verification (run_gate.py --verify-only) is the RTL check'
    try:
        if args.flow=='unified':
            vr=executable('vitis-run');vc=executable('v++')
            stages=[('csim',[vr,'--mode','hls','--csim']),('synth',[vc,'-c','--mode','hls']),('cosim',[vr,'--mode','hls','--cosim']),('package',[vr,'--mode','hls','--package'])]
            if args.start=='package':stages=[x for x in stages if x[0]=='package']
            if args.skip_cosim:stages=[x for x in stages if x[0]!='cosim']
            for stage,cmd in stages:
                execute(cmd+['--config','hls_config.cfg','--work_dir','gate_hls_work'],ROOT/'hls',logs/(stage+'.log'))
                receipt['steps'].append(stage)
                if stage=='synth':receipt['rf_bank_pipelining']=check_rf_bank(logs/'synth.log')
        else:
            execute([executable('vitis_hls'),'-f','run_hls.tcl'],ROOT/'hls',logs/'classic_hls.log');receipt['steps']+=['csim','synth','cosim','package']
        os.environ['GATE_JOBS']=str(args.jobs)
        selected_ip=ROOT/'hls/gate_hls_work/hls/impl/ip' if args.flow=='unified' else ROOT/'hls/gate_hls/sol1/impl/ip'
        os.environ['GATE_IP_DIR']=str(selected_ip.resolve())
        execute([executable('vivado'),'-mode','batch','-source','build_bd.tcl','-log','build.log','-journal','build.jou'],ROOT/'vivado',logs/'vivado.log')
        status=(ROOT/'vivado/reports/timing_status.txt').read_text()
        if 'TIMING MET' not in status or 'NOT MET' in status:raise RuntimeError('Timing is not met.')
        receipt['steps'].append('vivado_timing_met')
        ip=selected_ip
        headers=list(ip.rglob('xmoe_top_hw.h'))
        if len(headers)!=1:raise RuntimeError('Expected one generated HLS register-map header; found '+str(headers))
        execute([sys.executable,str(ROOT/'board/parse_regmap.py'),str(headers[0]),str(ROOT/'board/regmap.json')],ROOT,logs/'regmap.log')
        binary=ROOT/'board/firmware/moe_gate.bit.bin'
        execute([executable('bootgen'),'-image','moe_gate.bif','-arch','zynqmp','-o',str(binary),'-w'],ROOT/'vivado',logs/'bootgen.log')
        if not binary.exists() or binary.stat().st_size<100000:raise RuntimeError('Firmware binary missing/incomplete.')
        receipt.update(flow=args.flow,completed=True,bitstream_sha256=sha(binary),regmap_sha256=sha(ROOT/'board/regmap.json'),
            overlay_source_sha256=sha(ROOT/'board/firmware/pl.dtsi'),timing_status=status,
            clock_request_hz=int((ROOT/'board/firmware/pl_clk_hz.txt').read_text().strip()),
            scope='frontends in matched single bitstream; not a complete v5 receiver or measured energy')
        (ROOT/'board/firmware/shell.json').write_text('{"shell_type":"XRT_FLAT","num_slots":"1"}\n')
        (ROOT/'board/build_receipt.json').write_text(json.dumps(receipt,indent=2))
        print('BUILD COMPLETE. Copy this package to the KV260. Then run board/install_gate.sh and board/run_gate.py.',flush=True)
    except BaseException as e:
        receipt.update(completed=False,error=str(e));(logs/'failure.json').write_text(json.dumps(receipt,indent=2));raise
if __name__=='__main__':main()
