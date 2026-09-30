"""Build or natively verify the KV260 feature-finalisation/forest kernel."""
from pathlib import Path
import argparse
import os
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent))
from build_kv260 import executable,execute


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--native-only',action='store_true')
    ap.add_argument('--jobs',type=int,default=4)
    a=ap.parse_args();logs=ROOT/'build_logs';logs.mkdir(exist_ok=True)
    if a.native_only:
        compiler=shutil.which('g++')
        if not compiler:raise RuntimeError('g++ is required for native verification')
        program=logs/('classifier_native.exe' if os.name=='nt' else 'classifier_native')
        subprocess.run([compiler,'-O2','-std=c++14','-Wno-unknown-pragmas','-ffp-contract=off',
                        '-I'+str(ROOT/'hls/src'),str(ROOT/'hls/src/rf_top.cpp'),str(ROOT/'hls/tb/tb_rf.cpp'),'-o',str(program)],check=True)
        execute([str(program),str(ROOT/'hls/tb/rf_tests.bin')],ROOT,logs/'native.log')
        print((logs/'native.log').read_text().splitlines()[-1]);return
    vitis=executable('vitis-run');compiler=executable('v++');vivado=executable('vivado')
    for name,command in [('csim',[vitis,'--mode','hls','--csim']),('synth',[compiler,'-c','--mode','hls']),('package',[vitis,'--mode','hls','--package'])]:
        execute(command+['--config','hls_config.cfg','--work_dir','rf_hls_work'],ROOT/'hls',logs/(name+'.log'))
    header=ROOT/'hls/rf_hls_work/hls/impl/ip/drivers/rf_top_v1_0/src/xrf_top_hw.h'
    execute([sys.executable,str(ROOT.parent/'board/parse_regmap.py'),str(header),str(ROOT/'board/regmap_rf.json')],ROOT,logs/'regmap.log')
    (ROOT/'board/firmware').mkdir(exist_ok=True)
    os.environ['MOE_JOBS']=str(a.jobs)
    execute([vivado,'-mode','batch','-source','build_bd.tcl'],ROOT/'vivado',logs/'vivado.log')
    execute([executable('bootgen'),'-image','rf.bif','-arch','zynqmp','-o',str(ROOT/'board/firmware/rf.bit.bin'),'-w'],ROOT/'vivado',logs/'bootgen.log')
    print('Built classifier kernel. Board files:',Path('hardware/kv260/classifier_tail/board'))


if __name__=='__main__':
    main()
