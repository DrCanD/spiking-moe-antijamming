"""Produce board vectors and cross-check the C++ frontend (v3 numerics: Q1.17 rotations, Q20 state) against the Python integer oracle."""
import os,sys,json,hashlib,subprocess,re
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'software'))
import run_experiment as sim
from fixed_reference import run_v3 as run,quantize

def hashfile(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def statistics(sp,counts):
    n=len(sp);sf=n//20;ids=np.flatnonzero(sp);d=np.diff(ids);r=np.zeros(385,np.uint32)
    r[0]=len(ids);r[1]=len(d);r[2]=int(d.sum());sq=int((d*d).sum());r[3]=sq&0xffffffff;r[4]=sq>>32
    r[5:25]=np.bincount(np.minimum(ids//sf,19),minlength=20)
    if len(d):
        r[25:45]=np.bincount(np.minimum(ids[1:]//sf,19),weights=d,minlength=20).astype(np.uint32)
        r[45:65]=np.bincount(np.minimum(ids[1:]//sf,19),minlength=20)
    r[65:385]=counts.reshape(-1);return r

def main():
    cfg=sim.core.Config();meta=[];expected={}
    for ci,cond in enumerate(['fixed','random']):
        for ji,jam in enumerate(['none','narrowband','sweep','pulse','broadband','switching']):
            seed=sim.derive(310,ci,ji)
            if jam=='switching':f=sim.generate(cfg,cond,sim.CASES[1],seed,sim.derive(311,ci,ji));rx=f['rx'][:100000]
            else:rx=sim.core.generate_frame(cfg,cond,() if jam=='none' else (jam,),() if jam=='none' else (12,),seed)['rx']
            name=cond+'_'+jam;x=quantize(rx);p=ROOT/'vectors'/f'{name}.bin';x.tofile(p)
            sp,c,w,_=run(x,bin_samples=len(x)//20);sp.astype('<i2').tofile(ROOT/'vectors'/f'{name}_spikes.bin')
            expected[name]=(statistics(sp,c[0]),statistics(sp,c[1]),w)
            meta.append(dict(name=name,seed=seed,n_samples=len(x),input_sha256=hashfile(p),clipped_samples=int(np.count_nonzero((rx*1024>32767)|(rx*1024<-32768)))))
    for name in ['edge_silence','rtl_smoke']:
        x=np.zeros(4000,dtype='<i2')
        if name=='rtl_smoke':
            # Alternating polarities, active trajectories, then >256 samples of silence.
            x[20:1200]=np.rint(7000*np.sin(np.arange(1180)*.41)).astype('<i2')
            x[2300:3500]=np.rint(9000*np.sin(np.arange(1200)*.17)).astype('<i2')
        p=ROOT/'vectors'/f'{name}.bin';x.tofile(p)
        sp,c,w,_=run(x,bin_samples=200);sp.astype('<i2').tofile(ROOT/'vectors'/f'{name}_spikes.bin')
        expected[name]=(statistics(sp,c[0]),statistics(sp,c[1]),w)
        if name=='rtl_smoke':assert w[0]>0 and w[1]>0 and w[2]>0,w
        meta.append(dict(name=name,n_samples=4000,input_sha256=hashfile(p),clipped_samples=0))
    exe=ROOT/'evidence/tb_gate_native'
    subprocess.run(['g++','-O2','-std=c++14','-DN_REPL=1','-I'+str(ROOT/'hls/src'),str(ROOT/'hls/src/moe_top.cpp'),str(ROOT/'hls/tb/tb_gate.cpp'),'-o',str(exe)],check=True)
    names=[r['name'] for r in meta]
    out=subprocess.run([str(exe),str(ROOT/'vectors'),*names,'--generate'],check=True,capture_output=True,text=True).stdout
    print(out,end='',flush=True)
    bank={m.group(1):int(m.group(2)) for m in re.finditer(r'PASS (\S+) gate_iso_rom_bank bank_skips=(\d+)',out)}
    audit=[]
    for item in meta:
        name=item['name'];a,b,w=expected[name]
        for label,ref in [('dense',a),('gate',b)]:
            p=ROOT/'vectors'/f'{name}_{label}_res.bin';got=np.fromfile(p,dtype='<u4')
            np.testing.assert_array_equal(got[:385],ref)
            if label=='gate':np.testing.assert_array_equal(got[979:982],w[:3])
            else:assert got[979]==16*item['n_samples'] and got[980]==0
            item[label+'_expected_sha256']=hashfile(p)
        item['fft_expected_sha256']=hashfile(ROOT/'vectors'/f'{name}_fft_res.bin')
        item['spikes_sha256']=hashfile(ROOT/'vectors'/f'{name}_spikes.bin')
        assert bank[name]==int(w[5]),(name,bank.get(name),int(w[5]))
        audit.append(dict(name=name,python_cpp_dense_equal=True,python_cpp_gate_equal=True,encoder_equal=True,bank_skips=int(w[5]),
                          variants_equal=['dense_rom==dense','gate_iso==gate','gate_iso_rom==gate','gate_iso_rom_bank==gate']))
    (ROOT/'vectors/manifest.json').write_text(json.dumps(meta,indent=2))
    (ROOT/'evidence/native_verification.json').write_text(json.dumps(dict(vectors=len(meta),sample_count=sum(r['n_samples'] for r in meta),
        native_frontend_matches_python=True,numerics='v3: Q1.17 rotations (18-bit), Q20 state (24-bit registers); see evidence/width_sweep_summary.json',source_digest=json.loads((ROOT/'build_identity.json').read_text())['source_digest'],
        integer_oracle_sha256=hashfile(ROOT/'fixed_reference.py'),vector_manifest_sha256=hashfile(ROOT/'vectors/manifest.json'),fft_oracle='existing native C++ implementation; actual ap_int and RTL verification required by build flow',checks=audit),indent=2))
    print('PYTHON / C++ INTEGER FRONTEND MATCH',len(meta),flush=True)
if __name__=='__main__':main()
