"""Integer oracles, separate from the C++ HLS implementation.

run()    : frozen reference front-end: Q6.10 samples, Q*.20 resonator state, Q2.30 rotations (dense + lazy paths).
run_v3() : v3 hardware numerics: the same Q20 state, rotations re-quantized to Q1.17 (18-bit signed, one DSP48E2 B port),
           jump table built from the Q1.17 rotation. Chosen by tools/width_sweep.py: the narrowest width with identical router
           actions on all 288 quality streams (16-bit coefficients or a Q16 state changed at least one decision).
Lazy updates are a numerical variant, NOT bit-identical to repeated truncation.
"""
from pathlib import Path
import re
import numpy as np
import numba
ROOT=Path(__file__).resolve().parent
txt=(ROOT/'reference/moe_params.h').read_text()
def parse(name):return np.array([int(s) for s in re.search(name+r'\[16\]\s*=\s*\{([^}]+)',txt).group(1).replace('\n',' ').split(',') if s.strip()],np.int64)
CR=parse('RF_ROT_RE_Q');CI=parse('RF_ROT_IM_Q');A=(CR+1j*CI)/2**30
# Direct jump through up to 256 samples; beyond this the ideal contribution
# of the old state is <0.023 Q20 LSB for the reachable |z| <=1+roundoff.
# Dropping it does not reproduce the dense recurrence's rounding limit cycle.
LR=np.rint((A[:,None]**np.arange(257)[None,:]).real*2**30).astype(np.int64)
LI=np.rint((A[:,None]**np.arange(257)[None,:]).imag*2**30).astype(np.int64)

@numba.njit(cache=True,fastmath=False)
def run(x,cr=CR,ci=CI,lr=LR,li=LI,bin_samples=250):
    n=len(x);sp=np.zeros(n,np.int8);ref=int(x[0]);refr=0
    for t in range(n):
        if refr:refr-=1;continue
        d=int(x[t])-ref
        if d>=2048:sp[t]=1;ref+=2048;refr=2
        elif d<=-2048:sp[t]=-1;ref-=2048;refr=2
    counts=np.zeros((2,(n+bin_samples-1)//bin_samples,16),np.int64)
    updates=0;skips=0;long=0;disagree=0;maxerr=0
    final=np.zeros((2,16,2),np.int64)
    for k in range(16):
        dr=0;di=0;gr=0;gi=0;dp=False;gp=False;stamp=-1;safe=True
        for t in range(n):
            nr=(dr*cr[k]-di*ci[k])>>30;ni=(dr*ci[k]+di*cr[k])>>30
            dr=nr+int(sp[t])*69905;di=ni;active=di>=125829
            de=active and not dp;counts[0,t//bin_samples,k]+=int(de);dp=active;ge=False
            if sp[t]==0 and safe:skips+=1
            else:
                gap=t-stamp
                if gap>256:gr=0;gi=0;long+=1
                else:
                    nr=(gr*lr[k,gap]-gi*li[k,gap])>>30;ni=(gr*li[k,gap]+gi*lr[k,gap])>>30
                    gr=nr;gi=ni
                gr+=int(sp[t])*69905;stamp=t;active=gi>=125829;ge=active and not gp;gp=active
                counts[1,t//bin_samples,k]+=int(ge);updates+=1
                safe=abs(gr)+abs(gi)<125765
                maxerr=max(maxerr,abs(dr-gr),abs(di-gi))
            disagree+=int(de!=ge)
            assert abs(dr)<2**27 and abs(di)<2**27 and abs(gr)<2**27 and abs(gi)<2**27
        final[0,k]=np.array([dr,di]);final[1,k]=np.array([gr,gi])
    return sp,counts,np.array([updates,skips,long,disagree,maxerr]),final

def quantize(rx):return np.clip(np.rint(np.asarray(rx)*1024),-32768,32767).astype('<i2')

# ---------------------------------------------------------------- v3 numerics ----------------------------------------
V3_COEF_FRAC=17; V3_COEF_BITS=18; V3_STATE_BITS=24          # Q1.17 rotation; Q20 state held in 24-bit registers (|z| < 2^20)
V3_S_IN=69905; V3_TH=125829; V3_MARGIN=64                     # unchanged Q20 constants (state scale is unchanged)
CR3=np.rint(CR/2.0**(30-V3_COEF_FRAC)).astype(np.int64); CI3=np.rint(CI/2.0**(30-V3_COEF_FRAC)).astype(np.int64)
A3=(CR3+1j*CI3)/2**V3_COEF_FRAC
LR3=np.rint((A3[:,None]**np.arange(257)[None,:]).real*2**V3_COEF_FRAC).astype(np.int64)
LI3=np.rint((A3[:,None]**np.arange(257)[None,:]).imag*2**V3_COEF_FRAC).astype(np.int64)
assert max(np.abs(LR3).max(),np.abs(LI3).max())<=2**V3_COEF_FRAC and np.abs(np.r_[LR3[:,1:].ravel(),LI3[:,1:].ravel()]).max()<2**V3_COEF_FRAC

@numba.njit(cache=True,fastmath=False)
def _run_v3(sp,cr,ci,lr,li,F,sin,th,safe_th,lim,bin_samples):
    n=len(sp);counts=np.zeros((2,(n+bin_samples-1)//bin_samples,16),np.int64)
    updates=0;skips=0;long=0;disagree=0;maxerr=0;final=np.zeros((2,16,2),np.int64);allskip=np.ones(n,np.int64)
    for k in range(16):
        dr=0;di=0;gr=0;gi=0;dp=False;gp=False;stamp=-1;safe=True
        for t in range(n):
            nr=(dr*cr[k]-di*ci[k])>>F;ni=(dr*ci[k]+di*cr[k])>>F
            dr=nr+int(sp[t])*sin;di=ni;active=di>=th
            de=active and not dp;counts[0,t//bin_samples,k]+=int(de);dp=active;ge=False
            if sp[t]==0 and safe:skips+=1
            else:
                allskip[t]=0
                gap=t-stamp
                if gap>256:gr=0;gi=0;long+=1
                else:
                    nr=(gr*lr[k,gap]-gi*li[k,gap])>>F;ni=(gr*li[k,gap]+gi*lr[k,gap])>>F
                    gr=nr;gi=ni
                gr+=int(sp[t])*sin;stamp=t;active=gi>=th;ge=active and not gp;gp=active
                counts[1,t//bin_samples,k]+=int(ge);updates+=1
                safe=abs(gr)+abs(gi)<safe_th
                maxerr=max(maxerr,abs(dr-gr),abs(di-gi))
            disagree+=int(de!=ge)
            assert abs(dr)<lim and abs(di)<lim and abs(gr)<lim and abs(gi)<lim
        final[0,k,0]=dr;final[0,k,1]=di;final[1,k,0]=gr;final[1,k,1]=gi
    return counts,np.array([updates,skips,long,disagree,maxerr,allskip.sum()]),final

def spikes(x):
    """Delta encoder (theta 2048, refractory 2), identical to run()."""
    n=len(x);sp=np.zeros(n,np.int8);ref=int(x[0]);refr=0
    for t in range(n):
        if refr:refr-=1;continue
        d=int(x[t])-ref
        if d>=2048:sp[t]=1;ref+=2048;refr=2
        elif d<=-2048:sp[t]=-1;ref-=2048;refr=2
    return sp

def run_v3(x,bin_samples=250):
    """Same return format as run(): spikes, counts[dense|lazy], [updates,skips,long,disagree,maxerr,bank_skips], final states.
    bank_skips = samples in which every band is skipped (= the whole-bank skips of EN_BANK)."""
    sp=spikes(np.asarray(x))
    c,w,final=_run_v3(sp,CR3,CI3,LR3,LI3,V3_COEF_FRAC,V3_S_IN,V3_TH,V3_TH-V3_MARGIN,2**(V3_STATE_BITS-1),bin_samples)
    return sp,c,w,final

def write_lut():
    """hls/src/gate_lut.h for v3: Q1.17 rotations and the Q1.17 jump table, typed so HLS stores 18-bit ROM words."""
    lines=['// Generated by fixed_reference.write_lut() from the frozen Q2.30 rotations re-quantized to Q1.17. Do not edit.','#pragma once',
        '#include "moe_types.h"',f'#define RF3_COEF_FRAC {V3_COEF_FRAC}',f'#define RF3_COEF_BITS {V3_COEF_BITS}',f'#define RF3_STATE_BITS {V3_STATE_BITS}',
        f'#define RF3_S_IN {V3_S_IN}',f'#define RF3_TH {V3_TH}',
        f'#define GATE_MARGIN_Q {V3_MARGIN}','#define GATE_MAX_GAP 256','#define GATE_PROTOCOL 0x20260927u',
        'typedef sint<RF3_COEF_BITS> rfc_t;   // rotation coefficient Q1.17',
        'typedef sint<RF3_STATE_BITS> rfs_t;  // resonator state Q20']
    for name,arr in [('RF3_ROT_RE',CR3),('RF3_ROT_IM',CI3)]:
        lines.append('static const rfc_t '+name+'[16] = {'+','.join(map(str,arr))+'};')
    for name,arr in [('GATE_ROT_RE_Q',LR3),('GATE_ROT_IM_Q',LI3)]:
        a=arr.copy();a[:,0]=0          # gap 0 is never used (a band is updated at most once per sample); keeps every word in 18 bits
        lines.append('static const rfc_t '+name+'[16][257] = {')
        for row in a:lines.append('  {'+','.join(map(str,row))+'},')
        lines.append('};')
    (ROOT/'hls/src/gate_lut.h').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':write_lut()
