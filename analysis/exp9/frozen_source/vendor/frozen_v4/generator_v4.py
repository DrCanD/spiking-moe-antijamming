"""Separate channel/payload randomness; coded BPSK on continuous jammer phase."""
from dataclasses import replace
import numpy as np
import streaming_v3 as legacy
import link_v4

def generate(cfg,channel_seed,payload_seed,scenario,condition):
    if condition not in ('fixed','random'):raise ValueError(condition)
    s=replace(legacy.StreamSettings(),dwell_ms=scenario['dwell_ms'],dwell_jitter=scenario.get('dwell_jitter',.25),
        jsr_db=scenario.get('jsr_db',12.),relative_pulse_jsr_db=tuple(scenario.get('pulse_relative_db',[-6.,0.,6.])))
    b=legacy._validate(cfg,s);rng=np.random.default_rng(channel_seed)
    names=['none','narrowband','narrowband+pulse','narrowband','sweep','none']
    edges=[0]
    for _ in names:
        d=s.dwell_ms*(rng.uniform(1-s.dwell_jitter,1+s.dwell_jitter) if condition=='random' else 1.)
        end=edges[-1]+max(2*b,round(d*cfg.fs/1000/cfg.sps)*cfg.sps)
        if end%b==0:end+=cfg.sps
        edges.append(int(end))
    edges[-1]=int(np.ceil(edges[-1]/b))*b;n=edges[-1]
    bits,payloads,words,meta=link_v4.make_bits(payload_seed,n//cfg.sps)
    tx=np.repeat(2*bits.astype(float)-1,cfg.sps)
    snr=scenario.get('snr_db',10.);rx=tx+rng.normal(0,10**(-snr/20),n)
    signal_power=1.+10**(-snr/10);jsr=s.jsr_db+(rng.uniform(-4,4) if condition=='random' else 0.)
    ratio=float(rng.choice(s.relative_pulse_jsr_db));fa=(rng.uniform(*cfg.nb_f_range) if condition=='random' else .3)*cfg.fs/2
    fb=(rng.uniform(*cfg.sweep_f_range) if condition=='random' else .95)*cfg.fs/2
    phase0=rng.uniform(0,2*np.pi);idx=np.arange(n);phase=phase0+2*np.pi*fa*idx/cfg.fs
    sw0,sw1=edges[4],edges[5];u=np.arange(sw1-sw0)/cfg.fs;dur=(sw1-sw0)/cfg.fs
    phase[sw0:sw1]=phase0+2*np.pi*fa*sw0/cfg.fs+2*np.pi*(fa*u+.5*(fb-fa)*u*u/dur)
    duty_range=scenario.get('pulse_duty_range',list(cfg.pulse_duty_range))
    duty=float(rng.uniform(*duty_range)) if condition=='random' else float(scenario.get('fixed_pulse_duty',.2))
    period=int(rng.integers(cfg.pulse_period_range[0],cfg.pulse_period_range[1]+1)) if condition=='random' else 500
    offset=int(rng.integers(period));env=((idx+offset)%period)<max(1,int(period*duty))
    tone=np.sqrt(2*signal_power*10**(jsr/10))*np.sin(phase)
    pulse=rng.normal(0,np.sqrt(signal_power*10**((jsr+ratio)/10)/duty),n)*env
    if not scenario.get('clean',False):
        rx[edges[1]:edges[5]]+=tone[edges[1]:edges[5]];rx[edges[2]:edges[3]]+=pulse[edges[2]:edges[3]]
    else:names=['none']*6
    segs=[dict(name=nm,start_symbol=a//cfg.sps,end_symbol=z//cfg.sps) for nm,a,z in zip(names,edges[:-1],edges[1:])]
    transitions=[] if scenario.get('clean',False) else [v//cfg.sps for v in edges[1:-1]]
    return dict(rx=rx,bits=bits,payloads=payloads,words=words,coding=meta,segments=segs,transitions=transitions,
        metadata=dict(condition=condition,scenario=scenario,n_samples=n,block_samples=b,jsr_db=jsr,
          pulse_relative_jsr_db=ratio,pulse_duty=duty,pulse_period=period,tone_hz=fa,sweep_end_hz=fb,
          nominal_jsr_reference='ensemble signal plus AWGN power; per jammer component',
          continuous_tone_phase=True,perfect_bit_and_codeword_synchronization=True))
