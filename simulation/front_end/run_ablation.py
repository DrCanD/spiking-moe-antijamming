"""Matched train/test front-end sweeps: LIF count, robustness and representation."""
from pathlib import Path
from dataclasses import replace
import argparse
import hashlib
import itertools
import json
import sys
import numpy as np
from scipy.signal import welch
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import numba

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'simulation/streaming/frozen_v4'))
import core


@numba.njit(cache=True)
def lif_rate(spikes, weights, tau, threshold=1.0, refractory=3):
    n, m = len(spikes), weights.shape[1]
    if m == 0:
        return 0.0
    voltage = np.zeros(m)
    last = np.full(m,-refractory-1)
    total = 0
    decay = np.exp(-1.0/tau)
    for t in range(n):
        for j in range(m):
            voltage[j] *= decay
            if spikes[t] != 0:
                voltage[j] += weights[0 if spikes[t]>0 else 1,j]
            if voltage[j]>=threshold and t-last[j]>refractory:
                total += 1
                voltage[j] = 0.0
                last[j] = t
    return total/(n*m)


def features(x, cfg, neurons, tau, weight_seed, kind):
    sp = core.delta_encode(x, cfg.router_theta, cfg.enc_refractory)
    weights = np.random.default_rng(weight_seed).normal(0,.3,(2,neurons))
    four = core.spike_stats(sp,cfg.n_subframes)
    stats = [four[0],lif_rate(sp,weights,tau),four[1],four[2],four[3]]
    if kind == 'spike5':
        return np.array(stats)
    if kind.startswith('rf_'):
        threshold = .2 if kind=='rf_th_hi' else .12
        c = replace(cfg,rf_th=threshold)
        rf = core.snn_features(x,c)
        if kind=='rf_analog':
            sys.path.insert(0,str(ROOT/'simulation/streaming'))
            from common_v5 import receiver_v4
            return receiver_v4.Features(c).advance(x)['analog_delta']
        if kind=='rf_compact_lo':
            return rf[-4:]
        return rf
    if kind == 'full7':
        magnitude = np.abs(np.fft.rfft(x))
        hf = np.sum(magnitude[len(magnitude)//2:]**2)/(np.sum(magnitude**2)+1e-10)
        sharp = magnitude[1:].max()/(magnitude[1:].mean()+1e-10)
    else:
        if 'welch512' in kind:
            _, power = welch(x, nperseg=min(512,len(x)))
            magnitude = np.sqrt(power)
        else:
            size = 2048 if '2048' in kind else 512
            magnitude = np.abs(np.fft.rfft(x[:size]))
        hf = np.sum(magnitude[len(magnitude)//2:]**2)/(np.sum(magnitude**2)+1e-10)
        sharp = magnitude[1:].max()/(magnitude[1:].mean()+1e-10)
    result = stats+[hf,sharp]
    if kind.endswith('_drift'):
        result.append(core.isi_drift(sp,cfg.n_subframes))
    return np.array(result)


def dataset(settings, cfg, condition, seed, train):
    prefix = 'train' if train else 'test'
    clean = settings['n_'+prefix+'_clean']
    per = settings['n_'+prefix+'_per']
    base = (1000 if train else 9500) + seed*100000
    rows = [(core.generate_frame(cfg,condition,(),(),(700 if train else 9000)+seed*100000+i)['rx'],'none') for i in range(clean)]
    for k,jammer in enumerate(core.JT):
        for j,jsr in enumerate(settings[prefix+'_jsr']):
            for i in range(per):
                sd = base+i+k*(500 if train else 300)+j*(50 if train else 20)
                rows.append((core.generate_frame(cfg,condition,(jammer,),(jsr,),sd)['rx'],jammer))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--study',choices=['neuron_count','robustness','representation'],default='neuron_count')
    ap.add_argument('--config',type=Path)
    ap.add_argument('--output',type=Path)
    ap.add_argument('--smoke',action='store_true')
    a=ap.parse_args()
    settings=json.loads((ROOT/(a.config or Path('configs/front_end_'+a.study+'.json'))).read_text())
    cfg=core.Config(n_sym_test=settings['frame_samples']//10, snr_db=settings['snr_db'],
                    enc_refractory=settings['enc_refractory'])
    cfg=replace(cfg,**{k:settings[k] for k in ('rf_bands','rf_tau','rf_f_lo_hz','rf_f_hi_hz') if k in settings})
    if a.smoke:
        settings.update(seeds=[0],conditions=['random'],theta_list=[2.0],m_list=[0,16],
                        tau_list=[8.0],w_seeds=[42],feature_sets=['full7','spike5','rf_th_lo'],
                        train_jsr=[4,12],test_jsr=[6,14],n_train_clean=2,n_train_per=1,
                        n_test_clean=1,n_test_per=1,rf_trees=10)
        cfg=replace(cfg,n_sym_test=300)
    out=ROOT/(a.output or Path('runs/front_end_'+a.study));out.mkdir(parents=True,exist_ok=True)
    lock=dict(settings=settings,smoke=a.smoke,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if (out/'protocol.json').exists() and json.loads((out/'protocol.json').read_text())!=lock:
        raise RuntimeError('Output belongs to a different protocol; select another --output.')
    (out/'protocol.json').write_text(json.dumps(lock,indent=2))
    rows=json.loads((out/'records.json').read_text()) if (out/'records.json').exists() else []
    done={r['id'] for r in rows}
    conditions=settings.get('conditions',['fixed']);taus=settings.get('tau_list',[settings.get('tau_m',8.)]);weights=settings.get('w_seeds',[42])
    for condition,seed in itertools.product(conditions,settings['seeds']):
        train=dataset(settings,cfg,condition,seed,True);test=dataset(settings,cfg,condition,seed,False)
        for theta,neurons,tau,ws,kind in itertools.product(settings['theta_list'],settings['m_list'],taus,weights,settings['feature_sets']+['conventional7']):
            uid=f'{condition}|{seed}|{theta}|{neurons}|{tau}|{ws}|{kind}'
            if uid in done:continue
            c=replace(cfg,router_theta=theta)
            fun=lambda x:core.extract_conventional_features(x,c.n_subframes) if kind=='conventional7' else features(x,c,neurons,tau,ws,kind)
            X=np.array([fun(x) for x,y in train]);y=np.array([y for x,y in train]);Z=np.array([fun(x) for x,y in test]);truth=[y for x,y in test]
            scaler=StandardScaler().fit(X);model=RandomForestClassifier(n_estimators=settings['rf_trees'],max_depth=settings['rf_depth'],random_state=settings['rf_seed'],n_jobs=1)
            model.fit(scaler.transform(X),y);pred=model.predict(scaler.transform(Z))
            labels=['none']+core.JT
            rows.append(dict(id=uid,condition=condition,seed=seed,theta=theta,neurons=neurons,tau=tau,weight_seed=ws,
                             features=kind,accuracy=float(np.mean(pred==truth)),confusion=confusion_matrix(truth,pred,labels=labels).tolist(),labels=labels,
                             train_frames=len(train),test_frames=len(test)))
            temp=out/'records.tmp';temp.write_text(json.dumps(rows,indent=2));temp.replace(out/'records.json')
            print(uid,rows[-1]['accuracy'],flush=True)
    (out/'summary.json').write_text(json.dumps(dict(smoke=a.smoke,configurations=len(rows),records=rows),indent=2))


if __name__=='__main__':
    main()
