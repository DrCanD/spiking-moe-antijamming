#!/usr/bin/env python3
"""PS-side frame-rate finalisation (the 'PS' half of the split in Exp-5 README §1):
 - the 9 front-end-B features from the integer sufficient statistics + RF-bank counters (same formulas as
   exp5_kv260_package.py fx_spike_stats / rf_features_from_counts), the random-forest router from spec.json
   (scaler folded in), the override statistic rho = 1 - sum(e_sw^2)/sum(x^2), and the 7 conventional features of D2.
"""
import json, math
import numpy as np
from moe_ctl import R, MoE

FS = 1e6; N_SUB = 20; K = 16; F_LO, F_HI = 25e3, 475e3; IN_FRAC = 10; FFT_BINS = 513
FE_COLS = ['fin', 'isi', 'fano', 'sfr', 'isi_drift', 'rf_sharp', 'rf_hf', 'rf_drift', 'rf_adrift']


def spike_features(r, n):
    cnt = r[R['COUNT']]; isi_n = r[R['ISI_N']]; isi_sum = r[R['ISI_SUM']]; isi_sq = MoE.u64(r, R['ISI_SQ'])
    sf_cnt = np.array(r[R['SF_CNT']:R['SF_CNT'] + N_SUB], dtype=np.float64)
    sf_isi_sum = np.array(r[R['SF_ISI_SUM']:R['SF_ISI_SUM'] + N_SUB], dtype=np.float64)
    sf_isi_cnt = np.array(r[R['SF_ISI_CNT']:R['SF_ISI_CNT'] + N_SUB], dtype=np.float64)
    fin = cnt / n
    if isi_n > 2:
        m = isi_sum / isi_n; v = isi_sq / isi_n - m * m; isi_cv = math.sqrt(max(v, 0.0)) / (m + 1e-10)
    else:
        isi_cv = 0.0
    mu = sf_cnt.mean(); fano = sf_cnt.var() / (mu + 1e-10) if cnt > 0 else 0.0
    sfr = sf_cnt.max() / (sf_cnt.min() + 1.0) if cnt > 0 else 1.0
    means = [sf_isi_sum[k] / sf_isi_cnt[k] for k in range(N_SUB) if sf_isi_cnt[k] > 2]
    g = isi_sum / max(isi_n, 1)
    drift = float(np.std(means) / (g + 1e-10)) if (cnt >= 4 and len(means) > 1) else 0.0
    return [fin, isi_cv, fano, sfr, drift]


def rf_features(r, n):
    cnt = np.array(r[R['RF_CNT']:R['RF_CNT'] + N_SUB * K], dtype=np.float64).reshape(N_SUB, K)
    om = 2 * np.pi * np.linspace(F_LO, F_HI, K) / FS
    rates = cnt / ((n // N_SUB) * om / (2 * np.pi))[None, :]
    band = rates.mean(axis=0)
    sharp = float(np.max(band) / (np.mean(band) + 1e-10)); hf = float(np.sum(band[K // 2:]) / (np.sum(band) + 1e-10))
    tot = rates.sum(axis=1); active = tot > 0
    if active.sum() > 1:
        cen = (rates[active] * np.arange(K)[None, :]).sum(axis=1) / tot[active]
        cdrift = float(np.std(cen) / K); adrift = float(np.std(np.argmax(rates[active], axis=1)) / K)
    else:
        cdrift, adrift = 0.0, 0.0
    return [sharp, hf, cdrift, adrift]


def frontend_features(r, n=100000):
    return np.array(spike_features(r, n) + rf_features(r, n))


def forest_predict(forest, feats):
    votes = np.zeros(len(forest['classes']))
    for tree in forest['trees']:
        i = 0
        while 'leaf' not in tree[i]:
            i = tree[i]['l'] if feats[tree[i]['f']] <= tree[i]['thr'] else tree[i]['r']
        v = np.array(tree[i]['leaf'], dtype=float); votes += v / max(v.sum(), 1)
    return forest['classes'][int(np.argmax(votes))], votes / len(forest['trees'])


def override_rho(r):
    sx2 = MoE.u64(r, R['SUM_X2']); se2 = MoE.u64(r, R['SUM_E2_SW'])
    return 1.0 - se2 / max(sx2, 1)


def conv_features(r, n=100000):
    sc = 1 << IN_FRAC
    sx = MoE.s64(r, R['CONV_SX']); sx2 = MoE.u64(r, R['CONV_SX2']); sx3 = MoE.s64(r, R['CONV_SX3'])
    sx4 = r[R['CONV_SX4']] | (r[R['CONV_SX4'] + 1] << 32) | (r[R['CONV_SX4'] + 2] << 64)
    mu = sx / n; m2 = sx2 / n; m3 = sx3 / n; m4 = sx4 / n
    var_q = m2 - mu * mu; c4 = m4 - 4 * mu * m3 + 6 * mu * mu * m2 - 3 * mu ** 4
    rms = math.sqrt(m2) / sc; var = var_q / sc ** 2; kurt = c4 / (var_q ** 2 + 1e-10 * sc ** 4) - 3
    nseg = max(r[R['CONV_NSEG']], 1)
    P = np.array(r[R['CONV_P']:R['CONV_P'] + FFT_BINS], dtype=np.float64) / nseg / sc
    gm = math.exp(np.mean(np.log(P[1:] + 1e-10))); am = np.mean(P[1:])
    se = np.array([MoE.u64(r, R['CONV_SE'] + 2 * k) for k in range(N_SUB)], dtype=np.float64) / (n // N_SUB) / sc ** 2
    return np.array([rms, var, kurt, gm / (am + 1e-10), np.max(P[1:]) / (am + 1e-10), np.var(se) / (np.mean(se) + 1e-10), r[R['CONV_ZC']] / n])


def route(r, spec, protocol='fixed', n=100000, rho_min=0.25):
    """final-MoE routing decision on the PS from one frame's PL results (router + predictability override)."""
    feats = frontend_features(r, n)
    verdict, proba = forest_predict(spec[f'router_forest_{protocol}'], feats)
    rho = override_rho(r)
    path = verdict
    if verdict in ('none', 'broadband') and rho > rho_min:
        path = 'structured(override)'
    return {'features': dict(zip(FE_COLS, feats.tolist())), 'verdict': verdict, 'proba': proba.tolist(), 'rho': rho, 'path': path}
