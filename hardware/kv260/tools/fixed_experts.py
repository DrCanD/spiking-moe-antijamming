# extracted verbatim from exp5_kv260_package.py (fixed-point golden models)
import numpy as np
def spike_stats(spikes, n_subframes):
    n = len(spikes); pos = np.where(spikes == 1)[0]; neg = np.where(spikes == -1)[0]
    fr_in = (len(pos) + len(neg)) / n; all_sp = np.sort(np.concatenate([pos, neg])); isi_cv = 0.0
    if len(all_sp) > 2:
        isis = np.diff(all_sp).astype(float); isi_cv = np.std(isis) / (np.mean(isis) + 1e-10)
    sf_len = n // n_subframes; fano, sf_ratio = 0.0, 1.0
    if sf_len > 0 and len(all_sp) > 0:
        counts = np.array([np.sum((all_sp >= k * sf_len) & (all_sp < (k + 1) * sf_len)) for k in range(n_subframes)])
        mu = np.mean(counts); fano = np.var(counts) / (mu + 1e-10); sf_ratio = np.max(counts) / (np.min(counts) + 1.0)
    return fr_in, isi_cv, fano, sf_ratio

def isi_drift(spikes, n_subframes):
    n = len(spikes); sf_len = n // n_subframes; idx = np.where(spikes != 0)[0]
    if len(idx) < 4:
        return 0.0
    g = np.mean(np.diff(idx)); means = []
    for k in range(n_subframes):
        ii = idx[(idx >= k * sf_len) & (idx < (k + 1) * sf_len)]
        if len(ii) > 2:
            means.append(np.mean(np.diff(ii)))
    return float(np.std(means) / (g + 1e-10)) if len(means) > 1 else 0.0

def rf_features(rates):
    band = rates.mean(axis=0); K = len(band)
    sharp = float(np.max(band) / (np.mean(band) + 1e-10)); hf = float(np.sum(band[K // 2:]) / (np.sum(band) + 1e-10))
    tot = rates.sum(axis=1); active = tot > 0
    if active.sum() > 1:
        cen = (rates[active] * np.arange(K)[None, :]).sum(axis=1) / tot[active]
        cdrift = float(np.std(cen) / K); adrift = float(np.std(np.argmax(rates[active], axis=1)) / K)
    else:
        cdrift, adrift = 0.0, 0.0
    return sharp, hf, cdrift, adrift

def q(x, frac):
    """round-to-nearest quantisation to an integer grid with `frac` fractional bits."""
    return np.round(np.asarray(x, dtype=np.float64) * (1 << frac)).astype(np.int64)

def fx_encoder(x_q, theta_q, refr):
    """delta encoder in integers (Q6.10): identical control flow to the reference DeltaEncoder.encode.
    Emits +1/-1/0 per sample; the reference tracks `ref += theta*sign` with a refractory counter."""
    n = len(x_q); sp = np.zeros(n, dtype=np.int8); ref = np.int64(x_q[0]); last = -refr - 1
    for i in range(n):
        if i - last <= refr:
            continue
        d = x_q[i] - ref
        if d >= theta_q:
            sp[i] = 1; ref += theta_q; last = i
        elif d <= -theta_q:
            sp[i] = -1; ref -= theta_q; last = i
    return sp

def fx_spike_stats(sp, n_subframes):
    """integer sufficient statistics accumulated in hardware; the 5 spike features are finalised (divisions) at frame rate."""
    n = len(sp); idx = np.where(sp != 0)[0]; cnt = len(idx)
    isis = np.diff(idx).astype(np.int64) if cnt > 1 else np.array([], dtype=np.int64)
    sf_len = n // n_subframes
    sf_cnt = np.array([np.sum((idx >= k * sf_len) & (idx < (k + 1) * sf_len)) for k in range(n_subframes)], dtype=np.int64)
    # per-sub-frame ISI sums (for ISI drift): ISIs attributed to the sub-frame of their END spike
    sf_isi_sum = np.zeros(n_subframes, dtype=np.int64); sf_isi_cnt = np.zeros(n_subframes, dtype=np.int64)
    if cnt > 1:
        end_sf = np.minimum(idx[1:] // sf_len, n_subframes - 1)
        np.add.at(sf_isi_sum, end_sf, isis); np.add.at(sf_isi_cnt, end_sf, 1)
    stats = {'n': n, 'count': int(cnt), 'isi_sum': int(isis.sum()), 'isi_sq_sum': int((isis ** 2).sum()), 'isi_n': int(len(isis)),
             'sf_cnt': sf_cnt.tolist(), 'sf_isi_sum': sf_isi_sum.tolist(), 'sf_isi_cnt': sf_isi_cnt.tolist()}
    # finalisation (frame rate, floating point on the PS) — must equal the float feature definitions
    fin = cnt / n
    if len(isis) > 2:
        m = stats['isi_sum'] / stats['isi_n']; v = stats['isi_sq_sum'] / stats['isi_n'] - m * m
        isi_cv = np.sqrt(max(v, 0.0)) / (m + 1e-10)
    else:
        isi_cv = 0.0
    mu = sf_cnt.mean(); fano = sf_cnt.var() / (mu + 1e-10) if cnt > 0 else 0.0
    sfr = sf_cnt.max() / (sf_cnt.min() + 1.0) if cnt > 0 else 1.0
    means = [sf_isi_sum[k] / sf_isi_cnt[k] for k in range(n_subframes) if sf_isi_cnt[k] > 2]
    g = stats['isi_sum'] / max(stats['isi_n'], 1)
    drift = float(np.std(means) / (g + 1e-10)) if (cnt >= 4 and len(means) > 1) else 0.0
    return stats, {'fin': fin, 'isi': isi_cv, 'fano': fano, 'sfr': sfr, 'isi_drift': drift}

def fx_rf_bank(sp, fs, cfg, coef):
    """16 resonators, integer state Q12.20; rotation Q2.30; input s/tau as a Q.20 constant; threshold crossing counters."""
    n = len(sp); K = cfg.rf_bands; nsf = cfg.n_subframes; sf_len = n // nsf
    cr, ci, s_in_q, th_q = coef['rot_re_q'], coef['rot_im_q'], coef['s_in_q'], coef['th_q']
    zr = np.zeros(K, dtype=np.int64); zi = np.zeros(K, dtype=np.int64)
    cnt = np.zeros((nsf, K), dtype=np.int64); prev = np.zeros(K, dtype=bool)
    SH = cfg.rf_coef_frac
    for t in range(n):
        s = int(sp[t])
        # complex multiply in integers: (zr + i zi)(cr + i ci) >> SH  (round-to-nearest-even omitted: truncation toward -inf)
        nr = (zr * cr - zi * ci) >> SH
        ni = (zr * ci + zi * cr) >> SH
        zr, zi = nr + s * s_in_q, ni
        a = zi >= th_q
        cnt[min(t // sf_len, nsf - 1)] += (a & ~prev); prev = a
    return cnt

def rf_features_from_counts(cnt, cfg, fs):
    K = cfg.rf_bands; n_sub = cnt.shape[0]
    om = 2 * np.pi * np.linspace(cfg.rf_f_lo_hz, cfg.rf_f_hi_hz, K) / fs
    rates = cnt / ((cfg.n_sym_test * 10 // n_sub) * om / (2 * np.pi))[None, :]
    return rf_features(rates)

def fx_ale(x_q, n_taps, delay, mu_q, cfg, lut):
    """NLMS in integers. x: Q6.10; w: Q6.26; xx: sum of squares (Q12.20 as int64); reciprocal of (xx+eps) from a
    2^B-entry LUT over the mantissa; g = mu*err*recip in the weight format. Output e in Q6.10."""
    n = len(x_q); IF = cfg.in_frac; WF = cfg.ale_w_frac; B = cfg.ale_recip_bits
    w = np.zeros(n_taps, dtype=np.int64); e = x_q.copy().astype(np.int64)
    xx = np.int64(0)
    for i in range(delay + n_taps, n):
        win = x_q[i - delay - n_taps + 1:i - delay + 1][::-1].astype(np.int64)          # x[i-delay-k], k=0..n_taps-1
        acc = int(np.dot(w, win))                                                       # Q(6+6).(26+10)
        y_q = acc >> WF                                                                 # -> Q.10
        err = int(x_q[i]) - y_q
        e[i] = err
        xx = int(np.dot(win, win))                                                      # Q.20 power sum
        # reciprocal of (xx + eps) via normalisation: xx = m * 2^k, m in [1,2)
        v = xx + lut['eps_q']
        k = int(v).bit_length() - 1
        m_idx = ((v - (1 << k)) << B) >> k if k >= B else ((v - (1 << k)) << (B - k)) if k > 0 else 0
        recip_q = lut['table'][int(m_idx)]                                             # ~ 2^R / m, R = lut['R']
        # NLMS update in integers. Real units: dw = mu*err*x/(xx+eps). With mu = mu_q/2^16, err = err_q/2^10,
        # x = x_q/2^10, xx = v/2^20 and 1/v = recip_q*2^(-R-k):  dw_q (Q.26) = mu_q*err_q*recip_q*x_q * 2^(10-R-k).
        # Keep 16 extra fractional bits in the gain (gq16 = num * 2^(26-R-k)), then dw_q = (gq16 * x_q) >> 16.
        num = int(mu_q) * err * int(recip_q)
        sh = lut['R'] + k - 26
        gq = (num >> sh) if sh >= 0 else (num << -sh)
        w += (gq * win) >> 16
        np.clip(w, -(1 << 31), (1 << 31) - 1, out=w)
    return e.astype(np.int64)

def make_recip_lut(cfg):
    B = cfg.ale_recip_bits; R = 15
    table = [int(round((1 << R) / (1.0 + (j + 0.5) / (1 << B)))) for j in range(1 << B)]
    return {'table': table, 'R': R, 'B': B, 'eps_q': int(round(1e-6 * (1 << 20)))}

def fx_blanker_mask(x_q, sps, group, thr_sq_q):
    """energy blanker: per-group sum of squares vs (calibrated RMS threshold)^2 * win, all integers."""
    win = group * sps; ng = len(x_q) // win
    ss = np.array([int(np.dot(x_q[k * win:(k + 1) * win].astype(np.int64), x_q[k * win:(k + 1) * win].astype(np.int64))) for k in range(ng)], dtype=np.int64)
    return (ss > thr_sq_q).astype(np.int8), ss

def forest_predict(forest, feats):
    votes = np.zeros(len(forest['classes']))
    for tree in forest['trees']:
        i = 0
        while 'leaf' not in tree[i]:
            i = tree[i]['l'] if feats[tree[i]['f']] <= tree[i]['thr'] else tree[i]['r']
        v = np.array(tree[i]['leaf'], dtype=float); votes += v / max(v.sum(), 1)
    return forest['classes'][int(np.argmax(votes))]

