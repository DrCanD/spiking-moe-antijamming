# Unmodified AST-selected source definitions. No notebook top-level execution.
import numpy as np
from dataclasses import dataclass
SOURCE_SHA256 = {'notebook': '5cbed90b346d48fdc2f6eeb495f4e91d72b8164dc4ead4d50befad64a4581723', 'exp6a_final_moe.py': '62def611cebdd90eabd46fd8445f7d07e1ef36af7fc99cb2ff65bab43d86ada9'}

JT = ['broadband', 'narrowband', 'sweep', 'pulse']


@dataclass
class ChannelConfig:
    fs: float = 1e6
    symbol_rate: float = 1e5
    snr_db: float = 10.0
    jsr_db: float = 10.0
    n_symbols: int = 10000
    seed: int = 42


def generate_bpsk(cfg, rng=None):
    if rng is None: rng = np.random.default_rng(cfg.seed)
    bits = rng.integers(0, 2, size=cfg.n_symbols)
    symbols = 2 * bits - 1
    sps = int(cfg.fs / cfg.symbol_rate)
    signal = np.repeat(symbols, sps).astype(np.float64)
    signal /= np.sqrt(np.mean(signal**2))
    return bits, symbols, signal, sps


def add_awgn(signal, snr_db, rng):
    sp = np.mean(signal**2)
    np_ = sp / 10**(snr_db/10)
    return signal + rng.normal(0, np.sqrt(np_), len(signal))


def add_broadband_jam(signal, jsr_db, rng):
    sp = np.mean(signal**2)
    jp = sp * 10**(jsr_db/10)
    jam = rng.normal(0, np.sqrt(jp), len(signal))
    return signal + jam, jam


def add_narrowband_jam(signal, jsr_db, fs, rng, f_ratio=0.3):
    sp = np.mean(signal**2)
    jp = sp * 10**(jsr_db/10)
    f = f_ratio * fs / 2
    t = np.arange(len(signal)) / fs
    jam = np.sqrt(2*jp) * np.sin(2*np.pi*f*t + rng.uniform(0, 2*np.pi))
    return signal + jam, jam


def add_sweep_jam(signal, jsr_db, fs, rng):
    sp = np.mean(signal**2)
    jp = sp * 10**(jsr_db/10)
    t = np.arange(len(signal)) / fs
    f0, f1 = 0.05*fs/2, 0.95*fs/2
    phase = 2*np.pi*(f0*t + 0.5*(f1-f0)*t**2/t[-1])
    jam = np.sqrt(2*jp) * np.sin(phase)
    return signal + jam, jam


def add_pulse_jam(signal, jsr_db, rng, duty=0.2, period=500):
    sp = np.mean(signal**2)
    jp = sp * 10**(jsr_db/10) / duty
    n = len(signal)
    env = np.zeros(n)
    on = int(period * duty)
    for i in range(0, n, period):
        env[i:min(i+on, n)] = 1.0
    jam = env * rng.normal(0, np.sqrt(jp), n)
    return signal + jam, jam


JAM_FNS = {
    'broadband': lambda s, j, fs, rng: add_broadband_jam(s, j, rng),
    'narrowband': lambda s, j, fs, rng: add_narrowband_jam(s, j, fs, rng),
    'sweep': lambda s, j, fs, rng: add_sweep_jam(s, j, fs, rng),
    'pulse': lambda s, j, fs, rng: add_pulse_jam(s, j, rng),
}


def simulate_channel(cfg, jam_type='broadband'):
    rng = np.random.default_rng(cfg.seed)
    bits, symbols, tx, sps = generate_bpsk(cfg, rng)
    noisy = add_awgn(tx, cfg.snr_db, rng)
    if jam_type == 'none':
        return {'bits': bits, 'symbols': symbols, 'tx': tx, 'rx': noisy,
                'jam': np.zeros_like(tx), 'sps': sps, 'cfg': cfg}
    rx, jam = JAM_FNS[jam_type](noisy, cfg.jsr_db, cfg.fs, rng)
    return {'bits': bits, 'symbols': symbols, 'tx': tx, 'rx': rx,
            'jam': jam, 'sps': sps, 'cfg': cfg}


def demodulate_bpsk(rx, sps):
    ns = len(rx) // sps
    dec = np.mean(rx[:ns*sps].reshape(ns, sps), axis=1)
    return (dec > 0).astype(int), dec


def compute_ber(tx_bits, rx_bits):
    n = min(len(tx_bits), len(rx_bits))
    return np.sum(tx_bits[:n] != rx_bits[:n]) / n


@dataclass
class SpikeEncoderConfig:
    threshold: float = 0.04
    refractory: int = 2


class DeltaEncoder:
    def __init__(self, cfg=None):
        self.cfg = cfg or SpikeEncoderConfig()

    def encode(self, signal):
        n = len(signal)
        spikes = np.zeros(n, dtype=np.int8)
        ref = signal[0]
        last = -self.cfg.refractory - 1
        th = self.cfg.threshold
        refr = self.cfg.refractory
        for i in range(1, n):
            if i - last <= refr: continue
            d = signal[i] - ref
            if d > th:
                spikes[i] = 1; ref += th; last = i
            elif d < -th:
                spikes[i] = -1; ref -= th; last = i
        pos = np.where(spikes == 1)[0]
        neg = np.where(spikes == -1)[0]
        fr = (len(pos) + len(neg)) / n
        return spikes, pos, neg, fr


class PulseBlankerExpert:
    # Pulse blanker using per-group signal ENERGY for temporal localization.
    #
    # Why not spike rate? The delta encoder with threshold=0.04 saturates at
    # the refractory limit (~0.33) for both clean and jammed signals, because
    # even AWGN fluctuations exceed the threshold every sample. Spike rate
    # cannot differentiate ON vs OFF periods.
    #
    # Signal energy works because pulse jamming adds massive power during ON
    # (JSR=12 dB, duty=0.2 -> ON power ~80x baseline). RMS per symbol-group
    # is a natural, robust discriminator.
    #
    # The SNN still handles detection + classification. The pulse blanker
    # uses energy only for fine-grained temporal localization of ON regions.

    def __init__(self, blank_threshold_z=3.0, group_size=5):
        self.blank_z = blank_threshold_z
        self.group_size = group_size  # symbols per analysis window
        self.baseline_rms_mean = None
        self.baseline_rms_std = None

    def _compute_group_rms(self, signal, sps):
        # RMS energy per symbol-group
        win = self.group_size * sps
        n_groups = len(signal) // win
        rms = np.zeros(n_groups)
        for k in range(n_groups):
            seg = signal[k*win:(k+1)*win]
            rms[k] = np.sqrt(np.mean(seg**2))
        return rms

    def calibrate(self, clean_signals, sps):
        # Learn baseline RMS from clean signals
        all_rms = []
        for sig in clean_signals:
            rms = self._compute_group_rms(sig, sps)
            all_rms.append(rms)
        all_rms = np.concatenate(all_rms)
        self.baseline_rms_mean = np.mean(all_rms)
        self.baseline_rms_std = max(np.std(all_rms), 1e-6)
        print(f"  Pulse blanker calibrated: RMS baseline = "
              f"{self.baseline_rms_mean:.4f} +/- {self.baseline_rms_std:.4f}")

    def correct_symbols(self, jammed_signal, sps):
        # Detect high-energy groups (pulse ON) and erase those symbols
        n_total = len(jammed_signal)
        ns = n_total // sps
        soft = np.mean(jammed_signal[:ns*sps].reshape(ns, sps), axis=1)

        rms = self._compute_group_rms(jammed_signal, sps)
        z = (rms - self.baseline_rms_mean) / self.baseline_rms_std
        grp_mask = z > self.blank_z  # only high energy = jammed

        # map to symbol erasure mask
        erasure_mask = np.zeros(ns, dtype=bool)
        for k, is_j in enumerate(grp_mask):
            if is_j:
                s0 = k * self.group_size
                s1 = min((k+1) * self.group_size, ns)
                erasure_mask[s0:s1] = True

        surviving_idx = np.where(~erasure_mask)[0]
        surviving_bits = (soft[surviving_idx] > 0).astype(int)
        survival_rate = len(surviving_idx) / ns if ns > 0 else 0

        return {
            'soft': soft,
            'erasure_mask': erasure_mask,
            'surviving_bits': surviving_bits,
            'surviving_indices': surviving_idx,
            'survival_rate': survival_rate,
            'rms_values': rms,
            'grp_mask': grp_mask,
        }

    def power_mw(self):
        return 0.001


def extract_conventional_features(signal, n_subframes=20):
    n = len(signal)
    rms = np.sqrt(np.mean(signal**2))
    var = np.var(signal)
    mu = np.mean(signal)
    kurt = np.mean((signal - mu)**4) / (var**2 + 1e-10) - 3
    ft = np.abs(np.fft.rfft(signal))
    geo_mean = np.exp(np.mean(np.log(ft[1:] + 1e-10)))
    arith_mean = np.mean(ft[1:])
    flatness = geo_mean / (arith_mean + 1e-10)
    peak_ratio = np.max(ft[1:]) / (arith_mean + 1e-10)
    sf_len = n // n_subframes
    sf_energy = [np.mean(signal[k*sf_len:(k+1)*sf_len]**2) for k in range(n_subframes)]
    sf_var = np.var(sf_energy) / (np.mean(sf_energy) + 1e-10)
    zc = np.sum(np.diff(np.sign(signal)) != 0) / n
    return np.array([rms, var, kurt, flatness, peak_ratio, sf_var, zc])


def add_sweep_jam_rand(signal, jsr_db, fs, rng, cfg):
    sp = np.mean(signal ** 2); jp = sp * 10 ** (jsr_db / 10)
    n = len(signal); t = np.arange(n) / fs
    fa, fb = rng.uniform(cfg.sweep_f_range[0], cfg.sweep_f_range[1], 2) * fs / 2
    dur = rng.uniform(cfg.sweep_dur_range[0], cfg.sweep_dur_range[1]) * (t[-1] + 1 / fs)
    t0 = rng.uniform(0, 2 * dur)
    x = np.mod(t + t0, 2 * dur); frac = np.where(x < dur, x / dur, 2 - x / dur)
    f_inst = fa + (fb - fa) * frac
    phase = 2 * np.pi * np.cumsum(f_inst) / fs + rng.uniform(0, 2 * np.pi)
    jam = np.sqrt(2 * jp) * np.sin(phase)
    return signal + jam, jam


def add_pulse_jam_rand(signal, jsr_db, rng, cfg):
    sp = np.mean(signal ** 2)
    duty = rng.uniform(*cfg.pulse_duty_range)
    period = int(rng.integers(cfg.pulse_period_range[0], cfg.pulse_period_range[1] + 1))
    jp = sp * 10 ** (jsr_db / 10) / duty
    n = len(signal); on = max(1, int(period * duty)); off0 = int(rng.integers(0, period))
    env = np.zeros(n)
    for i in range(off0 - period, n, period):
        a, b = max(i, 0), min(i + on, n)
        if b > a:
            env[a:b] = 1.0
    jam = env * rng.normal(0, np.sqrt(jp), n)
    return signal + jam, jam


def simulate_channel_rand(R, cc, jam_type, cfg):
    rng = np.random.default_rng(cc.seed)
    bits, symbols, tx, sps = R['generate_bpsk'](cc, rng)
    noisy = R['add_awgn'](tx, cc.snr_db, rng)
    if jam_type == 'none':
        rx = noisy
    elif jam_type == 'broadband':
        rx, _ = R['add_broadband_jam'](noisy, cc.jsr_db, rng)
    elif jam_type == 'narrowband':
        rx, _ = R['add_narrowband_jam'](noisy, cc.jsr_db, cc.fs, rng, f_ratio=rng.uniform(*cfg.nb_f_range))
    elif jam_type == 'sweep':
        rx, _ = add_sweep_jam_rand(noisy, cc.jsr_db, cc.fs, rng, cfg)
    elif jam_type == 'pulse':
        rx, _ = add_pulse_jam_rand(noisy, cc.jsr_db, rng, cfg)
    else:
        raise ValueError(jam_type)
    return {'bits': bits, 'tx': tx, 'rx': rx, 'sps': sps, 'cfg': cc}


def make_sim(R, cfg, cond):
    return R['simulate_channel'] if cond == 'fixed' else (lambda cc, jam_type: simulate_channel_rand(R, cc, jam_type, cfg))


def _ale_py(signal, n_taps, delay, mu, eps):
    n = len(signal); w = np.zeros(n_taps); e = signal.copy()
    for i in range(delay + n_taps, n):
        acc = 0.0; xx = 0.0
        for k in range(n_taps):
            xk = signal[i - delay - k]
            acc += w[k] * xk; xx += xk * xk
        err = signal[i] - acc
        e[i] = err
        g = mu * err / (xx + eps)
        for k in range(n_taps):
            w[k] += g * signal[i - delay - k]
    return e


def soft_symbols(x, sps):
    n = len(x) // sps
    return np.mean(x[:n * sps].reshape(n, sps), axis=1)


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


def rf_bank_batched(spikes_signed, fs, cfg):
    F, T = spikes_signed.shape; K = cfg.rf_bands; nsf = cfg.n_subframes
    om = 2 * np.pi * np.linspace(cfg.rf_f_lo_hz, cfg.rf_f_hi_hz, K) / fs
    rot = (1.0 - 1.0 / cfg.rf_tau) * np.exp(1j * om)
    z = np.zeros((F, K), dtype=np.complex128); sf_len = T // nsf
    cnt = np.zeros((F, nsf, K)); prev = np.zeros((F, K), dtype=bool)
    s_in = spikes_signed.astype(np.float64) / cfg.rf_tau
    for t in range(T):
        z = z * rot + s_in[:, t][:, None]
        a = z.imag >= cfg.rf_th
        cnt[:, min(t // sf_len, nsf - 1), :] += a & ~prev; prev = a
    return cnt / (sf_len * om / (2 * np.pi))[None, None, :]


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


def frontend_B(R, cfg, signals, fs):
    enc = R['DeltaEncoder'](R['SpikeEncoderConfig'](threshold=cfg.router_theta, refractory=cfg.enc_refractory))
    T = len(signals[0]); spikes = np.zeros((len(signals), T), dtype=np.int8); rows = []
    for i, x in enumerate(signals):
        sp, _, _, _ = enc.encode(x); spikes[i] = sp
        fin, isi, fano, sfr = spike_stats(sp, cfg.n_subframes)
        rows.append([fin, isi, fano, sfr, isi_drift(sp, cfg.n_subframes)])
    rates = rf_bank_batched(spikes, fs, cfg)
    return np.array([rows[i] + list(rf_features(rates[i])) for i in range(len(signals))])


class RouterB:
    def __init__(self, R, cfg, cond, fs, sps):
        CC = R['ChannelConfig']; JT = R['JT']; sim = make_sim(R, cfg, cond); nsym = cfg.n_sym_test; off = cfg.router_seed_base
        sigs, ys = [], []
        for i in range(cfg.router_n_train_clean):
            sigs.append(sim(CC(snr_db=cfg.snr_db, n_symbols=nsym, seed=off + i), jam_type='none')['rx']); ys.append('none')
        for jt in JT:
            for jsr in cfg.router_train_jsr:
                for i in range(cfg.router_n_train_per):
                    seed = off + 1000 + i + JT.index(jt) * 700 + cfg.router_train_jsr.index(jsr) * 50
                    sigs.append(sim(CC(snr_db=cfg.snr_db, jsr_db=jsr, n_symbols=nsym, seed=seed), jam_type=jt)['rx']); ys.append(jt)
        X = frontend_B(R, cfg, sigs, fs); self.classes = ['none'] + list(JT)
        self.sc = StandardScaler().fit(X)
        self.clf = RandomForestClassifier(n_estimators=cfg.rf_trees, max_depth=cfg.rf_depth, random_state=cfg.rf_seed, n_jobs=cfg.n_jobs)
        self.clf.fit(self.sc.transform(X), ys)
        self.n_train = len(ys); self.train_acc = float(np.mean(self.clf.predict(self.sc.transform(X)) == np.array(ys)))
    def predict(self, X):
        P = self.clf.predict_proba(self.sc.transform(X)); cls = self.clf.classes_
        idx = np.argmax(P, axis=1)
        return cls[idx], P[np.arange(len(idx)), idx]


def jam_component(R, cfg, cond, noisy, jt, jsr, fs, rng):
    if jt == 'broadband':
        return R['add_broadband_jam'](noisy, jsr, rng)[1]
    if jt == 'narrowband':
        return R['add_narrowband_jam'](noisy, jsr, fs, rng, f_ratio=(rng.uniform(*cfg.nb_f_range) if cond == 'random' else 0.3))[1]
    if jt == 'sweep':
        return (add_sweep_jam_rand(noisy, jsr, fs, rng, cfg) if cond == 'random' else R['add_sweep_jam'](noisy, jsr, fs, rng))[1]
    if jt == 'pulse':
        return (add_pulse_jam_rand(noisy, jsr, rng, cfg) if cond == 'random' else R['add_pulse_jam'](noisy, jsr, rng))[1]
    raise ValueError(jt)


def compound_frame(R, cfg, cond, comp, jsr, seed):
    cc = R['ChannelConfig'](snr_db=cfg.snr_db, jsr_db=jsr, n_symbols=cfg.n_sym_test, seed=seed)
    rng = np.random.default_rng(seed)
    bits, symbols, tx, sps = R['generate_bpsk'](cc, rng)
    noisy = R['add_awgn'](tx, cc.snr_db, rng)
    j1 = jam_component(R, cfg, cond, noisy, comp[0], jsr, cc.fs, rng)
    j2 = jam_component(R, cfg, cond, noisy, comp[1], jsr, cc.fs, rng)
    return {'bits': bits, 'rx': noisy + j1 + j2, 'rx_only1': noisy + j1, 'rx_only2': noisy + j2, 'sps': sps}


from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

def ber_of(bits, soft, idx=None):
    b = (soft > 0).astype(int)
    return R['compute_ber'](bits if idx is None else bits[idx], b if idx is None else b[idx]) if len(b) else 0.5


def blank_calibrated(x):
    pr = blanker.correct_symbols(x, sps)
    ns_ = len(x) // sps
    return pr['soft'], pr['surviving_indices'], (float(np.mean(pr['grp_mask'])) if len(pr['grp_mask']) else 0.0), np.repeat(~pr['erasure_mask'][:ns_], sps)


def pulse_check(y):
    ns_ = len(y) // sps; soft = np.mean(y[:ns_ * sps].reshape(ns_, sps), axis=1)
    win = cfg.blanker_group * sps; ng = len(y) // win
    rms = np.sqrt(np.mean(y[:ng * win].reshape(ng, win) ** 2, axis=1))
    flag = rms > cfg.pulse_k * np.median(rms); frac = float(np.mean(flag)) if ng else 0.0
    er = np.zeros(ns_, dtype=bool)
    for g in np.where(flag)[0]:
        er[g * cfg.blanker_group:min((g + 1) * cfg.blanker_group, ns_)] = True
    return soft, np.where(~er)[0], frac


def final_moe(rx, bits, cls1, cond, use_override=True, use_cascade=True):
    """Returns (ber, survival, path). cls1 = router verdict on the full frame."""
    cls = cls1; path = cls1[:2]
    if use_override and cls1 in ('none', 'broadband'):
        e = apply_ale(rx, cfg, cfg.ale_mu_sw)
        rho = 1.0 - np.mean(e ** 2) / (np.mean(rx ** 2) + 1e-12)
        if rho > cfg.rho_min:
            cls = 'sweep'; path += f'->override(rho={rho:.2f})'
    if cls in ('narrowband', 'sweep'):
        y = apply_ale(rx, cfg, mu_of[cls]); path += '->ALE'
        if use_cascade:
            soft, surv_idx, frac = pulse_check(y)
            if cfg.blank_min_fraction < frac < cfg.blank_max_fraction:
                return ber_of(bits[surv_idx], soft[surv_idx]), len(surv_idx) / len(soft), path + '->blank'
        return ber_of(bits, soft_symbols(y, sps)), 1.0, path
    if cls == 'pulse':
        soft, surv_idx, frac, smask = blank_calibrated(rx); path += '->blank'
        if not use_cascade or len(surv_idx) < 200:
            return ber_of(bits[surv_idx], soft[surv_idx]) if len(surv_idx) else 0.5, len(surv_idx) / len(soft), path
        surv = rx[:len(smask)][smask]
        cls2 = routers[cond].predict(frontend_B(R, cfg, [surv], fs))[0][0]
        if cls2 in ('narrowband', 'sweep'):
            y = soft_symbols(apply_ale(surv, cfg, mu_of[cls2]), sps)
            return ber_of(bits[surv_idx], y[:len(surv_idx)]), len(surv_idx) / len(soft), path + f'->{cls2[:2]}->ALE'
        return ber_of(bits[surv_idx], soft[surv_idx]), len(surv_idx) / len(soft), path
    return ber_of(bits, soft_symbols(rx, sps)), 1.0, path + '->pass'


