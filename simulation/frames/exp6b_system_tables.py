# ============================================================================
#  SNN-MoE Anti-Jamming — TCCN R1 revision
#  EXP 6b/6: SYSTEM-LEVEL TABLES WITH THE FINAL MoE (frozen in Exp 6a)
#    A  dynamic jammer switching (Table IX protocol: 100-frame engagement, 10 trials): no correction, fixed notch,
#       fixed blanker, blind ALE, MoE with CONVENTIONAL-feature router (same experts), final MoE, oracle
#    B  pulse goodput vs JSR (Table V) and variable duty cycle (Table VI) with the calibrated pulse expert
#    C  Rayleigh block fading (Tables VII/VIII protocol): routing accuracy and BER, AWGN vs fading, no retraining
#    D  SNR robustness: router trained at 10 dB, tested at 4..16 dB, JSR 10
#    E  RS coded link (Table X protocol): BLER for RS(255,223/191/159), no-correction vs blanker vs erasure-aware
#    every unit under BOTH jammer-parameter protocols; reference protocols/seeds kept where they exist
#    frame     : 10,000 symbols = 100,000 samples (router and experts see the SAME frame)
#    primitives per frame: no-correction, fixed notch (fixed protocol only), classical blanker,
#              ALE-32 (NLMS, numba), paper LSTM, LSTM 0-20 (Exp 2 'fixed020'), monolithic LSTM
#              (one LSTM on all four jammer types), calibrated pulse-blanker expert (erasures),
#              front-end B router prediction + max-probability, energy-based JSR estimate
#    methods   : single-method baselines, oracle routing (true type), JSR-gated structured expert
#              (LSTM below tau_J, ALE above), MoE with front-end B routing, MoE + confidence
#              pass-through; clean frames audited for false actions
#    protocol  : Table II seeds (11000 + trial + type*2000), JSR 0..20 step 2, 100 trials,
#              'fixed' and 'random' jammer parameters
#    gate      : fixed, JSR 10, first 20 trials: no-corr NB 0.0433 / SW 0.0500 / BB 0.1718 / PL 0.0675,
#              paper-LSTM SW 0.0135, classical blanker PL 0.0674 (all_results.txt Table II)
#  Paste into ONE Colab cell and run. Re-paste to resume. GPU (T4 is enough).
# ============================================================================
import os, sys, json, time, ast, math, random, hashlib, subprocess, shutil
from pathlib import Path
from dataclasses import dataclass, field, asdict
from datetime import datetime

import numpy as np

try:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=False)
    ON_COLAB = True
except ImportError:
    ON_COLAB = False

MYDRIVE  = Path('/content/drive/MyDrive') if ON_COLAB else Path('.')
RESEARCH = MYDRIVE / 'Research' if ON_COLAB else Path('.') / 'Research'
RESEARCH.mkdir(parents=True, exist_ok=True)
PROJECT  = RESEARCH / 'SNN_MoE_AntiJamming_R1'
PROJECT.mkdir(parents=True, exist_ok=True)
REF_PROJECT = MYDRIVE / 'SNN_MoE_AntiJamming'

def _pip(pkg):
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', pkg], capture_output=True)
try:
    from tqdm.auto import tqdm
except ImportError:
    _pip('tqdm'); from tqdm.auto import tqdm
try:
    import sklearn, scipy  # noqa
except ImportError:
    _pip('scikit-learn'); _pip('scipy')
try:
    from reedsolo import RSCodec, ReedSolomonError
except ImportError:
    _pip('reedsolo'); from reedsolo import RSCodec, ReedSolomonError
try:
    import numba
    NUMBA_OK = True
except ImportError:
    _pip('numba')
    try:
        import numba; NUMBA_OK = True
    except ImportError:
        NUMBA_OK = False
if ON_COLAB and not Path('/usr/share/fonts/truetype/liberation').exists():
    subprocess.run(['apt-get', 'install', '-y', '-qq', 'fonts-liberation'], capture_output=True)
    try:
        import matplotlib.font_manager as _fm; _fm._load_fontmanager(try_read_cache=False)
    except Exception:
        pass

import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import torch

# ============================================================================
#  CONFIG
# ============================================================================
@dataclass
class Config:
    exp_index: int = 6
    exp_total: int = 6
    exp_name: str = "System-level tables with the final MoE: switching, goodput/duty, fading, SNR, coded link (Exp 6b)"
    exp_manifest: list = field(default_factory=lambda: [
        (1, "Front-end model settling -> locked: theta=2, no LIF, A=Welch-512+drift / B=RF bank", "done"),
        (2, "Expert x JSR diagnosis: LSTM 0-20 fixes the range; ALE wins above ~10 dB, harms below (R3-4)", "done"),
        (3, "MoE-level BER: ALE-128 type-matched mu beats LSTM everywhere; MoE == oracle (R3-3)", "done"),
        (4, "Compound jamming: cascade == oracle; NB+SW exposed the broadband confusion -> override (R3-5)", "done"),
        (5, "KV260 energy measurement of the final MoE (R3-2)", "pending"),
        (6, "6a done (final MoE == oracle); 6b: switching, goodput/duty, fading, SNR, coded link (THIS)", "THIS"),
    ])
    smoke: bool = False
    # ── protocol ──
    conditions: list = field(default_factory=lambda: ['fixed', 'random'])
    rho_min: float = 0.25                   # override: fraction of received power the ALE removes on a 'none'/'broadband' verdict
    # A: switching (reference scenario and seeds)
    switch_trials: int = 10
    switch_seed_base: int = 30000
    # B: goodput / duty
    gp_jsr_list: list = field(default_factory=lambda: [6, 8, 10, 12, 14, 16])
    gp_trials: int = 20
    gp_seed_base: int = 15000
    duty_list: list = field(default_factory=lambda: [0.10, 0.15, 0.20, 0.30, 0.40])
    duty_trials: int = 20
    duty_seed_base: int = 12000
    # C: fading
    fade_trials: int = 20
    fade_block: int = 100
    fade_cls_seed_base: int = 13000
    fade_ber_seed_base: int = 14000
    # D: SNR robustness
    snr_list: list = field(default_factory=lambda: [4, 7, 10, 13, 16])
    snr_trials: int = 30
    snr_seed_base: int = 16000
    # E: coded link
    rs_codes: list = field(default_factory=lambda: [(32, 'RS(255,223)'), (64, 'RS(255,191)'), (96, 'RS(255,159)'), (128, 'RS(255,127)')])
    rs_jsr_list: list = field(default_factory=lambda: [6, 8, 10, 12, 14, 16])
    rs_trials: int = 50
    rs_codewords: int = 20
    rs_seed_base: int = 40000
    blank_min_fraction: float = 0.01        # cascade: apply erasure only if the flagged fraction is in (min, max)
    blank_max_fraction: float = 0.6
    pulse_k: float = 2.0                    # within-frame pulse test: group RMS > k x median group RMS
    n_sym_test: int = 10000
    snr_db: float = 10.0
    seed_base: int = 11000                  # Table II protocol
    clean_seed_base: int = 30000            # new
    # ── experts ──
    lstm_hidden: int = 24
    lstm_seq_len: int = 200
    lstm_lr: float = 0.002
    lstm_epochs: int = 200
    lstm_batch: int = 128
    train_jsr_list: list = field(default_factory=lambda: [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
    train_n_per_jsr: int = 8                # structured LSTM (as Exp 2 'fixed020')
    mono_n_per_jsr: int = 4                 # monolithic LSTM: 4 types x 11 JSR x 4 = 176 pairs (same budget)
    train_seed_base: int = 40000
    mono_seed_base: int = 50000
    train_n_sym: int = 10000
    paper_train_jsr: list = field(default_factory=lambda: [4, 8, 12, 16])
    paper_train_n_per_jsr: int = 10
    paper_train_seed_base: int = 4000
    ale_taps: int = 128
    ale_delay: int = 20
    ale_mu_nb: float = 0.02                 # tones: small step -> low misadjustment (the 3a floor was misadjustment)
    ale_mu_sw: float = 0.05                 # sweeps: faster tracking
    ale_mu_only: float = 0.02               # router-free 'ale_only' baseline
    gate_thresholds: list = field(default_factory=lambda: [4.0, 6.0, 8.0, 10.0])   # JSR_est (dB) above which ALE is used
    gate_default: float = 6.0
    amb_frac: float = 0.3                   # quality gate: fraction of |soft| < amb_frac * rms(soft)
    conf_threshold: float = 0.5             # router max-probability below which the MoE passes through
    blanker_z: float = 3.0
    blanker_group: int = 5
    blanker_cal_seed_base: int = 600        # reference Phase 3: 30 clean frames of 10,000 samples
    blanker_cal_n: int = 30
    blanker_cal_samples: int = 10000
    # ── router (front-end B, full-frame) ──
    router_train_jsr: list = field(default_factory=lambda: [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
    router_n_train_clean: int = 50
    router_n_train_per: int = 5             # per (type, JSR) -> 50 + 4*11*5 = 270 frames of 100k samples
    router_seed_base: int = 60000
    router_theta: float = 2.0
    enc_refractory: int = 2
    rf_bands: int = 16
    rf_f_lo_hz: float = 25e3
    rf_f_hi_hz: float = 475e3
    rf_tau: float = 15.0
    rf_th: float = 0.12
    n_subframes: int = 20
    rf_trees: int = 100
    rf_depth: int = 10
    rf_seed: int = 42
    nb_f_range: tuple = (0.05, 0.95)
    sweep_f_range: tuple = (0.05, 0.95)
    sweep_dur_range: tuple = (0.5, 2.0)
    pulse_duty_range: tuple = (0.1, 0.4)
    pulse_period_range: tuple = (300, 800)
    # ── machine / cosmetic ──
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_jobs: int = -1
    verbose: bool = True
    recommended_runtime: str = 'GPU (T4 is enough); ~25-35 min (router features on 100k-sample frames dominate)'

cfg = Config()
if cfg.smoke:
    cfg.switch_trials = 1
    cfg.gp_jsr_list, cfg.gp_trials, cfg.duty_list, cfg.duty_trials = [10], 2, [0.2], 2
    cfg.fade_trials = 2
    cfg.snr_list, cfg.snr_trials = [4, 10], 2
    cfg.rs_codes, cfg.rs_jsr_list, cfg.rs_trials, cfg.rs_codewords = [(32, 'RS(255,223)')], [10], 2, 2
    cfg.n_sym_test = 2000
    cfg.lstm_epochs = 2
    cfg.train_jsr_list, cfg.train_n_per_jsr, cfg.mono_n_per_jsr, cfg.train_n_sym = [0, 10], 1, 1, 2000
    cfg.paper_train_jsr, cfg.paper_train_n_per_jsr = [4], 1
    cfg.router_train_jsr, cfg.router_n_train_clean, cfg.router_n_train_per = [0, 10], 4, 2
    cfg.blanker_cal_n, cfg.blanker_cal_samples = 3, 2000
    cfg.rf_bands = 8

_KEY_EXCLUDE = ('device', 'n_jobs', 'num_workers', 'backend', 'recommended_runtime', 'smoke', 'verbose',
                'exp_index', 'exp_total', 'exp_name', 'exp_manifest')
RUN_VERSION = 1
ANALYSIS_VERSION = 1

def run_key(cfg, data_fingerprint, ref_hash=None):
    payload = {k: v for k, v in asdict(cfg).items() if k not in _KEY_EXCLUDE}
    payload['run_version'] = RUN_VERSION; payload['data_fp'] = data_fingerprint
    if ref_hash:
        payload['ref_hash'] = ref_hash
    return hashlib.md5(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:8]

def write_json_atomic(obj, path):
    path = Path(path); tmp = path.with_suffix(path.suffix + '.tmp')
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=2, default=float); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)

def set_all_seeds(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

# ============================================================================
#  REFERENCE IMPORT
# ============================================================================
REF_NAMES = ['ChannelConfig', 'generate_bpsk', 'add_awgn', 'add_broadband_jam', 'add_narrowband_jam',
             'add_sweep_jam', 'add_pulse_jam', 'simulate_channel', 'demodulate_bpsk', 'compute_ber',
             'SpikeEncoderConfig', 'WatchdogConfig', 'DeltaEncoder', 'ResidualLSTMNet', 'LSTMExpert',
             'PulseBlankerExpert', 'apply_notch', 'apply_classical_blanker', 'extract_conventional_features',
             'simulate_rayleigh', 'add_pulse_jam_variable']
REF_ASSIGNS = ['JAM_FNS', 'JT', 'JT_ALL', 'JL', 'device']

def find_reference_notebook():
    for c in [MYDRIVE / 'Colab_Notebooks' / 'SNN_MoE_COMPLETE.ipynb', MYDRIVE / 'Colab Notebooks' / 'SNN_MoE_COMPLETE.ipynb',
              PROJECT / 'SNN_MoE_COMPLETE.ipynb']:
        if c.exists():
            return c
    hits = list(MYDRIVE.glob('*/SNN_MoE_COMPLETE.ipynb')) + list(MYDRIVE.glob('*/*/SNN_MoE_COMPLETE.ipynb'))
    if hits:
        return hits[0]
    raise FileNotFoundError("SNN_MoE_COMPLETE.ipynb not found under MyDrive")

def load_reference(nb_path):
    nb = json.load(open(nb_path)); src = ''
    for c in [c for c in nb['cells'] if c.get('cell_type') == 'code']:
        s = c['source']; s = ''.join(s) if isinstance(s, list) else s
        src += s + ('' if s.endswith('\n') else '\n')
    lines = src.splitlines(keepends=True); tree = ast.parse(src)
    def seg(node):
        start = node.lineno
        if getattr(node, 'decorator_list', None):
            start = min([start] + [d.lineno for d in node.decorator_list])
        return ''.join(lines[start - 1:node.end_lineno])
    imports, picked, found = [], [], set()
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(seg(node))
        elif isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in REF_NAMES:
            picked.append(seg(node)); found.add(node.name)
        elif isinstance(node, ast.Assign):
            tg = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if tg and tg[0] in REF_ASSIGNS:
                picked.append(seg(node)); found.add(tg[0])
    missing = (set(REF_NAMES) | set(REF_ASSIGNS)) - found
    if missing:
        raise RuntimeError(f"reference notebook lacks definitions: {sorted(missing)}")
    ns, skipped = {}, []
    for imp in imports:
        try:
            exec(imp, ns)
        except Exception as e:
            skipped.append(f"{imp.strip()}  ({e.__class__.__name__})")
    code = ''.join(picked); exec(code, ns)
    return ns, hashlib.sha256(code.encode()).hexdigest()[:16], skipped

# ============================================================================
#  RANDOMISED JAMMERS (Exp 1b/1c/2)
# ============================================================================
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

# ============================================================================
#  ALE (NLMS adaptive line enhancer) — numba if available, python fallback, equality-checked
# ============================================================================
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
_ale_fast = numba.njit(cache=False)(_ale_py) if NUMBA_OK else _ale_py

def apply_ale(signal, cfg, mu):
    return _ale_fast(np.ascontiguousarray(signal, dtype=np.float64), cfg.ale_taps, cfg.ale_delay, mu, 1e-6)

def soft_symbols(x, sps):
    n = len(x) // sps
    return np.mean(x[:n * sps].reshape(n, sps), axis=1)

def ambiguity(s, frac):
    r = np.sqrt(np.mean(s ** 2)) + 1e-12
    return float(np.mean(np.abs(s) < frac * r))

# ============================================================================
#  EXPERTS
# ============================================================================
def training_pairs(R, cfg, cond, types, jsr_list, n_per_jsr, seed_base, n_sym):
    CC = R['ChannelConfig']; sim = make_sim(R, cfg, cond); jammed, clean = [], []
    for jt in types:
        for jsr in jsr_list:
            for i in range(n_per_jsr):
                seed = seed_base + i + types.index(jt) * 500 + jsr_list.index(jsr) * 50
                rj = sim(CC(snr_db=cfg.snr_db, jsr_db=jsr, n_symbols=n_sym, seed=seed), jam_type=jt)
                rc = R['simulate_channel'](CC(snr_db=cfg.snr_db, n_symbols=n_sym, seed=seed), jam_type='none')
                jammed.append(rj['rx']); clean.append(rc['tx'])
    return jammed, clean

def new_expert(R, cfg):
    return R['LSTMExpert'](hidden=cfg.lstm_hidden, seq_len=cfg.lstm_seq_len, lr=cfg.lstm_lr, epochs=cfg.lstm_epochs, batch_size=cfg.lstm_batch)

def load_paper_expert(R, cfg, sps):
    path = REF_PROJECT / 'models' / 'lstm_expert_awgn.pt'
    if not path.exists():
        raise FileNotFoundError(f"paper LSTM weights not found: {path}")
    ex = new_expert(R, cfg)
    ex.model = R['ResidualLSTMNet'](cfg.lstm_hidden).to(R['device'])
    ex.model.load_state_dict(torch.load(path, map_location=R['device'])); ex.model.eval()
    jam, cln = training_pairs(R, cfg, 'fixed', ['narrowband', 'sweep'], cfg.paper_train_jsr, cfg.paper_train_n_per_jsr,
                              cfg.paper_train_seed_base, cfg.train_n_sym if cfg.smoke else 10000)
    X, _ = ex._make_seqs(jam, cln, sps)
    ex.in_mean, ex.in_std = float(np.mean(X)), float(max(np.std(X), 1e-6))
    return ex

def train_or_load_expert(R, cfg, name, cond, types, n_per_jsr, seed_base, exp_dirs, sps, ref_hash):
    """exp_dirs: list of experts/ folders to look in (Exp 2 first, then this experiment)."""
    tag = {'hidden': cfg.lstm_hidden, 'seq_len': cfg.lstm_seq_len, 'lr': cfg.lstm_lr, 'epochs': cfg.lstm_epochs,
           'batch': cfg.lstm_batch, 'train_jsr': cfg.train_jsr_list, 'n_per_jsr': n_per_jsr, 'seed_base': seed_base,
           'n_sym': cfg.train_n_sym, 'cond': cond, 'ref_hash': ref_hash, 'run_version': RUN_VERSION, 'smoke': cfg.smoke}
    if types != ['narrowband', 'sweep']:
        tag['types'] = types
    tag_hash = hashlib.md5(json.dumps(tag, sort_keys=True).encode()).hexdigest()[:8]
    ex = new_expert(R, cfg)
    for d in exp_dirs:
        pt, meta = d / f'lstm_{name}_{tag_hash}.pt', d / f'lstm_{name}_{tag_hash}.json'
        if pt.exists() and meta.exists():
            m = json.load(open(meta))
            ex.model = R['ResidualLSTMNet'](cfg.lstm_hidden).to(R['device'])
            ex.model.load_state_dict(torch.load(pt, map_location=R['device'])); ex.model.eval()
            ex.in_mean, ex.in_std = m['in_mean'], m['in_std']
            print(f"[EXPERT] {name}: loaded {pt} (trained {m['trained_at'][:16]}, best val {m['best_val']:.6f})")
            return ex
    d = exp_dirs[-1]; d.mkdir(parents=True, exist_ok=True)
    pt, meta = d / f'lstm_{name}_{tag_hash}.pt', d / f'lstm_{name}_{tag_hash}.json'
    print(f"[EXPERT] {name}: training on {cond} {types}, JSR {cfg.train_jsr_list}, {len(types) * len(cfg.train_jsr_list) * n_per_jsr} pairs ...")
    t0 = time.time()
    jam, cln = training_pairs(R, cfg, cond, types, cfg.train_jsr_list, n_per_jsr, seed_base, cfg.train_n_sym)
    set_all_seeds(42); ex.train_model(jam, cln, sps, verbose=cfg.verbose)
    torch.save(ex.model.state_dict(), pt)
    write_json_atomic({**tag, 'in_mean': ex.in_mean, 'in_std': ex.in_std, 'best_val': float(min(ex.val_losses)),
                       'train_losses': ex.losses, 'val_losses': ex.val_losses, 'trained_at': datetime.now().isoformat(),
                       'train_seconds': time.time() - t0}, meta)
    print(f"[EXPERT] {name}: done in {time.time() - t0:.0f}s, saved {pt.name}")
    return ex

# ============================================================================
#  ROUTER FRONT-END B (locked in Exp 1c), applied to the FULL frame
# ============================================================================
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

FE_COLS = ['fin', 'isi', 'fano', 'sfr', 'isi_drift', 'rf_sharp', 'rf_hf', 'rf_drift', 'rf_adrift']

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
    def __init__(self, R, cfg, cond, fs, sps, extra_sim=None):
        CC = R['ChannelConfig']; JT = R['JT']; sim = make_sim(R, cfg, cond); nsym = cfg.n_sym_test; off = cfg.router_seed_base
        sigs, ys = [], []
        for sim_k, seed_off in ([(sim, 0)] + ([(extra_sim, 100000)] if extra_sim is not None else [])):
            for i in range(cfg.router_n_train_clean):
                sigs.append(sim_k(CC(snr_db=cfg.snr_db, n_symbols=nsym, seed=off + seed_off + i), jam_type='none')['rx']); ys.append('none')
            for jt in JT:
                for jsr in cfg.router_train_jsr:
                    for i in range(cfg.router_n_train_per):
                        seed = off + seed_off + 1000 + i + JT.index(jt) * 700 + cfg.router_train_jsr.index(jsr) * 50
                        sigs.append(sim_k(CC(snr_db=cfg.snr_db, jsr_db=jsr, n_symbols=nsym, seed=seed), jam_type=jt)['rx']); ys.append(jt)
        X = frontend_B(R, cfg, sigs, fs); self.classes = ['none'] + list(JT)
        self.sc = StandardScaler().fit(X)
        self.clf = RandomForestClassifier(n_estimators=cfg.rf_trees, max_depth=cfg.rf_depth, random_state=cfg.rf_seed, n_jobs=cfg.n_jobs)
        self.clf.fit(self.sc.transform(X), ys)
        self.n_train = len(ys); self.train_acc = float(np.mean(self.clf.predict(self.sc.transform(X)) == np.array(ys)))
    def predict(self, X):
        P = self.clf.predict_proba(self.sc.transform(X)); cls = self.clf.classes_
        idx = np.argmax(P, axis=1)
        return cls[idx], P[np.arange(len(idx)), idx]

# ============================================================================
#  FIGURES
# ============================================================================
PAL = {'cobalt': '#1F63A8', 'ember': '#E1712F', 'fern': '#3B9A5B', 'amethyst': '#8151A1', 'teal': '#1B9AA6',
       'crimson': '#BE3B3B', 'amber': '#E4B23C', 'slate': '#4B6584', 'ink': '#23272E', 'graphite': '#767C86', 'mist': '#EBEEF2'}
SERIES = [PAL[k] for k in ['cobalt', 'amber', 'crimson', 'teal', 'ember', 'fern', 'amethyst']]
STYLES = ['-', '--', '-.', ':', '-', '--', '-.']; MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X']
IEEE_1COL, IEEE_2COL, IEEE_PT, IEEE_DPI = 3.5, 7.16, 8, 600

def apply_style():
    from matplotlib.font_manager import findfont, FontProperties
    fam = 'Times New Roman'
    try:
        findfont(FontProperties(family=fam), fallback_to_default=False)
    except Exception:
        fam = 'Liberation Serif'
        try:
            findfont(FontProperties(family=fam), fallback_to_default=False)
            print('[FONT] !! Times New Roman not installed; using Liberation Serif (metric-compatible)')
        except Exception:
            fam = 'DejaVu Serif'; print('[FONT] !! no Times/Liberation; DejaVu Serif used — regenerate before submission')
    mpl.rcParams.update({'font.family': 'serif', 'font.serif': [fam, 'DejaVu Serif'], 'font.size': IEEE_PT,
        'axes.labelsize': IEEE_PT, 'axes.titlesize': IEEE_PT, 'xtick.labelsize': IEEE_PT - 1, 'ytick.labelsize': IEEE_PT - 1,
        'legend.fontsize': IEEE_PT - 1, 'axes.prop_cycle': cycler(color=SERIES), 'axes.edgecolor': PAL['ink'],
        'axes.labelcolor': PAL['ink'], 'text.color': PAL['ink'], 'xtick.color': PAL['ink'], 'ytick.color': PAL['ink'],
        'grid.color': PAL['mist'], 'grid.linewidth': 0.6, 'axes.grid': True, 'axes.axisbelow': True,
        'axes.spines.top': False, 'axes.spines.right': False, 'lines.linewidth': 1.2, 'lines.markersize': 3.5,
        'axes.linewidth': 0.8, 'legend.frameon': False, 'figure.dpi': 110, 'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    return fam

def grey_proof(png):
    try:
        from PIL import Image
        Image.open(png).convert('L').save(str(png).replace('.png', '_grey.png'))
    except Exception as e:
        print(f'[GREY] skipped ({e})')

def make_figures(agg, cfg, JT, fig_dir, smoke, gate_key):
    fam = apply_style()
    series = [('none', 'no correction'), ('mono_lstm', 'monolithic LSTM'), ('lstm020_only', 'LSTM 0–20 only'),
              ('oracle_ale', 'oracle routing + ALE-128'), ('moeB_ale', 'MoE (front-end B) + ALE-128')]
    fig, axes = plt.subplots(len(JT), len(cfg.conditions), figsize=(IEEE_2COL, 1.75 * len(JT) + 0.6), sharex=True, sharey=True, squeeze=False)
    for ji, jt in enumerate(JT):
        for ci, cond in enumerate(cfg.conditions):
            ax = axes[ji, ci]
            for si, (m, lab) in enumerate(series):
                y = np.array([agg[f'{cond}|{jt}|{jsr}|{m}']['mean'] for jsr in cfg.jsr_list])
                c = np.array([agg[f'{cond}|{jt}|{jsr}|{m}']['ci95'] for jsr in cfg.jsr_list])
                y = np.maximum(y, 1e-5)
                ax.semilogy(cfg.jsr_list, y, color=SERIES[si], linestyle=STYLES[si], marker=MARKERS[si], label=lab)
                ax.fill_between(cfg.jsr_list, np.maximum(y - c, 1e-5), y + c, color=SERIES[si], alpha=0.12, linewidth=0)
            ax.set_title(f'({chr(97 + ji * len(cfg.conditions) + ci)}) {jt}, {cond} parameters', loc='left')
            if ji == len(JT) - 1:
                ax.set_xlabel('JSR (dB)')
            if ci == 0:
                ax.set_ylabel('BER' + (' (surviving symbols)' if jt == 'pulse' else ''))
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.0))
    if smoke:
        fig.suptitle('SMOKE — every number here is garbage', color=PAL['crimson'], fontsize=IEEE_PT + 1)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    pa = fig_dir / 'figR1_moe_ber_vs_jsr_ale128'
    fig.savefig(pa.with_suffix('.pdf')); fig.savefig(pa.with_suffix('.png'), dpi=IEEE_DPI); plt.show(); plt.close(fig)
    grey_proof(pa.with_suffix('.png'))
    return [str(pa.with_suffix('.pdf'))], fam


# ============================================================================
#  COMPOUND FRAMES: each jammer's power is set relative to the CLEAN signal (same rng order as single frames)
# ============================================================================
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


# ============================================================================
#  COMPOUND FRAMES: each jammer's power is set relative to the CLEAN signal (same rng order as single frames)
# ============================================================================
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



# ============================================================================
#  MAIN
# ============================================================================
def main():
    t_start = time.time(); RUN_ID = datetime.now().strftime('%Y%m%d_%H%M')
    print('=' * 78); print(f"[EXP] SNN-MoE TCCN R1 — Experiment {cfg.exp_index}/{cfg.exp_total}: {cfg.exp_name}"); print('=' * 78)
    if cfg.smoke:
        print('\n' + '!' * 78 + '\n!!  SMOKE MODE — every number below is garbage; this run only checks the wiring\n' + '!' * 78 + '\n')
    print('[PROGRESS] Manifest:')
    for idx, name, status in cfg.exp_manifest:
        print(f"  {'→' if status == 'THIS' else '✓' if status == 'done' else ' '} {idx}/{cfg.exp_total}: {name} [{status}]")
    print(f"[PERSIST] colab={ON_COLAB}  project={PROJECT}")
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'
    print(f"[HW] device={cfg.device}  gpu={gpu}  cpus={os.cpu_count()}  numba={'yes' if NUMBA_OK else 'NO (ALE ~100x slower)'}")

    nb_path = find_reference_notebook(); R, ref_hash, skipped = load_reference(nb_path)
    print(f"[REF] {nb_path}  sha256[:16]={ref_hash}")
    for s in skipped:
        print(f"[REF] import skipped: {s}")
    CC = R['ChannelConfig'](); SEC = R['SpikeEncoderConfig']()
    expect = {'fs': 1e6, 'symbol_rate': 1e5, 'snr_db': 10.0, 'n_symbols': 10000, 'enc_threshold': 0.04, 'enc_refractory': 2}
    got = {'fs': CC.fs, 'symbol_rate': CC.symbol_rate, 'snr_db': CC.snr_db, 'n_symbols': CC.n_symbols, 'enc_threshold': SEC.threshold, 'enc_refractory': SEC.refractory}
    bad = {k: (expect[k], got[k]) for k in expect if expect[k] != got[k]}
    if bad:
        raise RuntimeError(f"FIDELITY GATE FAILED — reference defaults differ: {bad}")
    print(f"[GATE] reference dataclass defaults match ({len(expect)} fields)")
    fs = CC.fs; JT = list(R['JT'])
    probe = R['simulate_channel'](R['ChannelConfig'](snr_db=cfg.snr_db, jsr_db=10, n_symbols=min(cfg.n_sym_test, 2000), seed=1), jam_type='narrowband')['rx']
    if not np.allclose(apply_ale(probe, cfg, cfg.ale_mu_nb), _ale_py(np.ascontiguousarray(probe), cfg.ale_taps, cfg.ale_delay, cfg.ale_mu_nb, 1e-6), rtol=1e-9, atol=1e-9):
        raise RuntimeError('ALE numba path differs from python path')
    print('[GATE] ALE numba == python on a real frame')
    deviations = [
        "final MoE frozen in Exp 6a: front-end B router + {ALE-128 mu_NB, ALE-128 mu_SW, calibrated blanker, pass-through} + override + cascade",
        "switching: reference scenario (100 frames) and seeds 30000+trial*1000+frame; 'fixed blanker' re-implemented from the reference eval (within-frame z>3)",
        "MoE-conv: the reference 7 conventional features (full FFT) as the router, SAME experts — routing-parity row",
        "goodput/duty: reference protocol and seeds; duty variation uses the reference add_pulse_jam_variable (fixed protocol) and random duty/period/phase (random protocol)",
        "fading: reference simulate_rayleigh (fixed jammers; block-faded signal, unfaded jammer); random protocol = same fading + randomised jammers; AWGN-trained router AND a fading-augmented router (AWGN + faded training frames) reported",
        "SNR: routers and blanker calibration at 10 dB, tested at other SNRs; override uses rho only (no SNR knowledge)",
        "coded link: reference protocol (255-byte codewords, 20 cw x 50 trials, seeds 40000+); erasures from the calibrated pulse expert; random protocol = random duty/period/phase",
    ]
    print('[DEVIATIONS]'); [print(f'   - {d}') for d in deviations]

    r0 = R['simulate_channel'](R['ChannelConfig'](snr_db=cfg.snr_db, n_symbols=cfg.n_sym_test, seed=cfg.seed_base), jam_type='none')
    data_fp = hashlib.sha256(np.ascontiguousarray(r0['rx']).tobytes()).hexdigest()[:12]; sps = r0['sps']
    key = run_key(cfg, data_fp, ref_hash)
    EXP_DIR = PROJECT / 'experiments' / 'exp6b_system_tables'; RUN_ROOT = EXP_DIR / 'runs'
    RUN_DIR = RUN_ROOT / (('smoke_' if cfg.smoke else '') + key); FIG_DIR = RUN_DIR / 'figures'
    for d in (EXP_DIR, RUN_ROOT, RUN_DIR, FIG_DIR):
        d.mkdir(parents=True, exist_ok=True)
    PARTIAL = RUN_DIR / 'partial.json'
    print(f"[RUN] key={key}  data_fp={data_fp}  dir={RUN_DIR}")

    blanker = R['PulseBlankerExpert'](blank_threshold_z=cfg.blanker_z, group_size=cfg.blanker_group)
    cal = [R['simulate_channel'](R['ChannelConfig'](snr_db=cfg.snr_db, n_symbols=cfg.blanker_cal_samples // 10, seed=cfg.blanker_cal_seed_base + i), jam_type='none')['rx'][:cfg.blanker_cal_samples]
           for i in range(cfg.blanker_cal_n)]
    blanker.calibrate(cal, sps)
    routers, conv_routers = {}, {}
    for cond in cfg.conditions:
        t0 = time.time(); routers[cond] = RouterB(R, cfg, cond, fs, sps)
        # conventional-feature router on the same training frames
        CCc = R['ChannelConfig']; sim = make_sim(R, cfg, cond); nsym = cfg.n_sym_test; off = cfg.router_seed_base; sigs, ys = [], []
        for i in range(cfg.router_n_train_clean):
            sigs.append(sim(CCc(snr_db=cfg.snr_db, n_symbols=nsym, seed=off + i), jam_type='none')['rx']); ys.append('none')
        for jt in JT:
            for jsr in cfg.router_train_jsr:
                for i in range(cfg.router_n_train_per):
                    seed = off + 1000 + i + JT.index(jt) * 700 + cfg.router_train_jsr.index(jsr) * 50
                    sigs.append(sim(CCc(snr_db=cfg.snr_db, jsr_db=jsr, n_symbols=nsym, seed=seed), jam_type=jt)['rx']); ys.append(jt)
        Xc = np.array([R['extract_conventional_features'](x) for x in sigs]); scc = StandardScaler().fit(Xc)
        clfc = RandomForestClassifier(n_estimators=cfg.rf_trees, max_depth=cfg.rf_depth, random_state=cfg.rf_seed, n_jobs=cfg.n_jobs).fit(scc.transform(Xc), ys)
        conv_routers[cond] = (scc, clfc)
        print(f"[ROUTER] {cond}: front-end B ({routers[cond].n_train} frames, train acc {routers[cond].train_acc:.3f}) + conventional router in {time.time() - t0:.0f}s")
    mu_of = {'narrowband': cfg.ale_mu_nb, 'sweep': cfg.ale_mu_sw}

    # ── final MoE (identical to Exp 6a) ────────────────────────────────────
    def ber_of(bits, soft, idx=None):
        b = (soft > 0).astype(int)
        return R['compute_ber'](bits if idx is None else bits[idx], b if idx is None else b[idx]) if len(b) else 0.5
    def blank_calibrated(x):
        pr = blanker.correct_symbols(x, sps); ns_ = len(x) // sps
        return pr['soft'], pr['surviving_indices'], (float(np.mean(pr['grp_mask'])) if len(pr['grp_mask']) else 0.0), np.repeat(~pr['erasure_mask'][:ns_], sps), pr['erasure_mask']
    def pulse_check(y):
        ns_ = len(y) // sps; soft = np.mean(y[:ns_ * sps].reshape(ns_, sps), axis=1)
        win = cfg.blanker_group * sps; ng = len(y) // win
        rms = np.sqrt(np.mean(y[:ng * win].reshape(ng, win) ** 2, axis=1)); flag = rms > cfg.pulse_k * np.median(rms)
        frac = float(np.mean(flag)) if ng else 0.0; er = np.zeros(ns_, dtype=bool)
        for g in np.where(flag)[0]:
            er[g * cfg.blanker_group:min((g + 1) * cfg.blanker_group, ns_)] = True
        return soft, np.where(~er)[0], frac
    def route_B(rx, cond):
        return str(routers[cond].predict(frontend_B(R, cfg, [rx], fs))[0][0])
    def route_conv(rx, cond):
        scc, clfc = conv_routers[cond]; return str(clfc.predict(scc.transform(R['extract_conventional_features'](rx).reshape(1, -1)))[0])
    def final_moe(rx, bits, cls1, cond):
        cls = cls1; path = cls1[:2]
        if cls1 in ('none', 'broadband'):
            e = apply_ale(rx, cfg, cfg.ale_mu_sw); rho = 1.0 - np.mean(e ** 2) / (np.mean(rx ** 2) + 1e-12)
            if rho > cfg.rho_min:
                cls = 'sweep'; path += '->override'
        if cls in ('narrowband', 'sweep'):
            y = apply_ale(rx, cfg, mu_of[cls]); path += '->ALE'
            soft, surv_idx, frac = pulse_check(y)
            if cfg.blank_min_fraction < frac < cfg.blank_max_fraction:
                return ber_of(bits[surv_idx], soft[surv_idx]), len(surv_idx) / len(soft), path + '->blank'
            return ber_of(bits, soft_symbols(y, sps)), 1.0, path
        if cls == 'pulse':
            soft, surv_idx, frac, smask, _ = blank_calibrated(rx); path += '->blank'
            if len(surv_idx) < 200:
                return (ber_of(bits[surv_idx], soft[surv_idx]) if len(surv_idx) else 0.5), len(surv_idx) / len(soft), path
            surv = rx[:len(smask)][smask]; cls2 = route_B(surv, cond)
            if cls2 in ('narrowband', 'sweep'):
                y = soft_symbols(apply_ale(surv, cfg, mu_of[cls2]), sps)
                return ber_of(bits[surv_idx], y[:len(surv_idx)]), len(surv_idx) / len(soft), path + f'->{cls2[:2]}->ALE'
            return ber_of(bits[surv_idx], soft[surv_idx]), len(surv_idx) / len(soft), path
        return ber_of(bits, soft_symbols(rx, sps)), 1.0, path + '->pass'
    def oracle_single(rx, bits, jt):
        if jt in mu_of:
            return ber_of(bits, soft_symbols(apply_ale(rx, cfg, mu_of[jt]), sps))
        if jt == 'pulse':
            soft, surv_idx, _, _, _ = blank_calibrated(rx); return ber_of(bits[surv_idx], soft[surv_idx])
        return ber_of(bits, soft_symbols(rx, sps))
    def fixed_blanker_ref(rx, bits):
        """reference eval_blanker_only: within-frame mean/std z>3 on 5-symbol groups."""
        ns_ = len(rx) // sps; soft = np.mean(rx[:ns_ * sps].reshape(ns_, sps), axis=1)
        win = 5 * sps; n_g = len(rx) // win
        rms = np.sqrt(np.mean(rx[:n_g * win].reshape(n_g, win) ** 2, axis=1)); mu, st = np.mean(rms), max(np.std(rms), 1e-6)
        er = np.zeros(ns_, dtype=bool)
        for k in range(n_g):
            if (rms[k] - mu) / st > 3.0:
                er[k * 5:min((k + 1) * 5, ns_)] = True
        idx = np.where(~er)[0]
        return ber_of(bits[idx], soft[idx]) if len(idx) else 0.5

    # ── resume ─────────────────────────────────────────────────────────────
    blob = {'results': {}, 'units_done': []}
    if PARTIAL.exists():
        try:
            blob = json.load(open(PARTIAL))
        except Exception:
            PARTIAL.replace(PARTIAL.with_suffix('.json.corrupt')); print('[PERSIST] partial.json truncated; quarantined')
    else:
        best, best_n = None, 0
        for sib in sorted(RUN_ROOT.glob('*/partial.json')):
            try:
                b = json.load(open(sib))
            except Exception:
                continue
            if b.get('run_version') == RUN_VERSION and b.get('ref_hash') == ref_hash and bool(b.get('smoke')) == cfg.smoke and len(b.get('units_done', [])) > best_n:
                best, best_n = sib, len(b['units_done'])
        if best:
            shutil.copy2(best, PARTIAL); blob = json.load(open(PARTIAL)); print(f"[RESUME] adopted {best_n} unit(s) from {best.parent.name}")
    blob.update({'run_version': RUN_VERSION, 'ref_hash': ref_hash, 'smoke': cfg.smoke})
    done = set(blob['units_done']); res = blob['results']
    def save():
        blob['units_done'] = sorted(done); write_json_atomic(blob, PARTIAL)

    scenario = ([('broadband', 10)] * 20 + [('narrowband', 10)] * 15 + [('sweep', 10)] * 15 + [('pulse', 10)] * 20 +
                [('broadband', 10)] * 10 + [('sweep', 10)] * 10 + [('pulse', 10)] * 10)
    segments = [('BB-1', 0, 20), ('NB', 20, 35), ('SW-1', 35, 50), ('PL-1', 50, 70), ('BB-2', 70, 80), ('SW-2', 80, 90), ('PL-2', 90, 100)]
    units = [f'A|{c}' for c in cfg.conditions] + [f'B|{c}' for c in cfg.conditions] + [f'C|{c}' for c in cfg.conditions] + \
            [f'D|{c}' for c in cfg.conditions] + [f'E|{c}' for c in cfg.conditions]
    print(f"[WORKLOAD] {len(units)} units: switching {cfg.switch_trials}x{len(scenario)} frames/cond, goodput {len(cfg.gp_jsr_list)}x{cfg.gp_trials}, duty {len(cfg.duty_list)}x{cfg.duty_trials}, "
          f"fading {len(JT)}x{cfg.fade_trials}x2, SNR {len(cfg.snr_list)}x{len(JT)}x{cfg.snr_trials}, coded {len(cfg.rs_codes)}x{len(cfg.rs_jsr_list)}x{cfg.rs_trials}x{cfg.rs_codewords} -> est ~15-20 min")
    pbar = tqdm(units, desc='units')
    for uid in pbar:
        if uid in done:
            continue
        part, cond = uid.split('|'); t_unit = time.time(); pbar.set_postfix_str(uid); sim = make_sim(R, cfg, cond); CCc = R['ChannelConfig']

        if part == 'A':   # ---------------- switching ----------------
            names = ['none', 'notch', 'blanker_fixed', 'ale_blind', 'moe_conv', 'moe_final', 'oracle']
            per = {m: np.zeros((cfg.switch_trials, len(scenario))) for m in names}
            route_ok_B = np.zeros((cfg.switch_trials, len(scenario))); route_ok_C = np.zeros((cfg.switch_trials, len(scenario)))
            for t in range(cfg.switch_trials):
                for fi, (jt, jsr) in enumerate(scenario):
                    r = sim(CCc(snr_db=cfg.snr_db, jsr_db=jsr, n_symbols=cfg.n_sym_test, seed=cfg.switch_seed_base + t * 1000 + fi), jam_type=jt)
                    rx, bits = r['rx'], r['bits']
                    per['none'][t, fi] = ber_of(bits, soft_symbols(rx, sps))
                    per['notch'][t, fi] = ber_of(bits, soft_symbols(R['apply_notch'](rx, fs), sps))
                    per['blanker_fixed'][t, fi] = fixed_blanker_ref(rx, bits)
                    per['ale_blind'][t, fi] = ber_of(bits, soft_symbols(apply_ale(rx, cfg, cfg.ale_mu_sw), sps))
                    cB = route_B(rx, cond); cC = route_conv(rx, cond)
                    per['moe_final'][t, fi] = final_moe(rx, bits, cB, cond)[0]
                    per['moe_conv'][t, fi] = final_moe(rx, bits, cC, cond)[0]
                    per['oracle'][t, fi] = oracle_single(rx, bits, jt)
                    okB = (cB in ('narrowband', 'sweep')) if jt in ('narrowband', 'sweep') else cB == jt
                    okC = (cC in ('narrowband', 'sweep')) if jt in ('narrowband', 'sweep') else cC == jt
                    route_ok_B[t, fi] = okB; route_ok_C[t, fi] = okC
            res[uid] = {'overall': {m: float(per[m].mean()) for m in names},
                        'segments': {sname: {m: float(per[m][:, a:b].mean()) for m in names} for sname, a, b in segments},
                        'routing_B': float(route_ok_B.mean()), 'routing_conv': float(route_ok_C.mean()),
                        'routing_B_segments': {sname: float(route_ok_B[:, a:b].mean()) for sname, a, b in segments}}
            pbar.write(f"  A {cond}: " + '  '.join(f"{m}={res[uid]['overall'][m]:.4f}" for m in names) + f"  routing B {res[uid]['routing_B']:.3f} conv {res[uid]['routing_conv']:.3f}  [{time.time() - t_unit:.0f}s]")

        elif part == 'B':   # ---------------- goodput + duty ----------------
            gp = {}
            for jsr in cfg.gp_jsr_list:
                g_none, g_er, sv, be = [], [], [], []
                for t in range(cfg.gp_trials):
                    r = sim(CCc(snr_db=cfg.snr_db, jsr_db=jsr, n_symbols=cfg.n_sym_test, seed=cfg.gp_seed_base + t), jam_type='pulse')
                    b0 = ber_of(r['bits'], soft_symbols(r['rx'], sps)); soft, idx, _, _, _ = blank_calibrated(r['rx']); b1 = ber_of(r['bits'][idx], soft[idx]); s_ = len(idx) / len(soft)
                    g_none.append(1 - b0); g_er.append(s_ * (1 - b1)); sv.append(s_); be.append(b1)
                gp[str(jsr)] = {'gp_none': float(np.mean(g_none)), 'gp_erasure': float(np.mean(g_er)), 'survival': float(np.mean(sv)), 'ber_erasure': float(np.mean(be))}
            duty = {}
            for d in cfg.duty_list:
                bo, be, sv = [], [], []
                for t in range(cfg.duty_trials):
                    rng = np.random.default_rng(cfg.duty_seed_base + t); cc = CCc(snr_db=cfg.snr_db, jsr_db=10, n_symbols=cfg.n_sym_test, seed=cfg.duty_seed_base + t)
                    bits, symbols, tx, _ = R['generate_bpsk'](cc, rng); noisy = R['add_awgn'](tx, cfg.snr_db, rng)
                    if cond == 'fixed':
                        rx, _ = R['add_pulse_jam_variable'](noisy, 10, rng, duty=d)
                    else:   # random phase and period, requested duty
                        period = int(rng.integers(cfg.pulse_period_range[0], cfg.pulse_period_range[1] + 1)); on = max(1, int(period * d)); off0 = int(rng.integers(0, period))
                        env = np.zeros(len(noisy))
                        for i in range(off0 - period, len(noisy), period):
                            a_, b_ = max(i, 0), min(i + on, len(noisy))
                            if b_ > a_:
                                env[a_:b_] = 1.0
                        rx = noisy + env * rng.normal(0, np.sqrt(np.mean(noisy ** 2) * 10 / d), len(noisy))
                    bo.append(ber_of(bits, soft_symbols(rx, sps))); soft, idx, _, _, _ = blank_calibrated(rx); be.append(ber_of(bits[idx], soft[idx])); sv.append(len(idx) / len(soft))
                duty[str(d)] = {'ber_none': float(np.mean(bo)), 'ber_erasure': float(np.mean(be)), 'survival': float(np.mean(sv))}
            res[uid] = {'goodput': gp, 'duty': duty}
            pbar.write(f"  B {cond}: goodput@10 none={gp[str(10)]['gp_none']:.3f} erasure={gp[str(10)]['gp_erasure']:.3f} surv={gp[str(10)]['survival']:.2f}; duty: " +
                       ' '.join(f"{d}:{duty[str(d)]['ber_erasure']:.4f}/{duty[str(d)]['survival']:.2f}" for d in cfg.duty_list) + f"  [{time.time() - t_unit:.0f}s]" if str(10) in gp else f"  B {cond} done")

        elif part == 'C':   # ---------------- fading ----------------
            def sim_fade(cc, jam_type):
                if cond == 'fixed':
                    return R['simulate_rayleigh'](cc, jam_type=jam_type, fade_block=cfg.fade_block)
                rng = np.random.default_rng(cc.seed); bits, symbols, tx, sps_ = R['generate_bpsk'](cc, rng); n = len(tx)
                nb_ = n // cfg.fade_block + 1; h = np.sqrt(rng.normal(0, 1 / np.sqrt(2), nb_) ** 2 + rng.normal(0, 1 / np.sqrt(2), nb_) ** 2)
                faded = np.zeros(n)
                for k in range(nb_):
                    faded[k * cfg.fade_block:min((k + 1) * cfg.fade_block, n)] = tx[k * cfg.fade_block:min((k + 1) * cfg.fade_block, n)] * h[k]
                noisy = R['add_awgn'](faded, cc.snr_db, rng)
                if jam_type == 'none':
                    return {'bits': bits, 'rx': noisy, 'sps': sps_}
                return {'bits': bits, 'rx': noisy + jam_component(R, cfg, cond, noisy, jam_type, cc.jsr_db, cc.fs, rng), 'sps': sps_}
            t0 = time.time(); router_aug = RouterB(R, cfg, cond, fs, sps, extra_sim=sim_fade)   # AWGN + faded training frames
            pbar.write(f"    [ROUTER] {cond}: fading-augmented front-end B router ({router_aug.n_train} frames) in {time.time() - t0:.0f}s")
            def route_aug(rx):
                return str(router_aug.predict(frontend_B(R, cfg, [rx], fs))[0][0])
            cls_acc, cls_acc_aug = {}, {}
            for jt in JT + ['none']:
                ok = 0; ok_aug = 0
                for t in range(cfg.fade_trials):
                    seed = (cfg.fade_cls_seed_base + 500 + t) if jt == 'none' else (cfg.fade_cls_seed_base + t + JT.index(jt) * 500)
                    cc = CCc(snr_db=cfg.snr_db, jsr_db=10, n_symbols=cfg.n_sym_test, seed=seed); r = sim_fade(cc, jt)
                    for p, acc in [(route_B(r['rx'], cond), 'a'), (route_aug(r['rx']), 'b')]:
                        hit = (p in ('narrowband', 'sweep')) if jt in ('narrowband', 'sweep') else (p == jt)
                        if acc == 'a':
                            ok += hit
                        else:
                            ok_aug += hit
                cls_acc[jt] = ok / cfg.fade_trials; cls_acc_aug[jt] = ok_aug / cfg.fade_trials
            ber = {}
            for jt in JT + ['none']:
                out = {'awgn_none': [], 'awgn_final': [], 'awgn_oracle': [], 'fade_none': [], 'fade_final': [], 'fade_final_aug': [], 'fade_oracle': []}
                for t in range(cfg.fade_trials):
                    seed = (cfg.fade_ber_seed_base + 900 + t) if jt == 'none' else (cfg.fade_ber_seed_base + t + JT.index(jt) * 500)
                    cc = CCc(snr_db=cfg.snr_db, jsr_db=10, n_symbols=cfg.n_sym_test, seed=seed)
                    ra = sim(cc, jam_type=jt); rf = sim_fade(cc, jt)
                    for tag, r in [('awgn', ra), ('fade', rf)]:
                        out[f'{tag}_none'].append(ber_of(r['bits'], soft_symbols(r['rx'], sps)))
                        out[f'{tag}_final'].append(final_moe(r['rx'], r['bits'], route_B(r['rx'], cond), cond)[0])
                        out[f'{tag}_oracle'].append(oracle_single(r['rx'], r['bits'], jt))
                    out['fade_final_aug'].append(final_moe(rf['rx'], rf['bits'], route_aug(rf['rx']), cond)[0])
                ber[jt] = {k: float(np.mean(v)) for k, v in out.items()}
            res[uid] = {'routing_acc': cls_acc, 'routing_acc_aug': cls_acc_aug, 'ber': ber}
            pbar.write(f"  C {cond}: routing " + ' '.join(f"{jt[:2]}={cls_acc[jt]:.2f}/{cls_acc_aug[jt]:.2f}" for jt in cls_acc) + '; BER fade final/aug: ' + ' '.join(f"{jt[:2]}={ber[jt]['fade_final']:.4f}/{ber[jt]['fade_final_aug']:.4f}" for jt in JT + ['none']) + f"  [{time.time() - t_unit:.0f}s]")

        elif part == 'D':   # ---------------- SNR robustness ----------------
            out = {}
            for snr in cfg.snr_list:
                for jt in JT:
                    b = {'none': [], 'final': [], 'oracle': [], 'route_ok': [], 'survival': []}
                    for t in range(cfg.snr_trials):
                        r = sim(CCc(snr_db=snr, jsr_db=10, n_symbols=cfg.n_sym_test, seed=cfg.snr_seed_base + t + JT.index(jt) * 500 + cfg.snr_list.index(snr) * 50), jam_type=jt)
                        p = route_B(r['rx'], cond); fb, sv, _ = final_moe(r['rx'], r['bits'], p, cond)
                        b['none'].append(ber_of(r['bits'], soft_symbols(r['rx'], sps))); b['final'].append(fb); b['oracle'].append(oracle_single(r['rx'], r['bits'], jt))
                        b['route_ok'].append((p in ('narrowband', 'sweep')) if jt in ('narrowband', 'sweep') else p == jt); b['survival'].append(sv)
                    out[f'{snr}|{jt}'] = {k: float(np.mean(v)) for k, v in b.items()}
            res[uid] = out
            pbar.write(f"  D {cond}: " + ' '.join(f"SNR{snr}:" + '/'.join(f"{out[f'{snr}|{jt}']['final']:.4f}" for jt in JT) for snr in cfg.snr_list) + f"  [{time.time() - t_unit:.0f}s]")

        elif part == 'E':   # ---------------- coded link ----------------
            def safe_dec(rs, data, erase_pos=None):
                r_ = rs.decode(data, erase_pos=erase_pos) if erase_pos is not None else rs.decode(data)
                return bytearray(r_[0]) if isinstance(r_, tuple) else bytearray(r_)
            out = {}
            for n_par, cname in cfg.rs_codes:
                k = 255 - n_par; rs = RSCodec(n_par)
                for jsr in cfg.rs_jsr_list:
                    bl = {'nocorr': [], 'blanker': [], 'erasure': []}; n_er = []
                    for t in range(cfg.rs_trials):
                        nf = {'nocorr': 0, 'blanker': 0, 'erasure': 0}
                        for cw in range(cfg.rs_codewords):
                            seed = cfg.rs_seed_base + jsr * 1000 + t * 100 + cw
                            data = np.random.default_rng(seed).integers(0, 256, size=k, dtype=np.uint8)
                            enc = np.array(list(rs.encode(data)), dtype=np.uint8); tx_bits = np.unpackbits(enc); nsym = len(tx_bits)
                            tx_s = np.repeat(2.0 * tx_bits - 1.0, sps); rng_c = np.random.default_rng(seed + 500000)
                            noisy = tx_s + rng_c.normal(0, np.sqrt(0.1), len(tx_s)); sp = np.mean(tx_s ** 2)
                            if cond == 'fixed':
                                per_, dt_, off0 = 500, 0.2, 0
                            else:
                                per_ = int(rng_c.integers(cfg.pulse_period_range[0], cfg.pulse_period_range[1] + 1)); dt_ = rng_c.uniform(*cfg.pulse_duty_range); off0 = int(rng_c.integers(0, per_))
                            env = np.zeros(len(noisy)); on = max(1, int(per_ * dt_))
                            for i in range(off0 - per_, len(noisy), per_):
                                a_, b_ = max(i, 0), min(i + on, len(noisy))
                                if b_ > a_:
                                    env[a_:b_] = 1.0
                            rx = noisy + env * rng_c.normal(0, np.sqrt(sp * 10 ** (jsr / 10) / dt_), len(noisy))
                            soft, idx, _, _, er_mask = blank_calibrated(rx); hard = (soft > 0).astype(np.uint8)[:nsym]
                            er_bytes = sorted(set(i // 8 for i in np.where(er_mask[:nsym])[0] if i // 8 < 255)); n_er.append(len(er_bytes))
                            rx_bytes = np.packbits(hard)[:255]
                            for m, arg in [('nocorr', (rx_bytes, None)), ('blanker', (np.packbits(np.where(er_mask[:nsym], 0, hard))[:255], None)), ('erasure', (rx_bytes, er_bytes))]:
                                try:
                                    dec = safe_dec(rs, arg[0], arg[1])[:k]
                                    if np.array(list(dec), dtype=np.uint8).tolist() != data.tolist():
                                        nf[m] += 1
                                except ReedSolomonError:
                                    nf[m] += 1
                        for m in bl:
                            bl[m].append(nf[m] / cfg.rs_codewords)
                    out[f'{cname}|{jsr}'] = {m: float(np.mean(bl[m])) for m in bl}; out[f'{cname}|{jsr}']['erased_bytes'] = float(np.mean(n_er))
            res[uid] = out
            pbar.write(f"  E {cond}: " + '  '.join(f"{cname}@10: {out[f'{cname}|10']['nocorr']:.3f}/{out[f'{cname}|10']['blanker']:.3f}/{out[f'{cname}|10']['erasure']:.3f}" for _, cname in cfg.rs_codes if f'{cname}|10' in out) + f"  [{time.time() - t_unit:.0f}s]")
        done.add(uid); save()
    pbar.close()

    # ── tables ─────────────────────────────────────────────────────────────
    print('\n' + '=' * 96); print("TABLE IX' — DYNAMIC JAMMER SWITCHING (100-frame engagement, overall BER; pulse frames: surviving symbols)"); print('=' * 96)
    names = ['none', 'notch', 'blanker_fixed', 'ale_blind', 'moe_conv', 'moe_final', 'oracle']
    print(f"{'cond':<8}" + ''.join(f"{m:>14}" for m in names) + f"{'route B':>9}{'route conv':>11}")
    for cond in cfg.conditions:
        a = res[f'A|{cond}']; print(f"{cond:<8}" + ''.join(f"{a['overall'][m]:>14.4f}" for m in names) + f"{a['routing_B']:>9.3f}{a['routing_conv']:>11.3f}")
        print(f"{'':<8}per segment (final MoE / oracle / routing B): " + '  '.join(f"{s}:{a['segments'][s]['moe_final']:.4f}/{a['segments'][s]['oracle']:.4f}/{a['routing_B_segments'][s]:.2f}" for s, _, _ in segments))
    print("\nTABLE V' — PULSE GOODPUT (calibrated pulse expert) and TABLE VI' — DUTY CYCLE @ JSR 10")
    for cond in cfg.conditions:
        b = res[f'B|{cond}']
        print(f"  {cond}: " + '  '.join(f"JSR{j}: surv {b['goodput'][str(j)]['survival']:.2f} GP {b['goodput'][str(j)]['gp_none']:.3f}->{b['goodput'][str(j)]['gp_erasure']:.3f}" for j in cfg.gp_jsr_list))
        print(f"  {cond}: " + '  '.join(f"duty {d:.0%}: BER {b['duty'][str(d)]['ber_none']:.4f}->{b['duty'][str(d)]['ber_erasure']:.4f} surv {b['duty'][str(d)]['survival']:.2f}" for d in cfg.duty_list))
    print("\nTABLE VII' — RAYLEIGH BLOCK FADING (JSR 10): routing accuracy AWGN-trained/fading-augmented | BER none -> final MoE (fading-aug router) (oracle)")
    for cond in cfg.conditions:
        c = res[f'C|{cond}']
        print(f"  {cond}: routing " + ' '.join(f"{jt[:2]}={c['routing_acc'][jt]:.2f}/{c['routing_acc_aug'][jt]:.2f}" for jt in JT + ['none']))
        for jt in JT + ['none']:
            b = c['ber'][jt]; print(f"    {jt:<11} AWGN {b['awgn_none']:.4f}->{b['awgn_final']:.4f} ({b['awgn_oracle']:.4f})   fading {b['fade_none']:.4f}->{b['fade_final']:.4f} / {b['fade_final_aug']:.4f} ({b['fade_oracle']:.4f})")
    print("\nSNR ROBUSTNESS (JSR 10; routers/blanker calibrated at 10 dB): final MoE BER (oracle) | routing ok")
    for cond in cfg.conditions:
        d = res[f'D|{cond}']
        for snr in cfg.snr_list:
            print(f"  {cond} SNR {snr:>2}: " + '  '.join(f"{jt[:2]} {d[f'{snr}|{jt}']['final']:.4f} ({d[f'{snr}|{jt}']['oracle']:.4f}) r{d[f'{snr}|{jt}']['route_ok']:.2f}" for jt in JT))
    print("\nTABLE X' — RS CODED LINK, BLER (no correction / blanker without erasure info / erasure-aware decoding)")
    for cond in cfg.conditions:
        e = res[f'E|{cond}']
        for _, cname in cfg.rs_codes:
            print(f"  {cond} {cname}: " + '  '.join(f"JSR{j}: {e[f'{cname}|{j}']['nocorr']:.3f}/{e[f'{cname}|{j}']['blanker']:.3f}/{e[f'{cname}|{j}']['erasure']:.3f} [{e[f'{cname}|{j}']['erased_bytes']:.0f}B]" for j in cfg.rs_jsr_list))

    summary = {'run_id': RUN_ID, 'run_key': key, 'run_version': RUN_VERSION, 'analysis_version': ANALYSIS_VERSION,
               'exp_index': cfg.exp_index, 'exp_total': cfg.exp_total, 'exp_name': cfg.exp_name, 'exp_manifest': cfg.exp_manifest,
               'smoke': cfg.smoke, 'config': asdict(cfg), 'reference': {'notebook': str(nb_path), 'sha256_16': ref_hash, 'imports_skipped': skipped},
               'deviations': deviations, 'data_fingerprint': data_fp, 'results': res, 'scenario': scenario, 'segments': segments,
               'routers': {c: {'n_train': routers[c].n_train, 'train_acc': routers[c].train_acc} for c in routers},
               'numba': NUMBA_OK, 'elapsed_min': (time.time() - t_start) / 60, 'timestamp': datetime.now().isoformat()}
    write_json_atomic(summary, RUN_DIR / 'summary.json'); write_json_atomic(summary, EXP_DIR / 'latest_summary.json')
    print(f"\n[DONE] {RUN_DIR / 'summary.json'}; elapsed {summary['elapsed_min']:.1f} min")
    return summary

summary = main()
