# ============================================================================
#  SNN-MoE Anti-Jamming — TCCN R1 revision
#  EXP 6a/6: FINAL MoE ASSEMBLY + REGENERATION OF THE HEADLINE TABLES
#    final MoE : front-end B router (full frame) -> {ALE-128 mu_NB, ALE-128 mu_SW, calibrated blanker, pass-through}
#                + PREDICTABILITY OVERRIDE: a 'none'/'broadband' verdict is checked by running the ALE (sweep step)
#                  and measuring the removed power fraction rho = 1 - P(e)/P(rx); rho > rho_min => structured
#                  (catches multi-tone compounds the classifier calls broadband and weak tones it calls clean)
#                + GUIDED CASCADE (Exp 4): structured -> within-frame pulse test -> erasure; pulse -> blank -> re-route
#    regenerates: Table II' (JSR 10, both protocols), BER vs JSR for the four types (100 trials), clean-frame audit,
#                compound table (50 trials), routing/override statistics — final MoE vs hard MoE (3b) vs oracle
#    system-level tables (switching, coded link, goodput/duty, fading) follow in Exp 6b
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
    import google.colab  # noqa: F401
    ON_COLAB = True
except ImportError:
    ON_COLAB = False

_default_frames = Path(__file__).resolve().parent if '__file__' in globals() else Path('simulation/frame_receiver')
FRAME_DIR = Path(os.environ.get('UAV_MOE_FRAME_DIR', str(_default_frames))).resolve()
PROJECT = Path(os.environ.get('UAV_MOE_OUTPUT_DIR', 'runs/frame_experiments')).resolve()
PROJECT.mkdir(parents=True, exist_ok=True)
REF_PROJECT = FRAME_DIR

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
    exp_name: str = "Final MoE assembly (override + cascade) and regeneration of Table II', BER-vs-JSR and compound tables (Exp 6a)"
    exp_manifest: list = field(default_factory=lambda: [
        (1, "Front-end model settling -> locked: theta=2, no LIF, A=Welch-512+drift / B=RF bank", "done"),
        (2, "Expert x JSR diagnosis: LSTM 0-20 fixes the range; ALE wins above ~10 dB, harms below (R3-4)", "done"),
        (3, "MoE-level BER: ALE-128 type-matched mu beats LSTM everywhere; MoE == oracle (R3-3)", "done"),
        (4, "Compound jamming: cascade == oracle; NB+SW exposed the broadband confusion -> override (R3-5)", "done"),
        (5, "KV260 energy measurement of the final MoE (R3-2)", "pending"),
        (6, "6a: final MoE + headline tables (THIS); 6b: switching, coded link, goodput/duty, fading", "THIS"),
    ])
    smoke: bool = False
    # ── protocol ──
    conditions: list = field(default_factory=lambda: ['fixed', 'random'])
    compounds: list = field(default_factory=lambda: [('narrowband', 'pulse'), ('sweep', 'pulse'), ('broadband', 'narrowband'), ('narrowband', 'sweep')])
    jsr_list: list = field(default_factory=lambda: [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
    compound_jsr_list: list = field(default_factory=lambda: [6, 10, 14])
    n_trials: int = 100
    n_compound_trials: int = 50
    n_clean_trials: int = 100
    rho_min: float = 0.25                   # override: fraction of received power the ALE removes on a 'none'/'broadband' verdict
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
    cfg.jsr_list = [0, 10]
    cfg.compound_jsr_list = [10]
    cfg.compounds = [('narrowband', 'pulse'), ('narrowband', 'sweep')]
    cfg.n_trials, cfg.n_compound_trials, cfg.n_clean_trials = 3, 3, 3
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
             'PulseBlankerExpert', 'apply_notch', 'apply_classical_blanker']
REF_ASSIGNS = ['JAM_FNS', 'JT', 'JT_ALL', 'JL', 'device']

def find_reference_notebook():
    path = FRAME_DIR / 'SNN_MoE_COMPLETE.ipynb'
    if not path.is_file():
        raise FileNotFoundError('Reference notebook missing; restore assets and run from the repository root or set UAV_MOE_FRAME_DIR')
    return path

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
    print(f"[PROGRESS] {sum(1 for _, _, s in cfg.exp_manifest if s == 'done')} done, 1 running, "
          f"{sum(1 for _, _, s in cfg.exp_manifest if s == 'pending')} remaining")
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
    fs = CC.fs; JT = list(R['JT']); snr_lin = 10 ** (cfg.snr_db / 10)
    probe = R['simulate_channel'](R['ChannelConfig'](snr_db=cfg.snr_db, jsr_db=10, n_symbols=min(cfg.n_sym_test, 2000), seed=1), jam_type='narrowband')['rx']
    e_fast = apply_ale(probe, cfg, cfg.ale_mu_nb); e_py = _ale_py(np.ascontiguousarray(probe), cfg.ale_taps, cfg.ale_delay, cfg.ale_mu_nb, 1e-6)
    if not np.allclose(e_fast, e_py, rtol=1e-9, atol=1e-9):
        raise RuntimeError('ALE numba path differs from python path')
    print('[GATE] ALE numba == python on a real frame')
    deviations = [
        "FINAL MoE = front-end B router + {ALE-128 mu_NB, ALE-128 mu_SW, calibrated blanker, pass-through}; no LSTM (ablation in Exp 3b)",
        f"predictability override on 'none'/'broadband' verdicts: ALE(mu_SW) removed-power fraction rho > {cfg.rho_min} => structured (sweep step)",
        "guided cascade (Exp 4): structured -> within-frame pulse test -> erasure; pulse -> blanker -> re-route survivors -> ALE",
        "hard MoE (Exp 3b, one expert per verdict, no override/cascade) and oracle routing kept as comparison rows",
        f"protocol: Table II seeds, {cfg.n_trials} trials; compounds {cfg.n_compound_trials} trials at JSR {cfg.compound_jsr_list}; both jammer-parameter protocols",
    ]
    print('[DEVIATIONS]'); [print(f'   - {d}') for d in deviations]

    r0 = R['simulate_channel'](R['ChannelConfig'](snr_db=cfg.snr_db, n_symbols=cfg.n_sym_test, seed=cfg.seed_base), jam_type='none')
    data_fp = hashlib.sha256(np.ascontiguousarray(r0['rx']).tobytes()).hexdigest()[:12]; sps = r0['sps']
    key = run_key(cfg, data_fp, ref_hash)
    EXP_DIR = PROJECT / 'experiments' / 'exp6a_final_moe'; RUN_ROOT = EXP_DIR / 'runs'
    RUN_DIR = RUN_ROOT / (('smoke_' if cfg.smoke else '') + key); FIG_DIR = RUN_DIR / 'figures'
    for d in (EXP_DIR, RUN_ROOT, RUN_DIR, FIG_DIR):
        d.mkdir(parents=True, exist_ok=True)
    PARTIAL = RUN_DIR / 'partial.json'
    print(f"[RUN] key={key}  data_fp={data_fp}  dir={RUN_DIR}")

    blanker = R['PulseBlankerExpert'](blank_threshold_z=cfg.blanker_z, group_size=cfg.blanker_group)
    cal = [R['simulate_channel'](R['ChannelConfig'](snr_db=cfg.snr_db, n_symbols=cfg.blanker_cal_samples // 10, seed=cfg.blanker_cal_seed_base + i), jam_type='none')['rx'][:cfg.blanker_cal_samples]
           for i in range(cfg.blanker_cal_n)]
    blanker.calibrate(cal, sps)
    routers = {}
    for cond in cfg.conditions:
        t0 = time.time(); routers[cond] = RouterB(R, cfg, cond, fs, sps)
        print(f"[ROUTER] {cond}: front-end B trained on {routers[cond].n_train} full frames (train acc {routers[cond].train_acc:.3f}) in {time.time() - t0:.0f}s")
    mu_of = {'narrowband': cfg.ale_mu_nb, 'sweep': cfg.ale_mu_sw}

    # ── the final MoE ──────────────────────────────────────────────────────
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

    def oracle_single(rx, bits, jt):
        if jt in mu_of:
            return ber_of(bits, soft_symbols(apply_ale(rx, cfg, mu_of[jt]), sps))
        if jt == 'pulse':
            soft, surv_idx, _, _ = blank_calibrated(rx); return ber_of(bits[surv_idx], soft[surv_idx])
        return ber_of(bits, soft_symbols(rx, sps))

    def oracle_compound(rx, bits, comp):
        a, b = comp
        if 'pulse' in comp:
            s_ = a if b == 'pulse' else b
            if s_ in mu_of:
                soft, surv_idx, _ = pulse_check(apply_ale(rx, cfg, mu_of[s_])); return ber_of(bits[surv_idx], soft[surv_idx])
            soft, surv_idx, _, _ = blank_calibrated(rx); return ber_of(bits[surv_idx], soft[surv_idx])
        if 'broadband' in comp:
            s_ = a if b == 'broadband' else b
            return ber_of(bits, soft_symbols(apply_ale(rx, cfg, mu_of[s_]), sps)) if s_ in mu_of else ber_of(bits, soft_symbols(rx, sps))
        return ber_of(bits, soft_symbols(apply_ale(rx, cfg, cfg.ale_mu_sw), sps))

    # ── resume ─────────────────────────────────────────────────────────────
    blob = {'per_frame': {}, 'units_done': []}
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
    done = set(blob['units_done'])

    units = [('single', cond, 'clean', None) for cond in cfg.conditions] + \
            [('single', cond, jt, jsr) for cond in cfg.conditions for jsr in cfg.jsr_list for jt in JT] + \
            [('compound', cond, '+'.join(c[:2].upper() for c in comp), jsr) for cond in cfg.conditions for comp in cfg.compounds for jsr in cfg.compound_jsr_list]
    print(f"[WORKLOAD] {len(units)} units ({len(cfg.conditions) * (1 + len(cfg.jsr_list) * len(JT))} single x {cfg.n_trials}, "
          f"{len(cfg.conditions) * len(cfg.compounds) * len(cfg.compound_jsr_list)} compound x {cfg.n_compound_trials}) -> est ~{(len(units) * 90 * 0.12) / 60:.0f} min")
    first_dt = None
    pbar = tqdm(units, desc='units', disable=len(units) < 2)
    for ui, (kind, cond, name, jsr) in enumerate(pbar):
        uid = f'{kind}|{cond}|{name}|{jsr}'
        if uid in done:
            continue
        t_unit = time.time(); pbar.set_postfix_str(uid); sim = make_sim(R, cfg, cond)
        frames = []
        if kind == 'single':
            n_tr = cfg.n_clean_trials if name == 'clean' else cfg.n_trials
            for t in range(n_tr):
                if name == 'clean':
                    frames.append(sim(R['ChannelConfig'](snr_db=cfg.snr_db, n_symbols=cfg.n_sym_test, seed=cfg.clean_seed_base + t), jam_type='none'))
                else:
                    frames.append(sim(R['ChannelConfig'](snr_db=cfg.snr_db, jsr_db=jsr, n_symbols=cfg.n_sym_test, seed=cfg.seed_base + t + JT.index(name) * 2000), jam_type=name))
        else:
            comp = next(c for c in cfg.compounds if '+'.join(x[:2].upper() for x in c) == name)
            frames = [compound_frame(R, cfg, cond, comp, jsr, cfg.seed_base + t + cfg.compounds.index(comp) * 2000) for t in range(cfg.n_compound_trials)]
        pred, pmax = routers[cond].predict(frontend_B(R, cfg, [f['rx'] for f in frames], fs))
        recs = []
        for k, f in enumerate(frames):
            rx, bits = f['rx'], f['bits']; c1 = str(pred[k])
            rec = {'pred1': c1, 'pmax': float(pmax[k]), 'none': ber_of(bits, soft_symbols(rx, sps))}
            rec['moe_hard'], _, _ = final_moe(rx, bits, c1, cond, use_override=False, use_cascade=False)
            rec['moe_final'], rec['surv_final'], rec['path'] = final_moe(rx, bits, c1, cond)
            if kind == 'single':
                rec['oracle'] = oracle_single(rx, bits, 'none' if name == 'clean' else name)
            else:
                rec['oracle'] = oracle_compound(rx, bits, comp)
            recs.append(rec)
        blob['per_frame'][uid] = recs; done.add(uid); blob['units_done'] = sorted(done); write_json_atomic(blob, PARTIAL)
        dt = time.time() - t_unit
        paths = {}
        for r in recs:
            paths[r['path']] = paths.get(r['path'], 0) + 1
        top = max(paths, key=paths.get)
        pbar.write(f"  unit {ui+1}/{len(units)} {uid:<32}: none={np.mean([r['none'] for r in recs]):.4f}  hard={np.mean([r['moe_hard'] for r in recs]):.4f}  "
                   f"final={np.mean([r['moe_final'] for r in recs]):.4f}  oracle={np.mean([r['oracle'] for r in recs]):.4f}  main path {top} ({100 * paths[top] / len(recs):.0f}%)  [{dt:.0f}s]")
        if first_dt is None:
            first_dt = dt; print(f"    [ETA] ~{first_dt * (len(units) - ui - 1) / 60:.0f} min remaining")
    pbar.close()

    # ── aggregate ──────────────────────────────────────────────────────────
    agg = {}
    for uid, recs in blob['per_frame'].items():
        for m in ['none', 'moe_hard', 'moe_final', 'oracle']:
            v = np.array([r[m] for r in recs]); n = len(v)
            agg[f'{uid}|{m}'] = {'mean': float(v.mean()), 'ci95': float(1.96 * v.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0, 'n': int(n), 'mean_first20': float(v[:20].mean())}
        kind, cond, name, jsr = uid.split('|')
        truth = 'none' if name == 'clean' else name
        agg[f'{uid}|meta'] = {'routing_acc': float(np.mean([r['pred1'] == truth for r in recs])) if kind == 'single' else None,
                              'override_rate': float(np.mean(['override' in r['path'] for r in recs])),
                              'erasure_rate': float(np.mean(['blank' in r['path'] for r in recs])),
                              'survival': float(np.mean([r['surv_final'] for r in recs])),
                              'paths': {p: sum(1 for r in recs if r['path'] == p) for p in set(r['path'] for r in recs)}}
    repro = {'pass': None, 'note': 'skipped in smoke'}
    if not cfg.smoke and 'fixed' in cfg.conditions and 10 in cfg.jsr_list:
        g = lambda jt: agg[f'single|fixed|{jt}|10|none']['mean_first20']
        checks = {'NB': (g('narrowband'), 0.0433), 'SW': (g('sweep'), 0.0500), 'BB': (g('broadband'), 0.1718), 'PL': (g('pulse'), 0.0675)}
        ok = all(abs(v - ref) <= 1e-4 for v, ref in checks.values())
        print('\n' + ('[REPRO] PASS' if ok else '[REPRO] !!! FAIL !!!') + ' — no-correction BER, fixed, JSR 10, first 20 trials: ' + '; '.join(f"{k} {v:.4f} (paper {ref})" for k, (v, ref) in checks.items()))
        repro = {'pass': ok, **{k: v[0] for k, v in checks.items()}}

    # ── tables ─────────────────────────────────────────────────────────────
    print('\n' + '=' * 100); print(f"TABLE II' — BER @ JSR 10 dB, {cfg.n_trials} trials (pulse: BER on surviving symbols)"); print('=' * 100)
    print(f"{'cond':<7}{'type':<11}{'none':>10}{'hard MoE':>10}{'final MoE':>11}{'oracle':>10}{'route acc':>11}{'override':>10}{'erasure':>9}{'survival':>10}")
    for cond in cfg.conditions:
        for jt in JT + ['clean']:
            uid = f'single|{cond}|{jt}|{10 if jt != "clean" else None}'; m = agg[f'{uid}|meta']
            print(f"{cond:<7}{jt:<11}{agg[f'{uid}|none']['mean']:>10.4f}{agg[f'{uid}|moe_hard']['mean']:>10.4f}{agg[f'{uid}|moe_final']['mean']:>11.4f}{agg[f'{uid}|oracle']['mean']:>10.4f}"
                  f"{m['routing_acc']:>11.2f}{m['override_rate']:>10.2f}{m['erasure_rate']:>9.2f}{m['survival']:>10.2f}")
    print('\nMEAN BER OVER JSR 0..20 (single jammers)')
    print(f"{'cond':<7}{'type':<11}{'none':>10}{'hard MoE':>10}{'final MoE':>11}{'oracle':>10}{'final/oracle':>13}")
    for cond in cfg.conditions:
        for jt in JT:
            mean = lambda m: np.mean([agg[f'single|{cond}|{jt}|{jsr}|{m}']['mean'] for jsr in cfg.jsr_list])
            print(f"{cond:<7}{jt:<11}{mean('none'):>10.5f}{mean('moe_hard'):>10.5f}{mean('moe_final'):>11.5f}{mean('oracle'):>10.5f}{(mean('moe_final') / max(mean('oracle'), 1e-9)):>13.2f}")
    print('\nCOMPOUND JAMMING (components at equal JSR)')
    print(f"{'cond':<7}{'compound':<9}{'JSR':>4}{'none':>10}{'hard MoE':>10}{'final MoE':>11}{'oracle':>10}{'override':>10}{'erasure':>9} | main path")
    for cond in cfg.conditions:
        for comp in cfg.compounds:
            cname = '+'.join(c[:2].upper() for c in comp)
            for jsr in cfg.compound_jsr_list:
                uid = f'compound|{cond}|{cname}|{jsr}'; m = agg[f'{uid}|meta']; top = max(m['paths'], key=m['paths'].get)
                print(f"{cond:<7}{cname:<9}{jsr:>4}{agg[f'{uid}|none']['mean']:>10.4f}{agg[f'{uid}|moe_hard']['mean']:>10.4f}{agg[f'{uid}|moe_final']['mean']:>11.4f}{agg[f'{uid}|oracle']['mean']:>10.4f}"
                      f"{m['override_rate']:>10.2f}{m['erasure_rate']:>9.2f} | {top} ({100 * m['paths'][top] / sum(m['paths'].values()):.0f}%)")
    print('\nCLEAN FRAMES: final MoE BER / override rate / erasure rate — must be ~0 / ~0 / ~0')
    for cond in cfg.conditions:
        uid = f'single|{cond}|clean|None'; m = agg[f'{uid}|meta']
        print(f"  {cond:<6}: BER {agg[f'{uid}|moe_final']['mean']:.5f}  override {m['override_rate']:.3f}  erasure {m['erasure_rate']:.3f}  routing acc {m['routing_acc']:.2f}")

    # ── figure: BER vs JSR, final MoE vs none vs oracle, 4 types x 2 protocols ────────────────
    fig_paths, fam = [], None
    try:
        fam = apply_style()
        fig, axes = plt.subplots(len(JT), len(cfg.conditions), figsize=(IEEE_2COL, 1.75 * len(JT) + 0.6), sharex=True, sharey=True, squeeze=False)
        series = [('none', 'no correction'), ('moe_hard', 'MoE, hard routing (Exp 3b)'), ('moe_final', 'final MoE (override + cascade)'), ('oracle', 'oracle expert')]
        for ji, jt in enumerate(JT):
            for ci, cond in enumerate(cfg.conditions):
                ax = axes[ji, ci]
                for si, (m, lab) in enumerate(series):
                    y = np.array([agg[f'single|{cond}|{jt}|{jsr}|{m}']['mean'] for jsr in cfg.jsr_list]); c = np.array([agg[f'single|{cond}|{jt}|{jsr}|{m}']['ci95'] for jsr in cfg.jsr_list])
                    y = np.maximum(y, 1e-5)
                    ax.semilogy(cfg.jsr_list, y, color=SERIES[si], linestyle=STYLES[si], marker=MARKERS[si], label=lab)
                    ax.fill_between(cfg.jsr_list, np.maximum(y - c, 1e-5), y + c, color=SERIES[si], alpha=0.12, linewidth=0)
                ax.set_title(f'({chr(97 + ji * len(cfg.conditions) + ci)}) {jt}, {cond} parameters', loc='left')
                if ji == len(JT) - 1:
                    ax.set_xlabel('JSR (dB)')
                if ci == 0:
                    ax.set_ylabel('BER' + (' (surviving symbols)' if jt == 'pulse' else ''))
        h, l = axes[0, 0].get_legend_handles_labels(); fig.legend(h, l, loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.0))
        if cfg.smoke:
            fig.suptitle('SMOKE — every number here is garbage', color=PAL['crimson'], fontsize=IEEE_PT + 1)
        fig.tight_layout(rect=[0, 0.05, 1, 1]); pa = FIG_DIR / 'figR1_final_moe_ber_vs_jsr'
        fig.savefig(pa.with_suffix('.pdf')); fig.savefig(pa.with_suffix('.png'), dpi=IEEE_DPI); plt.show(); plt.close(fig); grey_proof(pa.with_suffix('.png'))
        fig_paths = [str(pa.with_suffix('.pdf').relative_to(PROJECT))]
    except Exception as e:
        print(f"[FIG] figure generation failed ({e.__class__.__name__}: {e}); results are saved, figures are not")
    summary = {'run_id': RUN_ID, 'run_key': key, 'run_version': RUN_VERSION, 'analysis_version': ANALYSIS_VERSION,
               'exp_index': cfg.exp_index, 'exp_total': cfg.exp_total, 'exp_name': cfg.exp_name, 'exp_manifest': cfg.exp_manifest,
               'smoke': cfg.smoke, 'config': asdict(cfg), 'reference': {'notebook': 'simulation/frame_receiver/SNN_MoE_COMPLETE.ipynb', 'sha256_16': ref_hash, 'imports_skipped': skipped},
               'deviations': deviations, 'data_fingerprint': data_fp, 'repro_gate': repro, 'agg': agg,
               'routers': {c: {'n_train': routers[c].n_train, 'train_acc': routers[c].train_acc} for c in routers},
               'figures': fig_paths, 'figure_font': fam, 'numba': NUMBA_OK, 'elapsed_min': (time.time() - t_start) / 60, 'timestamp': datetime.now().isoformat()}
    write_json_atomic(summary, RUN_DIR / 'summary.json'); write_json_atomic(summary, EXP_DIR / 'latest_summary.json')
    with open(RUN_DIR / 'final_moe_ber.csv', 'w') as f:
        f.write('kind,cond,name,jsr,method,mean,ci95,n\n')
        for k_, a in agg.items():
            if 'mean' in a:
                kind, cond, name, jsr, m = k_.split('|'); f.write(f"{kind},{cond},{name},{jsr},{m},{a['mean']},{a['ci95']},{a['n']}\n")
    print(f"\n[DONE] {RUN_DIR / 'summary.json'}; elapsed {summary['elapsed_min']:.1f} min")
    return summary

summary = main()
