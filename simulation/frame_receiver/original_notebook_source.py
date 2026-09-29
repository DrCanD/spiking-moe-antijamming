# ============================================================
# Setup
# ============================================================
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "torch", "numpy", "matplotlib", "scipy", "scikit-learn"])

import numpy as np
import subprocess, sys
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'reedsolo', '-q'])

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
import matplotlib.pyplot as plt
from scipy.signal import welch
from sklearn.metrics import confusion_matrix, classification_report
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict
import time
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 11, 'axes.labelsize': 12,
    'axes.titlesize': 13, 'legend.fontsize': 9, 'figure.dpi': 120,
    'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'axes.grid': True, 'grid.alpha': 0.3,
})

# Project output directories
import os
DRIVE_ROOT = Path(os.environ.get('UAV_MOE_OUTPUT_DIR', 'runs/original_receiver'))
DRIVE_ROOT.mkdir(parents=True, exist_ok=True)
DRIVE_FIGS   = DRIVE_ROOT / 'figures'
DRIVE_TABLES = DRIVE_ROOT / 'tables'
DRIVE_MODELS = DRIVE_ROOT / 'models'
DRIVE_NB     = DRIVE_ROOT / 'notebooks'
for d in [DRIVE_FIGS, DRIVE_TABLES, DRIVE_MODELS, DRIVE_NB]:
    d.mkdir(exist_ok=True)

OUT = Path('results_v3')
OUT.mkdir(exist_ok=True)
print(f"Project outputs: {DRIVE_ROOT}")

JT = ['broadband', 'narrowband', 'sweep', 'pulse']
JT_ALL = ['none'] + JT  # 5 classes for classifier
JL = {'none':'Clean', 'broadband':'Broadband', 'narrowband':'Narrowband',
      'sweep':'Sweep', 'pulse':'Pulse'}
JC = {'none':'#888780', 'broadband':'#534AB7', 'narrowband':'#1D9E75',
      'sweep':'#D85A30', 'pulse':'#D4537E'}

print("Setup complete.")

# ============================================================
# Channel Model
# ============================================================
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

print("Channel model ready.")

# ============================================================
# SNN Watchdog + Classifier
# ============================================================
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

@dataclass
class SpikeEncoderConfig:
    threshold: float = 0.04
    refractory: int = 2

@dataclass
class WatchdogConfig:
    n_neurons: int = 64
    tau_m: float = 10.0
    v_th: float = 1.0
    v_reset: float = 0.0
    refr_period: int = 3
    anomaly_z: float = 2.5
    n_subframes: int = 20  # increased for better pulse localization

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

class LIFLayer:
    def __init__(self, n_in, n_out, cfg):
        self.cfg = cfg
        rng = np.random.default_rng(42)
        self.W = rng.normal(0, 0.3, (n_in, n_out))
        self.n_out = n_out

    def run(self, input_2ch, n_steps):
        cfg = self.cfg
        decay = np.exp(-1.0 / cfg.tau_m)
        v = np.zeros(self.n_out)
        last_sp = np.full(self.n_out, -cfg.refr_period - 1)
        total = 0
        for t in range(n_steps):
            v *= decay
            inp = input_2ch[t]
            if inp[0] != 0 or inp[1] != 0:
                v += inp @ self.W
            refr_ok = (t - last_sp) > cfg.refr_period
            fire = (v >= cfg.v_th) & refr_ok
            total += np.sum(fire)
            v[fire] = cfg.v_reset
            last_sp[fire] = t
        return total / (n_steps * self.n_out)


class SNNWatchdogClassifier:
    """SNN-based anomaly detector AND jamming classifier.

    Features extracted (7 total):
      - Input firing rate
      - LIF output firing rate
      - ISI coefficient of variation
      - Fano factor (burstiness)
      - HF spectral ratio
      - Spike rate max/min ratio across subframes
      - Spectral peak sharpness (narrowband indicator)
    """
    def __init__(self, enc_cfg=None, wdg_cfg=None):
        self.enc_cfg = enc_cfg or SpikeEncoderConfig()
        self.cfg = wdg_cfg or WatchdogConfig()
        self.encoder = DeltaEncoder(self.enc_cfg)
        self.lif = LIFLayer(2, self.cfg.n_neurons, self.cfg)
        self.baseline_mean = None
        self.baseline_std = None
        self.classifier = None
        self.scaler = StandardScaler()
        self.is_calibrated = False

    def extract_features(self, signal):
        """Extract 7 spike-domain features from a signal frame."""
        spikes, pos, neg, fr_in = self.encoder.encode(signal)
        n = len(signal)

        # LIF output
        inp = np.zeros((n, 2))
        inp[spikes == 1, 0] = 1.0
        inp[spikes == -1, 1] = 1.0
        fr_out = self.lif.run(inp, n)

        # ISI
        all_sp = np.sort(np.concatenate([pos, neg]))
        isi_cv = 0.0
        if len(all_sp) > 2:
            isis = np.diff(all_sp).astype(float)
            isi_cv = np.std(isis) / (np.mean(isis) + 1e-10)

        # sub-frame analysis
        nsf = self.cfg.n_subframes
        sf_len = n // nsf
        fano = 0.0
        sf_max_min_ratio = 1.0
        if sf_len > 0 and len(all_sp) > 0:
            counts = np.array([np.sum((all_sp >= k*sf_len) & (all_sp < (k+1)*sf_len))
                               for k in range(nsf)])
            mu = np.mean(counts)
            fano = np.var(counts) / (mu + 1e-10)
            cmin = np.min(counts)
            cmax = np.max(counts)
            sf_max_min_ratio = cmax / (cmin + 1.0)

        # spectral features
        hf_ratio = 0.5
        spectral_peak = 0.0
        if n > 128:
            ft = np.abs(np.fft.rfft(signal))
            mid = len(ft) // 2
            hf_ratio = np.sum(ft[mid:]**2) / (np.sum(ft**2) + 1e-10)
            # spectral peak sharpness: max peak / mean (narrowband = high)
            spectral_peak = np.max(ft[1:]) / (np.mean(ft[1:]) + 1e-10)

        return np.array([fr_in, fr_out, isi_cv, fano, hf_ratio,
                         sf_max_min_ratio, spectral_peak])

    def calibrate(self, clean_signals):
        """Learn baseline from clean signals."""
        feats = np.array([self.extract_features(s) for s in clean_signals])
        self.baseline_mean = np.mean(feats, axis=0)
        self.baseline_std = np.maximum(np.std(feats, axis=0), 1e-6)
        self.is_calibrated = True

    def train_classifier(self, signals, labels):
        """Train RF classifier on labeled signal frames.

        Args:
            signals: list of signal arrays
            labels: list of jam type strings ('none','broadband',...)
        """
        feats = np.array([self.extract_features(s) for s in signals])
        self.scaler.fit(feats)
        X = self.scaler.transform(feats)

        self.classifier = RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        self.classifier.fit(X, labels)

        # in-sample accuracy
        acc = self.classifier.score(X, labels)
        print(f"  Classifier trained: {len(signals)} samples, "
              f"in-sample acc: {acc:.3f}")
        return acc

    def detect_and_classify(self, signal):
        """Full pipeline: detect anomaly + classify type.

        Returns:
            is_anomaly: bool
            anomaly_score: float
            predicted_class: str ('none','broadband',...)
            class_proba: dict of class probabilities
            features: raw feature vector
        """
        feat = self.extract_features(signal)

        # detection via z-score
        z = np.abs(feat[:5] - self.baseline_mean[:5]) / self.baseline_std[:5]
        score = np.max(z)
        is_anomaly = score > self.cfg.anomaly_z

        # classification
        if self.classifier is not None:
            X = self.scaler.transform(feat.reshape(1, -1))
            pred = self.classifier.predict(X)[0]
            proba = dict(zip(self.classifier.classes_,
                             self.classifier.predict_proba(X)[0]))
        else:
            pred = 'none' if not is_anomaly else 'broadband'
            proba = {}

        return is_anomaly, score, pred, proba, feat

    def get_subframe_spike_rates(self, signal):
        """Get per-subframe spike rates for pulse blanker."""
        spikes, pos, neg, _ = self.encoder.encode(signal)
        n = len(signal)
        nsf = self.cfg.n_subframes
        sf_len = n // nsf
        all_sp = np.sort(np.concatenate([pos, neg]))

        rates = np.zeros(nsf)
        for k in range(nsf):
            lo, hi = k * sf_len, (k+1) * sf_len
            rates[k] = np.sum((all_sp >= lo) & (all_sp < hi)) / sf_len
        return rates, sf_len

    def power_mw(self):
        n_syn = 2 * self.cfg.n_neurons
        ops_s = 0.05 * n_syn * 1e5
        # RF classifier adds negligible compute (~100 comparisons)
        return ops_s * 20e-12 * 1e3 + 0.001  # +1 uW for RF

print(f"SNN Watchdog+Classifier ready: 64 LIF neurons, 7 features, RF classifier.")

# ============================================================
# Expert 1: Residual LSTM Denoiser (for structured jamming)
# ============================================================

class ResidualLSTMNet(nn.Module):
    def __init__(self, hidden=24, n_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden, 1)
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        noise_est, _ = self.lstm(x)
        noise_est = self.fc(noise_est)
        return x - noise_est


class LSTMExpert:
    def __init__(self, hidden=24, seq_len=200, lr=0.002, epochs=200, batch_size=128):
        self.hidden = hidden
        self.seq_len = seq_len
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.losses = []
        self.val_losses = []
        self.in_mean = 0.0
        self.in_std = 1.0

    def _to_symbols(self, signal, sps):
        ns = len(signal) // sps
        return np.mean(signal[:ns*sps].reshape(ns, sps), axis=1)

    def _make_seqs(self, jammed_list, clean_list, sps):
        X, Y = [], []
        stride = self.seq_len // 2
        for j, c in zip(jammed_list, clean_list):
            js = self._to_symbols(j, sps)
            cs = self._to_symbols(c, sps)
            n = min(len(js), len(cs))
            for s in range(0, n - self.seq_len, stride):
                X.append(js[s:s+self.seq_len])
                Y.append(cs[s:s+self.seq_len])
        return np.array(X), np.array(Y)

    def train_model(self, jammed_list, clean_list, sps, verbose=True):
        X, Y = self._make_seqs(jammed_list, clean_list, sps)
        self.in_mean, self.in_std = np.mean(X), max(np.std(X), 1e-6)
        Xn = (X - self.in_mean) / self.in_std
        Yn = (Y - self.in_mean) / self.in_std

        n = len(X)
        split = int(0.85 * n)
        idx = np.random.default_rng(42).permutation(n)

        Xt = torch.FloatTensor(Xn[idx[:split]]).unsqueeze(-1).to(device)
        Yt = torch.FloatTensor(Yn[idx[:split]]).unsqueeze(-1).to(device)
        Xv = torch.FloatTensor(Xn[idx[split:]]).unsqueeze(-1).to(device)
        Yv = torch.FloatTensor(Yn[idx[split:]]).unsqueeze(-1).to(device)

        self.model = ResidualLSTMNet(self.hidden).to(device)
        opt = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
        crit = nn.MSELoss()

        self.losses, self.val_losses = [], []
        best_val, best_state = float('inf'), None

        dl = DataLoader(TensorDataset(Xt, Yt), batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.epochs):
            self.model.train()
            el = 0.0
            for xb, yb in dl:
                pred = self.model(xb)
                loss = crit(pred, yb)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 3.0)
                opt.step()
                el += loss.item() * len(xb)

            tl = el / len(Xt)
            self.losses.append(tl)

            self.model.eval()
            with torch.no_grad():
                vl = crit(self.model(Xv), Yv).item()
            self.val_losses.append(vl)
            sched.step()

            if vl < best_val:
                best_val = vl
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

            if verbose and (epoch+1) % 50 == 0:
                print(f"    Epoch {epoch+1:3d}/{self.epochs} | "
                      f"T: {tl:.6f} V: {vl:.6f}")

        if best_state:
            self.model.load_state_dict(best_state)
            self.model.to(device)

        n_p = sum(p.numel() for p in self.model.parameters())
        if verbose:
            print(f"    Best val: {best_val:.6f} | {n_p:,} params")

    def correct_symbols(self, jammed_signal, sps):
        if self.model is None:
            return self._to_symbols(jammed_signal, sps)
        self.model.eval()
        js = self._to_symbols(jammed_signal, sps)
        n = len(js)
        corrected = np.zeros(n)
        counts = np.zeros(n)
        stride = self.seq_len // 2

        with torch.no_grad():
            wins, starts = [], []
            for s in range(0, n - self.seq_len + 1, stride):
                wins.append((js[s:s+self.seq_len] - self.in_mean) / self.in_std)
                starts.append(s)
            if not wins:
                return js
            batch = torch.FloatTensor(np.array(wins)).unsqueeze(-1).to(device)
            out = self.model(batch).cpu().numpy().squeeze(-1)
            out = out * self.in_std + self.in_mean
            for k, s in enumerate(starts):
                corrected[s:s+self.seq_len] += out[k]
                counts[s:s+self.seq_len] += 1

        mask = counts > 0
        corrected[mask] /= counts[mask]
        corrected[~mask] = js[~mask]
        return corrected

    def power_mw(self):
        if self.model is None: return 0
        n_p = sum(p.numel() for p in self.model.parameters())
        return n_p * 4 * 1e5 * 10e-12 * 1e3

print("Expert 1 (LSTM) ready.")

# ============================================================
# Expert 2: SNN Pulse Blanker (v3.3 - energy-based localization)
# ============================================================

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

print("Expert 2 (Pulse Blanker v3.3 - energy-based) ready.")

# ============================================================
# Expert 3: MMSE Estimator + Confidence
# ============================================================

class MMSEExpert:
    """Adaptive MMSE estimator for broadband Gaussian jamming.

    For BPSK + AWGN + broadband Gaussian jamming, the matched filter
    output is already the optimal linear estimator. We cannot improve
    the hard decision BER.

    But we CAN provide:
    1. Per-symbol soft reliability (confidence) score
    2. JSR estimate from SNN anomaly score
    3. Adaptive MMSE scaling for soft-output applications

    The confidence output is the real value here: it tells the upper
    layer which symbols are trustworthy and which need retransmission.
    """
    def __init__(self):
        self.jsr_calibration = None  # mapping: anomaly_score -> JSR estimate

    def calibrate_jsr_mapping(self, anomaly_scores, true_jsrs):
        """Learn mapping from SNN anomaly score to JSR estimate."""
        # simple linear regression
        from numpy.polynomial import polynomial as P
        self.jsr_calibration = P.polyfit(anomaly_scores, true_jsrs, deg=2)

    def estimate_jsr(self, anomaly_score):
        """Estimate JSR from anomaly score."""
        if self.jsr_calibration is None:
            return 10.0  # default
        from numpy.polynomial import polynomial as P
        return max(0, P.polyval(anomaly_score, self.jsr_calibration))

    def correct_symbols(self, jammed_signal, sps, snr_db, anomaly_score):
        """Apply MMSE scaling and compute per-symbol confidence."""
        js = np.mean(jammed_signal[:len(jammed_signal)//sps*sps].reshape(-1, sps), axis=1)

        # estimate effective SNR
        jsr_est = self.estimate_jsr(anomaly_score)
        # effective SNR = signal / (noise + jamming)
        snr_lin = 10**(snr_db/10)
        jsr_lin = 10**(jsr_est/10)
        eff_snr = snr_lin / (1 + jsr_lin * snr_lin / snr_lin)  # simplified

        # MMSE scaling: shrink soft decisions toward zero proportional to noise
        mmse_gain = eff_snr / (eff_snr + 1)
        corrected = mmse_gain * js

        # per-symbol confidence: based on distance from decision boundary
        # symbols far from 0 are more reliable
        confidence = np.abs(js) * np.sqrt(eff_snr) / (1 + np.sqrt(eff_snr))
        confidence = np.clip(confidence, 0, 1)

        mean_confidence = np.mean(confidence)

        return corrected, confidence, mean_confidence, jsr_est

    def power_mw(self):
        return 0.002  # analytical computation, no NN

print("Expert 3 (MMSE + Confidence) ready.")

# ============================================================
# MoE Anti-Jamming System (v3.3)
# ============================================================

class MoEAntiJamming:
    def __init__(self, snn, lstm_expert, pulse_expert, mmse_expert):
        self.snn = snn
        self.lstm_expert = lstm_expert
        self.pulse_expert = pulse_expert
        self.mmse_expert = mmse_expert

    def process(self, signal, sps, snr_db=10.0):
        is_anom, score, pred_class, proba, feat = self.snn.detect_and_classify(signal)
        ns = len(signal) // sps

        result = {
            'is_anomaly': is_anom, 'anomaly_score': score,
            'predicted_class': pred_class, 'class_proba': proba,
            'features': feat, 'expert_used': pred_class,
        }

        if pred_class == 'none' or not is_anom:
            soft = np.mean(signal[:ns*sps].reshape(ns, sps), axis=1)
            result['corrected_symbols'] = soft
            result['bits'] = (soft > 0).astype(int)
            result['confidence'] = np.ones(ns)
            result['mean_confidence'] = 1.0
            result['action'] = 'none'
            result['ber_bits'] = result['bits']
            result['ber_indices'] = np.arange(ns)

        elif pred_class in ('narrowband', 'sweep'):
            corrected = self.lstm_expert.correct_symbols(signal, sps)
            result['corrected_symbols'] = corrected
            result['bits'] = (corrected > 0).astype(int)
            result['confidence'] = np.minimum(np.abs(corrected) * 2, 1.0)
            result['mean_confidence'] = np.mean(result['confidence'])
            result['action'] = 'lstm_denoise'
            result['ber_bits'] = result['bits']
            result['ber_indices'] = np.arange(len(corrected))

        elif pred_class == 'pulse':
            pr = self.pulse_expert.correct_symbols(signal, sps)
            result['corrected_symbols'] = pr['soft']
            result['bits'] = (pr['soft'] > 0).astype(int)
            result['erasure_mask'] = pr['erasure_mask']
            result['survival_rate'] = pr['survival_rate']
            result['pulse_rms'] = pr['rms_values']
            result['pulse_grp_mask'] = pr['grp_mask']
            result['ber_bits'] = pr['surviving_bits']
            result['ber_indices'] = pr['surviving_indices']
            conf = np.ones(ns)
            conf[pr['erasure_mask'][:ns]] = 0.0
            result['confidence'] = conf
            result['mean_confidence'] = pr['survival_rate']
            result['action'] = 'erase_and_survive'

        elif pred_class == 'broadband':
            corrected, conf, mean_conf, jsr_est = self.mmse_expert.correct_symbols(
                signal, sps, snr_db, score)
            result['corrected_symbols'] = corrected
            result['bits'] = (corrected > 0).astype(int)
            result['confidence'] = conf
            result['mean_confidence'] = mean_conf
            result['jsr_estimate'] = jsr_est
            result['action'] = 'retransmit_request' if mean_conf < 0.5 else 'mmse_correct'
            result['ber_bits'] = result['bits']
            result['ber_indices'] = np.arange(len(corrected))

        return result

    def total_power_mw(self, active_expert='lstm'):
        snn_p = self.snn.power_mw()
        if active_expert == 'lstm': return snn_p + self.lstm_expert.power_mw()
        elif active_expert == 'pulse': return snn_p + self.pulse_expert.power_mw()
        elif active_expert == 'mmse': return snn_p + self.mmse_expert.power_mw()
        return snn_p

print("MoE (v3.3) ready.")

# ============================================================
# Metrics
# ============================================================
def compute_snr_improvement(clean, jammed, corrected):
    nb = jammed - clean
    na = corrected - clean
    snr_b = 10*np.log10(np.mean(clean**2)/(np.mean(nb**2)+1e-15))
    snr_a = 10*np.log10(np.mean(clean**2)/(np.mean(na**2)+1e-15))
    return snr_b, snr_a, snr_a - snr_b

def evaluate_detection(preds, gt):
    tp = sum(p and g for p,g in zip(preds, gt))
    fp = sum(p and not g for p,g in zip(preds, gt))
    fn = sum(not p and g for p,g in zip(preds, gt))
    tn = sum(not p and not g for p,g in zip(preds, gt))
    tpr = tp/(tp+fn) if (tp+fn)>0 else 0
    fpr = fp/(fp+tn) if (fp+tn)>0 else 0
    prec = tp/(tp+fp) if (tp+fp)>0 else 0
    f1 = 2*prec*tpr/(prec+tpr) if (prec+tpr)>0 else 0
    return {'tpr':tpr,'fpr':fpr,'prec':prec,'f1':f1}

print("Metrics ready.")

# ============================================================
# Phase 1: SNN Calibration + Classifier Training
# ============================================================
t0 = time.time()
print("="*70)
print("PHASE 1: SNN Calibration + Classifier Training")
print("="*70)

frame_size = 10000
sps_ref = 10
jsr_levels = [4, 8, 12, 16, 20]

snn = SNNWatchdogClassifier(
    SpikeEncoderConfig(threshold=0.04, refractory=2),
    WatchdogConfig(n_neurons=64, anomaly_z=2.5, tau_m=8.0, n_subframes=20))

# calibrate on clean
print("\nCalibrating on clean signals...")
cal_sigs = [simulate_channel(ChannelConfig(snr_db=10, n_symbols=frame_size//10, seed=500+i),
            jam_type='none')['rx'][:frame_size] for i in range(30)]
snn.calibrate(cal_sigs)

# build classification dataset
print("Building classification dataset...")
cls_signals, cls_labels = [], []

# clean
for i in range(50):
    r = simulate_channel(ChannelConfig(snr_db=10, n_symbols=frame_size//10, seed=700+i),
                          jam_type='none')
    cls_signals.append(r['rx'][:frame_size])
    cls_labels.append('none')

# each jam type at multiple JSR levels
for jt in JT:
    for jsr in jsr_levels:
        for i in range(10):
            seed = 1000 + i + JT.index(jt)*500 + jsr_levels.index(jsr)*50
            r = simulate_channel(ChannelConfig(snr_db=10, jsr_db=jsr,
                                  n_symbols=frame_size//10, seed=seed), jam_type=jt)
            cls_signals.append(r['rx'][:frame_size])
            cls_labels.append(jt)

print(f"  Dataset: {len(cls_signals)} frames "
      f"({cls_labels.count('none')} clean + "
      f"{sum(cls_labels.count(j) for j in JT)} jammed)")

# train classifier
snn.train_classifier(cls_signals, cls_labels)

print(f"Phase 1 done in {time.time()-t0:.1f}s")

# ============================================================
# Phase 2: LSTM Expert Training (NB + Sweep only)
# ============================================================
t0 = time.time()
print("\n" + "="*70)
print("PHASE 2: LSTM Expert Training")
print("="*70)

n_sym_train = 10000
n_per_jsr = 10

train_jammed, train_clean = [], []
for jt in ['narrowband', 'sweep']:
    for jsr in [4, 8, 12, 16]:
        for i in range(n_per_jsr):
            seed = 4000 + i + ['narrowband','sweep'].index(jt)*500 + [4,8,12,16].index(jsr)*50
            rj = simulate_channel(ChannelConfig(snr_db=10, jsr_db=jsr,
                                   n_symbols=n_sym_train, seed=seed), jam_type=jt)
            rc = simulate_channel(ChannelConfig(snr_db=10, n_symbols=n_sym_train,
                                   seed=seed), jam_type='none')
            train_jammed.append(rj['rx'])
            train_clean.append(rc['tx'])

print(f"  Training data: {len(train_jammed)} pairs (NB + Sweep, JSR 4-16 dB)")

lstm_expert = LSTMExpert(hidden=24, seq_len=200, lr=0.002, epochs=200, batch_size=128)
lstm_expert.train_model(train_jammed, train_clean, sps_ref, verbose=True)

print(f"Phase 2 done in {time.time()-t0:.1f}s")

# ============================================================
# Phase 3: Pulse Blanker + MMSE Calibration (v3.3)
# ============================================================
print("\n" + "="*70)
print("PHASE 3: Pulse Blanker + MMSE Calibration")
print("="*70)

pulse_expert = PulseBlankerExpert(blank_threshold_z=3.0, group_size=5)

# calibrate on clean signals (energy-based, no encoder needed)
clean_sigs_pb = [simulate_channel(ChannelConfig(snr_db=10, n_symbols=frame_size//10, seed=600+i),
                  jam_type='none')['rx'][:frame_size] for i in range(30)]
pulse_expert.calibrate(clean_sigs_pb, sps_ref)

# sanity check: compare clean vs pulse-jammed RMS
r_clean = simulate_channel(ChannelConfig(snr_db=10, n_symbols=frame_size//10, seed=9998),
                            jam_type='none')
r_pulse = simulate_channel(ChannelConfig(snr_db=10, jsr_db=12,
                            n_symbols=frame_size//10, seed=9999), jam_type='pulse')
rms_c = pulse_expert._compute_group_rms(r_clean['rx'][:frame_size], sps_ref)
rms_p = pulse_expert._compute_group_rms(r_pulse['rx'][:frame_size], sps_ref)
print(f"  Sanity: clean RMS range = [{rms_c.min():.3f}, {rms_c.max():.3f}]")
print(f"  Sanity: pulse RMS range = [{rms_p.min():.3f}, {rms_p.max():.3f}]")
print(f"  Sanity: max pulse / max clean = {rms_p.max()/rms_c.max():.1f}x")

pr_test = pulse_expert.correct_symbols(r_pulse['rx'][:frame_size], sps_ref)
print(f"  Sanity: survival={pr_test['survival_rate']:.1%}, "
      f"groups flagged={np.sum(pr_test['grp_mask'])}/{len(pr_test['grp_mask'])}")

# MMSE calibration (unchanged)
mmse_expert = MMSEExpert()
asc, jvc = [], []
for jsr in np.arange(2, 22, 2):
    for i in range(5):
        r = simulate_channel(ChannelConfig(snr_db=10, jsr_db=jsr,
                              n_symbols=frame_size//10, seed=8000+i), jam_type='broadband')
        _, sc, _, _, _ = snn.detect_and_classify(r['rx'][:frame_size])
        asc.append(sc); jvc.append(jsr)
mmse_expert.calibrate_jsr_mapping(np.array(asc), np.array(jvc))
print(f"  MMSE calibrated on {len(asc)} points")

moe = MoEAntiJamming(snn, lstm_expert, pulse_expert, mmse_expert)
print("\nMoE system assembled (v3.3).")

# ============================================================
# Exp 1: Classification Accuracy
# ============================================================
t0 = time.time()
print("EXP 1: SNN Classification")

n_test_per = 15
test_signals, test_labels = [], []

for i in range(n_test_per):
    r = simulate_channel(ChannelConfig(snr_db=10, n_symbols=frame_size//10, seed=9000+i),
                          jam_type='none')
    test_signals.append(r['rx'][:frame_size])
    test_labels.append('none')

for jt in JT:
    for jsr in [6, 10, 14, 18]:
        for i in range(n_test_per):
            seed = 9500 + i + JT.index(jt)*300 + [6,10,14,18].index(jsr)*20
            r = simulate_channel(ChannelConfig(snr_db=10, jsr_db=jsr,
                                  n_symbols=frame_size//10, seed=seed), jam_type=jt)
            test_signals.append(r['rx'][:frame_size])
            test_labels.append(jt)

pred_labels = []
for sig in test_signals:
    _, _, pred, _, _ = snn.detect_and_classify(sig)
    pred_labels.append(pred)

# confusion matrix
labels_order = JT_ALL
cm = confusion_matrix(test_labels, pred_labels, labels=labels_order)
acc = np.mean(np.array(test_labels) == np.array(pred_labels))

print(f"\nOverall accuracy: {acc:.3f}")
print(f"\nClassification report:")
print(classification_report(test_labels, pred_labels, target_names=[JL[j] for j in labels_order]))

# plot confusion matrix
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(cm, cmap='Blues')
ax.set_xticks(range(5)); ax.set_yticks(range(5))
ax.set_xticklabels([JL[j] for j in labels_order], rotation=45, ha='right')
ax.set_yticklabels([JL[j] for j in labels_order])
for i in range(5):
    for j in range(5):
        ax.text(j, i, str(cm[i,j]), ha='center', va='center',
                color='white' if cm[i,j] > cm.max()/2 else 'black', fontsize=12)
ax.set(xlabel='Predicted', ylabel='True', title=f'SNN classification (accuracy: {acc:.3f})')
plt.colorbar(im, ax=ax)
plt.tight_layout(); fig.savefig(OUT/'fig1_confusion.png'); plt.show()
print(f"Done in {time.time()-t0:.1f}s")

# ============================================================
# Exp 2: MoE BER vs JSR (v3.3 energy-based erasure)
# ============================================================
t0 = time.time()
print("EXP 2: MoE BER (energy-based erasure)")

jsr_range = np.arange(0, 22, 2)
n_trials = 20
n_sym_test = 10000

ber_results = {}
for jt in JT:
    ber_results[jt] = {m: [] for m in ['none','lstm_only','moe']}
    for m in ['none','lstm_only','moe']:
        ber_results[jt][f'{m}_ci'] = []

for jsr in jsr_range:
    for jt in JT:
        b = {'none':[], 'lstm_only':[], 'moe':[]}
        for trial in range(n_trials):
            cfg = ChannelConfig(snr_db=10, jsr_db=jsr, n_symbols=n_sym_test,
                                seed=5000+trial+JT.index(jt)*2000)
            r = simulate_channel(cfg, jam_type=jt)
            sps = r['sps']

            d, _ = demodulate_bpsk(r['rx'], sps)
            b['none'].append(compute_ber(r['bits'], d))

            ls = lstm_expert.correct_symbols(r['rx'], sps)
            b['lstm_only'].append(compute_ber(r['bits'], (ls > 0).astype(int)))

            res = moe.process(r['rx'], sps, snr_db=10.0)
            bi = res['ber_indices']
            bb = res['ber_bits']
            true_surv = r['bits'][bi] if len(bi) <= len(r['bits']) else r['bits'][:len(bb)]
            b['moe'].append(compute_ber(true_surv, bb))

        for m in ['none','lstm_only','moe']:
            arr = np.array(b[m])
            ber_results[jt][m].append(max(np.mean(arr), 1e-5))
            ber_results[jt][f'{m}_ci'].append(1.96*np.std(arr)/np.sqrt(n_trials))

    pl = ber_results['pulse']
    print(f"  JSR={jsr:2d} | Pulse: nc={pl['none'][-1]:.4f} moe={pl['moe'][-1]:.4f}")

print(f"Done in {time.time()-t0:.1f}s")

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
for idx_j, jt in enumerate(JT):
    ax = axes[idx_j//2][idx_j%2]
    for m, mk, c, lbl in [('none','^','k','No correction'),
                           ('lstm_only','s','#888780','LSTM only (v2)'),
                           ('moe','o','#1D9E75','MoE (v3.3)')]:
        y = np.array(ber_results[jt][m])
        ci = np.array(ber_results[jt][f'{m}_ci'])
        ax.semilogy(jsr_range, y, f'{mk}-', color=c, label=lbl, ms=4, lw=1.5)
        ax.fill_between(jsr_range, np.maximum(y-ci,1e-5), y+ci, color=c, alpha=0.1)
    ttl = f'{JL[jt]}'
    if jt == 'pulse': ttl += ' (erasure BER on surviving sym)'
    ax.set(xlabel='JSR (dB)', ylabel='BER', title=ttl)
    ax.legend(fontsize=8); ax.set_ylim(1e-5, 0.55)
plt.suptitle('BER: No correction vs LSTM-only vs MoE (SNR=10 dB)', fontsize=13)
plt.tight_layout(); fig.savefig(OUT/'fig2_ber_moe.png'); plt.show()

# ============================================================
# BER Gain: MoE vs No Correction
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# gain ratio
ax = axes[0]
for jt in JT:
    ratio = np.array(ber_results[jt]['none']) / np.maximum(np.array(ber_results[jt]['moe']), 1e-5)
    ax.plot(jsr_range, ratio, 'o-', color=JC[jt], label=JL[jt], ms=4, lw=1.5)
ax.axhline(y=1, color='k', ls='--', lw=0.8, alpha=0.5)
ax.set(xlabel='JSR (dB)', ylabel='BER gain (no_corr / MoE)', title='(a) MoE correction gain')
ax.legend()

# MoE vs LSTM-only
ax = axes[1]
for jt in JT:
    moe_ber = np.array(ber_results[jt]['moe'])
    lstm_ber = np.array(ber_results[jt]['lstm_only'])
    improvement = (lstm_ber - moe_ber) / np.maximum(lstm_ber, 1e-5) * 100
    ax.plot(jsr_range, improvement, 'o-', color=JC[jt], label=JL[jt], ms=4, lw=1.5)
ax.axhline(y=0, color='k', ls='--', lw=0.8, alpha=0.5)
ax.set(xlabel='JSR (dB)', ylabel='BER reduction (%)', title='(b) MoE vs LSTM-only improvement')
ax.legend()

plt.tight_layout(); fig.savefig(OUT/'fig2b_ber_gain.png'); plt.show()

# ============================================================
# Exp 3: Routing Analysis
# ============================================================
print("EXP 3: Expert Routing")

routing_counts = {jt: defaultdict(int) for jt in JT}
confidence_by_type = {jt: [] for jt in JT}
actions_by_type = {jt: defaultdict(int) for jt in JT}

for jt in JT:
    for jsr in [6, 10, 14]:
        for i in range(15):
            cfg = ChannelConfig(snr_db=10, jsr_db=jsr, n_symbols=n_sym_test,
                                seed=7000+i+JT.index(jt)*500)
            r = simulate_channel(cfg, jam_type=jt)
            res = moe.process(r['rx'], r['sps'])
            routing_counts[jt][res['predicted_class']] += 1
            confidence_by_type[jt].append(res['mean_confidence'])
            actions_by_type[jt][res['action']] += 1

print("\nRouting distribution:")
for jt in JT:
    total = sum(routing_counts[jt].values())
    print(f"  {JL[jt]:12s}: ", end='')
    for cls in JT_ALL:
        cnt = routing_counts[jt].get(cls, 0)
        if cnt > 0:
            print(f"{JL[cls]}={cnt/total*100:.0f}%  ", end='')
    print()

print("\nMean confidence by type:")
for jt in JT:
    print(f"  {JL[jt]:12s}: {np.mean(confidence_by_type[jt]):.3f}")

print("\nActions taken:")
for jt in JT:
    total = sum(actions_by_type[jt].values())
    print(f"  {JL[jt]:12s}: ", end='')
    for act, cnt in actions_by_type[jt].items():
        print(f"{act}={cnt/total*100:.0f}%  ", end='')
    print()

# routing bar chart
fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(len(JT))
width = 0.15
for ci, cls in enumerate(JT_ALL):
    vals = [routing_counts[jt].get(cls, 0) / sum(routing_counts[jt].values()) * 100
            for jt in JT]
    ax.bar(x + ci*width, vals, width, label=f'Routed to {JL[cls]}', color=JC[cls], alpha=0.8)
ax.set_xticks(x + 2*width); ax.set_xticklabels([JL[jt] for jt in JT])
ax.set(ylabel='Routing %', title='Expert routing distribution by true jamming type')
ax.legend(fontsize=8); ax.set_ylim(0, 105)
plt.tight_layout(); fig.savefig(OUT/'fig3_routing.png'); plt.show()

# ============================================================
# Exp 4: Pulse Blanker (v3.3 energy-based erasure)
# ============================================================
print("EXP 4: Pulse Blanker Detail")

pulse_jsr_range = np.arange(2, 22, 2)
p_ber_orig, p_ber_eras, p_surv = [], [], []

for jsr in pulse_jsr_range:
    bo, be, sv = [], [], []
    for trial in range(15):
        cfg = ChannelConfig(snr_db=10, jsr_db=jsr, n_symbols=10000, seed=4400+trial)
        r = simulate_channel(cfg, jam_type='pulse')
        sps = r['sps']
        d, _ = demodulate_bpsk(r['rx'], sps)
        bo.append(compute_ber(r['bits'], d))
        pr = pulse_expert.correct_symbols(r['rx'], sps)
        surv_true = r['bits'][pr['surviving_indices']]
        be.append(compute_ber(surv_true, pr['surviving_bits']))
        sv.append(pr['survival_rate'])
    p_ber_orig.append(np.mean(bo))
    p_ber_eras.append(np.mean(be))
    p_surv.append(np.mean(sv))
    gain = np.mean(bo) / max(np.mean(be), 1e-5)
    print(f"  JSR={jsr:2d} | orig={np.mean(bo):.4f} erasure={np.mean(be):.4f} "
          f"({gain:.1f}x) surv={np.mean(sv):.1%}")

# detailed viz at JSR=12
cfg = ChannelConfig(snr_db=10, jsr_db=12, n_symbols=2000, seed=4444)
r = simulate_channel(cfg, jam_type='pulse')
sps = r['sps']
pr = pulse_expert.correct_symbols(r['rx'], sps)
orig_bits, orig_soft = demodulate_bpsk(r['rx'], sps)
ber_o = compute_ber(r['bits'], orig_bits)
ber_e = compute_ber(r['bits'][pr['surviving_indices']], pr['surviving_bits'])
print(f"\nDetail JSR=12: BER {ber_o:.4f}->{ber_e:.4f} ({ber_o/max(ber_e,1e-5):.1f}x), "
      f"surv={pr['survival_rate']:.0%}, flagged={np.sum(pr['grp_mask'])}/{len(pr['grp_mask'])}")

# true pulse envelope for comparison
n_samp = len(r['rx'])
true_env = np.zeros(n_samp)
for k in range(0, n_samp, 500):
    true_env[k:min(k+100, n_samp)] = 1.0

fig, axes = plt.subplots(5, 1, figsize=(13, 14),
                         gridspec_kw={'height_ratios': [2, 1.2, 0.8, 0.8, 2]})

sl = slice(0, 5000)
axes[0].plot(r['tx'][sl], 'k-', lw=0.5, alpha=0.3, label='Clean')
axes[0].plot(r['rx'][sl], '-', color=JC['pulse'], lw=0.4, alpha=0.6, label='Jammed')
axes[0].set(ylabel='Amp', title='Pulse-jammed (JSR=12 dB)'); axes[0].legend(fontsize=8)

# RMS values per group (energy-based detection)
ng = min(len(pr['rms_values']), 100)
thresh_rms = pulse_expert.baseline_rms_mean + pulse_expert.blank_z * pulse_expert.baseline_rms_std
cg = [JC['pulse'] if pr['grp_mask'][k] else '#888780' for k in range(ng)]
axes[1].bar(range(ng), pr['rms_values'][:ng], color=cg, alpha=0.7, width=1.0)
axes[1].axhline(y=thresh_rms, color='k', ls='--', lw=1, label=f'Threshold (z={pulse_expert.blank_z})')
axes[1].axhline(y=pulse_expert.baseline_rms_mean, color='#888780', ls=':', lw=0.8, label='Baseline')
axes[1].set(ylabel='RMS', title=f'Group RMS energy (win={pulse_expert.group_size} sym)')
axes[1].legend(fontsize=8)

# true vs detected ON
win = pulse_expert.group_size * sps
true_grp = np.array([np.mean(true_env[k*win:(k+1)*win]) > 0.3 for k in range(ng)])
axes[2].fill_between(range(ng), 0, true_grp.astype(float), color='#888780', alpha=0.3, label='True ON')
axes[2].fill_between(range(ng), 0, pr['grp_mask'][:ng].astype(float), color=JC['pulse'], alpha=0.5, label='Detected')
axes[2].set(ylabel='ON', title='True vs detected pulse ON', ylim=(-0.1, 1.3)); axes[2].legend(fontsize=8)

ns_show = min(500, len(pr['erasure_mask']))
axes[3].fill_between(range(ns_show), 0, pr['erasure_mask'][:ns_show].astype(float),
                     color=JC['pulse'], alpha=0.4)
axes[3].set(ylabel='Erased', title=f'Erasure mask (surv={pr["survival_rate"]:.0%})', ylim=(-0.1,1.1))

axes[4].plot(orig_soft[:ns_show], '-', color='#888780', lw=0.5, alpha=0.4, label='All soft')
sm = ~pr['erasure_mask'][:ns_show]
axes[4].plot(np.where(sm)[0], orig_soft[:ns_show][sm], '.', color='#1D9E75', ms=2, label='Surviving')
axes[4].plot(r['symbols'][:ns_show], 'k--', lw=0.3, alpha=0.3, label='True')
axes[4].set(xlabel='Symbol', ylabel='Soft', title=f'BER {ber_o:.4f}->{ber_e:.4f}')
axes[4].legend(fontsize=8)
plt.tight_layout(); fig.savefig(OUT/'fig4_pulse.png'); plt.show()

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].semilogy(pulse_jsr_range, p_ber_orig, '^-', color='k', label='No corr (all)', ms=4)
axes[0].semilogy(pulse_jsr_range, p_ber_eras, 'o-', color='#1D9E75', label='Erasure (surv)', ms=4)
axes[0].set(xlabel='JSR (dB)', ylabel='BER', title='(a) Pulse erasure BER')
axes[0].legend(); axes[0].set_ylim(1e-5, 0.5)
axes[1].plot(pulse_jsr_range, [s*100 for s in p_surv], 'o-', color=JC['pulse'], ms=4)
axes[1].axhline(y=80, color='k', ls='--', lw=0.8, alpha=0.5, label='Expected (duty=20%)')
axes[1].set(xlabel='JSR (dB)', ylabel='Survival %', title='(b) Survival rate')
axes[1].legend(); axes[1].set_ylim(0, 105)
plt.tight_layout(); fig.savefig(OUT/'fig4b_pulse_ber.png'); plt.show()

# ============================================================
# Exp 5: Power
# ============================================================
powers = {
    'SNN watchdog\n(always-on)': snn.power_mw(),
    'MoE + LSTM\n(structured jam)': moe.total_power_mw('lstm'),
    'MoE + Blanker\n(pulse jam)': moe.total_power_mw('pulse'),
    'MoE + MMSE\n(broadband)': moe.total_power_mw('mmse'),
    'Edge CNN\n(always-on)': 50.0,
    'DRL agent\n(always-on)': 120.0,
}

fig, ax = plt.subplots(figsize=(9, 4.5))
cs = ['#1D9E75', '#534AB7', '#D4537E', '#D85A30', '#888780', '#888780']
bars = ax.bar(powers.keys(), powers.values(), color=cs, edgecolor='white', lw=0.5)
ax.set_ylabel('Power (mW)'); ax.set_title('Power consumption by configuration')
ax.set_yscale('log'); ax.set_ylim(0.005, 500)
for b, v in zip(bars, powers.values()):
    ax.text(b.get_x()+b.get_width()/2, v*1.5, f'{v:.3f}',
            ha='center', va='bottom', fontsize=8, fontweight='bold')
plt.tight_layout(); fig.savefig(OUT/'fig5_power.png'); plt.show()

print("Power summary:")
for n, v in powers.items():
    print(f"  {n.replace(chr(10),' '):<30s} {v:>8.3f} mW")
snn_p = snn.power_mw()
print(f"\n  SNN vs CNN:   {50.0/snn_p:.0f}x more efficient")
print(f"  Worst-case MoE (LSTM) vs DRL: {120.0/moe.total_power_mw('lstm'):.0f}x more efficient")

# ============================================================
# Exp 6: Training Curves + Feature Importance
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

ax = axes[0]
ax.plot(lstm_expert.losses, color='#1D9E75', lw=1, label='Train')
ax.plot(lstm_expert.val_losses, color='#1D9E75', lw=1, ls='--', alpha=0.7, label='Validation')
ax.set(xlabel='Epoch', ylabel='MSE Loss', title='LSTM expert training')
ax.legend(); ax.set_yscale('log')

# RF feature importance
ax = axes[1]
feat_names = ['FR_in', 'FR_out', 'ISI_CV', 'Fano', 'HF_ratio', 'SF_ratio', 'Spec_peak']
imp = snn.classifier.feature_importances_
bars = ax.barh(feat_names, imp, color='#534AB7', alpha=0.8)
ax.set(xlabel='Importance', title='SNN feature importance (RF classifier)')

plt.tight_layout(); fig.savefig(OUT/'fig6_training.png'); plt.show()

# ============================================================
# Exp 7: SNR Heatmap (v3.3)
# ============================================================
print("Computing SNR grid...")
jsr_grid = np.arange(2, 22, 2)
snr_grid = np.arange(2, 22, 2)
n_tr = 8
snr_imp = {jt: np.zeros((len(snr_grid), len(jsr_grid))) for jt in JT}

for i, sv in enumerate(snr_grid):
    for j, jv in enumerate(jsr_grid):
        for jt in JT:
            imps = []
            for t in range(n_tr):
                cfg = ChannelConfig(snr_db=sv, jsr_db=jv, n_symbols=5000, seed=9000+t)
                r = simulate_channel(cfg, jam_type=jt)
                sps = r['sps']
                res = moe.process(r['rx'], sps, snr_db=sv)
                if jt == 'pulse' and 'erasure_mask' in res:
                    surv = ~res['erasure_mask']
                    n_s = min(len(r['symbols']), len(surv))
                    surv = surv[:n_s]
                    cs = r['symbols'][:n_s][surv].astype(float)
                    ja = np.mean(r['rx'][:n_s*sps].reshape(n_s, sps), axis=1)[surv]
                    co = res['corrected_symbols'][:n_s][surv]
                    if len(cs) > 10:
                        _, _, iv = compute_snr_improvement(cs, ja, co)
                    else:
                        iv = 0
                else:
                    cs = r['symbols'][:len(res['corrected_symbols'])].astype(float)
                    n_s = len(cs)
                    ja = np.mean(r['rx'][:n_s*sps].reshape(n_s, sps), axis=1)
                    _, _, iv = compute_snr_improvement(cs, ja, res['corrected_symbols'][:n_s])
                imps.append(iv)
            snr_imp[jt][i, j] = np.mean(imps)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for idx_j, jt in enumerate(JT):
    ax = axes[idx_j//2][idx_j%2]
    vmax = max(np.max(snr_imp[jt]), 5)
    im = ax.imshow(snr_imp[jt], aspect='auto', origin='lower', cmap='RdYlGn',
                   vmin=-3, vmax=vmax,
                   extent=[jsr_grid[0]-1, jsr_grid[-1]+1, snr_grid[0]-1, snr_grid[-1]+1])
    ax.set(xlabel='JSR (dB)', ylabel='SNR (dB)', title=f'{JL[jt]}')
    plt.colorbar(im, ax=ax, label='dB')
plt.suptitle('MoE SNR improvement (dB)', fontsize=13)
plt.tight_layout(); fig.savefig(OUT/'fig7_snr_heatmap.png'); plt.show()
for jt in JT:
    print(f"  {JL[jt]:12s} max SNR: {np.max(snr_imp[jt]):.2f} dB")

# ============================================================
# Exp 8: Waveform Examples (all 4 types, MoE corrected)
# ============================================================
cfg = ChannelConfig(snr_db=10, jsr_db=12, n_symbols=2000, seed=7777)

fig, axes = plt.subplots(4, 3, figsize=(15, 12))
col_titles = ['Clean', 'Jammed', 'MoE corrected']
sl = slice(0, 2500)

for idx, jt in enumerate(JT):
    r = simulate_channel(cfg, jam_type=jt)
    sps = r['sps']
    res = moe.process(r['rx'], sps)
    corr_samples = np.repeat(res['corrected_symbols'], sps)[:len(r['tx'])]

    axes[idx][0].plot(r['tx'][sl], 'k-', lw=0.6)
    axes[idx][1].plot(r['rx'][sl], '-', color=JC[jt], lw=0.4, alpha=0.7)
    axes[idx][2].plot(corr_samples[sl], '-', color='#1D9E75', lw=0.6)
    axes[idx][2].plot(r['tx'][sl], 'k--', lw=0.3, alpha=0.3)

    expert_name = res['expert_used']
    ber_orig = compute_ber(r['bits'], demodulate_bpsk(r['rx'], sps)[0])
    ber_moe = compute_ber(r['bits'], res['bits'])
    axes[idx][0].set_ylabel(f"{JL[jt]}\nBER: {ber_orig:.3f}->{ber_moe:.3f}",
                            fontsize=9, fontweight='bold')

    for a in axes[idx]:
        a.set_ylim(-5, 5)

for j, t in enumerate(col_titles):
    axes[0][j].set_title(t, fontsize=11)
axes[-1][1].set_xlabel('Sample')
plt.suptitle('MoE correction across all jamming types (JSR=12 dB)', fontsize=13)
plt.tight_layout(); fig.savefig(OUT/'fig8_waveforms.png'); plt.show()

# ============================================================
# R2 EXPERIMENTS (single cell - all results printed at end)
# ============================================================
import gc
from scipy.signal import iirnotch, lfilter
t_r2_start = time.time()

# ============================================================
# 1. CLASSICAL BASELINES (notch, LMS-32, classical blanker)
# ============================================================
print("="*70)
print("R2-EXP 1: Classical Signal Processing Baselines")
print("="*70)

def apply_notch(signal, fs, f0_ratio=0.3, Q=30):
    f0 = f0_ratio * fs / 2
    b, a = iirnotch(f0, Q, fs)
    return lfilter(b, a, signal)

def apply_lms(signal, n_taps=32, mu=0.01):
    n = len(signal)
    w = np.zeros(n_taps)
    y = np.zeros(n)
    e = np.zeros(n)
    for i in range(n_taps, n):
        x = signal[i-n_taps:i][::-1]
        y[i] = np.dot(w, x)
        e[i] = signal[i] - y[i]
        w += 2 * mu * e[i] * x
    return e

def apply_classical_blanker(signal, sps, group_size=5, z_thresh=3.0):
    ns = len(signal) // sps
    soft = np.mean(signal[:ns*sps].reshape(ns, sps), axis=1)
    win = group_size * sps
    n_groups = len(signal) // win
    rms = np.zeros(n_groups)
    for k in range(n_groups):
        seg = signal[k*win:(k+1)*win]
        rms[k] = np.sqrt(np.mean(seg**2))
    mu_rms, std_rms = np.mean(rms), max(np.std(rms), 1e-6)
    z = (rms - mu_rms) / std_rms
    grp_mask = z > z_thresh
    erasure = np.zeros(ns, dtype=bool)
    for k, is_j in enumerate(grp_mask):
        if is_j:
            s0 = k * group_size
            s1 = min((k+1) * group_size, ns)
            erasure[s0:s1] = True
    surv_idx = np.where(~erasure)[0]
    surv_bits = (soft[surv_idx] > 0).astype(int)
    return surv_bits, surv_idx, soft, erasure

jsr_range_bl = np.arange(0, 22, 2)
n_trials_bl = 20
n_sym_bl = 10000

baseline_ber = {}
for jt in JT:
    baseline_ber[jt] = {}
    for m in ['none', 'notch', 'lms32', 'blanker', 'lstm_only', 'moe']:
        baseline_ber[jt][m] = []
        baseline_ber[jt][f'{m}_ci'] = []

for jsr in jsr_range_bl:
    for jt in JT:
        b = {m: [] for m in ['none', 'notch', 'lms32', 'blanker', 'lstm_only', 'moe']}
        for trial in range(n_trials_bl):
            cfg = ChannelConfig(snr_db=10, jsr_db=jsr, n_symbols=n_sym_bl,
                                seed=11000+trial+JT.index(jt)*2000)
            r = simulate_channel(cfg, jam_type=jt)
            sps = r['sps']

            d, _ = demodulate_bpsk(r['rx'], sps)
            b['none'].append(compute_ber(r['bits'], d))

            rx_notch = apply_notch(r['rx'], r['cfg'].fs)
            dn, _ = demodulate_bpsk(rx_notch, sps)
            b['notch'].append(compute_ber(r['bits'], dn))

            rx_lms = apply_lms(r['rx'], n_taps=32, mu=0.005)
            dl, _ = demodulate_bpsk(rx_lms, sps)
            b['lms32'].append(compute_ber(r['bits'], dl))

            sb, si, _, _ = apply_classical_blanker(r['rx'], sps)
            if len(si) > 0:
                b['blanker'].append(compute_ber(r['bits'][si], sb))
            else:
                b['blanker'].append(0.5)

            ls = lstm_expert.correct_symbols(r['rx'], sps)
            b['lstm_only'].append(compute_ber(r['bits'], (ls > 0).astype(int)))

            res = moe.process(r['rx'], sps, snr_db=10.0)
            bi = res['ber_indices']
            bb = res['ber_bits']
            true_surv = r['bits'][bi] if len(bi) <= len(r['bits']) else r['bits'][:len(bb)]
            b['moe'].append(compute_ber(true_surv, bb))

        for m in ['none', 'notch', 'lms32', 'blanker', 'lstm_only', 'moe']:
            arr = np.array(b[m])
            baseline_ber[jt][m].append(max(np.mean(arr), 1e-5))
            baseline_ber[jt][f'{m}_ci'].append(1.96*np.std(arr)/np.sqrt(n_trials_bl))

    print(f"  JSR={jsr:2d} dB done")

print("Baselines complete.")

# ============================================================
# 2. VARIABLE DUTY CYCLE
# ============================================================
print("\n" + "="*70)
print("R2-EXP 2: Variable Duty Cycle")
print("="*70)

duty_cycles = [0.10, 0.15, 0.20, 0.30, 0.40]
n_trials_dc = 20
n_sym_dc = 10000

def add_pulse_jam_variable(signal, jsr_db, rng, duty, period=500):
    sp = np.mean(signal**2)
    jp = sp * 10**(jsr_db/10) / duty
    n = len(signal)
    env = np.zeros(n)
    on = int(period * duty)
    for i in range(0, n, period):
        env[i:min(i+on, n)] = 1.0
    jam = env * rng.normal(0, np.sqrt(jp), n)
    return signal + jam, jam

dc_results = {'duty': [], 'ber_none': [], 'ber_erasure': [],
              'survival': [], 'ber_none_ci': [], 'ber_erasure_ci': []}

for duty in duty_cycles:
    bo, be, sv = [], [], []
    for trial in range(n_trials_dc):
        rng = np.random.default_rng(12000+trial)
        cfg = ChannelConfig(snr_db=10, jsr_db=10, n_symbols=n_sym_dc, seed=12000+trial)
        bits, symbols, tx, sps = generate_bpsk(cfg, rng)
        noisy = add_awgn(tx, 10, rng)
        rx, jam = add_pulse_jam_variable(noisy, 10, rng, duty=duty)

        d, _ = demodulate_bpsk(rx, sps)
        bo.append(compute_ber(bits, d))

        pr = pulse_expert.correct_symbols(rx, sps)
        surv_true = bits[pr['surviving_indices']]
        be.append(compute_ber(surv_true, pr['surviving_bits']))
        sv.append(pr['survival_rate'])

    dc_results['duty'].append(duty)
    dc_results['ber_none'].append(np.mean(bo))
    dc_results['ber_erasure'].append(np.mean(be))
    dc_results['survival'].append(np.mean(sv))
    dc_results['ber_none_ci'].append(1.96*np.std(bo)/np.sqrt(n_trials_dc))
    dc_results['ber_erasure_ci'].append(1.96*np.std(be)/np.sqrt(n_trials_dc))
    print(f"  duty={duty:.0%}: BER none={np.mean(bo):.4f} erasure={np.mean(be):.4f} surv={np.mean(sv):.1%}")

# ============================================================
# 3. RAYLEIGH FLAT FADING
# ============================================================
print("\n" + "="*70)
print("R2-EXP 3: Rayleigh Flat Fading")
print("="*70)

def simulate_rayleigh(cfg, jam_type, fade_block=100):
    rng = np.random.default_rng(cfg.seed)
    bits, symbols, tx, sps = generate_bpsk(cfg, rng)
    n = len(tx)
    n_blocks = n // fade_block + 1
    h_real = rng.normal(0, 1/np.sqrt(2), n_blocks)
    h_imag = rng.normal(0, 1/np.sqrt(2), n_blocks)
    h_mag = np.sqrt(h_real**2 + h_imag**2)
    faded = np.zeros(n)
    for k in range(n_blocks):
        s0 = k * fade_block
        s1 = min((k+1) * fade_block, n)
        faded[s0:s1] = tx[s0:s1] * h_mag[k]
    noisy = add_awgn(faded, cfg.snr_db, rng)
    if jam_type == 'none':
        return {'bits': bits, 'symbols': symbols, 'tx': tx, 'rx': noisy,
                'jam': np.zeros(n), 'sps': sps, 'cfg': cfg}
    rx, jam = JAM_FNS[jam_type](noisy, cfg.jsr_db, cfg.fs, rng)
    return {'bits': bits, 'symbols': symbols, 'tx': tx, 'rx': rx,
            'jam': jam, 'sps': sps, 'cfg': cfg}

n_trials_fade = 20
n_sym_fade = 10000

# Classification under fading
fade_cls_correct = 0
fade_cls_total = 0
fade_routing_correct = 0

for jt in JT:
    correct = 0
    for trial in range(n_trials_fade):
        cfg = ChannelConfig(snr_db=10, jsr_db=10, n_symbols=n_sym_fade,
                            seed=13000+trial+JT.index(jt)*500)
        r = simulate_rayleigh(cfg, jam_type=jt)
        _, _, pred, _, _ = snn.detect_and_classify(r['rx'])
        if pred == jt:
            correct += 1
            fade_routing_correct += 1
        fade_cls_total += 1
    fade_cls_correct += correct
    print(f"  {JL[jt]:12s}: {correct}/{n_trials_fade} = {correct/n_trials_fade:.1%}")

clean_correct = 0
for trial in range(n_trials_fade):
    cfg = ChannelConfig(snr_db=10, n_symbols=n_sym_fade, seed=13500+trial)
    r = simulate_rayleigh(cfg, jam_type='none')
    _, _, pred, _, _ = snn.detect_and_classify(r['rx'])
    if pred == 'none':
        clean_correct += 1
    fade_cls_total += 1
fade_cls_correct += clean_correct

fade_accuracy = fade_cls_correct / fade_cls_total
fade_routing = fade_routing_correct / (n_trials_fade * len(JT))
print(f"  Clean:       {clean_correct}/{n_trials_fade}")
print(f"  Overall: {fade_accuracy:.1%}, Routing: {fade_routing:.1%}")

# BER under fading
fade_ber = {}
for jt in JT:
    b_awgn = {'none': [], 'moe': []}
    b_fade = {'none': [], 'moe': []}
    for trial in range(n_trials_fade):
        cfg = ChannelConfig(snr_db=10, jsr_db=10, n_symbols=n_sym_fade,
                            seed=14000+trial+JT.index(jt)*500)
        r_a = simulate_channel(cfg, jam_type=jt)
        sps = r_a['sps']
        d, _ = demodulate_bpsk(r_a['rx'], sps)
        b_awgn['none'].append(compute_ber(r_a['bits'], d))
        res_a = moe.process(r_a['rx'], sps)
        b_awgn['moe'].append(compute_ber(r_a['bits'][res_a['ber_indices']], res_a['ber_bits']))

        r_f = simulate_rayleigh(cfg, jam_type=jt)
        d, _ = demodulate_bpsk(r_f['rx'], sps)
        b_fade['none'].append(compute_ber(r_f['bits'], d))
        res_f = moe.process(r_f['rx'], sps)
        b_fade['moe'].append(compute_ber(r_f['bits'][res_f['ber_indices']], res_f['ber_bits']))

    fade_ber[jt] = {
        'awgn_none': np.mean(b_awgn['none']), 'awgn_moe': np.mean(b_awgn['moe']),
        'fade_none': np.mean(b_fade['none']), 'fade_moe': np.mean(b_fade['moe']),
    }
    print(f"  {JL[jt]:12s}: AWGN moe={fade_ber[jt]['awgn_moe']:.4f} Rayl moe={fade_ber[jt]['fade_moe']:.4f}")

# ============================================================
# 4. NO-SNN ABLATION
# ============================================================
print("\n" + "="*70)
print("R2-EXP 4: No-SNN Ablation")
print("="*70)

from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.preprocessing import StandardScaler as SS

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

abl_feats = np.array([extract_conventional_features(s) for s in cls_signals])
abl_scaler = SS().fit(abl_feats)
abl_X = abl_scaler.transform(abl_feats)
abl_clf = RFC(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
abl_clf.fit(abl_X, cls_labels)

abl_preds = []
for sig in test_signals:
    feat = extract_conventional_features(sig)
    X = abl_scaler.transform(feat.reshape(1, -1))
    abl_preds.append(abl_clf.predict(X)[0])

abl_acc = np.mean(np.array(test_labels) == np.array(abl_preds))
print(f"  SNN accuracy:          {acc:.3f}")
print(f"  Conventional accuracy: {abl_acc:.3f}")
print(f"  Difference:            {acc - abl_acc:+.3f}")

abl_routing = {}
for jt in JT:
    correct = sum(1 for i, tl in enumerate(test_labels) if tl == jt and abl_preds[i] == jt)
    total = sum(1 for tl in test_labels if tl == jt)
    abl_routing[jt] = correct / total if total > 0 else 0
    print(f"  {JL[jt]:12s} routing: SNN={routing_counts[jt].get(jt,0)/sum(routing_counts[jt].values()):.1%} Conv={abl_routing[jt]:.1%}")

# ============================================================
# 5. PULSE GOODPUT
# ============================================================
print("\n" + "="*70)
print("R2-EXP 5: Pulse Goodput")
print("="*70)

gp_jsr_range = [6, 8, 10, 12, 14, 16]
gp_results = []
n_trials_gp = 20

for jsr in gp_jsr_range:
    gp_none, gp_eras, survs, ber_e_list = [], [], [], []
    for trial in range(n_trials_gp):
        cfg = ChannelConfig(snr_db=10, jsr_db=jsr, n_symbols=10000, seed=15000+trial)
        r = simulate_channel(cfg, jam_type='pulse')
        sps = r['sps']
        d, _ = demodulate_bpsk(r['rx'], sps)
        ber_nc = compute_ber(r['bits'], d)
        gp_none.append(1.0 * (1 - ber_nc))
        pr = pulse_expert.correct_symbols(r['rx'], sps)
        surv_true = r['bits'][pr['surviving_indices']]
        ber_er = compute_ber(surv_true, pr['surviving_bits'])
        gp_eras.append(pr['survival_rate'] * (1 - ber_er))
        survs.append(pr['survival_rate'])
        ber_e_list.append(ber_er)
    gp_results.append({'jsr': jsr, 'gp_none': np.mean(gp_none), 'gp_erasure': np.mean(gp_eras),
                        'survival': np.mean(survs), 'ber_erasure': np.mean(ber_e_list)})
    print(f"  JSR={jsr:2d}: surv={np.mean(survs):.1%} GP_none={np.mean(gp_none):.3f} GP_eras={np.mean(gp_eras):.3f}")

# ============================================================
# COMBINED RESULTS OUTPUT
# ============================================================
elapsed = time.time() - t_r2_start
print("\n\n")
print("="*80)
print("R2 EXPERIMENTS - COMPLETE RESULTS")
print(f"Total time: {elapsed:.1f}s")
print("="*80)

# TABLE II
jsr10_bl = list(jsr_range_bl).index(10)
print("\n--- TABLE II: BER @ JSR=10 dB, SNR=10 dB ---")
print(f"{'Type':<12s} {'No Corr':>8s} {'Notch':>8s} {'LMS-32':>8s} {'Blanker':>8s} {'LSTM':>8s} {'MoE':>8s}")
print("-"*68)
for jt in JT:
    row = f"{JL[jt]:<12s}"
    for m in ['none', 'notch', 'lms32', 'blanker', 'lstm_only', 'moe']:
        v = baseline_ber[jt][m][jsr10_bl]
        row += f" {v:>8.4f}" if v >= 0.001 else f" {'<0.001':>8s}"
    ci_row = f"{'(95% CI)':<12s}"
    for m in ['none', 'notch', 'lms32', 'blanker', 'lstm_only', 'moe']:
        c = baseline_ber[jt][f'{m}_ci'][jsr10_bl]
        ci_row += f" +/-{c:.4f}"
    print(row)
    print(ci_row)

# TABLE V
print("\n--- TABLE V: Pulse Goodput @ SNR=10 dB ---")
print(f"{'JSR':>6s} {'Survival':>10s} {'BER(surv)':>10s} {'GP(none)':>10s} {'GP(eras)':>10s} {'GP+RS':>10s}")
print("-"*60)
for g in gp_results:
    be = g['ber_erasure']
    be_str = f"{be:.4f}" if be >= 0.001 else "<0.001"
    gp_rs = g['gp_erasure'] * 223/255
    print(f"{g['jsr']:>4d} dB {g['survival']:>9.1%} {be_str:>10s} {g['gp_none']:>10.3f} {g['gp_erasure']:>10.3f} {gp_rs:>10.3f}")

# TABLE VI
print("\n--- TABLE VI: Variable Duty Cycle @ JSR=10 dB ---")
print(f"{'Duty':>6s} {'BER(none)':>10s} {'BER(eras)':>10s} {'Survival':>10s}")
print("-"*40)
for i, d in enumerate(duty_cycles):
    be = dc_results['ber_erasure'][i]
    be_str = f"{be:.4f}" if be >= 0.001 else "<0.001"
    print(f"{d:>5.0%} {dc_results['ber_none'][i]:>10.4f} {be_str:>10s} {dc_results['survival'][i]:>9.1%}")

# TABLE VII
print("\n--- TABLE VII: AWGN vs Rayleigh @ JSR=10 dB ---")
print(f"{'Type':<12s} {'AWGN none':>10s} {'AWGN MoE':>10s} {'Rayl none':>10s} {'Rayl MoE':>10s}")
print("-"*55)
for jt in JT:
    fb = fade_ber[jt]
    print(f"{JL[jt]:<12s} {fb['awgn_none']:>10.4f} {fb['awgn_moe']:>10.4f} {fb['fade_none']:>10.4f} {fb['fade_moe']:>10.4f}")
print(f"\nClassification: AWGN {acc:.1%} -> Rayleigh {fade_accuracy:.1%}")
print(f"Routing:        AWGN 100% -> Rayleigh {fade_routing:.1%}")

# ABLATION
print("\n--- NO-SNN ABLATION ---")
print(f"SNN accuracy:          {acc:.3f}")
print(f"Conventional accuracy: {abl_acc:.3f}")
print(f"Routing comparison:")
for jt in JT:
    snn_r = routing_counts[jt].get(jt,0)/sum(routing_counts[jt].values())
    print(f"  {JL[jt]:<12s}: SNN={snn_r:.1%} Conv={abl_routing[jt]:.1%}")

# FULL BER TABLE (all JSR, for reference)
print("\n--- FULL BER vs JSR (MoE, all types) ---")
print(f"{'JSR':>4s}", end="")
for jt in JT:
    print(f" {JL[jt]:>10s}", end="")
print()
for j, jsr in enumerate(jsr_range_bl):
    print(f"{jsr:>3d}", end=" ")
    for jt in JT:
        v = baseline_ber[jt]['moe'][j]
        print(f" {v:>10.4f}" if v >= 0.001 else f" {'<0.001':>10s}", end="")
    print()


# ============================================================
# R2-EXP 6: LSTM Retrained under Rayleigh Fading
# ============================================================
print("\n" + "="*70)
print("R2-EXP 6: LSTM Retrained under Rayleigh Fading")
print("="*70)

# --- Build fading-augmented training data ---
def simulate_rayleigh_training(cfg, jam_type, fade_block=100):
    """Rayleigh fading for training data generation."""
    rng = np.random.default_rng(cfg.seed)
    bits, symbols, tx, sps = generate_bpsk(cfg, rng)
    n = len(tx)
    n_blocks = n // fade_block + 1
    h_real = rng.normal(0, 1/np.sqrt(2), n_blocks)
    h_imag = rng.normal(0, 1/np.sqrt(2), n_blocks)
    h_mag = np.sqrt(h_real**2 + h_imag**2)
    faded = np.zeros(n)
    for k in range(n_blocks):
        s0 = k * fade_block
        s1 = min((k+1) * fade_block, n)
        faded[s0:s1] = tx[s0:s1] * h_mag[k]
    noisy = add_awgn(faded, cfg.snr_db, rng)
    if jam_type == 'none':
        return {'bits': bits, 'symbols': symbols, 'tx': tx, 'rx': noisy,
                'jam': np.zeros(n), 'sps': sps, 'cfg': cfg}
    rx, jam = JAM_FNS[jam_type](noisy, cfg.jsr_db, cfg.fs, rng)
    return {'bits': bits, 'symbols': symbols, 'tx': tx, 'rx': rx,
            'jam': jam, 'sps': sps, 'cfg': cfg}

# Generate mixed AWGN + Rayleigh training pairs for NB and Sweep
print("Building fading-augmented LSTM training data...")
fade_train_pairs = []
for jt in ['narrowband', 'sweep']:
    for jsr in [4, 8, 12, 16]:
        for trial in range(10):
            seed_base = 20000 + JT.index(jt)*1000 + jsr*100 + trial
            # AWGN pair
            cfg_a = ChannelConfig(snr_db=10, jsr_db=jsr, n_symbols=10000, seed=seed_base)
            r_a = simulate_channel(cfg_a, jam_type=jt)
            # Rayleigh pair
            cfg_f = ChannelConfig(snr_db=10, jsr_db=jsr, n_symbols=10000, seed=seed_base+50000)
            r_f = simulate_rayleigh_training(cfg_f, jam_type=jt)
            fade_train_pairs.append((r_a, jt))  # keep AWGN pairs
            fade_train_pairs.append((r_f, jt))  # add fading pairs

print(f"  Training pairs: {len(fade_train_pairs)} (50% AWGN + 50% Rayleigh)")

# Build dataset
fade_X_list, fade_Y_list = [], []
for r, jt in fade_train_pairs:
    sps = r['sps']
    ns = len(r['rx']) // sps
    soft_rx = np.mean(r['rx'][:ns*sps].reshape(ns, sps), axis=1)
    soft_tx = np.mean(r['tx'][:ns*sps].reshape(ns, sps), axis=1)
    for start in range(0, ns - 200, 200):
        fade_X_list.append(soft_rx[start:start+200])
        fade_Y_list.append(soft_tx[start:start+200])

fade_X = np.array(fade_X_list)
fade_Y = np.array(fade_Y_list)
print(f"  Sequences: {len(fade_X)}")

# Train/val split
n_train = int(0.85 * len(fade_X))
idx = np.random.default_rng(42).permutation(len(fade_X))
train_idx, val_idx = idx[:n_train], idx[n_train:]

fade_Xt = torch.tensor(fade_X[train_idx], dtype=torch.float32).unsqueeze(-1)
fade_Yt = torch.tensor(fade_Y[train_idx], dtype=torch.float32).unsqueeze(-1)
fade_Xv = torch.tensor(fade_X[val_idx], dtype=torch.float32).unsqueeze(-1)
fade_Yv = torch.tensor(fade_Y[val_idx], dtype=torch.float32).unsqueeze(-1)

# New LSTM model for fading
fade_lstm = ResidualLSTMNet(hidden=24, n_layers=1).to(device)
fade_opt = torch.optim.Adam(fade_lstm.parameters(), lr=1e-3)
fade_sched = torch.optim.lr_scheduler.CosineAnnealingLR(fade_opt, T_max=200)
criterion = nn.MSELoss()

best_val = float('inf')
best_state = None

print("  Training fading-augmented LSTM...")
for epoch in range(1, 201):
    fade_lstm.train()
    fade_opt.zero_grad()
    out = fade_lstm(fade_Xt.to(device))
    loss = criterion(out, fade_Yt.to(device))
    loss.backward()
    fade_opt.step()
    fade_sched.step()

    fade_lstm.eval()
    with torch.no_grad():
        val_out = fade_lstm(fade_Xv.to(device))
        val_loss = criterion(val_out, fade_Yv.to(device)).item()
    if val_loss < best_val:
        best_val = val_loss
        best_state = {k: v.cpu().clone() for k, v in fade_lstm.state_dict().items()}
    if epoch % 50 == 0:
        print(f"    Epoch {epoch:3d}/200 | T: {loss.item():.6f} V: {val_loss:.6f}")

fade_lstm.load_state_dict(best_state)
fade_lstm.eval()
print(f"  Best val loss: {best_val:.6f}")

# Create fading-aware expert
class FadingLSTMExpert:
    def __init__(self, model):
        self.model = model
    def correct_symbols(self, signal, sps):
        ns = len(signal) // sps
        soft = np.mean(signal[:ns*sps].reshape(ns, sps), axis=1)
        x = torch.tensor(soft, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
        with torch.no_grad():
            y = self.model(x).squeeze().cpu().numpy()
        return y

fade_lstm_expert = FadingLSTMExpert(fade_lstm)

# --- Evaluate: Original LSTM vs Fading-retrained LSTM under Rayleigh ---
print("\nEvaluating AWGN-trained vs Fading-trained LSTM under Rayleigh...")
n_trials_fr = 20
n_sym_fr = 10000

fade_retrain_ber = {}
for jt in ['narrowband', 'sweep']:
    b_orig_awgn, b_orig_fade = [], []
    b_retrain_awgn, b_retrain_fade = [], []
    b_none_awgn, b_none_fade = [], []
    for trial in range(n_trials_fr):
        seed = 25000 + trial + JT.index(jt)*500

        # AWGN
        cfg = ChannelConfig(snr_db=10, jsr_db=10, n_symbols=n_sym_fr, seed=seed)
        r_a = simulate_channel(cfg, jam_type=jt)
        sps = r_a['sps']
        d, _ = demodulate_bpsk(r_a['rx'], sps)
        b_none_awgn.append(compute_ber(r_a['bits'], d))
        # original LSTM
        ls_a = lstm_expert.correct_symbols(r_a['rx'], sps)
        b_orig_awgn.append(compute_ber(r_a['bits'], (ls_a > 0).astype(int)))
        # retrained LSTM
        ls_ar = fade_lstm_expert.correct_symbols(r_a['rx'], sps)
        ns = min(len(r_a['bits']), len(ls_ar))
        b_retrain_awgn.append(compute_ber(r_a['bits'][:ns], (ls_ar[:ns] > 0).astype(int)))

        # Rayleigh
        cfg_f = ChannelConfig(snr_db=10, jsr_db=10, n_symbols=n_sym_fr, seed=seed+50000)
        r_f = simulate_rayleigh_training(cfg_f, jam_type=jt)
        d, _ = demodulate_bpsk(r_f['rx'], sps)
        b_none_fade.append(compute_ber(r_f['bits'], d))
        # original LSTM
        ls_f = lstm_expert.correct_symbols(r_f['rx'], sps)
        b_orig_fade.append(compute_ber(r_f['bits'], (ls_f > 0).astype(int)))
        # retrained LSTM
        ls_fr = fade_lstm_expert.correct_symbols(r_f['rx'], sps)
        ns = min(len(r_f['bits']), len(ls_fr))
        b_retrain_fade.append(compute_ber(r_f['bits'][:ns], (ls_fr[:ns] > 0).astype(int)))

    fade_retrain_ber[jt] = {
        'none_awgn': np.mean(b_none_awgn), 'none_fade': np.mean(b_none_fade),
        'orig_awgn': np.mean(b_orig_awgn), 'orig_fade': np.mean(b_orig_fade),
        'retrain_awgn': np.mean(b_retrain_awgn), 'retrain_fade': np.mean(b_retrain_fade),
        'none_awgn_ci': 1.96*np.std(b_none_awgn)/np.sqrt(n_trials_fr),
        'orig_fade_ci': 1.96*np.std(b_orig_fade)/np.sqrt(n_trials_fr),
        'retrain_fade_ci': 1.96*np.std(b_retrain_fade)/np.sqrt(n_trials_fr),
    }
    fb = fade_retrain_ber[jt]
    print(f"  {JL[jt]:12s}:")
    print(f"    No corr:      AWGN={fb['none_awgn']:.4f}  Rayleigh={fb['none_fade']:.4f}")
    print(f"    AWGN LSTM:     AWGN={fb['orig_awgn']:.4f}  Rayleigh={fb['orig_fade']:.4f}")
    print(f"    Fading LSTM:   AWGN={fb['retrain_awgn']:.4f}  Rayleigh={fb['retrain_fade']:.4f}")

# Also re-evaluate BB and PL under fading (these don't use LSTM)
for jt in ['broadband', 'pulse']:
    b_none_f, b_moe_f = [], []
    for trial in range(n_trials_fr):
        cfg = ChannelConfig(snr_db=10, jsr_db=10, n_symbols=n_sym_fr,
                            seed=26000+trial+JT.index(jt)*500)
        r_f = simulate_rayleigh_training(cfg, jam_type=jt)
        sps = r_f['sps']
        d, _ = demodulate_bpsk(r_f['rx'], sps)
        b_none_f.append(compute_ber(r_f['bits'], d))
        res = moe.process(r_f['rx'], sps)
        b_moe_f.append(compute_ber(r_f['bits'][res['ber_indices']], res['ber_bits']))
    fade_retrain_ber[jt] = {
        'none_fade': np.mean(b_none_f),
        'moe_fade': np.mean(b_moe_f),
    }
    print(f"  {JL[jt]:12s}: no corr={np.mean(b_none_f):.4f} MoE={np.mean(b_moe_f):.4f}")

# --- Print combined fading table ---
print("\n" + "="*80)
print("TABLE VIII: LSTM Retrained under Rayleigh Fading (JSR=10 dB, SNR=10 dB)")
print("="*80)
print(f"{'Type':<12s} {'No Corr':>10s} {'AWGN LSTM':>10s} {'Fade LSTM':>10s} {'Gain':>8s}")
print("-"*50)
for jt in ['narrowband', 'sweep']:
    fb = fade_retrain_ber[jt]
    gain = fb['orig_fade'] / max(fb['retrain_fade'], 1e-5)
    print(f"{JL[jt]:<12s} {fb['none_fade']:>10.4f} {fb['orig_fade']:>10.4f} "
          f"{fb['retrain_fade']:>10.4f} {gain:>7.1f}x")
for jt in ['broadband', 'pulse']:
    fb = fade_retrain_ber[jt]
    print(f"{JL[jt]:<12s} {fb['none_fade']:>10.4f} {'N/A':>10s} {fb['moe_fade']:>10.4f} {'--':>8s}")

print("\nAWGN performance preserved:")
for jt in ['narrowband', 'sweep']:
    fb = fade_retrain_ber[jt]
    print(f"  {JL[jt]:12s}: orig={fb['orig_awgn']:.4f} retrained={fb['retrain_awgn']:.4f}")


# ============================================================
# R3-EXP 1: Dynamic Jammer-Switching
# ============================================================
print("\n" + "="*70)
print("R3-EXP 1: Dynamic Jammer-Switching Experiment")
print("="*70)

from scipy.signal import iirnotch, lfilter

scenario_frames = (
    [('broadband', 10)] * 20 +
    [('narrowband', 10)] * 15 +
    [('sweep', 10)] * 15 +
    [('pulse', 10)] * 20 +
    [('broadband', 10)] * 10 +
    [('sweep', 10)] * 10 +
    [('pulse', 10)] * 10
)
n_frames = len(scenario_frames)
n_trials_switch = 10

def eval_lstm_only(r, sps):
    ls = lstm_expert.correct_symbols(r['rx'], sps)
    return compute_ber(r['bits'], (ls > 0).astype(int))

def eval_notch_only(r, sps):
    f0 = 0.3 * r['cfg'].fs / 2
    b, a = iirnotch(f0, 30, r['cfg'].fs)
    rx_n = lfilter(b, a, r['rx'])
    d, _ = demodulate_bpsk(rx_n, sps)
    return compute_ber(r['bits'], d)

def eval_blanker_only(r, sps):
    ns = len(r['rx']) // sps
    soft = np.mean(r['rx'][:ns*sps].reshape(ns, sps), axis=1)
    win = 5 * sps; n_g = len(r['rx']) // win
    rms = np.array([np.sqrt(np.mean(r['rx'][k*win:(k+1)*win]**2)) for k in range(n_g)])
    mu, st = np.mean(rms), max(np.std(rms), 1e-6)
    er = np.zeros(ns, dtype=bool)
    for k in range(n_g):
        if (rms[k]-mu)/st > 3.0:
            er[k*5:min((k+1)*5,ns)] = True
    si = np.where(~er)[0]
    if len(si)==0: return 0.5
    return compute_ber(r['bits'][si], (soft[si]>0).astype(int))

def eval_moe(r, sps):
    res = moe.process(r['rx'], sps, snr_db=10.0)
    bi, bb = res['ber_indices'], res['ber_bits']
    tb = r['bits'][bi] if len(bi)<=len(r['bits']) else r['bits'][:len(bb)]
    return compute_ber(tb, bb)

def eval_no_corr(r, sps):
    d, _ = demodulate_bpsk(r['rx'], sps)
    return compute_ber(r['bits'], d)

# Conv-feature router
from sklearn.preprocessing import StandardScaler as SS2
def extract_cf(signal, nsf=20):
    n=len(signal); rms=np.sqrt(np.mean(signal**2)); var=np.var(signal)
    mu=np.mean(signal); kurt=np.mean((signal-mu)**4)/(var**2+1e-10)-3
    ft=np.abs(np.fft.rfft(signal)); gm=np.exp(np.mean(np.log(ft[1:]+1e-10)))
    am=np.mean(ft[1:]); fl=gm/(am+1e-10); pr=np.max(ft[1:])/(am+1e-10)
    sl=n//nsf; se=[np.mean(signal[k*sl:(k+1)*sl]**2) for k in range(nsf)]
    sv=np.var(se)/(np.mean(se)+1e-10); zc=np.sum(np.diff(np.sign(signal))!=0)/n
    return np.array([rms,var,kurt,fl,pr,sv,zc])

print("Training conv router...")
from sklearn.ensemble import RandomForestClassifier as RFC2
cf = np.array([extract_cf(s) for s in cls_signals])
cs = SS2().fit(cf); cX = cs.transform(cf)
conv_clf = RFC2(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
conv_clf.fit(cX, cls_labels)

def eval_conv_moe(r, sps):
    feat = extract_cf(r['rx']); X = cs.transform(feat.reshape(1,-1))
    pred = conv_clf.predict(X)[0]
    if pred in ['narrowband','sweep']:
        ls = lstm_expert.correct_symbols(r['rx'], sps)
        return compute_ber(r['bits'], (ls>0).astype(int))
    elif pred == 'pulse':
        pr = pulse_expert.correct_symbols(r['rx'], sps)
        return compute_ber(r['bits'][pr['surviving_indices']], pr['surviving_bits'])
    else:
        d, _ = demodulate_bpsk(r['rx'], sps)
        return compute_ber(r['bits'], d)

methods_sw = {'No correction': eval_no_corr, 'LSTM-only': eval_lstm_only,
              'Fixed notch': eval_notch_only, 'Fixed blanker': eval_blanker_only,
              'Conv-MoE': eval_conv_moe, 'SNN-MoE': eval_moe}

all_sw = {m: np.zeros((n_trials_switch, n_frames)) for m in methods_sw}
sw_routing = np.zeros((n_trials_switch, n_frames), dtype=object)

print(f"Running {n_frames} frames x {n_trials_switch} trials...")
for trial in range(n_trials_switch):
    for fi, (jt, jsr) in enumerate(scenario_frames):
        cfg = ChannelConfig(snr_db=10, jsr_db=jsr, n_symbols=10000, seed=30000+trial*1000+fi)
        r = simulate_channel(cfg, jam_type=jt); sps = r['sps']
        for mn, mf in methods_sw.items():
            all_sw[mn][trial, fi] = mf(r, sps)
        _, _, pred, _, _ = snn.detect_and_classify(r['rx'])
        sw_routing[trial, fi] = pred
    if (trial+1)%5==0: print(f"  Trial {trial+1}/{n_trials_switch}")

segments = [('BB-1',0,20,'broadband'),('NB',20,35,'narrowband'),('SW-1',35,50,'sweep'),
            ('PL-1',50,70,'pulse'),('BB-2',70,80,'broadband'),('SW-2',80,90,'sweep'),
            ('PL-2',90,100,'pulse')]

print("\n--- Jammer-Switching Results ---")
for m in methods_sw:
    print(f"  {m:<15s}: {np.mean(all_sw[m]):.4f}")

# Routing accuracy
rt_ok = sum(1 for t in range(n_trials_switch) for fi,(jt,_) in enumerate(scenario_frames)
            if (jt in ['narrowband','sweep'] and sw_routing[t,fi] in ['narrowband','sweep'])
            or sw_routing[t,fi]==jt)
print(f"  Routing: {rt_ok}/{n_trials_switch*n_frames} = {rt_ok/(n_trials_switch*n_frames):.1%}")


# ============================================================
# R3-EXP 2: RS(255,223) Coded-Link
# ============================================================
print("\n" + "="*70)
print("R3-EXP 2: RS(255,223) Coded-Link Validation")
print("="*70)

from reedsolo import RSCodec, ReedSolomonError
rs_codec = RSCodec(32)
n_rs, k_rs = 255, 223
n_trials_coded = 50
n_codewords = 20
jsr_coded = [6, 8, 10, 12, 14, 16]

def pulse_blank(rx, sps, gs=5, zt=3.0):
    ns=len(rx)//sps; soft=np.mean(rx[:ns*sps].reshape(ns,sps),axis=1)
    win=gs*sps; ng=len(rx)//win
    rms=np.array([np.sqrt(np.mean(rx[k*win:(k+1)*win]**2)) for k in range(ng)])
    mu,st=np.mean(rms),max(np.std(rms),1e-6)
    er=np.zeros(ns,dtype=bool)
    for k in range(ng):
        if (rms[k]-mu)/st>zt: er[k*gs:min((k+1)*gs,ns)]=True
    return soft, (soft>0).astype(np.uint8), er

results_rs = {}
for jsr in jsr_coded:
    tbler = {m:[] for m in ['raw','nocorr_rs','lstm_rs','blnk_rs','moe_rs']}
    for trial in range(n_trials_coded):
        nf = {m:0 for m in tbler}; nt = {m:0 for m in tbler}
        for cw in range(n_codewords):
            seed = 40000+jsr*1000+trial*100+cw
            rng_d = np.random.default_rng(seed)
            data = rng_d.integers(0,256,size=k_rs,dtype=np.uint8)
            enc = np.array(list(rs_codec.encode(data)),dtype=np.uint8)
            tx_bits = np.unpackbits(enc); nsym = len(tx_bits)
            sym = 2.0*tx_bits.astype(float)-1.0; sps=10
            tx_s = np.repeat(sym,sps)
            rng_c = np.random.default_rng(seed+500000)
            noisy = tx_s + rng_c.normal(0,np.sqrt(0.1),len(tx_s))
            sp=np.mean(tx_s**2); jp=sp*10**(jsr/10)/0.2; per=500
            env=np.zeros(len(noisy)); on=int(per*0.2)
            for i in range(0,len(noisy),per): env[i:min(i+on,len(noisy))]=1.0
            rx = noisy + env*rng_c.normal(0,np.sqrt(jp),len(noisy))

            ns=len(rx)//sps; soft=np.mean(rx[:ns*sps].reshape(ns,sps),axis=1)
            hard=(soft[:nsym]>0).astype(np.uint8)
            _, _, er_mask = pulse_blank(rx, sps)

            # raw
            nf['raw'] += (1 if np.sum(hard!=tx_bits[:nsym])>0 else 0); nt['raw']+=1
            # nocorr+RS
            try:
                dec=np.array(list(rs_codec.decode(np.packbits(hard)[:n_rs])),dtype=np.uint8)[:k_rs]
                nf['nocorr_rs']+=(1 if np.sum(dec!=data)>0 else 0)
            except ReedSolomonError: nf['nocorr_rs']+=1
            nt['nocorr_rs']+=1
            # lstm+RS
            lx=torch.tensor(soft[:nsym],dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
            with torch.no_grad(): ly=lstm_expert.model(lx).squeeze().cpu().numpy()
            lb=(ly[:nsym]>0).astype(np.uint8)
            try:
                dec=np.array(list(rs_codec.decode(np.packbits(lb)[:n_rs])),dtype=np.uint8)[:k_rs]
                nf['lstm_rs']+=(1 if np.sum(dec!=data)>0 else 0)
            except ReedSolomonError: nf['lstm_rs']+=1
            nt['lstm_rs']+=1
            # blanker+RS (no erasure info)
            bh=hard.copy(); bh[er_mask[:nsym]]=0
            try:
                dec=np.array(list(rs_codec.decode(np.packbits(bh)[:n_rs])),dtype=np.uint8)[:k_rs]
                nf['blnk_rs']+=(1 if np.sum(dec!=data)>0 else 0)
            except ReedSolomonError: nf['blnk_rs']+=1
            nt['blnk_rs']+=1
            # MoE+RS (erasure-aware)
            eb = sorted(set(idx//8 for idx in np.where(er_mask[:nsym])[0] if idx//8<n_rs))
            try:
                dec=np.array(list(rs_codec.decode(np.packbits(hard)[:n_rs],erase_pos=eb)),dtype=np.uint8)[:k_rs]
                nf['moe_rs']+=(1 if np.sum(dec!=data)>0 else 0)
            except ReedSolomonError: nf['moe_rs']+=1
            nt['moe_rs']+=1

        for m in tbler: tbler[m].append(nf[m]/max(nt[m],1))

    results_rs[jsr] = {m: {'bler':np.mean(tbler[m]), 'ci':1.96*np.std(tbler[m])/np.sqrt(n_trials_coded)}
                        for m in tbler}
    rc=results_rs[jsr]
    print(f"  JSR={jsr:2d}: NoCorr={rc['nocorr_rs']['bler']:.3f} LSTM={rc['lstm_rs']['bler']:.3f} "
          f"Blnk={rc['blnk_rs']['bler']:.3f} MoE*={rc['moe_rs']['bler']:.3f}")


# ============================================================
# COMPREHENSIVE RESULTS + DRIVE SAVE
# ============================================================
import shutil
print("\n\n")
print("="*80)
print("ALL EXPERIMENTS COMPLETE - SAVING TO DRIVE")
print("="*80)

# --- Save all figures ---
print("\n[1/4] Saving figures...")
for fig_file in OUT.glob('*.png'):
    shutil.copy2(fig_file, DRIVE_FIGS / fig_file.name)
    print(f"  -> figures/{fig_file.name}")

# --- Save all tables as single text file ---
print("\n[2/4] Saving tables...")
with open(DRIVE_TABLES / 'all_results.txt', 'w') as f:
    f.write(f"SNN-MoE Anti-Jamming Complete Results\n")
    f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M')}\n")
    f.write("="*80 + "\n\n")

    # Table II: BER
    jsr10_bl = list(jsr_range_bl).index(10)
    f.write("TABLE II: BER @ JSR=10 dB, SNR=10 dB\n")
    f.write(f"{'Type':<12s} {'No Corr':>8s} {'Notch':>8s} {'Blanker':>8s} {'LSTM':>8s} {'MoE':>8s}\n")
    f.write("-"*56 + "\n")
    for jt in JT:
        row = f"{JL[jt]:<12s}"
        for m in ['none','notch','blanker','lstm_only','moe']:
            v = baseline_ber[jt][m][jsr10_bl]
            row += f" {v:>8.4f}" if v >= 0.001 else f" {'<0.001':>8s}"
        f.write(row + "\n")

    # Table V: Goodput
    f.write(f"\nTABLE V: Pulse Goodput @ SNR=10 dB\n")
    f.write(f"{'JSR':>6s} {'Survival':>10s} {'GP(none)':>10s} {'GP(eras)':>10s}\n")
    for g in gp_results:
        f.write(f"{g['jsr']:>4d} dB {g['survival']:>9.1%} {g['gp_none']:>10.3f} {g['gp_erasure']:>10.3f}\n")

    # Table VI: Variable DC
    f.write(f"\nTABLE VI: Variable Duty Cycle @ JSR=10 dB\n")
    f.write(f"{'Duty':>6s} {'BER(none)':>10s} {'BER(eras)':>10s} {'Survival':>10s}\n")
    for i, d in enumerate(duty_cycles):
        be = dc_results['ber_erasure'][i]
        be_str = f"{be:.4f}" if be >= 0.001 else "<0.001"
        f.write(f"{d:>5.0%} {dc_results['ber_none'][i]:>10.4f} {be_str:>10s} {dc_results['survival'][i]:>9.1%}\n")

    # Table VII: Fading
    f.write(f"\nTABLE VII: AWGN vs Rayleigh @ JSR=10 dB\n")
    f.write(f"{'Type':<12s} {'AWGN MoE':>10s} {'Rayl MoE':>10s}\n")
    for jt in JT:
        fb = fade_ber[jt]
        f.write(f"{JL[jt]:<12s} {fb['awgn_moe']:>10.4f} {fb['fade_moe']:>10.4f}\n")
    f.write(f"Classification: AWGN {acc:.1%} -> Rayleigh {fade_accuracy:.1%}\n")

    # Table VIII: Fading retrain
    f.write(f"\nTABLE VIII: Fading Retrain (JSR=10 dB)\n")
    f.write(f"{'Type':<12s} {'No Corr':>10s} {'AWGN LSTM':>10s} {'Fade LSTM':>10s}\n")
    for jt in ['narrowband','sweep']:
        fb = fade_retrain_ber[jt]
        f.write(f"{JL[jt]:<12s} {fb['none_fade']:>10.4f} {fb['orig_fade']:>10.4f} {fb['retrain_fade']:>10.4f}\n")

    # Table IX: Jammer switching
    f.write(f"\nTABLE IX: Dynamic Jammer-Switching (Overall BER)\n")
    for m in methods_sw:
        f.write(f"  {m:<15s}: {np.mean(all_sw[m]):.4f}\n")
    f.write(f"  Routing: {rt_ok}/{n_trials_switch*n_frames}\n")

    # Table X: Coded-link
    f.write(f"\nTABLE X: RS(255,223) Coded-Link BLER\n")
    f.write(f"{'JSR':>5s} {'NoCorr+RS':>10s} {'LSTM+RS':>10s} {'Blnk+RS':>10s} {'MoE+RS*':>10s}\n")
    for jsr in jsr_coded:
        rc = results_rs[jsr]
        def fb(v): return "<0.001" if v<0.001 else f"{v:.3f}"
        f.write(f"{jsr:>4d}dB {fb(rc['nocorr_rs']['bler']):>10s} {fb(rc['lstm_rs']['bler']):>10s} "
                f"{fb(rc['blnk_rs']['bler']):>10s} {fb(rc['moe_rs']['bler']):>10s}\n")

    # Ablation
    f.write(f"\nNO-SNN ABLATION\n")
    f.write(f"SNN accuracy: {acc:.3f}\n")
    f.write(f"Conventional accuracy: {abl_acc:.3f}\n")

    # Power
    f.write(f"\nPOWER ESTIMATES\n")
    for n, v in powers.items():
        f.write(f"  {n.replace(chr(10),' '):<30s} {v:>8.3f} mW\n")

print(f"  -> tables/all_results.txt")

# --- Save models ---
print("\n[3/4] Saving models...")
torch.save(lstm_expert.model.state_dict(), DRIVE_MODELS / 'lstm_expert_awgn.pt')
torch.save(fade_lstm.state_dict(), DRIVE_MODELS / 'lstm_expert_fading.pt')
print(f"  -> models/lstm_expert_awgn.pt")
print(f"  -> models/lstm_expert_fading.pt")

# --- Save notebook ---
print("\n[4/4] Saving notebook...")
try:
    nb_src = Path('/content/SNN_MoE_AntiJamming_v33_R2_single__3_.ipynb')
    if nb_src.exists():
        shutil.copy2(nb_src, DRIVE_NB / 'notebook_latest.ipynb')
        print(f"  -> notebooks/notebook_latest.ipynb")
except: pass

# --- Print everything for chat ---
print("\n\n")
print("="*80)
print("COMPLETE RESULTS FOR CHAT")
print("="*80)

print(f"\n--- TABLE II: BER @ JSR=10 dB ---")
print(f"{'Type':<12s} {'No Corr':>8s} {'Notch':>8s} {'Blanker':>8s} {'LSTM':>8s} {'MoE':>8s}")
for jt in JT:
    row = f"{JL[jt]:<12s}"
    for m in ['none','notch','blanker','lstm_only','moe']:
        v = baseline_ber[jt][m][jsr10_bl]
        row += f" {v:>8.4f}" if v >= 0.001 else f" {'<0.001':>8s}"
    print(row)

print(f"\n--- TABLE IX: Jammer-Switching ---")
for m in methods_sw:
    print(f"  {m:<15s}: {np.mean(all_sw[m]):.4f} +/- {np.std(np.mean(all_sw[m],axis=1)):.4f}")
print(f"  Routing: {rt_ok}/{n_trials_switch*n_frames}")

print(f"\n--- TABLE X: RS Coded-Link BLER ---")
print(f"{'JSR':>5s} {'NoCorr':>8s} {'LSTM':>8s} {'Blnk':>8s} {'MoE*':>8s}")
for jsr in jsr_coded:
    rc=results_rs[jsr]
    def fb(v): return "<0.001" if v<0.001 else f"{v:.3f}"
    print(f"{jsr:>4d}dB {fb(rc['nocorr_rs']['bler']):>8s} {fb(rc['lstm_rs']['bler']):>8s} "
          f"{fb(rc['blnk_rs']['bler']):>8s} {fb(rc['moe_rs']['bler']):>8s}")

print(f"\n--- TABLE VIII: Fading Retrain ---")
for jt in ['narrowband','sweep']:
    fb=fade_retrain_ber[jt]
    print(f"  {JL[jt]:12s}: none={fb['none_fade']:.4f} awgn_lstm={fb['orig_fade']:.4f} fade_lstm={fb['retrain_fade']:.4f}")

print(f"\n--- Ablation ---")
print(f"  SNN: {acc:.3f}, Conv: {abl_acc:.3f}")

print(f"\nAll saved to: {DRIVE_ROOT}")
print("="*80)


# ============================================================
# R3-EXP 2 v3: RS Coded-Link - FIXED blanker + RS decode
# PASTE into Colab cell (variables in memory from main notebook)
# ============================================================
from reedsolo import RSCodec, ReedSolomonError
import time
t0 = time.time()

print("="*70)
print("R3-EXP 2: RS Coded-Link, Multiple Code Rates")
print("="*70)

def robust_blanker(rx, sps, gs=5, z_thresh=3.0):
    """Blanker using median/MAD (robust to 20% ON contamination)."""
    ns = len(rx) // sps
    soft = np.mean(rx[:ns*sps].reshape(ns, sps), axis=1)
    win = gs * sps
    ng = len(rx) // win
    rms = np.array([np.sqrt(np.mean(rx[k*win:(k+1)*win]**2)) for k in range(ng)])
    # Robust: median and MAD instead of mean/std
    med = np.median(rms)
    mad = np.median(np.abs(rms - med)) * 1.4826  # scale to match std
    mad = max(mad, 1e-6)
    z = (rms - med) / mad
    er = np.zeros(ns, dtype=bool)
    for k in range(ng):
        if z[k] > z_thresh:
            er[k*gs:min((k+1)*gs, ns)] = True
    return (soft > 0).astype(np.uint8), er

def safe_rs_decode(rs, data, erase_pos=None):
    """Handle both old and new reedsolo return formats."""
    if erase_pos is not None:
        result = rs.decode(data, erase_pos=erase_pos)
    else:
        result = rs.decode(data)
    # reedsolo may return (decoded, remainder, errata) or just bytearray
    if isinstance(result, tuple):
        return bytearray(result[0])
    return bytearray(result)

codes = [
    (32,  'RS(255,223)', 0.875),
    (64,  'RS(255,191)', 0.749),
    (96,  'RS(255,159)', 0.624),
    (128, 'RS(255,127)', 0.498),
]

n_trials = 50
n_cw = 20
jsr_list = [6, 8, 10, 12, 14, 16]

all_rs = {}
for n_parity, code_name, rate in codes:
    k = 255 - n_parity
    rs = RSCodec(n_parity)
    print(f"\n--- {code_name}: k={k}, {n_parity} erasure cap, rate={rate:.3f} ---")

    code_results = {}
    for jsr in jsr_list:
        bler_trials = {'nocorr': [], 'blanker': [], 'erasure': []}

        for trial in range(n_trials):
            nf = {'nocorr': 0, 'blanker': 0, 'erasure': 0}
            nt = 0
            last_n_erased = 0

            for cw in range(n_cw):
                seed = 40000 + jsr*1000 + trial*100 + cw
                rng_d = np.random.default_rng(seed)
                data = rng_d.integers(0, 256, size=k, dtype=np.uint8)
                enc_bytes = rs.encode(data)
                enc = np.array(list(enc_bytes), dtype=np.uint8)
                tx_bits = np.unpackbits(enc)
                nsym = len(tx_bits)

                sym = 2.0 * tx_bits.astype(float) - 1.0
                sps = 10
                tx_s = np.repeat(sym, sps)
                rng_c = np.random.default_rng(seed + 500000)
                noisy = tx_s + rng_c.normal(0, np.sqrt(0.1), len(tx_s))

                sp = np.mean(tx_s**2)
                jp = sp * 10**(jsr/10) / 0.2
                per = 500
                env = np.zeros(len(noisy))
                on_len = int(per * 0.2)
                for i in range(0, len(noisy), per):
                    env[i:min(i+on_len, len(noisy))] = 1.0
                rx = noisy + env * rng_c.normal(0, np.sqrt(jp), len(noisy))

                # Demod + robust blanker
                hard, er_mask = robust_blanker(rx, sps)
                hard = hard[:nsym]
                erased_sym = np.where(er_mask[:nsym])[0]
                erased_bytes = sorted(set(idx // 8 for idx in erased_sym if idx // 8 < 255))
                last_n_erased = len(erased_bytes)
                nt += 1

                rx_bytes = np.packbits(hard)[:255]

                # 1: No correction + RS
                try:
                    dec = safe_rs_decode(rs, rx_bytes)[:k]
                    if np.array(list(dec), dtype=np.uint8).tolist() != data.tolist():
                        nf['nocorr'] += 1
                except ReedSolomonError:
                    nf['nocorr'] += 1

                # 2: Blanker + RS (no erasure info)
                bh = hard.copy()
                bh[er_mask[:nsym]] = 0
                try:
                    dec = safe_rs_decode(rs, np.packbits(bh)[:255])[:k]
                    if np.array(list(dec), dtype=np.uint8).tolist() != data.tolist():
                        nf['blanker'] += 1
                except ReedSolomonError:
                    nf['blanker'] += 1

                # 3: MoE erasure-aware RS
                try:
                    dec = safe_rs_decode(rs, rx_bytes, erase_pos=erased_bytes)[:k]
                    if np.array(list(dec), dtype=np.uint8).tolist() != data.tolist():
                        nf['erasure'] += 1
                except ReedSolomonError:
                    nf['erasure'] += 1

            for m in bler_trials:
                bler_trials[m].append(nf[m] / nt)

        code_results[jsr] = {m: np.mean(bler_trials[m]) for m in bler_trials}
        code_results[jsr]['n_erased'] = last_n_erased

        rc = code_results[jsr]
        def fb(v): return "<0.001" if v < 0.001 else f"{v:.3f}"
        print(f"  JSR={jsr:2d}: NoCorr={fb(rc['nocorr'])} Blanker={fb(rc['blanker'])} "
              f"Erasure*={fb(rc['erasure'])} [{last_n_erased} erased bytes]")

    all_rs[code_name] = code_results

elapsed = time.time() - t0
print(f"\nDone in {elapsed:.1f}s")

# --- Tables ---
print("\n" + "="*80)
print("BLER at JSR=10 dB, SNR=10 dB")
print("="*80)
print(f"{'Code':<14s} {'Rate':>6s} {'Cap':>5s} {'NoCorr':>8s} {'Blanker':>8s} {'MoE*':>8s}")
print("-"*50)
for n_parity, code_name, rate in codes:
    rc = all_rs[code_name].get(10, {})
    def fb(v): return "<0.001" if v < 0.001 else f"{v:.3f}"
    print(f"{code_name:<14s} {rate:>6.3f} {n_parity:>5d} "
          f"{fb(rc.get('nocorr',1)):>8s} {fb(rc.get('blanker',1)):>8s} {fb(rc.get('erasure',1)):>8s}")

print(f"\n--- RS(255,191) across JSR ---")
print(f"{'JSR':>5s} {'NoCorr':>8s} {'Blanker':>8s} {'MoE*':>8s}")
for jsr in jsr_list:
    rc = all_rs['RS(255,191)'][jsr]
    def fb(v): return "<0.001" if v < 0.001 else f"{v:.3f}"
    print(f"{jsr:>4d}dB {fb(rc['nocorr']):>8s} {fb(rc['blanker']):>8s} {fb(rc['erasure']):>8s}")

print(f"\n--- Throughput at JSR=10 ---")
for n_parity, code_name, rate in codes:
    k = 255 - n_parity
    bler = all_rs[code_name].get(10, {}).get('erasure', 1)
    eff = (1.0 - bler) * k * 8
    print(f"  {code_name}: {eff:.0f} bits/cw (BLER={bler:.3f})")

# Save
results_file = DRIVE_TABLES / 'coded_link_results.txt'
with open(results_file, 'w') as f:
    f.write("RS Coded-Link BLER\n\n")
    for n_parity, code_name, rate in codes:
        f.write(f"\n{code_name}:\n")
        for jsr in jsr_list:
            rc = all_rs[code_name][jsr]
            def fb(v): return "<0.001" if v < 0.001 else f"{v:.3f}"
            f.write(f"  JSR={jsr}: NoCorr={fb(rc['nocorr'])} Blanker={fb(rc['blanker'])} MoE={fb(rc['erasure'])} [{rc['n_erased']} bytes]\n")
print(f"\nSaved: {results_file}")

print("\n" + "="*80)
print("COPY ALL ABOVE INTO CHAT")
print("="*80)