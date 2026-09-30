"""Original receiver definitions. Importing this module does not start a run."""
import subprocess, sys

import numpy as np

import subprocess, sys

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import os

JT = ['broadband', 'narrowband', 'sweep', 'pulse']

JT_ALL = ['none'] + JT  # 5 classes for classifier

JL = {'none':'Clean', 'broadband':'Broadband', 'narrowband':'Narrowband',
      'sweep':'Sweep', 'pulse':'Pulse'}

JC = {'none':'#888780', 'broadband':'#534AB7', 'narrowband':'#1D9E75',
      'sweep':'#D85A30', 'pulse':'#D4537E'}

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

import gc

from scipy.signal import iirnotch, lfilter

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

from scipy.signal import iirnotch, lfilter

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

from sklearn.preprocessing import StandardScaler as SS2

def extract_cf(signal, nsf=20):
    n=len(signal); rms=np.sqrt(np.mean(signal**2)); var=np.var(signal)
    mu=np.mean(signal); kurt=np.mean((signal-mu)**4)/(var**2+1e-10)-3
    ft=np.abs(np.fft.rfft(signal)); gm=np.exp(np.mean(np.log(ft[1:]+1e-10)))
    am=np.mean(ft[1:]); fl=gm/(am+1e-10); pr=np.max(ft[1:])/(am+1e-10)
    sl=n//nsf; se=[np.mean(signal[k*sl:(k+1)*sl]**2) for k in range(nsf)]
    sv=np.var(se)/(np.mean(se)+1e-10); zc=np.sum(np.diff(np.sign(signal))!=0)/n
    return np.array([rms,var,kurt,fl,pr,sv,zc])

from sklearn.ensemble import RandomForestClassifier as RFC2

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

from reedsolo import RSCodec, ReedSolomonError

def pulse_blank(rx, sps, gs=5, zt=3.0):
    ns=len(rx)//sps; soft=np.mean(rx[:ns*sps].reshape(ns,sps),axis=1)
    win=gs*sps; ng=len(rx)//win
    rms=np.array([np.sqrt(np.mean(rx[k*win:(k+1)*win]**2)) for k in range(ng)])
    mu,st=np.mean(rms),max(np.std(rms),1e-6)
    er=np.zeros(ns,dtype=bool)
    for k in range(ng):
        if (rms[k]-mu)/st>zt: er[k*gs:min((k+1)*gs,ns)]=True
    return soft, (soft>0).astype(np.uint8), er

import shutil

from reedsolo import RSCodec, ReedSolomonError

import time

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

