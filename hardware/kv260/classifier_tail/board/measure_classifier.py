#!/usr/bin/env python3
"""D6 power protocol — energy per router inference (feature finalisation + Random Forest) on the KV260.
   The IP repeats the per-frame classification tail at a paced rate far above the link's 10 frames/s so that it rises
   out of the INA260 noise; energy per inference = P_dyn / rate, then per input sample = E_inf / n_samples.
   Modes: R0 overhead (loop + pace), R1 features only, R2 features + forest 'fixed', R3 features + forest 'random'.
   sudo python3 measure_classifier.py kv260_package/vectors [--block 60] [--repeats 3] [--target auto] [--out power_records.json]
   --target = paced inferences per second per replica ('auto' = 90 % of the slowest unpaced mode, so that every mode runs at
   the same rate); all modes are paced to that rate, plus R3 at half rate as a linearity check (E/inference must not change)."""
import argparse, glob, json, math, os, sys, time
import numpy as np
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rf_ctl import RF, words_from_expected, MODES, RC_RUN, Q
from measurement_support import find_hwmon, set_performance, log_power, F_CLK, F_CLK_SRC

FRAMES = ['fixed_narrowband_0', 'fixed_sweep_0', 'fixed_pulse_0', 'fixed_broadband_0']     # one per replica


def calibrate(m, mode, target, n_infer=2000, iters=4):
    pace = 0; hist = []
    for _ in range(iters):
        dt = m.call(RC_RUN, arg0=n_infer, mode=mode, pace=pace); rate = n_infer / dt; hist.append((pace, rate))
        if target <= 0: break
        cyc = F_CLK / rate; want = F_CLK / target
        new = max(0, int(round(pace + (want - cyc))))
        if new == pace: break
        pace = new
    return pace, hist


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('vdir'); ap.add_argument('--block', type=float, default=60.0); ap.add_argument('--repeats', type=int, default=3)
    ap.add_argument('--target', default='auto'); ap.add_argument('--warmup', type=float, default=3.0); ap.add_argument('--out', default='power_records.json')
    a = ap.parse_args(); Path(a.out).parent.mkdir(parents=True, exist_ok=True); set_performance(); pw = find_hwmon(); m = RF()
    m.call(0); res = m.results(); n_repl = res[Q["N_REPL"]]
    for q, name in enumerate(FRAMES[:n_repl]):
        e = json.load(open(os.path.join(a.vdir, f'{name}_expected.json'))); w, n = words_from_expected(e); m.load_words(w, n, replica=q)
    print(f'[MEAS-RF] N_REPL={n_repl}, frames {FRAMES[:n_repl]}, block {a.block:.0f} s x {a.repeats}, target {a.target} inf/s/replica, PL clock {F_CLK / 1e6:.3f} MHz ({F_CLK_SRC}), sensor {pw}')
    # unpaced rates first (sets the achievable ceiling)
    unpaced = {}
    for name, mode in MODES.items():
        p, h = calibrate(m, mode, 0); unpaced[name] = h[-1][1]; print(f'    unpaced {name}: {h[-1][1]:.0f} inf/s per replica ({F_CLK / h[-1][1]:.0f} cycles/inference)')
    target = 100.0 * math.floor(0.9 * min(unpaced.values()) / 100.0) if str(a.target).lower() == 'auto' else float(a.target)
    if target > min(unpaced.values()): sys.exit(f'[MEAS-RF] target {target:.0f} inf/s exceeds the slowest unpaced mode ({min(unpaced.values()):.0f} inf/s) — lower --target')
    plan = [(name, mode, target) for name, mode in MODES.items()] + [('R3_half_rate', MODES['R3_features+forest_random'], target / 2)]
    print(f'[MEAS-RF] paced target {target:.0f} inf/s per replica; {len(plan)} runs x ({1 + 2 * a.repeats} blocks x {a.block:.0f} s) = about {len(plan) * (1 + 2 * a.repeats) * a.block / 60:.0f} min')
    out = {'n_repl': int(n_repl), 'frames': FRAMES[:n_repl], 'n_samples': m.n, 'block_s': a.block, 'repeats': a.repeats, 'target': target, 'unpaced_rates': unpaced, 'f_clk_hz': F_CLK, 'designs': {}, 'started': time.strftime('%Y-%m-%d %H:%M:%S')}
    for name, mode, target in plan:
        pace, hist = calibrate(m, mode, target); rate = hist[-1][1]
        n_infer = int(math.ceil((a.block + a.warmup + 5.0) * rate))
        print(f'[{name}] mode={mode} pace={pace} rate={rate:.0f} inf/s/replica -> {n_infer} inferences per run')
        idles, runs, rates = [], [], []
        idles.append(log_power(pw, a.block))
        for k in range(a.repeats):
            m.start(RC_RUN, arg0=n_infer, mode=mode, pace=pace); time.sleep(a.warmup)
            runs.append(log_power(pw, a.block)); dt = m.wait(timeout=a.block * 3 + 60); rates.append(n_infer / dt)
            idles.append(log_power(pw, a.block))
            pi = np.mean(idles[-2]), np.mean(idles[-1]); pr = np.mean(runs[-1])
            print(f'    run {k + 1}/{a.repeats}: idle {pi[0]:.1f}/{pi[1]:.1f} mW  run {pr:.1f} mW  delta {pr - 0.5 * (pi[0] + pi[1]):+.1f} mW  rate {rates[-1]:.0f} inf/s')
        im = np.array([np.mean(v) for v in idles]); rm = np.array([np.mean(v) for v in runs]); deltas = rm - 0.5 * (im[:-1] + im[1:])
        d = {'mode': mode, 'pace': pace, 'rate_per_replica': float(np.mean(rates)), 'rates': rates, 'n_infer': n_infer, 'idle_mW': im.tolist(), 'run_mW': rm.tolist(),
             'delta_mW': deltas.tolist(), 'delta_mean_mW': float(deltas.mean()), 'delta_std_mW': float(deltas.std(ddof=1) if len(deltas) > 1 else 0.0),
             'idle_samples_mW': [list(map(float, v)) for v in idles], 'run_samples_mW': [list(map(float, v)) for v in runs]}
        out['designs'][name] = d
        print(f'[{name}] P_dyn = {d["delta_mean_mW"]:.1f} +/- {d["delta_std_mW"]:.1f} mW (all replicas + overhead)')
        json.dump(out, open(a.out, 'w'))
    # ── per-replica energy per inference, overhead subtracted ──
    d0 = out['designs']['R0_overhead']; p0, s0 = d0['delta_mean_mW'], d0['delta_std_mW']
    print('\n design                        rate(inf/s)  P_total(mW)    P_block/replica(mW)   E/inference(nJ)   E per input sample(pJ)')
    for name, d in out['designs'].items():
        pb = (d['delta_mean_mW'] - p0) / n_repl; sb = math.sqrt(d['delta_std_mW'] ** 2 + s0 ** 2) / n_repl
        e_inf = pb * 1e-3 / d['rate_per_replica'] * 1e9; e_s = e_inf * 1e3 / m.n
        d.update({'per_replica_mW': pb, 'per_replica_std_mW': sb, 'nJ_per_inference': e_inf, 'pJ_per_sample': e_s})
        print(f" {name:28s} {d['rate_per_replica']:11.0f}  {d['delta_mean_mW']:8.1f}+/-{d['delta_std_mW']:.1f}   {pb:9.2f}+/-{sb:.2f}      {e_inf:12.1f}      {e_s:12.3f}")
    r1 = out['designs']['R1_features']; r3 = out['designs']['R3_features+forest_random']; r2 = out['designs']['R2_features+forest_fixed']
    for lab, ra, rb in (('forest random alone (R3-R1)', r3, r1), ('forest fixed alone (R2-R1)', r2, r1)):
        pb = ra['per_replica_mW'] - rb['per_replica_mW']; e = pb * 1e-3 / ra['rate_per_replica'] * 1e9
        print(f' {lab:28s} {"":11s}  {"":14s}   {pb:9.2f}             {e:12.1f}      {e * 1e3 / m.n:12.3f}')
    r3h = out['designs']['R3_half_rate']
    print(f" linearity check: R3 at {r3['rate_per_replica']:.0f} inf/s -> {r3['nJ_per_inference']:.1f} nJ/inference; at {r3h['rate_per_replica']:.0f} inf/s -> {r3h['nJ_per_inference']:.1f} nJ/inference")
    json.dump(out, open(a.out, 'w')); print(f'[MEAS-RF] saved {a.out}')


if __name__ == '__main__':
    main()
