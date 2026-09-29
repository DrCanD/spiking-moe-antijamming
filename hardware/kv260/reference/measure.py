#!/usr/bin/env python3
"""Power measurement protocol (Exp-5 README §5, adapted to the single-bitstream IP):
   for each design (block-enable set): pace calibration to the target sample rate, then
   [idle B s][run B s][idle B s][run B s]...[idle B s]  (repeats runs, bracketed by idles), SOM input power sampled at 10 Hz
   from the on-board INA260 (hwmon 'ina260_u14', 10 mW LSB; 120-s block means have ~3.6 mW std on this board).
   P_dyn(design) = mean(run) - mean(bracketing idles);  E/sample = P_dyn / achieved rate.
   D0 (blk_en = 0: replay + delay chain + pacing only) is the overhead to subtract from D1..D5; divide by N_REPL for one datapath.
   sudo python3 measure.py --vector <name_in_q6_10.bin> [--designs ...] [--block 120] [--repeats 5] [--target 1.0] [--out measure_results.json]
"""
import argparse, glob, json, os, time, math
import numpy as np
from moe_ctl import MoE, R, DESIGNS, CMD_RUN

def pl_clock_hz():
    """PL clock the kernel set (debugfs), else what the firmware overlay asked for (pl_clk_hz.txt), else the design default."""
    for f in ('/sys/kernel/debug/clk/pl0_ref/clk_rate',
              os.path.join(os.path.dirname(os.path.abspath(__file__)), 'firmware', 'pl_clk_actual_hz.txt'),
              os.path.join(os.path.dirname(os.path.abspath(__file__)), 'firmware', 'pl_clk_hz.txt')):
        try:
            return float(open(f).read().strip()), f
        except (OSError, ValueError):
            pass
    return 83.332e6, 'default'


F_CLK, F_CLK_SRC = pl_clock_hz()


def find_hwmon(name='ina260_u14'):
    for d in glob.glob('/sys/class/hwmon/hwmon*'):
        try:
            if open(os.path.join(d, 'name')).read().strip() == name and os.path.exists(os.path.join(d, 'power1_input')):
                return os.path.join(d, 'power1_input')
        except OSError:
            pass
    raise FileNotFoundError('INA260 hwmon (ina260_u14) not found')


def set_performance():
    for f in glob.glob('/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor'):
        try: open(f, 'w').write('performance')
        except OSError: pass


def log_power(path, seconds, hz=10.0):
    vals = []; t0 = time.time(); k = 0
    while True:
        t = t0 + k / hz
        now = time.time()
        if now < t: time.sleep(t - now)
        if time.time() - t0 >= seconds: break
        vals.append(int(open(path).read()) / 1000.0)      # uW -> mW
        k += 1
    return vals


def calibrate_pace(m, blk_en, n, target_msps, frames=20, iters=3):
    pace = 0; hist = []
    for _ in range(iters):
        dt = m.call(CMD_RUN, arg0=frames, blk_en=blk_en, pace=pace, n=n)
        rate = frames * n / dt / 1e6; hist.append((pace, rate))
        cyc = F_CLK / (rate * 1e6)                          # cycles per sample now
        want = F_CLK / (target_msps * 1e6)
        new = int(round(pace + (want - cyc)))
        if new < 0: new = 0
        if new == pace: break
        pace = new
    return pace, hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--vector', required=True, help='<name>_in_q6_10.bin to replay (a jammed frame, e.g. fixed_narrowband_0)')
    ap.add_argument('--designs', nargs='*', default=list(DESIGNS.keys()))
    ap.add_argument('--block', type=float, default=120.0); ap.add_argument('--repeats', type=int, default=5)
    ap.add_argument('--target', type=float, default=1.0, help='target sample rate, MS/s (pace is calibrated per design)')
    ap.add_argument('--warmup', type=float, default=5.0); ap.add_argument('--out', default='measure_results.json')
    a = ap.parse_args()
    set_performance(); pw = find_hwmon()
    m = MoE(); x = np.fromfile(a.vector, dtype='<i2'); n = len(x); m.load(x)
    r = m.results(); n_repl = r[R['N_REPL']]
    print(f'[MEAS] vector {os.path.basename(a.vector)} ({n} samples), N_REPL={n_repl}, block {a.block:.0f} s x {a.repeats} runs, target {a.target} MS/s, sensor {pw}')
    print(f'[MEAS] PL clock {F_CLK/1e6:.3f} MHz (from {F_CLK_SRC})')
    out = {'vector': a.vector, 'n_samples': n, 'n_repl': int(n_repl), 'block_s': a.block, 'repeats': a.repeats, 'target_msps': a.target,
           'f_clk_hz': F_CLK, 'f_clk_source': F_CLK_SRC, 'designs': {}, 'started': time.strftime('%Y-%m-%d %H:%M:%S')}
    for name in a.designs:
        blk_en = DESIGNS[name]
        pace, hist = calibrate_pace(m, blk_en, n, a.target)
        rate = hist[-1][1]
        frames = int(math.ceil((a.block + a.warmup + 5.0) * rate * 1e6 / n))
        print(f'[{name}] blk_en={blk_en} pace={pace} rate={rate:.4f} MS/s (calib {[(p, round(q, 4)) for p, q in hist]}) -> {frames} frames per run')
        idles, runs, rates = [], [], []
        idles.append(log_power(pw, a.block))
        for k in range(a.repeats):
            m.start(CMD_RUN, arg0=frames, blk_en=blk_en, pace=pace, n=n)
            time.sleep(a.warmup)
            runs.append(log_power(pw, a.block))
            dt = m.wait(timeout=a.block * 3 + 60); rates.append(frames * n / dt / 1e6)
            idles.append(log_power(pw, a.block))
            pi = np.mean(idles[-2]), np.mean(idles[-1]); pr = np.mean(runs[-1])
            print(f'    run {k + 1}/{a.repeats}: idle {pi[0]:.1f}/{pi[1]:.1f} mW  run {pr:.1f} mW  delta {pr - 0.5 * (pi[0] + pi[1]):+.1f} mW  rate {rates[-1]:.4f} MS/s')
        im = np.array([np.mean(v) for v in idles]); rm = np.array([np.mean(v) for v in runs])
        deltas = rm - 0.5 * (im[:-1] + im[1:])
        d = {'blk_en': blk_en, 'pace': pace, 'rate_msps': float(np.mean(rates)), 'rates': rates, 'frames_per_run': frames,
             'idle_mW': im.tolist(), 'run_mW': rm.tolist(), 'delta_mW': deltas.tolist(), 'delta_mean_mW': float(deltas.mean()), 'delta_std_mW': float(deltas.std(ddof=1) if len(deltas) > 1 else 0.0),
             'energy_per_sample_nJ_total': float(deltas.mean() / np.mean(rates) / 1e3), 'idle_samples_mW': [list(map(float, v)) for v in idles], 'run_samples_mW': [list(map(float, v)) for v in runs]}
        out['designs'][name] = d
        print(f'[{name}] P_dyn = {d["delta_mean_mW"]:.1f} +/- {d["delta_std_mW"]:.1f} mW (all replicas + overhead)  E/sample = {d["energy_per_sample_nJ_total"]:.2f} nJ (total)')
        json.dump(out, open(a.out, 'w'))
    # ── per-block table: subtract D0 overhead, divide by N_REPL ──
    if 'D0_empty' in out['designs']:
        d0 = out['designs']['D0_empty']['delta_mean_mW']; s0 = out['designs']['D0_empty']['delta_std_mW']
        print('\n design               blk_en  rate(MS/s)  P_total(mW)   P_block/replica(mW)   E/sample/replica(nJ)')
        for name, d in out['designs'].items():
            pb = (d['delta_mean_mW'] - d0) / n_repl; sb = math.sqrt(d['delta_std_mW'] ** 2 + s0 ** 2) / n_repl
            d['per_replica_mW'] = pb; d['per_replica_std_mW'] = sb; d['per_replica_nJ_per_sample'] = pb / d['rate_msps'] / 1e3
            print(f" {name:20s} {d['blk_en']:6d}  {d['rate_msps']:9.4f}  {d['delta_mean_mW']:8.1f}+/-{d['delta_std_mW']:.1f}   {pb:8.2f}+/-{sb:.2f}          {pb / d['rate_msps'] / 1e3:8.3f}")
        json.dump(out, open(a.out, 'w'))
    print(f'[MEAS] saved {a.out}')


if __name__ == '__main__':
    main()
