#!/usr/bin/env python3
"""Matched-energy measurement on the EXISTING v3 bitstream: spike front end + experts vs the router-free rule receiver.

Same hardware, same regmap, same INA260 protocol as run_gate.py (120 s blocks, idle bracket before and after every
run, randomised mode order, >= 5 repeats, rate matched within 1 %). New here:
  * board verification of the blocks the v3 run never checked: ALE_NB, ALE_SW, blanker (bit-exact against the
    expected files of prepare_matched_vectors.py) and the two measured combinations FE_v3+ALE_NB, FFT1024+ALE_NB;
  * --probe: maximum replay rate of every mode at pace 0, and the common target that all modes can hold;
  * measured modes D0_empty, FE_v3, FFT1024, ALE_NB, ALE_SW, BLANK, FE_v3+ALE_NB, FFT1024+ALE_NB;
  * additivity test: E(FE_v3+ALE_NB) - E(FE_v3) - E(ALE_NB) (D0-subtracted, paired per repeat); same for FFT.
    If the blocks add, a receiver's datapath energy is the sum of its blocks weighted by their activity.

Order of use (in the package root on the KV260):
  sudo python3 board/run_matched.py --verify-only
  sudo python3 board/run_matched.py --probe
  sudo python3 board/run_matched.py --quick --target <probe recommendation>
  sudo python3 board/run_matched.py --overnight --target <same value>
"""
import argparse, math, platform, random, sys, zipfile
from datetime import datetime
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_gate as g
from moe_ctl import MoE, R, ALL_MODES, CMD_RUN, CAP_SPIKES, CAP_E_NB, CAP_E_SW, CAP_MASK, CAP_SS_LO, CAP_SS_MID, CAP_SS_HI
from moe_ctl import EN_FE, EN_CONV, EN_ALE_NB, EN_ALE_SW, EN_BLANK, EN_GATE, EN_ISO, EN_ROMISO, EN_BANK

ROOT = g.ROOT
FE_V3 = EN_FE | EN_GATE | EN_ISO | EN_ROMISO | EN_BANK          # 481, the v3 front end as published
MATCHED = {'D0_empty': 0, 'FE_v3': FE_V3, 'FFT1024': EN_CONV, 'ALE_NB': EN_ALE_NB, 'ALE_SW': EN_ALE_SW, 'BLANK': EN_BLANK,
           'FE_v3+ALE_NB': FE_V3 | EN_ALE_NB, 'FFT1024+ALE_NB': EN_CONV | EN_ALE_NB}
for _k, _v in MATCHED.items():
    assert ALL_MODES.get(_k, _v) == _v, _k
ALL_MODES.update(MATCHED)          # run_gate.calibrate / measure_run look modes up in this (shared) dict
MODES = list(MATCHED)
ADDITIVITY = [('FE_v3+ALE_NB', 'FE_v3', 'ALE_NB'), ('FFT1024+ALE_NB', 'FFT1024', 'ALE_NB')]
ADDITIVE_BAND = 0.10               # pre-set: blocks add if the 95 % CI of the interaction lies within +-10 % of the parts
ALE_NB_WORDS = [385, 386, 387, 388, 392]
ALE_SW_WORDS = [385, 386, 389, 390, 393]
FE_WORDS = list(range(385)) + [979, 980, 981]
FFT_WORDS = list(range(394, 958))
WARM_MODE = 'FE_v3+ALE_NB'


def log(*a): print(*a, flush=True)


# ─────────────────────────────────────────── verification of ALE / blanker / combinations ─────────────────────
def verify_matched(m, identity, items, out):
    res = {'started_utc': g.utc(), 'completed': False, 'checks': []}
    g.save(out / 'board_matched_verification.json', res)

    def words(got, exp, idx, what):
        bad = [i for i in idx if got[i] != exp[i]]
        g.require(not bad, f'{what}: first differing result words {bad[:8]}')

    def stream(got, exp, what):
        g.require(len(got) == len(exp), f'{what}: capture length {len(got)} != {len(exp)}')
        bad = np.flatnonzero(got != exp)
        g.require(bad.size == 0, f'{what}: {bad.size} differing samples, first at {bad[:5].tolist()}')

    def run(blk_en, cap):
        m.verify(cap, blk_en); r = np.asarray(m.results(), dtype=np.uint32); g.check_identity(r, identity)
        g.require(int(r[R['BLK_EN']]) == blk_en, 'blk_en echo mismatch'); return r

    for item in items:
        name = item['name']; stem = ROOT / 'vectors' / name
        load = lambda kind, dt: np.fromfile(f'{stem}_{kind}.bin', dtype=dt)
        g.require(g.sha(f'{stem}.bin') == item['input_sha256'], name + ' input hash mismatch')
        for kind in ('alenb_res', 'alenb_cap', 'alesw_res', 'alesw_cap', 'blank_res', 'blank_cap'):
            g.require(g.sha(f'{stem}_{kind}.bin') == item[kind + '_sha256'], f'{name} {kind} hash mismatch (run prepare_matched_vectors.py / copy again)')
        for kind, key in (('gate_res', 'gate_expected_sha256'), ('fft_res', 'fft_expected_sha256'), ('spikes', 'spikes_sha256')):
            g.require(g.sha(f'{stem}_{kind}.bin') == item[key], f'{name} {kind} hash mismatch')
        x = np.fromfile(f'{stem}.bin', dtype='<i2'); n = len(x); G = n // 50
        g.require(n == item['n_samples'], name + ' length mismatch'); m.load(x)
        nb_r, nb_c = load('alenb_res', '<u4'), load('alenb_cap', '<i2')
        sw_r, sw_c = load('alesw_res', '<u4'), load('alesw_cap', '<i2')
        bl_r, bl_c = load('blank_res', '<u4'), load('blank_cap', '<i2')
        gate_r, fft_r, spikes = load('gate_res', '<u4'), load('fft_res', '<u4'), load('spikes', '<i2')
        # single blocks
        r = run(EN_ALE_NB, CAP_E_NB); words(r, nb_r, ALE_NB_WORDS, f'{name} ALE_NB'); stream(m.read_capture(n), nb_c, f'{name} ALE_NB e')
        res['checks'].append({'vector': name, 'mode': 'ALE_NB', 'passed': True})
        r = run(EN_ALE_SW, CAP_E_SW); words(r, sw_r, ALE_SW_WORDS, f'{name} ALE_SW'); stream(m.read_capture(n), sw_c, f'{name} ALE_SW e')
        res['checks'].append({'vector': name, 'mode': 'ALE_SW', 'passed': True})
        parts = []
        for cap in (CAP_MASK, CAP_SS_LO, CAP_SS_MID, CAP_SS_HI):
            r = run(EN_BLANK, cap); words(r, bl_r, [391], f'{name} BLANK'); parts.append(m.read_capture(G))
        stream(np.concatenate(parts), bl_c, f'{name} BLANK mask/sumsq')
        res['checks'].append({'vector': name, 'mode': 'BLANK', 'passed': True, 'flagged_groups': int(bl_r[391]), 'groups': G})
        # combinations that are measured: every block must reproduce its single-block result exactly
        r = run(FE_V3 | EN_ALE_NB, CAP_SPIKES)
        words(r, gate_r, FE_WORDS, f'{name} FE_v3+ALE_NB (FE part)'); words(r, nb_r, ALE_NB_WORDS, f'{name} FE_v3+ALE_NB (ALE part)')
        stream(m.read_capture(n), spikes, f'{name} FE_v3+ALE_NB spikes')
        run(FE_V3 | EN_ALE_NB, CAP_E_NB); stream(m.read_capture(n), nb_c, f'{name} FE_v3+ALE_NB e')
        res['checks'].append({'vector': name, 'mode': 'FE_v3+ALE_NB', 'passed': True})
        r = run(EN_CONV | EN_ALE_NB, CAP_E_NB)
        words(r, fft_r, FFT_WORDS, f'{name} FFT1024+ALE_NB (FFT part)'); words(r, nb_r, ALE_NB_WORDS, f'{name} FFT1024+ALE_NB (ALE part)')
        stream(m.read_capture(n), nb_c, f'{name} FFT1024+ALE_NB e')
        res['checks'].append({'vector': name, 'mode': 'FFT1024+ALE_NB', 'passed': True})
        g.save(out / 'board_matched_verification.json', res)
        log(f'[VERIFY+] {name} PASS  (ALE_NB, ALE_SW, BLANK {int(bl_r[391])}/{G} grup, FE_v3+ALE_NB, FFT1024+ALE_NB)')
    res.update(completed=True, finished_utc=g.utc(), vectors=len(items)); g.save(out / 'board_matched_verification.json', res)
    return res


# ─────────────────────────────────────────── rate probe ───────────────────────────────────────────────────────
def probe(m, vectors, modes, out):
    table = {}
    for name in vectors:
        x = np.fromfile(ROOT / 'vectors' / f'{name}.bin', dtype='<i2'); n = len(x); m.load(x); table[name] = {}
        for mode in modes:
            dt = m.call(CMD_RUN, arg0=3, blk_en=ALL_MODES[mode], pace=0, n=n)            # short pilot
            frames = max(3, math.ceil(3.0 / max(dt / 3, 1e-6)))                          # then ~3 s
            dt = m.call(CMD_RUN, arg0=frames, blk_en=ALL_MODES[mode], pace=0, n=n)
            g.require(m.frames_done() == frames, 'probe frame counter mismatch')
            rate = frames * n / dt / 1e6; table[name][mode] = rate
            log(f'[PROBE] {name:18s} {mode:16s} pace=0 -> en yuksek {rate:.4f} MS/s')
    worst = {mode: min(table[v][mode] for v in vectors) for mode in modes}
    limit = min(worst.values()); limiting = min(worst, key=worst.get)
    target = min(1.0, math.floor(limit * 0.95 * 20) / 20)                               # 5 % margin, 0.05 MS/s grid
    result = {'utc': g.utc(), 'vectors': vectors, 'max_rate_msps': table, 'worst_case_msps': worst,
              'limiting_mode': limiting, 'recommended_target_msps': target}
    g.save(out / 'probe.json', result)
    log(f'[PROBE] en yavas mod: {limiting} ({limit:.4f} MS/s). Onerilen ortak hedef: --target {target:.2f}')
    return result


# ─────────────────────────────────────────── analysis ─────────────────────────────────────────────────────────
def analyze(data):
    nr = data['n_repl']
    out = {'scope': 'SOM input power of datapath blocks on the v3 bitstream at one common replay rate; excludes RF/ADC, PS-side '
                    'control (router MLP, median thresholds, RS decoding) and anything not in the PL datapath',
           'units': 'nJ per sample per replica = (mW / MS/s) / N_REPL; D0-subtracted values remove replay, control and pacing',
           'complete': data.get('complete', False), 'diagnostic_only': data['settings']['quick'],
           'target_msps': data['settings']['target'], 'additivity_band': ADDITIVE_BAND, 'vectors': {}}
    for name in data['settings']['vectors']:
        reps = [r for r in data['repeats'] if r['vector'] == name and r.get('complete')]
        if not reps: continue
        per = {k: [] for k in MODES}; inter = {c: [] for c, _, _ in ADDITIVITY}; parts = {c: [] for c, _, _ in ADDITIVITY}
        for r in reps:
            by = {x['mode']: x for x in r['runs']}
            adj = {k: g.energy_nj(v['mean_mW'] - v['idle_bracket_mW'], v['rate_msps']) for k, v in by.items()}
            for k, v in by.items():
                per[k].append({'som_input_nJ_all_replicas': g.energy_nj(v['mean_mW'], v['rate_msps']),
                               'idle_adjusted_nJ_all_replicas': adj[k],
                               'D0_subtracted_nJ_per_replica': (adj[k] - adj['D0_empty']) / nr, 'rate_msps': v['rate_msps']})
            for combo, a, b in ADDITIVITY:
                if all(k in adj for k in (combo, a, b, 'D0_empty')):
                    d0 = adj['D0_empty']; ea, eb, ec = (adj[a] - d0) / nr, (adj[b] - d0) / nr, (adj[combo] - d0) / nr
                    inter[combo].append(ec - ea - eb); parts[combo].append(ea + eb)
        item = {'modes': {k: {f: g.interval([row[f] for row in rows]) for f in rows[0]} for k, rows in per.items() if rows}, 'additivity': {}}
        for combo, a, b in ADDITIVITY:
            if not inter[combo]: continue
            st = g.interval(inter[combo]); s = float(np.mean(parts[combo])); ci = st['ci95']
            verdict = 'unresolved (need >= 2 repeats)'
            if ci is not None and s > 0:
                if -ADDITIVE_BAND * s <= ci[0] and ci[1] <= ADDITIVE_BAND * s: verdict = 'additive within +-10 %'
                elif ci[0] > ADDITIVE_BAND * s or ci[1] < -ADDITIVE_BAND * s: verdict = 'NOT additive'
                else: verdict = 'unresolved (CI wider than +-10 % band)'
            item['additivity'][combo] = {'interaction_nJ_per_replica': st, 'sum_of_parts_nJ_per_replica': s,
                                         'relative_interaction': (st['mean'] / s) if s > 0 else None, 'verdict': verdict}
        e = {k: item['modes'][k]['D0_subtracted_nJ_per_replica']['mean'] for k in item['modes']}
        if all(k in e for k in ('FE_v3', 'ALE_SW', 'ALE_NB', 'BLANK')):
            ale = max(e['ALE_NB'], e['ALE_SW'])
            # rule receiver: ALE(mu_sw) every sample + 2 pulse tests;  spike MoE: FE every sample + ALE on duty d + (1+d) pulse tests
            rule = e['ALE_SW'] + 2 * e['BLANK']; denom = ale + e['BLANK']
            item['composition'] = {'rule_datapath_nJ_per_replica': rule,
                                   'spike_moe_datapath_nJ_per_replica(d)': f"{e['FE_v3']:.4f} + d*{ale:.4f} + (1+d)*{e['BLANK']:.4f}",
                                   'break_even_ALE_duty': ((rule - e['FE_v3'] - e['BLANK']) / denom) if denom > 0 else None,
                                   'note': 'ALE term uses max(ALE_NB, ALE_SW); break-even d: the spike MoE uses less datapath energy '
                                           'than the rule receiver when its ALE duty is below this value. Receiver duties come from Exp 9b.'}
        out['vectors'][name] = item
    return out


def print_summary(an, name, done, total):
    v = an['vectors'].get(name)
    if not v: return
    parts = [f"{k} {s['D0_subtracted_nJ_per_replica']['mean']:.2f}" for k, s in v['modes'].items() if k != 'D0_empty']
    log(f'[OZET] {name}: {done}/{total} tekrar | kopya basina, D0 cikarilmis (nJ/ornek): ' + ' | '.join(parts))
    for combo, a in v['additivity'].items():
        st = a['interaction_nJ_per_replica']; ci = st.get('ci95'); rel = a['relative_interaction']
        log(f"[OZET]   toplanabilirlik {combo}: etkilesim {st['mean']:+.3f} nJ" + (f" [%95 GA {ci[0]:+.3f}, {ci[1]:+.3f}]" if ci else '')
            + (f", parcalarin toplamina gore {100 * rel:+.1f} %" if rel is not None else '') + f" -> {a['verdict']}")
    c = v.get('composition')
    if c and c['break_even_ALE_duty'] is not None:
        log(f"[OZET]   kural alicisi veri yolu {c['rule_datapath_nJ_per_replica']:.2f} nJ; spike MoE ancak ALE doluluk orani "
            f"< {c['break_even_ALE_duty']:.2f} ise daha az harcar")


def write_archive(out):
    with zipfile.ZipFile(out / 'KV260_MATCHED_RESULTS.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file() and p.suffix not in ('.zip', '.tmp'): z.write(p, p.relative_to(out))
        for rel in ['build_identity.json', 'board/build_receipt.json', 'evidence/native_verification.json',
                    'evidence/campaign/matched_native_verification.json', 'evidence/matched_regeneration.json',
                    'evidence/matched_provenance.json', 'hls/tb/tb_matched.cpp',
                    'prepare_matched_vectors.py', 'tools/fixed_experts.py',
                    'vectors/manifest_matched.json', 'board/run_matched.py']:
            if (ROOT / rel).exists(): z.write(ROOT / rel, 'provenance/' + rel)


# ─────────────────────────────────────────── main ─────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    what = ap.add_mutually_exclusive_group(required=True)
    what.add_argument('--verify-only', action='store_true'); what.add_argument('--probe', action='store_true')
    what.add_argument('--quick', action='store_true'); what.add_argument('--overnight', action='store_true')
    ap.add_argument('--target', type=float, help='common replay rate, MS/s per replica (from --probe)')
    ap.add_argument('--vectors', nargs='+', default=g.DEFAULT_VECTORS)
    ap.add_argument('--modes', nargs='+', default=MODES, help='subset of ' + ', '.join(MODES) + ' (D0_empty is always measured)')
    ap.add_argument('--block', type=float); ap.add_argument('--repeats', type=int); ap.add_argument('--warmup', type=float, default=5)
    ap.add_argument('--out', type=Path); ap.add_argument('--order-seed', type=int, default=2809026)
    a = ap.parse_args()
    a.block = a.block or (5 if a.quick else 120); a.repeats = a.repeats or (1 if a.quick else 5)
    modes = ['D0_empty'] + [k for k in MODES if k in a.modes and k != 'D0_empty']
    g.require(all(k in MATCHED for k in a.modes), 'unknown mode in --modes')
    if a.quick or a.overnight:
        g.require(a.target is not None and 0 < a.target <= 1.0, 'give --target (the value --probe recommends)')
        g.require(a.block >= 2 and a.repeats >= 1 and a.warmup >= 0, 'invalid measurement settings')
        g.require(not a.overnight or (a.block >= 120 and a.repeats >= 5), 'overnight requires >= 120 s blocks and >= 5 repeats')
    out = (a.out or ROOT / 'board/results_matched' / datetime.now().strftime('%Y%m%d_%H%M%S')).resolve()
    out.mkdir(parents=True, exist_ok=False)
    m = None; governors = {}
    try:
        receipt, identity, clk = g.preflight()
        manifest = g.readjson(ROOT / 'vectors/manifest.json'); items = g.readjson(ROOT / 'vectors/manifest_matched.json')
        known = {r['name'] for r in items}
        g.require(len(a.vectors) == len(set(a.vectors)) and all(v in known for v in a.vectors), 'unknown/duplicate vectors')
        m = MoE()
        log('[VERIFY] v3 on uc ve FFT: 14 vektor x 7 mod (onceki dogrulama, degismedi)')
        g.verify_vectors(m, identity, manifest, out)
        log(f'[VERIFY+] ALE_NB, ALE_SW, BLANK ve iki birlesik mod: {len(items)} vektor')
        verify_matched(m, identity, items, out)
        if a.verify_only:
            log('MATCHED BOARD VERIFICATION COMPLETE', out); return
        if a.probe:
            probe(m, a.vectors, modes, out); log('PROBE COMPLETE', out); return
        sensor = g.find_sensor(); governors = g.read_governors(); g.require(governors, 'CPU governor information unavailable')
        for p in governors: Path(p).write_text('performance')
        data = {'started_utc': g.utc(), 'complete': False, 'settings': dict({k: v for k, v in vars(a).items() if k != 'out'}, modes=modes),
                'n_repl': identity['default_replicas'], 'blk_en': {k: MATCHED[k] for k in modes},
                'environment': {'platform': platform.platform(), 'clock_hz': clk, 'sensor': str(sensor), 'sensor_units': 'microW converted to mW',
                                'governors_before': governors, 'governors_during': g.read_governors(),
                                'temperatures_start_C': g.temperatures(), 'bitstream_sha256': receipt['bitstream_sha256']},
                'calibration': {}, 'repeats': []}
        g.save(out / 'measurement.json', data)
        rng = random.Random(a.order_seed)
        g.PROG = g.Progress(a, len(a.vectors), len(modes))
        log(f'[PLAN] {len(a.vectors)} vektor x {a.repeats} tekrar x {len(modes)} mod ({", ".join(modes)}), blok {a.block:.0f} s, '
            f'ortak hiz {a.target} MS/s/kopya -> tahmini toplam {g.fmt_dur(g.PROG.total)}')
        for vi, name in enumerate(a.vectors):
            x = np.fromfile(ROOT / 'vectors' / f'{name}.bin', dtype='<i2'); n = len(x); m.load(x)
            cal = {}; data['calibration'][name] = cal
            log(f'[VEKTOR {vi + 1}/{len(a.vectors)}] {name}: {n} ornek; {len(modes)} mod ayni hiza kalibre ediliyor | {g.PROG.status()}')
            for mode in modes:
                hist = []; cal[mode] = {'history': hist}
                try: pace, rate = g.calibrate(m, mode, n, a.target, clk, hist)
                finally: g.save(out / 'calibration.json', data['calibration'])
                cal[mode].update(pace=pace, rate_msps=rate); log(f'[RATE] {name} {mode}: {rate:.6f} MS/s, pace={pace}')
            cr = [v['rate_msps'] for v in cal.values()]
            g.require(max(cr) / min(cr) - 1 <= .01, 'calibrated modes differ by >1 % in rate; comparison blocked')
            g.PROG.add(g.Progress.CAL_S)
            thermal_s = 2 if a.quick else 60; warm = WARM_MODE if WARM_MODE in cal else modes[-1]; cc = cal[warm]
            log(f'[ISINMA] {name}: {thermal_s} s {warm} (ilk bos referans oncesi sicaklik dengesi) | {g.PROG.status()}')
            m.call(CMD_RUN, arg0=max(1, math.ceil(thermal_s * cc['rate_msps'] * 1e6 / n)), blk_en=ALL_MODES[warm], pace=cc['pace'], n=n)
            g.PROG.add(thermal_s)
            for rep in range(a.repeats):
                order = modes.copy(); rng.shuffle(order)
                record = {'vector': name, 'repeat': rep, 'order': order, 'complete': False, 'runs': [], 'idles': []}; data['repeats'].append(record)
                log(f'[TEKRAR] {name} tekrar {rep + 1}/{a.repeats}, mod sirasi: {" -> ".join(order)} | {g.PROG.status()}')
                g.CTX.clear(); g.CTX.update(vector=f'{name} ({vi + 1}/{len(a.vectors)})', rep=f'{rep + 1}/{a.repeats}', n=n, last_idle_mW=None)
                record['idles'].append(g.sample_power(sensor, a.block, m, False)); g.save(out / 'measurement.json', data)
                g.PROG.add(a.block); g.CTX['last_idle_mW'] = record['idles'][-1]['mean_mW']
                for mi, mode in enumerate(order):
                    g.CTX.update(mode=mode, mode_i=f'{mi + 1}/{len(order)}')
                    cc = cal[mode]; row = g.measure_run(m, sensor, mode, n, cc['pace'], cc['rate_msps'], a); record['runs'].append(row)
                    g.PROG.add(g.PROG.per_run); g.save(out / 'measurement.json', data)
                    idle = g.sample_power(sensor, a.block, m, False); record['idles'].append(idle)
                    g.PROG.add(a.block); g.CTX['last_idle_mW'] = idle['mean_mW']
                    row['idle_bracket_mW'] = (record['idles'][-2]['mean_mW'] + idle['mean_mW']) / 2
                    dmw = row['mean_mW'] - row['idle_bracket_mW']; nr = data['n_repl']
                    log(f"[POWER] {name} {rep + 1} {mode}: {row['mean_mW']:.2f} mW; bosa gore {dmw:+.2f} mW "
                        f"(kopya basina {g.energy_nj(dmw / nr, row['rate_msps']):+.3f} nJ/ornek, D0 cikarilmamis) | {g.PROG.status()}")
                    g.save(out / 'measurement.json', data)
                rates = [r['rate_msps'] for r in record['runs']]
                g.require(max(rates) / min(rates) - 1 <= .01, 'measured modes differ by >1 % in rate; this repeat is invalid')
                g.require(g.clock_hz(receipt) == clk, 'PL clock changed during measurement')
                record['complete'] = True; g.save(out / 'measurement.json', data)
                an = analyze(data); g.save(out / 'analysis_matched.json', an)
                try: print_summary(an, name, sum(1 for r in data['repeats'] if r['vector'] == name and r.get('complete')), a.repeats)
                except Exception as e: log('[OZET] yazilamadi:', e)
        data.update(complete=True, finished_utc=g.utc()); g.save(out / 'measurement.json', data)
        g.save(out / 'analysis_matched.json', analyze(data))
        log('COMPLETED:', out / 'KV260_MATCHED_RESULTS.zip')
    except BaseException as exc:
        g.save(out / 'failure.json', {'utc': g.utc(), 'type': type(exc).__name__, 'error': str(exc)})
        log('STOPPED:', str(exc), '\nEvidence:', out)
        raise
    finally:
        for p, value in governors.items():
            try: Path(p).write_text(value)
            except OSError: pass
        if m is not None: m.close()
        write_archive(out)


if __name__ == '__main__':
    main()
