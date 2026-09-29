"""RF-bank word-width sweep against the fixed-point quality protocol (same 288 streams and receiver as validate_fixed.py).

Reference : the frozen dense front-end (Q2.30 rotation, Q20 state) = fixed_reference.run, dense path.
Candidate : dense and lazy (gate) paths with the rotation quantized to Q1.F (F+1-bit signed coefficient) and the state
            to Q.S (threshold, spike input and sleeping margin rescaled), jump table rebuilt from the quantized rotation.
Pass rule (per panel x condition, identical to the build gate): new_failures == 0, different_action_streams == 0,
            undetected(candidate) <= undetected(reference); for both the dense and the lazy candidate.
    python tools/width_sweep.py --configs 17:20 15:20 17:16 --workers 2
"""
from pathlib import Path
import sys, json, argparse, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import numba
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'software'))
import fixed_reference as fr

TH20, SIN20, MARGIN20 = 125829, 69905, 64


def tables(F):
    cr = np.rint(fr.CR / 2.0 ** (30 - F)).astype(np.int64); ci = np.rint(fr.CI / 2.0 ** (30 - F)).astype(np.int64)
    a = (cr + 1j * ci) / 2.0 ** F
    p = a[:, None] ** np.arange(257)[None, :]
    return cr, ci, np.rint(p.real * 2.0 ** F).astype(np.int64), np.rint(p.imag * 2.0 ** F).astype(np.int64)


def scaled(S, margin_lsb):
    k = 2.0 ** (S - 20)
    th = int(round(TH20 * k)); sin = int(round(SIN20 * k))
    return th, sin, th - margin_lsb


@numba.njit(cache=True, fastmath=False)
def run_w(sp, cr, ci, lr, li, F, sin, th, safe_th, bin_samples=250):
    n = len(sp)
    counts = np.zeros((2, (n + bin_samples - 1) // bin_samples, 16), np.int64)
    updates = 0; maxabs = 0; bank_skip = 0
    allsafe = np.ones(n, np.int64)
    for k in range(16):
        dr = 0; di = 0; gr = 0; gi = 0; dp = False; gp = False; stamp = -1; safe = True
        for t in range(n):
            nr = (dr * cr[k] - di * ci[k]) >> F; ni = (dr * ci[k] + di * cr[k]) >> F
            dr = nr + int(sp[t]) * sin; di = ni; active = di >= th
            counts[0, t // bin_samples, k] += int(active and not dp); dp = active
            if not (sp[t] == 0 and safe):
                allsafe[t] = 0
                gap = t - stamp
                if gap > 256: gr = 0; gi = 0
                else:
                    nr = (gr * lr[k, gap] - gi * li[k, gap]) >> F; ni = (gr * li[k, gap] + gi * lr[k, gap]) >> F
                    gr = nr; gi = ni
                gr += int(sp[t]) * sin; stamp = t; active = gi >= th
                counts[1, t // bin_samples, k] += int(active and not gp); gp = active
                updates += 1
                safe = abs(gr) + abs(gi) < safe_th
            m = max(abs(dr), abs(di), abs(gr), abs(gi))
            if m > maxabs: maxabs = m
    for t in range(n): bank_skip += allsafe[t]
    return counts, updates, maxabs, bank_skip


def one(task):
    panel, cond, ms, ci_, tr, configs = task
    sim = fr_sim()
    cfg = sim.core.Config(); cidx = ['fixed', 'random'].index(cond)
    ch = sim.derive(20 if panel == 'regression' else 220, cidx, ms, ci_, tr)
    pay = sim.derive(30 if panel == 'regression' else 230, cidx, ms, ci_, tr)
    f = sim.generate(cfg, cond, sim.CASES[ci_], ch, pay); x = fr.quantize(f['rx']); rx = x.astype(float) / 1024
    sp, c_ref, _, _ = fr.run(x)
    import pickle, receiver_v5
    nb = len(rx) // 5000; om = 2 * np.pi * np.linspace(25e3, 475e3, 16) / 1e6
    bank = pickle.loads((ROOT / 'software/models' / f'{cond}_{ms}.pkl').read_bytes()); cache = {}

    def score(counts):
        feats = []
        for b in range(nb):
            rates = counts[b * 20:(b + 1) * 20] / (250 * om[None, :] / (2 * np.pi))
            feats.append({'spike': np.array(sim.first_features(sp[b * 5000:(b + 1) * 5000], cfg) + list(sim.core.rf_features(rates)))})
        res = receiver_v5.process_one(rx, cfg, bank, feats, 'spike', 'refresh_guard8', sim.BASE['rescue'])
        s, _ = sim.io.link_v4.evaluate(res['soft'], res['flags']['block_refine2'], 'block_refine2', f['bits'], f['payloads'], f['words'], f['coding'], 500, 1e5, f['transitions'], cache)
        return dict(success=[w['success'] for w in s['words']], undetected=s['undetected_wrong'], actions=res['actions_by_block'])
    ref = score(c_ref[0])
    out = dict(panel=panel, condition=cond, n=len(x), ref=ref, cand={})
    for (F, S, M) in configs:
        cr, ci, lr, li = tables(F); th, sin, safe_th = scaled(S, M)
        spa = sp.astype(np.int8)
        counts, updates, maxabs, bskip = run_w(spa, cr, ci, lr, li, F, sin, th, safe_th)
        key = f'F{F}_S{S}_M{M}'
        out['cand'][key] = dict(dense=score(counts[0]), lazy=score(counts[1]), cells_dense=int(np.count_nonzero(counts[0] != c_ref[0])),
                                cells_lazy=int(np.count_nonzero(counts[1] != c_ref[0])), updates=int(updates), maxabs=int(maxabs), bank_skip=int(bskip))
    return out


_SIM = []


def fr_sim():
    if not _SIM:
        import run_experiment as sim
        _SIM.append(sim)
    return _SIM[0]


def summarize(rows, configs):
    summ = {}
    for (F, S, M) in configs:
        key = f'F{F}_S{S}_M{M}'; per = []
        for panel in ('regression', 'fresh'):
            for cond in ('fixed', 'random'):
                rs = [r for r in rows if r['panel'] == panel and r['condition'] == cond]
                d = dict(panel=panel, condition=cond, streams=len(rs))
                for path in ('dense', 'lazy'):
                    pairs = [(a, b) for r in rs for a, b in zip(r['ref']['success'], r['cand'][key][path]['success'])]
                    d[f'{path}_new_failures'] = sum(a and not b for a, b in pairs)
                    d[f'{path}_recoveries'] = sum((not a) and b for a, b in pairs)
                    d[f'{path}_different_action_streams'] = sum(r['ref']['actions'] != r['cand'][key][path]['actions'] for r in rs)
                    d[f'{path}_undetected'] = sum(r['cand'][key][path]['undetected'] for r in rs)
                    d[f'{path}_different_count_cells'] = sum(r['cand'][key][f'cells_{path}'] for r in rs)
                d['ref_failed'] = sum(not a for r in rs for a in r['ref']['success']); d['words'] = sum(len(r['ref']['success']) for r in rs)
                d['ref_undetected'] = sum(r['ref']['undetected'] for r in rs)
                d['max_state_abs'] = max(r['cand'][key]['maxabs'] for r in rs)
                d['band_skip'] = 1 - sum(r['cand'][key]['updates'] for r in rs) / (16 * sum(r['n'] for r in rs))
                d['bank_skip'] = sum(r['cand'][key]['bank_skip'] for r in rs) / sum(r['n'] for r in rs)
                d['pass'] = all(d[f'{p}_new_failures'] == 0 and d[f'{p}_different_action_streams'] == 0 and d[f'{p}_undetected'] <= d['ref_undetected'] for p in ('dense', 'lazy'))
                per.append(d)
        summ[key] = dict(coef_bits=F + 1, coef_frac=F, state_frac=S, margin_lsb=M, state_bits_needed=int(max(p['max_state_abs'] for p in per)).bit_length() + 1,
                         passed=all(p['pass'] for p in per), rows=per)
    return summ


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--configs', nargs='+', default=['17:20:64']); ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--limit', type=int, default=0); ap.add_argument('--out', default=str(ROOT / 'evidence/width_sweep_summary.json'))
    a = ap.parse_args()
    configs = [tuple(int(v) for v in (c + ':64' if c.count(':') == 1 else c).split(':')) for c in a.configs]
    tasks = [(panel, cond, ms, ci, tr, configs) for panel, nt in [('regression', 4), ('fresh', 2)] for cond in ['fixed', 'random'] for ms in [811, 821, 823]
             for ci in range(8) for tr in range(nt)]
    if a.limit: tasks = tasks[::max(1, len(tasks) // a.limit)][:a.limit]
    t0 = time.time(); rows = []
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for i, r in enumerate(pool.map(one, tasks)):
            rows.append(r)
            if (i + 1) % 24 == 0 or i + 1 == len(tasks):
                bad = {k: sum(r2['ref']['actions'] != r2['cand'][k]['dense']['actions'] or r2['ref']['actions'] != r2['cand'][k]['lazy']['actions'] for r2 in rows) for k in rows[0]['cand']}
                print(f'[SWEEP] {i + 1}/{len(tasks)} akis, {time.time() - t0:.0f} s; karar degisen akis sayisi: {bad}', flush=True)
    s = summarize(rows, configs)
    Path(a.out).write_text(json.dumps(dict(protocol='validate_fixed.py streams and receiver; reference = frozen dense Q2.30/Q20', streams=len(tasks), configs=s), indent=1))
    for k, v in s.items():
        print(f'{k}: coef {v["coef_bits"]} bit, state needs {v["state_bits_needed"]} bit -> {"PASS" if v["passed"] else "FAIL"}')
        for p in v['rows']:
            print(f'   {p["panel"]:10s} {p["condition"]:6s} newfail d/l {p["dense_new_failures"]}/{p["lazy_new_failures"]}  actions d/l {p["dense_different_action_streams"]}/{p["lazy_different_action_streams"]}'
                  f'  undet ref/d/l {p["ref_undetected"]}/{p["dense_undetected"]}/{p["lazy_undetected"]}  cells d/l {p["dense_different_count_cells"]}/{p["lazy_different_count_cells"]}'
                  f'  band_skip {p["band_skip"]:.3f} bank_skip {p["bank_skip"]:.3f}')
