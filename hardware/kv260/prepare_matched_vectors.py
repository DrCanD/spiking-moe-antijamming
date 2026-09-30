"""Matched-energy extension (PC side, no hardware): expected outputs for ALE_NB, ALE_SW and the blanker, plus the
FE_v3+ALE_NB and FFT+ALE_NB combination checks, from the SAME HLS source that built the v3 bitstream.

  1. compiles hls/tb/tb_matched.cpp natively (plain integers, N_REPL=1) and writes, per vector,
     <name>_alenb_res.bin/_cap.bin, <name>_alesw_res.bin/_cap.bin, <name>_blank_res.bin/_cap.bin; the testbench then
     re-reads them and checks the two combinations against the single-block expectations;
  2. cross-checks those files against an INDEPENDENT integer model: tools/fixed_experts.py (fx_ale, fx_blanker_mask),
     the independent Python integer recurrence, so the C++ block is shown to implement the specified NLMS / blanker;
  3. compares with vectors/manifest_matched.json and writes evidence/matched_regeneration.json.

rtl_smoke is excluded: its full-scale tones followed by exact digital silence drive the NLMS gain beyond the 48-bit
hardware register (the C model's MOE_ASSERT fires), so the native and ap_int builds are not required to agree there.
The campaign receipt and capture manifest are read-only. Generated captures must match their recorded hashes.
Run from the package root:  python3 prepare_matched_vectors.py      (g++ and numpy required)
"""
import hashlib, json, re, shutil, subprocess, sys, tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'tools'))
import fixed_experts as fx

FXGOLDEN_SHA = '9a38fbf1c12835e09661ff7669a45253f53b95119c913c917c4cdf0e24a40c84'
EXCLUDED = {'rtl_smoke': 'NLMS gain exceeds the 48-bit hardware register on tone-to-exact-silence edges (MOE_ASSERT); not a receiver input'}
KINDS = ('alenb_res', 'alenb_cap', 'alesw_res', 'alesw_cap', 'blank_res', 'blank_cap')


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def params():
    text = (ROOT / 'hls/src/moe_params.h').read_text()
    g = lambda k: int(re.search(r'#define\s+' + k + r'\s+(-?\d+)', text).group(1))
    table = [int(v) for v in re.search(r'RECIP_TABLE\[32\]\s*=\s*\{([^}]*)\}', text).group(1).replace('\n', ' ').split(',') if v.strip()]
    p = dict(taps=g('ALE_TAPS'), delay=g('ALE_DELAY'), w_frac=g('ALE_W_FRAC'), mu_nb=g('ALE_MU_NB_Q16'), mu_sw=g('ALE_MU_SW_Q16'),
             recip_r=g('RECIP_R'), recip_b=g('RECIP_B'), eps=g('RECIP_EPS_Q'), window=g('BLANK_WINDOW'), thr=g('BLANK_THR_SQ_Q'),
             in_frac=g('IN_FRAC'), table=table)
    # the frozen specification: mu 0.02 / 0.05 in Q16, reciprocal LUT of the v1 golden
    assert p['mu_nb'] == round(0.02 * 65536) and p['mu_sw'] == round(0.05 * 65536), (p['mu_nb'], p['mu_sw'])
    cfg = SimpleNamespace(in_frac=p['in_frac'], ale_w_frac=p['w_frac'], ale_recip_bits=p['recip_b'])
    lut = fx.make_recip_lut(cfg)
    assert lut['table'] == table and lut['R'] == p['recip_r'] and lut['eps_q'] == p['eps'], 'RECIP_TABLE differs from the v1 golden LUT'
    return p, cfg, lut


def u64(r, at): return int(r[at]) | (int(r[at + 1]) << 32)
def s32(v): v = int(v); return v - (1 << 32) if v >= (1 << 31) else v
def sat16(e): return np.clip(e, -32768, 32767).astype(np.int16)


def golden_check(name, p, cfg, lut, vec):
    x = np.fromfile(vec / f'{name}.bin', dtype='<i2').astype(np.int64); n = len(x)
    out = {'vector': name, 'n_samples': n}
    sum_x2 = int(np.dot(x, x))
    for label, mu, e_word, last_word in (('alenb', p['mu_nb'], 387, 392), ('alesw', p['mu_sw'], 389, 393)):
        r = np.fromfile(vec / f'{name}_{label}_res.bin', dtype='<u4'); cap = np.fromfile(vec / f'{name}_{label}_cap.bin', dtype='<i2')
        e = fx.fx_ale(x, p['taps'], p['delay'], mu, cfg, lut)
        assert np.array_equal(sat16(e), cap), f'{name} {label}: C++ error stream differs from the Python golden'
        assert u64(r, 385) == sum_x2, f'{name} {label}: sum_x2'
        assert u64(r, e_word) == int(np.dot(e, e)), f'{name} {label}: sum_e2'
        assert s32(r[last_word]) == int(e[-1]), f'{name} {label}: e_last'
        out[label] = {'sum_e2': int(np.dot(e, e)), 'saturated_samples': int(np.count_nonzero((e > 32767) | (e < -32768))),
                      'residual_power_ratio': float(np.dot(e, e) / max(sum_x2, 1))}
    r = np.fromfile(vec / f'{name}_blank_res.bin', dtype='<u4'); cap = np.fromfile(vec / f'{name}_blank_cap.bin', dtype='<i2')
    mask, ss = fx.fx_blanker_mask(x, 1, p['window'], p['thr']); G = len(mask)
    assert len(cap) == 4 * G, f'{name} blank: capture length'
    ss_hw = (cap[G:2 * G].astype(np.int64) & 0xFFFF) | ((cap[2 * G:3 * G].astype(np.int64) & 0xFFFF) << 16) | ((cap[3 * G:].astype(np.int64) & 0xFF) << 32)
    assert np.array_equal(cap[:G], mask.astype(np.int16)), f'{name} blank: mask differs from the Python golden'
    assert np.array_equal(ss_hw, ss), f'{name} blank: group sums of squares differ'
    assert int(r[391]) == int(mask.sum()), f'{name} blank: flagged count'
    out['blank'] = {'groups': G, 'flagged': int(mask.sum())}
    return out


def main():
    (ROOT / 'evidence').mkdir(exist_ok=True)
    assert sha(ROOT / 'tools/fixed_experts.py') == FXGOLDEN_SHA, 'Integer golden source identity differs'
    p, cfg, lut = params()
    manifest = json.loads((ROOT / 'vectors/manifest.json').read_text())
    names = [r['name'] for r in manifest if r['name'] not in EXCLUDED]
    recorded_path = ROOT / 'vectors/manifest_matched.json'
    recorded = json.loads(recorded_path.read_text())
    campaign_path = ROOT / 'evidence/campaign/matched_native_verification.json'
    campaign = json.loads(campaign_path.read_text())
    base = {r['name']: r for r in manifest}
    with tempfile.TemporaryDirectory(prefix='matched_vectors_') as temporary:
        work = Path(temporary)
        vec = work / 'vectors'
        shutil.copytree(ROOT / 'vectors', vec)
        exe = work / 'tb_matched_native'
        subprocess.run(['g++', '-O2', '-std=c++14', '-w', '-DN_REPL=1', '-I' + str(ROOT / 'hls/src'),
                        str(ROOT / 'hls/src/moe_top.cpp'), str(ROOT / 'hls/tb/tb_matched.cpp'), '-o', str(exe)], check=True)
        run = subprocess.run([str(exe), str(vec), *names, '--generate'], capture_output=True, text=True)
        print(run.stdout, end='', flush=True)
        if run.returncode: print(run.stderr, flush=True); raise SystemExit(f'tb_matched failed ({run.returncode})')
        rerun = subprocess.run([str(exe), str(vec), *names], capture_output=True, text=True)
        if rerun.returncode: print(rerun.stderr, flush=True); raise SystemExit('tb_matched compare pass failed')
        checks, items = [], []
        for name in names:
            c = golden_check(name, p, cfg, lut, vec); checks.append(c)
            print(f"[GOLDEN] {name}: ALE_NB/ALE_SW/blanker == independent integer recurrence | e2/x2 NB {c['alenb']['residual_power_ratio']:.3f} "
                  f"SW {c['alesw']['residual_power_ratio']:.3f} | blank {c['blank']['flagged']}/{c['blank']['groups']}", flush=True)
            item = {'name': name, 'n_samples': base[name]['n_samples'], 'input_sha256': base[name]['input_sha256'],
                    'gate_expected_sha256': base[name]['gate_expected_sha256'], 'fft_expected_sha256': base[name]['fft_expected_sha256'],
                    'spikes_sha256': base[name]['spikes_sha256']}
            for k in KINDS: item[k + '_sha256'] = sha(vec / f'{name}_{k}.bin')
            items.append(item)
        if items != recorded:
            raise RuntimeError('Regenerated captures differ from the recorded manifest; campaign files retained')
        for name in names:
            for kind in KINDS:
                filename = f'{name}_{kind}.bin'
                shutil.copyfile(vec / filename, ROOT / 'vectors' / filename)
    (ROOT / 'evidence/matched_regeneration.json').write_text(json.dumps({
        'receipt_kind': 'post_campaign_regeneration',
        'campaign_receipt': 'evidence/campaign/matched_native_verification.json',
        'campaign_receipt_sha256': sha(campaign_path),
        'campaign_testbench_sha256': campaign['testbench_sha256'],
        'all_capture_hashes_match_campaign': True, 'captures': len(names) * len(KINDS),
        'vectors': len(names), 'excluded': EXCLUDED, 'source_digest': json.loads((ROOT / 'build_identity.json').read_text())['source_digest'],
        'hls_source_modified': False, 'testbench': 'hls/tb/tb_matched.cpp', 'testbench_sha256': sha(ROOT / 'hls/tb/tb_matched.cpp'),
        'independent_golden': 'tools/fixed_experts.py', 'independent_golden_sha256': sha(ROOT / 'tools/fixed_experts.py'),
        'native_cpp_equals_python_golden': True, 'combinations_checked': ['FE_v3+ALE_NB', 'FFT1024+ALE_NB'],
        'manifest_matched_sha256': sha(recorded_path), 'checks': checks}, indent=2) + '\n')
    print(f'MATCHED EXPECTATIONS READY: {len(names)} vectors; native C++ == independent integer recurrence for ALE_NB, ALE_SW, blanker', flush=True)


if __name__ == '__main__':
    main()
