#!/usr/bin/env python3
"""Bit-exact check of the D6 IP on silicon: for every <name>_expected.json, load the frame block, run VERIFY for both forests
and several sub-frame rotations, compare the 9 float64 features (bit for bit) and the forest verdicts with the PS
reference (features_ps == Colab, and expected.json's router_verdict for the frame's own protocol).
   sudo python3 verify_classifier.py kv260_package/vectors --spec kv260_package/spec.json [--rots 0 7 13]"""
import argparse, glob, json, os, struct, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rf_ctl import RF, words_from_expected, rotate_words, CLASSES, M_FOREST_FIXED, M_FOREST_RANDOM, Q
from features_ps import frontend_features, forest_predict


def bits(x): return struct.unpack('<Q', struct.pack('<d', float(x)))[0]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('vdir'); ap.add_argument('--spec', required=True); ap.add_argument('--rots', nargs='*', type=int, default=[0, 7, 13]); ap.add_argument('--names', nargs='*')
    a = ap.parse_args(); spec = json.load(open(a.spec)); m = RF()
    files = sorted(f for f in glob.glob(os.path.join(a.vdir, '*_expected.json')) if not f.endswith('_conv_expected.json'))
    if a.names: files = [f for f in files if os.path.basename(f).replace('_expected.json', '') in a.names]
    ok = 0; reports = []
    for f in files:
        name = os.path.basename(f).replace('_expected.json', ''); e = json.load(open(f)); w, n = words_from_expected(e)
        m.load_words(w, n, replica=-1)
        feat_mm = 0; verd_mm = 0; own_ok = None; t0 = time.time(); ulp_max = 0
        for rot in a.rots:
            wr = rotate_words(w, rot); ref_f = frontend_features(wr, n)
            for mode, fname in ((M_FOREST_FIXED, 'fixed'), (M_FOREST_RANDOM, 'random')):
                v, feats, votes, chk = m.verify(rot, mode)
                ref_v, ref_p = forest_predict(spec[f'router_forest_{fname}'], ref_f)
                for i in range(9):
                    if bits(feats[i]) != bits(ref_f[i]):
                        feat_mm += 1; ulp_max = max(ulp_max, abs(bits(feats[i]) - bits(ref_f[i])))
                if CLASSES[v] != ref_v: verd_mm += 1
                if rot == 0 and fname == e.get('protocol') and e.get('router_verdict') is not None: own_ok = (CLASSES[v] == e['router_verdict'])
        dt = time.time() - t0
        passed = (feat_mm == 0 and verd_mm == 0 and own_ok is not False)
        ok += passed
        print(f"[{name}] {'PASS' if passed else 'FAIL'}  feature bit-mismatches={feat_mm} (max ulp {ulp_max})  verdict mismatches={verd_mm}/{2 * len(a.rots)}  "
              f"own-protocol verdict {'ok' if own_ok else own_ok}  ({dt * 1e3 / (2 * len(a.rots)):.1f} ms per inference incl. AXI)")
        reports.append({'name': name, 'pass': passed, 'feature_mismatches': feat_mm, 'max_ulp': ulp_max, 'verdict_mismatches': verd_mm, 'own_ok': own_ok})
    print(f'[BOARD-RF] {ok}/{len(files)} frames bit-exact -> {"PASS" if ok == len(files) else "FAIL"}')
    json.dump({'reports': reports, 'passed': ok, 'total': len(files), 'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')},
              open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'verification.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
