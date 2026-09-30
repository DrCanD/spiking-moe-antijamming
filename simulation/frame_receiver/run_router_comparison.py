"""Paired frame receiver, compound recovery and router-free baseline studies."""
from pathlib import Path
import argparse
import hashlib
import json
import pickle
import sys
from dataclasses import asdict, replace
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'simulation/streaming/frozen_v4'))
import core


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False))
    temp.replace(path)


def metric(bits, output):
    keep = ~output['erasure_mask']
    errors = int(np.count_nonzero(((output['soft'] > 0) != bits) & keep))
    n = int(keep.sum())
    return dict(retained_bits=n, retained_errors=errors, input_bits=len(bits),
                conditional_ber=errors / n if n else None,
                retained_fraction=n / len(bits), correct_retained_per_input=(n-errors)/len(bits),
                path=output['path'], counts=output['counts'], predicted_class=output['pred1'])


def rule_output(receiver, rx, mu, gated):
    cfg = receiver.cfg
    residual = core.apply_ale(rx, cfg, mu)
    rho = float(1 - np.mean(residual**2) / (np.mean(rx**2) + 1e-12))
    y = residual if not gated or rho > cfg.rho_min else rx
    soft, candidate, fraction = receiver.pulse_check(y)
    mask = candidate if cfg.blank_min_fraction < fraction < cfg.blank_max_fraction else np.zeros(len(soft), bool)
    return dict(soft=soft, erasure_mask=mask, pred1=None,
                path='rule->ALE' if y is residual else 'rule->pass',
                counts=dict(ale_samples=len(rx), ale_calls=1, router_calls=0, pulse_tests=1))


def panel(settings, study):
    for condition in settings['conditions']:
        if study != 'compound':
            for label in core.JT:
                for jsr in settings['jsr_list']:
                    for trial in range(settings['n_trials']):
                        seed = settings['seed_base'] + trial + core.JT.index(label)*2000
                        yield 'single', condition, (label,), jsr, trial, seed
        for i, components in enumerate(settings['compounds']):
            for jsr in settings['compound_jsr_list']:
                for trial in range(settings['n_compound_trials']):
                    seed = settings['seed_base'] + i*2000 + trial
                    yield 'compound', condition, tuple(components), jsr, trial, seed
        for trial in range(settings['n_clean_trials']):
            yield 'clean', condition, (), 0, trial, settings['clean_seed_base']+trial


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--study', choices=['receivers', 'compound', 'routing'], default='receivers')
    ap.add_argument('--config', type=Path, default=Path('configs/frame_router_comparison.json'))
    ap.add_argument('--output', type=Path, default=Path('runs/frame_router_comparison'))
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    settings = json.loads((ROOT/a.config).read_text())
    cfg = core.Config(**{k:v for k,v in settings.items() if k in core.Config.__dataclass_fields__})
    if a.smoke:
        settings.update(conditions=['random'], jsr_list=[10], compound_jsr_list=[10],
                        n_trials=1, n_compound_trials=1, n_clean_trials=1,
                        compounds=[['narrowband','pulse'],['sweep','pulse']])
        cfg = replace(cfg, n_sym_test=1000, router_n_train_clean=2, router_n_train_per=1,
                      router_train_jsr=[0,10,20], rf_trees=10, blanker_cal_n=2)
    out = ROOT/a.output
    out.mkdir(parents=True, exist_ok=True)
    identity = dict(config=settings, receiver_config=asdict(cfg), smoke=a.smoke, study=a.study,
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    lock = out/'protocol.json'
    if lock.exists() and json.loads(lock.read_text()) != identity:
        raise RuntimeError('Output belongs to a different protocol; choose a new --output.')
    write(lock, identity)
    rows = json.loads((out/'records.json').read_text()) if (out/'records.json').exists() else []
    completed = {r['id'] for r in rows}
    receivers = {}
    for condition in settings['conditions']:
        path = out/'models'/f'{condition}.pkl'
        if path.exists():
            routers = pickle.loads(path.read_bytes())
        else:
            routers = core.fit_routers(cfg, condition, progress=lambda x: print(x, flush=True))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(pickle.dumps(routers, protocol=5))
        receivers[condition] = core.Receiver(cfg, routers)
    for kind, condition, components, jsr, trial, seed in panel(settings, a.study):
        uid = '|'.join([kind,condition,'+'.join(components) or 'none',str(jsr),str(trial)])
        if uid in completed:
            continue
        frame = core.generate_frame(cfg, condition, components, (jsr,)*len(components), seed)
        rx, bits = frame['rx'], frame['bits']
        receiver = receivers[condition]
        outputs = receiver.evaluate_all(rx)
        for label, mu in [('slow',cfg.ale_mu_nb), ('fast',cfg.ale_mu_sw)]:
            outputs[f'chain_{label}'] = rule_output(receiver, rx, mu, False)
            outputs[f'rule_{label}'] = rule_output(receiver, rx, mu, True)
        # Oracle is an explicitly labelled evaluation arm; truth never enters learned arms.
        label = components[0] if components else 'none'
        oracle = receiver.evaluate_all(rx, methods=('full',), predictions={'snn':(label,1.0)})['full']
        outputs['oracle'] = oracle
        row = dict(id=uid, kind=kind, condition=condition, components=components, jsr_db=jsr,
                   trial=trial, seed=seed, rx_sha256=hashlib.sha256(rx.tobytes()).hexdigest(),
                   methods={k:metric(bits,v) for k,v in outputs.items()})
        rows.append(row)
        write(out/'records.json', rows)
        print(f'{len(rows)} frames | {uid}', flush=True)
    summary = {}
    for method in rows[0]['methods']:
        values = [r['methods'][method] for r in rows]
        retained = sum(v['retained_bits'] for v in values)
        errors = sum(v['retained_errors'] for v in values)
        total = sum(v['input_bits'] for v in values)
        summary[method] = dict(frames=len(values), retained_bit_ber=errors/retained if retained else None,
                               retained_fraction=retained/total, correct_retained_per_input=(retained-errors)/total,
                               ale_samples=sum(v['counts'].get('ale_samples',0) for v in values))
    write(out/'summary.json', dict(smoke=a.smoke, methods=summary))
    print('Completed:', a.output, flush=True)


if __name__ == '__main__':
    main()
