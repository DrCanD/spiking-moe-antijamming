"""Recompute the measured classifier cost and its streaming amortisation."""
from pathlib import Path
import json
import math

ROOT=Path(__file__).resolve().parents[2]


def main():
    data=json.loads((ROOT/'results/hardware/classifier_tail/power_records.json').read_text())
    zero=data['designs']['R0_overhead'];replicas=data['n_repl'];result={}
    for name in ['R3_features+forest_random','R3_half_rate']:
        d=data['designs'][name]
        idle=d['idle_mW'];runs=d['run_mW']
        deltas=[run-(idle[i]+idle[i+1])/2 for i,run in enumerate(runs)]
        assert max(abs(a-b) for a,b in zip(deltas,d['delta_mW']))<1e-8
        energy=(d['delta_mean_mW']-zero['delta_mean_mW'])*1e6/(replicas*d['rate_per_replica'])
        sigma=math.hypot(d['delta_std_mW'],zero['delta_std_mW'])*1e6/(replicas*d['rate_per_replica'])
        assert abs(energy-d['nJ_per_inference'])<1e-8
        result[name]=dict(rate_per_s=d['rate_per_replica'],energy_uJ=energy/1000,std_uJ=sigma/1000)
    conservative=max(v['energy_uJ'] for v in result.values())
    # The recorded streaming router makes about 0.3 calls per 5,000-sample block.
    result['amortised_nJ_per_sample']=[conservative*1000*calls/5000 for calls in [.298,.317]]
    print(json.dumps(result,indent=2))
    print('PASS: idle-bracket contrasts, replica/rate units and archived inference energies')


if __name__=='__main__':
    main()
