"""JSR-dependent comparison of paper/retrained LSTM experts and adaptive filters."""
from pathlib import Path
from dataclasses import asdict
import argparse
import json
import hashlib
import numpy as np
import run_receiver_comparison as library


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',type=Path,default=Path('configs/expert_comparison.json'))
    ap.add_argument('--output',type=Path,default=Path('runs/expert_comparison'))
    ap.add_argument('--smoke',action='store_true')
    a=ap.parse_args();root=Path(__file__).resolve().parents[2]
    settings=json.loads((root/a.config).read_text())
    cfg=library.Config()
    for key,value in settings.items():
        if key in cfg.__dataclass_fields__:setattr(cfg,key,value)
    cfg.smoke=a.smoke
    cfg.ale_mu_nb=cfg.ale_mu_sw=settings.get('ale_mu',.2)
    if a.smoke:
        cfg.n_sym_test=cfg.train_n_sym=400;cfg.lstm_epochs=1;cfg.train_jsr_list=[0,10]
        cfg.train_n_per_jsr=1;cfg.paper_train_jsr=[4];cfg.paper_train_n_per_jsr=1
        settings.update(conditions=['random'],jsr_list=[0,10],n_trials=1,n_trials_ale=1)
    output=root/a.output;output.mkdir(parents=True,exist_ok=True)
    R,ref_hash,_=library.load_reference(library.find_reference_source())
    lock=dict(settings=settings,smoke=a.smoke,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),reference_definitions_sha256_16=ref_hash)
    if (output/'protocol.json').exists() and json.loads((output/'protocol.json').read_text())!=lock:
        raise RuntimeError('Output belongs to a different protocol; choose a new --output.')
    (output/'protocol.json').write_text(json.dumps(lock,indent=2))
    experts={}
    for name,condition in [('fixed020','fixed'),('rand020','random')]:
        experts[name]=library.train_or_load_expert(R,cfg,name,condition,['narrowband','sweep'],cfg.train_n_per_jsr,cfg.train_seed_base,[output/'models'],10,ref_hash)
    experts['paper']=library.load_paper_expert(R,cfg,10)
    rows=[]
    for condition in settings['conditions']:
        sim=library.make_sim(R,cfg,condition)
        for jammer in settings.get('jam_types',['narrowband','sweep']):
            for jsr in settings['jsr_list']:
                for trial in range(settings['n_trials']):
                    seed=settings['seed_base']+trial+R['JT'].index(jammer)*2000
                    f=sim(R['ChannelConfig'](snr_db=cfg.snr_db,jsr_db=jsr,n_symbols=cfg.n_sym_test,seed=seed),jammer)
                    bits,rx,sps=f['bits'],f['rx'],f['sps'];results={'raw':float(np.mean((library.soft_symbols(rx,sps)>0)!=bits))}
                    for name,expert in experts.items():
                        results[name]=float(np.mean((expert.correct_symbols(rx,sps)>0)!=bits))
                    if trial<settings.get('n_trials_ale',settings['n_trials']):
                        results['ale']=float(np.mean((library.soft_symbols(library.apply_ale(rx,cfg,cfg.ale_mu_nb),sps)>0)!=bits))
                    rows.append(dict(condition=condition,jammer=jammer,jsr_db=jsr,trial=trial,seed=seed,ber=results))
                    (output/'records.json').write_text(json.dumps(rows,indent=2))
                    print(condition,jammer,jsr,trial,flush=True)
    summary={}
    for row in rows:
        for method,ber in row['ber'].items():
            key=f"{row['condition']}|{row['jammer']}|{row['jsr_db']}|{method}"
            summary.setdefault(key,[]).append(ber)
    (output/'summary.json').write_text(json.dumps(dict(smoke=a.smoke,mean_ber={k:float(np.mean(v)) for k,v in summary.items()}),indent=2))


if __name__=='__main__':
    main()
