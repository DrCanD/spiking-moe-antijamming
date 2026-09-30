"""Matched learning/adaptive baselines, oracle routing and confidence/JSR gates."""
from pathlib import Path
from dataclasses import replace
import argparse
import hashlib
import json
import numpy as np
import run_receiver_comparison as library
import run_router_comparison as frame_library
from run_router_comparison import core,metric,write


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',type=Path,default=Path('configs/routing_baselines.json'))
    ap.add_argument('--output',type=Path,default=Path('runs/routing_baselines'))
    ap.add_argument('--smoke',action='store_true')
    a=ap.parse_args();root=Path(__file__).resolve().parents[2]
    settings=json.loads((root/a.config).read_text());cfg=library.Config()
    for k,v in settings.items():
        if k in cfg.__dataclass_fields__:setattr(cfg,k,v)
    if 'ale_mu' in settings:cfg.ale_mu_nb=cfg.ale_mu_sw=settings['ale_mu']
    cfg.smoke=a.smoke
    if a.smoke:
        cfg.n_sym_test=cfg.train_n_sym=400;cfg.lstm_epochs=1;cfg.train_n_per_jsr=cfg.mono_n_per_jsr=1
        cfg.train_jsr_list=[0,10];cfg.paper_train_jsr=[4];cfg.paper_train_n_per_jsr=1
        cfg.router_train_jsr=[0,10];cfg.router_n_train_clean=2;cfg.router_n_train_per=1
        cfg.rf_trees=10;cfg.blanker_cal_n=2
        settings.update(conditions=['random'],jsr_list=[0,10],n_trials=1,n_clean_trials=1)
    c=core.Config(**{k:getattr(cfg,k) for k in core.Config.__dataclass_fields__ if hasattr(cfg,k)})
    output=root/a.output;output.mkdir(parents=True,exist_ok=True)
    R,ref_hash,_=library.load_reference(library.find_reference_source())
    lock=dict(settings=settings,smoke=a.smoke,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if (output/'protocol.json').exists() and json.loads((output/'protocol.json').read_text())!=lock:
        raise RuntimeError('Output belongs to a different protocol; choose a new --output.')
    write(output/'protocol.json',lock)
    rows=json.loads((output/'records.json').read_text()) if (output/'records.json').exists() else []
    done={r['id'] for r in rows}
    paper=library.load_paper_expert(R,cfg,c.sps)
    for condition in settings['conditions']:
        router=core.fit_routers(c,condition);receiver=core.Receiver(c,router)
        structured=library.train_or_load_expert(R,cfg,'structured',condition,['narrowband','sweep'],cfg.train_n_per_jsr,cfg.train_seed_base,[output/'models'],c.sps,ref_hash)
        mono=library.train_or_load_expert(R,cfg,'monolithic',condition,list(core.JT),cfg.mono_n_per_jsr,cfg.mono_seed_base,[output/'models'],c.sps,ref_hash)
        panel=[('none',0,t,settings.get('clean_seed_base',30000)+t) for t in range(settings['n_clean_trials'])]
        panel += [(jam,jsr,t,settings['seed_base']+t+core.JT.index(jam)*2000) for jam in core.JT for jsr in settings['jsr_list'] for t in range(settings['n_trials'])]
        for label,jsr,trial,seed in panel:
            uid=f'{condition}|{label}|{jsr}|{trial}'
            if uid in done:continue
            f=core.generate_frame(c,condition,() if label=='none' else (label,),() if label=='none' else (jsr,),seed)
            rx,bits=f['rx'],f['bits'];out=receiver.evaluate_all(rx,methods=('none','ale02','ale05','hard','full','conv_full'))
            soft={name:ex.correct_symbols(rx,c.sps) for name,ex in [('lstm_paper',paper),('lstm_structured',structured),('lstm_monolithic',mono)]}
            def pack_output(values,path):
                return dict(soft=values,erasure_mask=np.zeros(len(bits),bool),path=path,counts={},pred1=None)
            for name,v in soft.items():out[name]=pack_output(v,name)
            pred,confidence=router.predict_signal('snn',rx)
            def routed(verdict,values,name):
                if verdict in ('narrowband','sweep'):return pack_output(values,name+'->denoise')
                return receiver.evaluate_all(rx,methods=('hard',),predictions={'snn':(verdict,1.0)})['hard']
            out['moe_lstm']=routed(pred,soft['lstm_structured'],'moe_lstm')
            out['oracle_lstm']=routed(label,soft['lstm_structured'],'oracle_lstm')
            out['oracle_ale']=receiver.evaluate_all(rx,methods=('hard',),predictions={'snn':(label,1.0)})['hard']
            estimate=10*np.log10(max(np.mean(rx**2)/(1+10**(-cfg.snr_db/10))-1,1e-10))
            for threshold in settings.get('gate_thresholds',[4.,6.,8.,10.]):
                values=out['hard']['soft'] if estimate>=threshold else soft['lstm_structured']
                out[f'jsr_gate_{threshold:g}']=routed(pred,values,'jsr_gate')
            out['confidence_gate']=out['hard'] if confidence>=cfg.conf_threshold else out['none']
            if condition=='fixed':
                notch=R['apply_notch'](rx,c.fs)
                out['notch']=pack_output(core.soft_symbols(notch,c.sps),'notch')
            rows.append(dict(id=uid,condition=condition,jammer=label,jsr_db=jsr,trial=trial,seed=seed,
                             prediction=pred,confidence=confidence,jsr_estimate_db=float(estimate),
                             rx_sha256=hashlib.sha256(rx.tobytes()).hexdigest(),methods={k:metric(bits,v) for k,v in out.items()}))
            write(output/'records.json',rows);print(uid,flush=True)
    pooled={}
    for r in rows:
        for name,v in r['methods'].items():
            key=f"{r['condition']}|{r['jammer']}|{r['jsr_db']}|{name}"
            d=pooled.setdefault(key,dict(retained_bits=0,retained_errors=0,input_bits=0))
            for k in d:d[k]+=v[k]
    for d in pooled.values():d['ber']=d['retained_errors']/d['retained_bits'] if d['retained_bits'] else None
    write(output/'summary.json',dict(smoke=a.smoke,results=pooled))


if __name__=='__main__':
    main()
