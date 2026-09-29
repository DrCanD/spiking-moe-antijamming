"""Recompute Exp 9a/9b summaries from saved records. No receiver rerun.

Requires Python and NumPy. Optional --frozen-source checks the 15 source hashes.
Frame BER is conditional on surviving symbols; streaming success is the saved
RS/CRC payload-success indicator. Work counters are not energy measurements.
"""
from pathlib import Path
import argparse
import collections
import csv
import hashlib
import itertools
import json
import numpy as np

ROOT = Path(__file__).resolve().parent


def read(name):
    return json.loads((ROOT/name).read_text(encoding="utf-8"))


def write_csv(name, rows):
    with (ROOT/name).open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def frame_stats(rows, arm, survival_key):
    ber = np.array([r[arm] for r in rows], dtype=float)
    survival = np.array([r[survival_key] for r in rows], dtype=float) if survival_key else np.ones(len(rows))
    retained = np.rint(10000*survival).astype(np.int64)
    errors_float = 10000*survival*ber
    errors = np.rint(errors_float).astype(np.int64)
    assert np.max(np.abs(errors-errors_float)) < 1e-6
    assert np.all((errors >= 0) & (errors <= retained))
    return dict(frames=len(rows), macro_ber=float(ber.mean()),
                retained_bit_ber=float(errors.sum()/retained.sum()) if retained.sum() else None,
                retained_fraction=float(survival.mean()), zero_output_frames=int((retained==0).sum()),
                correct_retained_per_input=float((retained-errors).sum()/(10000*len(rows))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen-source", type=Path)
    args = parser.parse_args()
    ap, a = read("exp9a_partial.json"), read("exp9a_summary.json")
    data = ap["per_frame"]
    assert len(data) == len(ap["units_done"]) == 120
    assert set(data) == set(ap["units_done"])
    assert ap["ref_hash"] == a["reference"]["sha256_16"]
    arms = ["none","moe_hard","moe_final","ale_nb","ale_sw","chain_nb","chain_sw","rule_nb","rule_sw","oracle"]
    checked, maxdiff = 0, 0.0
    for unit, rows in data.items():
        expected = 50 if unit.startswith("compound|") else 100
        assert sorted(r["t"] for r in rows) == list(range(expected))
        for arm in arms:
            v=np.array([r[arm] for r in rows]);assert np.all(np.isfinite(v))
            diff=abs(float(v.mean())-a["agg"][unit+"|"+arm]["mean"])
            assert diff<1e-12
            maxdiff=max(maxdiff,diff);checked+=1
    groups=collections.defaultdict(list)
    for unit,rows in data.items():
        kind,cond,case,jsr=unit.split("|")
        groups[(kind,cond,case)].extend(rows)
    metric_rows=[]
    for group,rows in sorted(groups.items()):
        for arm,sv in [("moe_final","surv_moe"),("rule_sw","surv_rule_sw"),("rule_nb","surv_rule_nb"),
                       ("chain_sw","surv_chain_sw"),("chain_nb","surv_chain_nb"),("ale_sw",None),("ale_nb",None),("none",None)]:
            metric_rows.append(dict(kind=group[0],condition=group[1],case=group[2],arm=arm,**frame_stats(rows,arm,sv)))
    write_csv("exp9a_quality_by_regime.csv",metric_rows)
    allframes=[r for rows in data.values() for r in rows]
    global_a=[]
    for arm,sv in [("moe_final","surv_moe"),("rule_sw","surv_rule_sw"),("rule_nb","surv_rule_nb"),
                   ("chain_sw","surv_chain_sw"),("chain_nb","surv_chain_nb"),("ale_sw",None),("ale_nb",None),("none",None)]:
        cell_mean=float(np.mean([np.mean([r[arm] for r in rows]) for rows in data.values()]))
        if arm in a["baseline_scores"]:assert abs(cell_mean-a["baseline_scores"][arm])<1e-12
        global_a.append(dict(arm=arm,equal_cell_macro_ber=cell_mean,**frame_stats(allframes,arm,sv)))
    write_csv("exp9a_global_weightings.csv",global_a)
    a_work_cells=float(np.mean([np.mean([r["work_moe"] for r in rows]) for rows in data.values()]))
    a_work_frames=float(np.mean([r["work_moe"] for r in allframes]))
    cross4b = dict(available=False)
    if (ROOT/"exp4b_partial.json").exists():
        old=read("exp4b_partial.json")["per_frame"]; comparisons=0; maximum=0.
        mapping={"none":"none","snn_final":"moe_final","snn_hard":"moe_hard","snn_surv":"surv_moe","oracle":"oracle"}
        for unit,rows in old.items():
            newunit=unit.replace("clean|","single|",1) if unit.startswith("clean|") else unit
            new={r["t"]:r for r in data[newunit]}
            for r in rows:
                for lhs,rhs in mapping.items():
                    diff=abs(r[lhs]-new[r["t"]][rhs]);maximum=max(maximum,diff);comparisons+=1
        assert maximum<1e-12
        cross4b=dict(available=True,scalar_comparisons=comparisons,max_abs_difference=maximum,
                     scope="saved per-frame scalar outputs; not waveform or source-code equivalence")

    b=read("exp9b_summary.json");streams=read("exp9b_results.json")["streams"]
    assert len(streams)==640==len({r["unit"] for r in streams})==len({r["rx_sha"] for r in streams})
    methods=list(b["receivers"]);summary_rows={(r["condition"],r["receiver"]):r for r in b["results"]}
    all_b=[];seed_b=[]
    for cond in ["fixed","random","pooled"]:
        selected=[r for r in streams if cond=="pooled" or r["cond"]==cond]
        samples=sum(r["n"] for r in selected);blocks=sum(len(r["bseg"]) for r in selected)
        words=sum(len(r["wcls"]) for r in selected)
        always_products=sum(r["v"]["rule_slow"]["ale_tap_products"] for r in selected)
        for method in methods:
            failed=undetected=products=calls=0
            for r in selected:
                v=r["v"][method]
                assert len(v["succ"])==len(r["wcls"]) and set(v["succ"])<=set("01")
                assert len(v["act"])==len(r["bseg"])
                failed+=v["succ"].count("0");undetected+=v["undetected"]
                products+=v["ale_tap_products"];calls+=v["router_calls"]
            pct=100*failed/words
            row=dict(condition=cond,receiver=method,streams=len(selected),words=words,failed_words=failed,
                     bler_pct=pct,undetected_wrong=undetected,input_samples=samples,
                     ale_tap_products=products,ale_products_per_sample=products/samples,
                     ale_work_ratio_to_rule=products/always_products,
                     ale_work_reduction_pct=100*(1-products/always_products),router_calls=calls,
                     router_calls_per_block=calls/blocks)
            if cond=="pooled":assert abs(pct-b["pooled_all_bler_pct"][method])<1e-10
            else:
                s=summary_rows[(cond,method)]
                for x,y in [("bler_pct","bler_all_pct"),("ale_products_per_sample","ale_tap_products_per_sample"),
                            ("router_calls_per_block","router_calls_per_block")]:assert abs(row[x]-s[y])<1e-10
                assert undetected==s["undetected"]
            all_b.append(row)
        for seed in b["config"]["model_seeds"]:
            rr=[r for r in selected if r["ms"]==seed];nw=sum(len(r["wcls"]) for r in rr)
            sn=sum(r["v"]["spike9_refresh"]["succ"].count("0") for r in rr)
            ru=sum(r["v"]["rule_slow"]["succ"].count("0") for r in rr)
            seed_b.append(dict(condition=cond,model_seed=seed,streams=len(rr),words=nw,snn_failed=sn,rule_failed=ru,
                               rule_minus_snn_pp=100*(ru-sn)/nw))
    write_csv("exp9b_recomputed_summary.csv",all_b);write_csv("exp9b_by_model_seed.csv",seed_b)
    ci=[]
    # Exploratory model-seed-cluster bootstrap, conditional on the chosen rule.
    # Enumerate all 5^5 equiprobable resampling sequences, preserving whole
    # streams and all words within each selected model seed. For pooled data,
    # fixed/random rows with the same model-seed label stay in the same block.
    draws=np.array(list(itertools.product(range(5),repeat=5)))
    for cond in ["fixed","random","pooled"]:
        rows=[r for r in seed_b if r["condition"]==cond]
        numerator=np.array([r["rule_failed"]-r["snn_failed"] for r in rows])
        denominator=np.array([r["words"] for r in rows])
        boot=100*numerator[draws].sum(axis=1)/denominator[draws].sum(axis=1)
        lo,hi=np.quantile(boot,[.025,.975])
        ci.append(dict(condition=cond,rule_minus_snn_pp=100*numerator.sum()/denominator.sum(),
                       ci95_lo=float(lo),ci95_hi=float(hi),clusters=5,resampling_sequences=len(draws),
                       status="exploratory_seed_cluster_bootstrap_selected_baseline_not_confirmatory"))
    write_csv("exp9b_exploratory_seed_cluster_ci.csv",ci)
    source_check=dict(available=False)
    if args.frozen_source:
        details=[]
        for name,want in b["reference"]["files_sha256"].items():
            path=args.frozen_source/name
            got=hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
            details.append(dict(file=name,expected=want,observed=got,match=got==want))
        assert all(r["match"] for r in details)
        source_check=dict(available=True,matched=len(details),total=len(details),details=details)
    audit=dict(exp9a=dict(run_key=a["run_key"],frames=len(allframes),cells=len(data),means_verified=checked,
                         max_mean_difference=maxdiff,ale_passes_equal_cell_average=a_work_cells,
                         ale_passes_actual_frame_average=a_work_frames,
                         reported_regime_verdict_counts=dict(collections.Counter(v["verdict"] for v in a["verdicts"])),
                         crosscheck_exp4b=cross4b),
               exp9b=dict(run_key=b["run_key"],streams=len(streams),unique_waveform_hashes=len({r["rx_sha"] for r in streams}),
                         words_per_receiver=sum(len(r["wcls"]) for r in streams),source_check=source_check,
                         seed_cluster_ci=ci),
               files_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
                             ["exp9a_partial.json","exp9a_summary.json","exp9b_results.json","exp9b_summary.json"]},
               limitations=["This analysis does not rerun signal processing or decoding.",
                            "Global baselines were chosen using these evaluation panels; full alternatives retained.",
                            "Frame-mode BER/retention and streaming coded-word BLER describe different protocols.",
                            "Tap products and full-frame pass counts exclude front-end/router/masking/decoder energy.",
                            "Five model-seed blocks give an exploratory interval, not a comprehensive training-variance study.",
                            "Exact Exp 9a/9b wrapper source snapshots were not retrieved; 9b frozen dependency hashes were verified."])
    (ROOT/"audit_summary.json").write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"exp9a":audit["exp9a"],"exp9b_words":audit["exp9b"]["words_per_receiver"],
                      "exp9b_frozen_sources_matched":source_check.get("matched"),"seed_cluster_ci":ci},ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
