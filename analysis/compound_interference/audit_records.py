"""Audit saved compound comparison frame records; no model fitting or receiver rerun.

Run in the directory containing the original summary.json and partial.json.
Requires NumPy. Empty-output frames have undefined conditional BER, not 0.5.
Original values remain unchanged in the source files.
"""
from pathlib import Path
import collections
import csv
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parent
BOOTSTRAPS = 20000
BOOT_SEED = 9284242
ARMS = ("snn", "conv")


def arm_stats(rows, arm, n_symbols, guard=False):
    b = np.asarray([r[f"{arm}_final"] for r in rows], dtype=float)
    s = np.asarray([r[f"{arm}_surv"] for r in rows], dtype=float)
    if guard:
        empty = s == 0
        b = np.where(empty, [r["none"] for r in rows], b)
        s = np.where(empty, 1.0, s)
    retained = np.rint(n_symbols * s).astype(np.int64)
    errors_float = n_symbols * s * b
    errors = np.rint(errors_float).astype(np.int64)
    assert np.max(np.abs(retained - n_symbols * s)) < 1e-7
    assert np.max(np.abs(errors - errors_float)) < 1e-7
    assert np.all((errors >= 0) & (errors <= retained))
    empty = retained == 0
    out = {
        "n_frames": len(rows), "input_symbols": len(rows) * n_symbols,
        "retained_symbols": int(retained.sum()), "retained_bit_errors": int(errors.sum()),
        "retention": float(s.mean()), "zero_output_frames": int(empty.sum()),
        "zero_output_fraction": float(empty.mean()),
        "conditional_ber_micro": float(errors.sum() / retained.sum()) if retained.sum() else None,
        "conditional_ber_macro_nonempty": float(b[~empty].mean()) if (~empty).any() else None,
        "correct_retained_per_input": float((retained - errors).sum() / (len(rows) * n_symbols)),
        "observed_error_per_input": float(errors.sum() / (len(rows) * n_symbols)),
        "erased_per_input": float(1 - s.mean()),
        "legacy_mean_score_including_empty_penalty": float(b.mean()),
    }
    assert abs(out["correct_retained_per_input"] + out["observed_error_per_input"]
               + out["erased_per_input"] - 1) < 1e-12
    return out


def main():
    summary = json.loads((ROOT / "summary.json").read_text())
    raw = json.loads((ROOT / "partial.json").read_text())
    units = raw["per_frame"]
    config = summary["config"]
    n_symbols = config["n_sym_test"]
    assert len(units) == len(raw["units_done"]) == 32
    assert set(units) == set(raw["units_done"])
    assert raw["ref_hash"] == summary["reference"]["sha256_16"]
    means_checked, max_mean_difference = 0, 0.0
    cell_rows, pooled_groups = [], collections.defaultdict(list)
    n_empty, n_empty_score_half, n_nonempty_score_half = 0, 0, 0
    for uid, rows in sorted(units.items()):
        kind, condition, compound, jsr = uid.split("|")
        expected = config["n_compound_trials"] if kind == "compound" else config["n_clean_trials"]
        assert sorted(r["t"] for r in rows) == list(range(expected))
        for key in ["none", "snn_hard", "snn_final", "conv_hard", "conv_final", "oracle"]:
            values = np.asarray([r[key] for r in rows])
            assert np.all(np.isfinite(values)) and np.all((values >= 0) & (values <= 1))
            error = abs(float(values.mean()) - summary["agg"][f"{uid}|{key}"]["mean"])
            max_mean_difference = max(max_mean_difference, error)
            assert error < 1e-12
            means_checked += 1
        for arm in ARMS:
            stats = arm_stats(rows, arm, n_symbols)
            cell_rows.append({"unit": uid, "arm": arm, **stats})
            for row in rows:
                assert 0 <= row[f"{arm}_surv"] <= 1
                is_empty = row[f"{arm}_surv"] == 0
                half = row[f"{arm}_final"] == 0.5
                n_empty += is_empty
                n_empty_score_half += is_empty and half
                n_nonempty_score_half += not is_empty and half
        if kind == "compound":
            pooled_groups[(condition, compound)].extend(dict(r, jsr=int(jsr)) for r in rows)
    pooled_rows, blocks, counterfactual_rows = [], [], []
    rng = np.random.default_rng(BOOT_SEED)
    for (condition, compound), rows in sorted(pooled_groups.items()):
        group_out = {"condition": condition, "compound": compound}
        for arm in ARMS:
            stats = arm_stats(rows, arm, n_symbols)
            pooled_rows.append({**group_out, "arm": arm, **stats})
            counterfactual_rows.append({**group_out, "arm": arm,
                "status": "post_hoc_zero_output_passthrough_counterfactual_not_executed",
                **arm_stats(rows, arm, n_symbols, guard=True)})
        # The reference seed formula reuses t across JSR levels. Resample the
        # entire t block, keeping SNN/conv and all three JSR levels together.
        by_t = collections.defaultdict(list)
        for row in rows:
            by_t[row["t"]].append(row)
        assert all(sorted(r["jsr"] for r in v) == config["compound_jsr_list"] for v in by_t.values())
        differences = []
        for t in sorted(by_t):
            values = by_t[t]
            differences.append(np.mean([r["snn_surv"] * (1-r["snn_final"])
                                        - r["conv_surv"] * (1-r["conv_final"]) for r in values]))
        differences = np.asarray(differences)
        samples = differences[rng.integers(0, len(differences), size=(BOOTSTRAPS, len(differences)))].mean(axis=1)
        lo, hi = np.quantile(samples, [0.025, 0.975])
        blocks.append({**group_out, "metric": "correct_retained_per_input_snn_minus_conv",
            "difference": float(differences.mean()), "ci95_lo": float(lo), "ci95_hi": float(hi),
            "n_trial_blocks": len(differences), "jsr_per_block": 3,
            "bootstrap_replicates": BOOTSTRAPS, "status": "exploratory_fixed_router_no_multiplicity_adjustment"})
    report = {
        "run_key": summary["run_key"], "units": len(units), "frames": sum(map(len, units.values())),
        "means_reconstructed": means_checked, "maximum_mean_difference": max_mean_difference,
        "empty_output_arm_frames": n_empty, "empty_frames_recorded_as_half": n_empty_score_half,
        "nonempty_arm_frames_recorded_as_half": n_nonempty_score_half,
        "source_sha256": {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                          for name in ("summary.json", "partial.json")},
        "original_repro_check_scope": "102 saved aggregate comparisons; not an independent per-sample rerun",
        "limits": ["No compound comparison execution source snapshot found; exact source parity not independently verified.",
                   "Local reference notebook SHA differs from run reference; source inspection is lineage evidence only.",
                   "No oracle survival masks, bit positions or FEC decoder outputs in saved per-frame records.",
                   "Correct retained fraction is an uncoded accounting metric, not packet goodput or post-FEC success.",
                   "The conditional BER of empty-output frames is undefined; 0.5 is retained only as a legacy score.",
                   "Trial-block bootstrap preserves reference seed reuse across JSR; it does not cover RF training variability.",
                   "Zero-output passthrough is a post-hoc counterfactual from saved no-correction outputs, not a new validated receiver."]}
    (ROOT / "audit_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for name, rows in [("corrected_cell_metrics.csv", cell_rows), ("corrected_pooled_metrics.csv", pooled_rows),
                       ("paired_trial_block_ci.csv", blocks), ("zero_output_guard_counterfactual.csv", counterfactual_rows)]:
        with (ROOT/name).open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    for row in blocks:
        print(row["condition"], row["compound"],
              "correct-retained delta SNN-conv:",
              f'{row["difference"]:.6f} [{row["ci95_lo"]:.6f}, {row["ci95_hi"]:.6f}]')


if __name__ == "__main__":
    main()
