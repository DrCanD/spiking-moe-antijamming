# Source and evidence map

| Result | Run or audit | Recorded evidence |
| --- | --- | --- |
| Front-end ablations | `simulation/front_end/run_ablation.py` | `results/experiments/front_end*/` and `configs/front_end_*.json` |
| Expert and routing comparisons | `simulation/frame_receiver/run_expert_comparison.py`, `run_receiver_comparison.py`, `run_routing_baselines.py` | `results/experiments/{expert_jsr,frame_ber,frame_ber_ale128,final_receiver}/` |
| Compound recovery | `simulation/frame_receiver/run_router_comparison.py --study compound`; `analysis/compound_interference/audit_records.py` | `analysis/compound_interference/{partial,summary}.json`, `analysis/router_comparison/frame_records.json` |
| Switching, fading, retention and coding | `simulation/frame_receiver/run_system_metrics.py` | `results/experiments/system_metrics/` |
| Streaming router-free comparison | `simulation/streaming/run_comparison.py --study router`; `analysis/router_comparison/audit_records.py` | `analysis/router_comparison/{stream_records,stream_summary}.json` |
| Band count and matched feature dimensions | `simulation/streaming/run_comparison.py --study bands` or `--study features` | `results/experiments/{band_count,rf_counterpart}/` |
| Frozen refresh validation | `simulation/streaming/run_v5.py` | `results/streaming/{development,validation}/` and the frozen protocol |
| State-bound gate and fixed-point quality | `hardware/kv260/{validate_fixed,prepare_vectors,build_kv260}.py` | `hardware/kv260/evidence/`, `results/hardware/KV260_GATE_RESULTS.zip` |
| Matched block energy | `hardware/kv260/prepare_matched_vectors.py`, `board/run_matched.py`; `analysis/energy/reproduce.py` | `results/hardware/KV260_MATCHED_RESULTS.zip` and the gate archive |
| Classifier inference energy | `hardware/kv260/classifier_tail/`; `analysis/energy/classifier_tail.py` | `results/hardware/classifier_tail/{power_records,verification}.json` |

## Source identity

The gate's synthesizable HLS sources retain the recorded build digest `90e0039e5aaad84606f142049be6d337fe62a4f9ef597cfc5f363c82bd7c067f`. Frozen numerical snapshots retain their recorded source hashes. Importable original receiver definitions replace the executed notebook; the definitions selected by the frame programs are unchanged. Historical summaries label the original run-time notebook hash as `reference.recorded_notebook.sha256_16`. The separate `reference.distributed_source` field identifies the current Python source and its full SHA-256. These are different source artifacts.

The matched-vector generator was recovered, and its integer golden has recorded SHA-256 `9a38fbf1c12835e09661ff7669a45253f53b95119c913c917c4cdf0e24a40c84`. The reconstructed native testbench produces all **78 captures** with the archived hashes.

The original campaign receipt is preserved byte for byte at `hardware/kv260/evidence/campaign/matched_native_verification.json`, extracted from `provenance/evidence/matched_native_verification.json` in `results/hardware/KV260_MATCHED_RESULTS.zip`. It records the campaign testbench hash `c65a90c428ad740a6345674e3d41fb40587dea7ba8b4a02cfcb135c1ce6fdd3c`; that original testbench source was not recovered. The distributed regeneration testbench has hash `b193ecca36ed2a5b0da9dd5dc76f27f9691bfa60da68f976b689e986b54627f0`. `hardware/kv260/evidence/matched_provenance.json` maps these identities. A new native run writes `hardware/kv260/evidence/matched_regeneration.json`; it does not overwrite the campaign receipt or the recorded capture manifest. The checked regeneration reference is stored separately as `matched_regeneration_reference.json`.

Classifier-tail sources, exported forests and the 44-frame binary test set were recovered from the measured design. Native verification gives zero feature and verdict mismatches across both forests and all twenty subframe rotations. `extract_vectors.py` restores the frame-word inputs for board verification and measurement.

Intermediate ablation entry points were rebuilt around the archived grids and shared numerical kernels. These runnable studies accompany the historical summaries; their original launchers were not recovered. Final frame/system programs, the frozen refresh runner and hardware kernels come from recorded source snapshots. Full historical refresh-validation shards and model checkpoints are not bundled; the runner regenerates them in a new output directory.

## Denominators and energy scope

Frame BER is conditional on retained bits; retained fractions and zero-output frames are reported alongside it. Streaming BLER uses RS/CRC payload success. These denominators remain separate.

Hardware energy uses paired SOM-input-power contrasts, shared D0 subtraction, replica normalization and achieved replay rates. The measured front-end-plus-ALE contrast is distinct from the receiver estimate weighted by streaming activity. Classifier-tail cost is measured separately and amortised using recorded classifier activity. RF/ADC, transfer, decoding and flight power are outside these datapath comparisons.

The portable raw-energy auditor changes evidence filenames and temporary-directory names only. Its numerical output and CSV agree with the recorded audit. `analysis/energy/source_identity.json` records the original and current auditor hashes.

## Distribution

Project input/output paths are relative. Device files and vendor tools are resolved on the machine executing the hardware workflow. Binary models, captures and test vectors are restored from `assets/binary-evidence.zip` and checked against `assets/manifest.json`. Restored assets and new verification receipts are ignored by Git. Proprietary AMD tool installations and generated bitstreams are not bundled.

`scripts/update_manifests.py` generates the asset manifest, restored-file ignore rules, `hardware/kv260/PACKAGE_SHA256.json` and `MANIFEST_SHA256.txt` together. It uses Git's distributed file list and the archived asset bytes, excluding local build outputs and new run receipts. `--check` detects stale generated files without changing them. Binary restoration is independent of documentation edits. CI checks both manifests, restoration and a clean checkout.

Code is licensed under [MIT](LICENSE). Recorded data, model parameters and measurement evidence are licensed under [CC BY 4.0](DATA_LICENSE.md). A tagged release can be archived for a DOI; add the DOI after that archive has been created.
