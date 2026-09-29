# Reproducibility map and release checks

This page records which frozen artifact supports each manuscript result. Results from unlike protocols are not pooled to form a new effect size.

| Manuscript result | Evidence in this repository | Reproduction and boundary |
| --- | --- | --- |
| Randomized compound pulse-containing frames retain approximately 69% of input bits | `analysis/exp4b/{partial.json,summary.json}`; `analysis/exp9/{exp9a_partial.json,exp9a_summary.json}` | `python analysis/exp4b/audit_exp4b.py` recomputes conditional BER, retained-bit fractions, empty outputs and paired trial-block intervals from saved frame records. It does not reconstruct missing sample masks or refit the router. |
| 640-stream spike routing vs router-free rules, and 39.7–43.9% fewer counted ALE operations | `analysis/exp9/{exp9b_results.json,exp9b_summary.json}` and its 15-file `frozen_source/` snapshot | `python analysis/exp9/audit_exp9.py --frozen-source analysis/exp9/frozen_source` checks per-codeword successes, model-seed contrasts and counted work. This is a saved-record audit, not a waveform rerun. Frame BER and stream BLER have different denominators. |
| Frozen v5 classifier-refresh development (720 streams) and validation (1,200 new streams) | `simulation/v5/`; `evidence/v5/{development,validation}/`, plus `verification.json` | Runner source bytes match all 22 hashes in each archived lock. Run `verify_v5.py` and the profile commands in `README.md` for fresh results. Checked-in tables/locks are summaries; full saved per-stream shards and model checkpoints were not present in the recovered local snapshot. |
| Fixed-point state-bound gate on KV260 | `hardware/kv260_v3/` | `python hardware/kv260_v3/verify_package.py` checks 437 delivered files. Vitis/Vivado 2025.2 and a KV260 are required for new synthesis/board measurements. Hardware evidence includes board verification reports in both raw archives. |
| FFT-1024 vs gated front end, measured with the same ALE continuously active, 9.7–22.9% lower incremental energy | `evidence/hardware/KV260_MATCHED_RESULTS.zip`, `evidence/hardware/KV260_GATE_RESULTS.zip`; `analysis/hardware/` | `python analysis/hardware/reproduce.py` unpacks both original archives in a temporary directory, checks raw sensor means, rates, D0 subtraction, paired intervals, build identity, and exact output agreement. Achieved matched rates were 0.7471–0.7533 MS/s. These are controlled block combinations, not a complete receiver. |
| 23–38% lower activity-weighted datapath energy estimate | Same two hardware archives and `analysis/hardware/exp9b_activity_for_energy.json` | The raw audit reports its composition formula and sensitivity calculations separately from direct block measurements. Classifier, RS/CRC, RF/ADC, streaming transfer, and flight power are outside this estimate. |
| Five numerical figures | `figures/UAV_R1_Figures.m` and `figures/UAV_R1_Figure_Data.mat` | MATLAB R2020b or newer. The fifth MAT panel is the archived earlier v3 front-end kernel study; the later matched front-end-plus-ALE result is reported in the manuscript table and hardware audit, not backfilled into this figure. MATLAB export and visual inspection should be repeated at journal size. |

## Provenance notes

- Frame programs `exp6a_final_moe.py` and `exp6b_system_tables.py` come from the revision-era Colab notebook workflow. Their local Drive directories and the supplied original LSTM weights need to be staged before they run. `original_notebook_source.py` and the original notebook are included to trace the pre-revision receiver.
- The notebook copy has its executed outputs and execution counters removed. Its code-cell source is unchanged, so the selected-definition hash used by the frame programs is preserved.
- Exp 1–4b and the exact Exp 9a/9b full generation launchers were not present in the recovered local source archive. We include their recorded summaries, the available final frame programs, and saved-record audit scripts without claiming source-complete regeneration of every intermediate result.
- The matched KV260 runner was archived from the actual campaign alongside build identity, board verification and a manifest of expected captures. The extra matched expected-capture binaries and the original matched-vector generator are not in this local snapshot. A new matched board run requires recreating these files and checking their manifest hashes first.
- The v3 package README was frozen before final board measurements and still says that bitstream generation and power collection were pending. The two later raw board archives are the subsequent measured evidence. Do not rewrite the frozen package README without invalidating its original integrity manifest.
- The repository intentionally excludes proprietary AMD installation files, synthesized bitstreams, board credentials, local Vitis build directories, and logs with machine paths. The original HLS C++ sources and generated input/expected-output vectors are supplied.
- Two older narratives have distinct accounting scopes: the original approximately 300× analytical claim is withdrawn, and the measured percentages here use paired SOM-input power with a shared D0 reference. Do not divide the two numbers to infer an efficiency change.

## Before a public release

1. Choose the code and data license; none has been assigned on the author's behalf.
2. Recover and review missing early experiment launchers and full v5 per-stream shards if a source-complete, record-complete public archive is intended. The present repo is explicit about that scope.
3. Review notebook outputs and model weights for rights and size, and check the MATLAB/PPTX figure exports at final print size.
4. Create a repository under the author's GitHub account, publish a specific immutable release/tag, and optionally archive it with a DOI. Verify the public URL before changing `\section*{Data and Code Availability}` in the manuscript.
5. Link that version in the point-by-point reviewer response. State direct hardware block measurements separately from projections and simulation operation counts.

## Suggested manuscript statement after publication

Replace `<tagged repository URL>` only after the release is publicly reachable:

```tex
\section*{Data and Code Availability}
The transmitted signals and interference are synthetically generated. The receiver simulations, frozen configurations and seeds, saved result summaries, KV260 source, and raw board-power records are available at \url{<tagged repository URL>}. The repository documents the separate frame, streaming, and hardware protocols and the scope of each reproducibility check.
```

The present manuscript still uses its existing availability sentence. It should not assert an uncreated URL.
