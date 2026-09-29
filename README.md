# Spiking Front End MoE for Physics Aware Anti-Jamming

Code and results for **“Spiking-Front-End Mixture of Experts for Physics-Aware Anti-Jamming on a Baseband Digital Link.”**

The receiver uses spike timing to route received baseband samples to mitigation methods matched to interference structure. It combines compound-interference recovery with selective computation and a measured FPGA front end.

## Main results

| Result | Observation |
| --- | --- |
| Compound interference | The spike-routed receiver retains **68.7–69.4% of input bits** in randomized tone-plus-pulse and sweep-plus-pulse mixtures, while the tested conventional-feature cascade erases almost all samples. |
| Streaming workload | Across **640 streams**, refresh routing uses **39.7–43.9% fewer ALE operations** than the selected router-free rule. Pooled BLER is **7.68% versus 7.97%**. |
| Classifier calls | A frozen, independent **1,200-stream validation** cuts classifier calls by **62.9%** while satisfying every prespecified descriptive quality cell. |
| KV260 energy | On a common bitstream and matched replay rate, gated front end plus continuously active ALE uses **9.7–22.9% less measured incremental energy** than FFT-1024 plus the same ALE. Measured block costs combined with streaming activity give a **23–38% datapath energy estimate**. |

The energy measurement is a paired SOM-input-power contrast for selected blocks. The activity-weighted receiver estimate is reported separately from that direct measurement.

## Repository contents

| Directory | Contents |
| --- | --- |
| [`simulation/v5/`](simulation/v5/) | Frozen v4 core and v5 causal refresh/rescue experiment, including fixed configurations and source locks. |
| [`simulation/frames/`](simulation/frames/) | Original receiver notebook, Exp 6a/6b frame programs, and original LSTM weights. |
| [`analysis/exp4b/`](analysis/exp4b/) and [`analysis/exp9/`](analysis/exp9/) | Saved frame/stream records and scripts that recompute compound, BLER, and operation-count results. |
| [`hardware/kv260_v3/`](hardware/kv260_v3/) | Identified KV260 HLS/Vivado source, fixed-point checks, vectors, board runner, and matched-campaign runner. |
| [`evidence/hardware/`](evidence/hardware/) and [`analysis/hardware/`](analysis/hardware/) | Raw board-power archives, paired-energy audit, and exact result reproduction. |
| [`figures/`](figures/) | MATLAB plot source and data, plus editable architecture slides. |

## Reproduce the recorded comparisons

From the repository root, with Python 3.10+:

```bash
python scripts/restore_assets.py
python hardware/kv260_v3/verify_package.py
python analysis/hardware/reproduce.py
python analysis/exp9/audit_exp9.py --frozen-source analysis/exp9/frozen_source
python analysis/exp4b/audit_exp4b.py
```

The hardware energy audit uses only the Python standard library. The Exp 4b/9 audits require NumPy. `REPRODUCIBILITY.md` maps each manuscript result to its source, saved evidence, and protocol.

For a new v5 simulation run, use an isolated Python environment and the pinned requirements:

```bash
cd simulation/v5
python -m pip install -r frozen_v4/requirements.txt
python verify_v5.py --output ../../verification_local.json
python run_v5.py --profile development --output ../../run_development --workers 2
python run_v5.py --profile validation --output ../../run_validation \
  --selection ../../run_development/selection.json --workers 2
```

The simulation is CPU-based. Development covers 720 streams; validation uses 1,200 new streams and the locked development selection. The detailed protocol is in [`simulation/v5/README_TR.md`](simulation/v5/README_TR.md).

## Hardware provenance

The included v3 HLS source matches the recorded build digest `90e0039e5aaad84606f142049be6d337fe62a4f9ef597cfc5f363c82bd7c067f` in both board campaigns. The 437-file package passes `verify_package.py`. The paired raw-power analysis checks rate matching, idle/D0 references, five repeats per vector, and the reported confidence intervals directly from recorded samples.

See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for experimental details and the manuscript availability wording to use once a public release has a stable URL.
