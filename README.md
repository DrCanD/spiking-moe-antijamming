# Spiking Front End MoE for Physics-Aware Anti-Jamming

Receiver simulations, frozen configurations, FPGA sources and measured results for **“Spiking-Front-End Mixture of Experts for Physics-Aware Anti-Jamming on a Baseband Digital Link.”** 

The receiver uses spike timing to identify interference structure, selects the appropriate mitigation path, and preserves useful samples under compound interference. Causal refresh routing reduces classifier and adaptive-filter work. The KV260 implementation verifies the state-bound resonator gate and measures its energy advantage against an FFT front end.

## Main results

| Result | Observation |
| --- | --- |
| Useful-data retention | **68.7–69.4% of input bits retained** in randomized tone-plus-pulse and sweep-plus-pulse mixtures; the tested conventional-feature cascade erases almost all samples. |
| Adaptive-filter workload | **39.7–43.9% fewer ALE operations** over **640 streams**, with pooled BLER **7.68% versus 7.97%** for the selected router-free rule. |
| Classifier activity | **62.9% fewer classifier calls** in frozen **1,200-stream validation**, satisfying every prespecified descriptive quality cell. |
| Measured energy | **9.7–22.9% lower incremental energy** for the gated front end plus continuously active ALE, compared with FFT-1024 plus the same ALE on the same bitstream at a matched replay rate. |
| Receiver datapath estimate | **23–38% lower energy**, combining measured block costs with streaming activity. |

## Layout

| Directory | Purpose |
| --- | --- |
| [`simulation/front_end/`](simulation/front_end/) | Neuron-count, robustness and representation ablations. |
| [`simulation/frame_receiver/`](simulation/frame_receiver/) | Expert, routing, compound-interference and system comparisons; original receiver definitions and LSTM weights. |
| [`simulation/streaming/`](simulation/streaming/) | Causal receiver, frozen refresh validation, band-count and matched-feature comparisons. |
| [`configs/`](configs/) | Recorded parameter grids and random-seed settings. |
| [`hardware/kv260/`](hardware/kv260/) | Gate HLS/Vivado source, fixed-point checks, expected captures and board measurement runners. |
| [`hardware/kv260/classifier_tail/`](hardware/kv260/classifier_tail/) | Feature-finalisation and Random Forest hardware, both frozen forests, native tests and classifier power protocol. |
| [`analysis/`](analysis/) | Frame/stream record audits and raw-energy reproduction. |
| [`results/`](results/) | Recorded simulation summaries and raw hardware-power evidence. |
| [`figures/`](figures/) | PDF figures and editable architecture slides. |

## Setup

Run commands from the repository root using Python 3.12. Simulation uses the CPU; PyTorch can use a GPU for LSTM training.

```bash
python -m venv .venv
```

Activate the environment with `.\.venv\Scripts\Activate.ps1` in PowerShell or `source .venv/bin/activate` in Bash, then:

```bash
python -m pip install -r requirements.txt
python scripts/restore_assets.py
```

For streaming alone, use `simulation/streaming/frozen_v4/requirements.txt`.

## Run simulations

| Study | Command |
| --- | --- |
| Neuron count | `python simulation/front_end/run_ablation.py --study neuron_count` |
| Front-end robustness | `python simulation/front_end/run_ablation.py --study robustness` |
| Representation comparison | `python simulation/front_end/run_ablation.py --study representation` |
| LSTM and adaptive experts | `python simulation/frame_receiver/run_expert_comparison.py` |
| Final frame receiver | `python simulation/frame_receiver/run_receiver_comparison.py --config configs/receiver_comparison.json` |
| Earlier routing baselines | `python simulation/frame_receiver/run_routing_baselines.py --config configs/routing_baselines.json --output runs/routing_baselines` |
| ALE-128 routing ablation | `python simulation/frame_receiver/run_routing_baselines.py --config configs/routing_ale128.json --output runs/routing_ale128` |
| Compound recovery | `python simulation/frame_receiver/run_router_comparison.py --study compound --output runs/compound_recovery` |
| Frame router-free comparison | `python simulation/frame_receiver/run_router_comparison.py` |
| Switching, fading, retention and coded link | `python simulation/frame_receiver/run_system_metrics.py --config configs/system_metrics.json` |
| Stream router-free comparison | `python simulation/streaming/run_comparison.py --study router` |
| Resonator-band count | `python simulation/streaming/run_comparison.py --study bands` |
| Matched feature dimensions | `python simulation/streaming/run_comparison.py --study features` |
| Common-workload front ends | `python analysis/energy/run_front_end_comparison.py` |

Add `--smoke` for a short wiring check. Set `--output runs/<study>` for a separate result directory. Saved models, trial records and protocol locks support continuation where implemented.

The frozen development/validation experiment has its own source and protocol locks:

```bash
python simulation/streaming/verify_v5.py --output runs/verification.json
python simulation/streaming/run_v5.py --profile development --output runs/development --workers 2
python simulation/streaming/run_v5.py --profile validation --output runs/validation --selection runs/development/selection.json --workers 2
```

Development covers **720 streams**. Validation uses **1,200 new streams** with the locked development selection.

## Reproduce recorded results

```bash
python hardware/kv260/verify_package.py
python analysis/router_comparison/audit_records.py --frozen-source analysis/router_comparison/frozen_source
python analysis/compound_interference/audit_records.py
python analysis/energy/reproduce.py
python analysis/energy/classifier_tail.py
```

The raw-energy audit checks sensor means, rate matching, idle brackets, D0 subtraction and paired confidence intervals. Direct block measurements and the activity-weighted receiver estimate retain their distinct accounting scopes.

## FPGA verification and measurement

```bash
python hardware/kv260/prepare_matched_vectors.py
python hardware/kv260/classifier_tail/build_classifier.py --native-only
python hardware/kv260/build_kv260.py --jobs 4
```

Matched-vector regeneration reproduces **all 78 recorded captures byte for byte**. Classifier native verification checks **44 frames**, all nine features and both forests over all twenty subframe rotations. New synthesis requires AMD Vitis/Vivado 2025.2; new power measurements require a KV260. Board runners are [`run_gate.py`](hardware/kv260/board/run_gate.py), [`run_matched.py`](hardware/kv260/board/run_matched.py) and [`measure_classifier.py`](hardware/kv260/classifier_tail/board/measure_classifier.py).

See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for source/evidence mapping. Code is distributed under the **[MIT license](LICENSE)**.
