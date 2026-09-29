# KV260 hardware package — final-MoE datapath (Exp 5 hardware part)

One Vitis HLS IP (`moe_top`) that carries the frozen final MoE datapath of Exp 6a in fixed point, **bit-exact
with the Exp-5 golden models** (`spec.json` / `vectors/` on Drive), replicated `N_REPL` times behind a sample
delay chain so that the dynamic power rises well above the board's 3.6 mW block-mean noise floor. A single
bitstream serves the whole power table: the design under test (D0…D5) is selected at run time through
block-enable bits.

| design | `blk_en` | blocks (all replicas) |
|---|---|---|
| D0 empty | 0 | replay from URAM + delay chain + pacing (overhead reference) |
| D1 front-end B | 1 | delta encoder + integer spike statistics + 16 resonate-and-fire neurons (spiking watchdog) |
| D2 conventional | 2 | Hann-1024 / FFT-1024 (Q15, unscaled) / Welch |X| accumulation + moments, sub-frame energies, zero crossings |
| D3 ALE-128 | 4 (NB) · 12 (NB+SW) | NLMS adaptive line enhancer, 128 taps, delay 20, Q6.26 weights, LUT reciprocal |
| D4 blanker | 16 | 50-sample group sum of squares vs the calibrated threshold |
| D5 full MoE | 29 | D1 + 2×ALE + D4 |

## Layout
```
golden/    fxgolden.py (verbatim Exp-5 fixed-point models), conv_golden.py (D2 model), gen_params.py -> hls/src/moe_params.h,
           make_vectors.py (synthetic vectors, same file formats as Drive), convert_expected.py (expected.json -> flat words for C)
hls/       src/moe_top.cpp, moe_blocks.h, moe_types.h, moe_params.h ; tb/tb_moe.cpp ; run_hls.tcl ; hls_config.cfg ; Makefile
vivado/    build_bd.tcl (KV260 block design + bitstream + reports), power_saif.tcl (report_power cross-check)
board/     moe_ctl.py (AXI-Lite driver via /dev/mem), parse_regmap.py, features_ps.py (PS half: features, forest, rho),
           verify_board.py (bit-exact check on silicon + router), measure.py (power protocol), pwlog.sh,
           firmware/ (pl.dtsi, shell.json, make_firmware.bat, install_firmware.sh)
vectors/   14 synthetic frames incl. edge cases (silence, LSB noise, clipping, Nyquist+DC), verified here
```

## Verification status (this package, before any FPGA tool)
* `hls/Makefile tb_c` (plain integers, N_REPL=16): **14/14 frames bit-exact** — spikes, 320 RF counters, both ALE streams
  (100 000 samples each), blanker mask + sums of squares, D2 moments/energies/zc/513 Welch bins; block enables do not interact;
  RUN mode reproduces the VERIFY checksum and the replicas' checksums differ (delay chain effective).
* `tb_ap` (the ap_int<N> word lengths that Vitis HLS synthesises, with range assertions enabled): bit-exact on the same set.
* `board/features_ps.py` reproduces the golden 9 front-end features and the 7 conventional features from the result words with
  zero error (validated with the C model's result block).
* Word lengths (measured maxima over the 14 frames, incl. JSR 20 dB and edge cases): RF state |z| < 2^19 (kept 28 b),
  ALE accumulator < 2^42 (56 b), error < 2^17 (32 b), gain gq < 2^33 (48 b, asserted), weights < 2^24 (32 b, clipped as in the model).

## Step 1 — vectors
Download `Research/SNN_MoE_AntiJamming_R1/kv260_package/` from Drive (spec.json, vectors/). Then once:
```
python3 golden/convert_expected.py <path>/kv260_package/vectors
```
(Synthetic vectors for quick checks are already in `vectors/`; regenerate with `python3 golden/make_vectors.py vectors`.)

## Step 2 — Vitis HLS 2025.2 (Windows, from `hls/`)
Vitis 2025.2 ships no `vitis_hls` executable; use the unified flow (cmd: `call C:\AMDDesignTools\2025.2\Vitis\settings64.bat` first):
```
vitis-run --mode hls --csim    --config hls_config.cfg --work_dir moe_hls_work  > csim.log 2>&1
v++ -c --mode hls              --config hls_config.cfg --work_dir moe_hls_work  > synth.log 2>&1
vitis-run --mode hls --package --config hls_config.cfg --work_dir moe_hls_work  > package.log 2>&1
```
(`run_hls.tcl` is the equivalent classic-flow script for installations that still have `vitis_hls`.) Edit `hls_config.cfg` to change
N_REPL (`-DN_REPL=8` in both cflags lines), the vectors directory (`tb.file=`) or the csim frame set (`csim.argv=`).
C simulation must print `[TB] … -> PASS`. Synthesis report: `moe_hls_work/hls/syn/report/moe_top_csynth.rpt` (classic flow: `moe_hls/sol1/syn/report/`) — the per-instance
table (`fe_step`, `ale_shared_step`, `ale_step`, `blank_step`, `conv_step`/`conv_fft`, `rep_step_*`) gives the LUT/FF/DSP/BRAM
column of the paper's table per block. Packaged IP: `hls/moe_hls_work/hls/impl/ip` (Vivado's build_bd.tcl finds it); register map: `moe_hls_work/hls/impl/ip/drivers/moe_top_v1_0/src/xmoe_top_hw.h`.

## Step 3 — Vivado 2025.2 (from `vivado/`, ~80 min on 8 jobs)
```
vivado -mode batch -source build_bd.tcl -log build.log -journal build.jou
```
Outputs `moe_vivado/moe_vivado.runs/impl_1/design_1_wrapper.bit`, `reports/utilization_hier.rpt`, `timing.rpt`, `power_vectorless.rpt`,
`reports/timing_status.txt`, `design_1.xsa`, `moe.bif`, and it rewrites `board/firmware/pl.dtsi` + `pl_clk_hz.txt` with the PL clock it
constrained. The last log lines print `TIMING MET` / `NOT MET` — only a MET bitstream goes to the board.
PL clock: `MOE_PL_MHZ` env var, default 89 → the PS picks divider 18, 1499.985/18 = **83.332 MHz** (12.0 ns period) (the IP is synthesised for 10 ns; at 100 MHz the
first build missed by WNS −0.406 ns on conv_step's 64×64 multiplier, so the measurement clock is 83.332 MHz — quote that in the paper).
If it still fails, `$env:MOE_PL_MHZ=80` and rerun. If utilisation fails, rebuild the IP with `-DN_REPL=4`.

## Step 4 — firmware (Windows -> board)
```
cd board\firmware ; make_firmware.bat            -> moe.bit.bin (bootgen)
scp moe.bit.bin pl.dtsi pl_clk_hz.txt moe.dtbo shell.json install_firmware.sh unload_firmware.sh ubuntu@<board-ip>:~/moe/firmware/
scp -r ..\*.py ubuntu@<board-ip>:~/moe/          (moe_ctl.py, features_ps.py, verify_board.py, measure.py, parse_regmap.py)
scp <hls>\moe_hls_work\hls\impl\ip\drivers\moe_top_v1_0\src\xmoe_top_hw.h ubuntu@<board-ip>:~/moe/
scp -r <path>\kv260_package ubuntu@<board-ip>:~/moe/        (spec.json + vectors)
```
On the board (no internet needed; dtc is present on the Kria Ubuntu image, otherwise the shipped moe.dtbo is used):
```
cd ~/moe/firmware && sudo ./install_firmware.sh            # compiles moe.dtbo, xmutil loadapp moe, checks pl0_ref <= pl_clk_hz.txt
cd ~/moe && python3 parse_regmap.py xmoe_top_hw.h regmap.json
sudo python3 moe_ctl.py                                     # smoke: magic ok, N_REPL=…
sudo python3 verify_board.py kv260_package/vectors --spec kv260_package/spec.json     # 30 frames bit-exact on silicon + router verdicts
```

## Step 5 — measurement
```
sudo python3 measure.py --vector kv260_package/vectors/fixed_narrowband_0_in_q6_10.bin --block 120 --repeats 5 --target 1.0
```
Per design: pace calibration to 1.0 MS/s (the achieved rate is recorded), then idle/run/idle… blocks at 10 Hz on the SOM input
INA260. Output `measure_results.json` + a table: P_dyn (all replicas, ± std over runs), per-replica block power
(D_k − D0)/N_REPL and energy per sample. Quick pass first with `--block 60 --repeats 3`. Keep the board otherwise idle
(gdm stopped, performance governor is set by the script).

## Step 6 — report_power cross-check (optional, for the paper's "analytical vs measured" remark)
See `vivado/power_saif.tcl`: RTL co-simulation of one frame with a SAIF dump, out-of-context implementation of the IP, `report_power`
with real switching activity, per block module.

## What goes into the paper
Table: block | ops/s (README Exp 5 §5) | LUT/FF/DSP/BRAM (HLS/Vivado hierarchical report) | P_dyn per datapath (mW, measured, ± over 5 runs)
| E/sample (nJ) — for D1 vs D2 (R2-4/R3-2), D3, D4, D5. State explicitly: same bitstream, same PL clock (83.332 MHz, timing met; `measure_results.json` records `f_clk_hz` as read from the kernel), same replay,
P_dyn = P(run) − P(idle) at the achieved sample rate, N_REPL replicas, overhead D0 subtracted; the fixed-point datapath is bit-exact
with the simulated system on 30 test frames (verify_board_report.json).

## D6 — per-frame classification tail (feature finalisation + Random Forest), `hls_rf/`
The D1/D2 measurements cover the streaming, per-sample part of the front-ends. The router's per-frame tail (9 feature
finalisations in float64, exactly as the Colab/PS reference, and the 100-tree Random Forest) is a second, small IP
(`rf_top`, one hardware instance of the tail, `N_REPL_RF` = 1 frame buffer) that repeats the inference at a paced rate;
energy per inference = P_dyn / rate, per input sample = E_inf / 100 000. The tail is not tied to a sample rate, so the
measured power is raised by running the single instance at a high inference rate rather than by replicating hardware
(a 4-replica version with automatic loop pipelining was 2.6x over the LUT budget: five copies of the float64 dividers).
Synthesis uses `syn.compile.pipeline_loops=0`; only the marked loops are pipelined and ALLOCATION pragmas pin ddiv/dsqrt/dmul/dadd. Modes: R0 overhead, R1 features, R2 + forest
'fixed', R3 + forest 'random'. Verified here with g++ (`hls_rf/Makefile`): 44 frames (30 Drive + 14 synthetic), features
bit-identical to the float64 reference, 1760 forest verdicts (20 rotations x 2 forests) identical.
```
cd hls_rf
vitis-run --mode hls --csim    --config hls_rf.cfg --work_dir rf_hls_work        # ~1 min -> PASS
v++ -c --mode hls              --config hls_rf.cfg --work_dir rf_hls_work        # ~2-4 min; check utilization in rf_hls_work\hls\syn\report\rf_top_csynth.rpt
vitis-run --mode hls --package --config hls_rf.cfg --work_dir rf_hls_work
cd ..\vivado ; vivado -mode batch -source build_bd_rf.tcl -log build_rf.log -journal build_rf.jou   # TIMING MET expected
cd ..\board\firmware_rf ; .\make_firmware.bat ; scp rf.bit.bin pl.dtsi pl_clk_hz.txt rf.dtbo shell.json install_firmware.sh unload_firmware.sh ubuntu@<board-ip>:~/moe/firmware_rf/
scp ..\rf_ctl.py ..\verify_rf.py ..\measure_rf.py ..\parse_regmap.py ubuntu@<board-ip>:~/moe/
scp ..\..\hls_rf\rf_hls_work\hls\impl\ip\drivers\rf_top_v1_0\src\xrf_top_hw.h ubuntu@<board-ip>:~/moe/
```
On the board: `cd ~/moe/firmware_rf && sudo ./install_firmware.sh` (unloads the moe app, loads rf), `python3 parse_regmap.py
xrf_top_hw.h regmap_rf.json`, `sudo python3 rf_ctl.py`, `sudo python3 verify_rf.py kv260_package/vectors --spec kv260_package/spec.json`,
`sudo python3 measure_rf.py kv260_package/vectors --block 60 --repeats 3` (paced target chosen automatically = 90 % of the
slowest unpaced mode; about 35 min; `--target <inf/s>` overrides).
