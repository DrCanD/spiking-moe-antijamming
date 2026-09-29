# Vitis HLS 2025.x — classic Tcl flow:   vitis_hls -f run_hls.tcl        (run from the hls/ directory)
# (Vitis Unified alternative: v++ -c --mode hls --config hls_config.cfg --work_dir moe_hls_work)
#   1. C simulation  : the testbench replays the vectors through the IP and checks every stream bit for bit
#   2. C synthesis   : 100 MHz, xck26 (KV260 SOM)
#   3. export        : IP catalog ZIP for the Vivado block design
# Set VECTORS to the directory with the Exp-5 vectors (Drive: Research/SNN_MoE_AntiJamming_R1/kv260_package/vectors)
# after running  python3 golden/convert_expected.py <that dir>  once.
cd [file dirname [file normalize [info script]]]
set VECTORS "../vectors";  if {[info exists ::env(MOE_VECTORS)]}  { set VECTORS $::env(MOE_VECTORS) }
set VECTORS [file normalize $VECTORS]
set N_REPL 8;              if {[info exists ::env(MOE_N_REPL)]}   { set N_REPL $::env(MOE_N_REPL) }
set CSIM_SET "syn_narrowband_0 syn_pulse_0 edge_silence"
if {[info exists ::env(MOE_CSIM_SET)]} { set CSIM_SET $::env(MOE_CSIM_SET) }
set DO_COSIM 0;            if {[info exists ::env(MOE_COSIM)]}    { set DO_COSIM $::env(MOE_COSIM) }
if {![file isdirectory $VECTORS]} { error "vectors directory not found: $VECTORS (set MOE_VECTORS)" }
puts "\[HLS\] vectors=$VECTORS  N_REPL=$N_REPL  csim set={$CSIM_SET}"

set CFLAGS "-Isrc -std=c++14 -DN_REPL=$N_REPL"
open_project -reset moe_hls
set_top moe_top
add_files src/moe_top.cpp -cflags $CFLAGS
add_files -tb tb/tb_moe.cpp -cflags $CFLAGS
add_files -tb $VECTORS
open_solution -reset sol1 -flow_target vivado
set_part {xck26-sfvc784-2LV-c}
create_clock -period 10.0 -name default
config_compile -pipeline_loops 0
config_rtl -reset control
# ── 1. C simulation (bit-exact check against the golden vectors; ~2-3 min per frame with ap_int) ──
set vdir [file tail $VECTORS]
csim_design -argv "$vdir $CSIM_SET" -clean
# ── 2. synthesis ──
csynth_design
# ── 3. optional RTL co-simulation on one short vector (slow); enable with MOE_COSIM=1 ──
if {$DO_COSIM} { cosim_design -argv "$vdir --quick syn_narrowband_0" -trace_level port }
# ── 4. export IP ──
file mkdir ../vivado
export_design -format ip_catalog -rtl verilog -vendor xilinx.com -library hls -ipname moe_top -version 1.0 -output ../vivado/moe_top_ip.zip
puts "\n\[HLS\] done: report = moe_hls/sol1/syn/report/moe_top_csynth.rpt ; IP dir = moe_hls/sol1/impl/ip (used directly by vivado/build_bd.tcl) ; register map = moe_hls/sol1/impl/ip/drivers/moe_top_v1_0/src/xmoe_top_hw.h"
exit
