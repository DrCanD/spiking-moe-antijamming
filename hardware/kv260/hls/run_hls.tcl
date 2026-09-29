cd [file dirname [file normalize [info script]]]
set flags "-Isrc -std=c++14 -DUSE_AP_INT -DN_REPL=4"
open_project -reset gate_hls
set_top moe_top
add_files src/moe_top.cpp -cflags $flags
add_files -tb tb/tb_gate.cpp -cflags $flags
add_files -tb ../vectors
open_solution -reset sol1 -flow_target vivado
set_part {xck26-sfvc784-2LV-c}
create_clock -period 10.0 -name default
set_clock_uncertainty 1.0
config_compile -pipeline_loops 0
config_rtl -reset control
csim_design -clean -argv "vectors fixed_none fixed_narrowband fixed_sweep fixed_pulse fixed_broadband fixed_switching random_none random_narrowband random_sweep random_pulse random_broadband random_switching edge_silence rtl_smoke"
csynth_design
cosim_design -argv "vectors rtl_smoke" -trace_level port
export_design -format ip_catalog -rtl verilog
exit
