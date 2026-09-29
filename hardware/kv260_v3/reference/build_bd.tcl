# Vivado 2025.2 — KV260 block design around the HLS IP, bitstream, reports.   Run from the vivado/ directory:
#   vivado -mode batch -source build_bd.tcl -log build.log -journal build.jou
# Inputs : the packaged IP left by the HLS flow (unified v++/vitis-run: ../hls/moe_hls_work/hls/impl/ip ; classic: ../hls/moe_hls/sol1/impl/ip)
# Outputs: moe_vivado/moe_vivado.runs/impl_1/design_1_wrapper.bit, reports/*.rpt, reports/timing_status.txt, moe.bif (for bootgen),
#          design_1.xsa, ../board/firmware/pl.dtsi + pl_clk_hz.txt (overlay with the PL clock actually constrained here)
# PL clock: MOE_PL_MHZ env var (default 89 -> the PS picks divider 18: 1499.985/18 = 83.332 MHz). The HLS IP was synthesised for 10 ns; at
#           100 MHz the first build failed timing by 0.41 ns (WNS -0.406, paths in conv_step's 64x64 multiplier), hence the default.
set here [file dirname [file normalize [info script]]]
cd $here
set part   xck26-sfvc784-2LV-c
set jobs   [expr {[info exists ::env(MOE_JOBS)] ? $::env(MOE_JOBS) : 8}]
set pl_mhz [expr {[info exists ::env(MOE_PL_MHZ)] ? $::env(MOE_PL_MHZ) : 89}]

# ── IP repository: the packaged IP that Vitis HLS leaves in <solution>/impl/ip (no unzip needed) ──
set ip_dir ""
foreach cand {../hls/moe_hls_work/hls/impl/ip ../hls/moe_hls/sol1/impl/ip ip_repo/moe_top} {
    if {[file exists [file normalize $cand]/component.xml]} { set ip_dir [file normalize $cand]; break }
}
if {$ip_dir eq ""} { error "HLS IP not found: run the HLS flow first (unified: hls/moe_hls_work/hls/impl/ip, classic: hls/moe_hls/sol1/impl/ip)" }
puts "\[BD\] IP repository: $ip_dir"
puts "\[BD\] PL clock request: $pl_mhz MHz (MOE_PL_MHZ)"

create_project moe_vivado ./moe_vivado -part $part -force
# board files (installed by the user): pick the newest KV260 SOM + carrier
set bp [lindex [get_board_parts -quiet -latest_file_version "*kv260_som*"] 0]
if {$bp ne ""} {
    set_property board_part $bp [current_project]
    catch { set_property board_connections [list som240_1_connector xilinx.com:kv260_carrier:som240_1_connector:1.4] [current_project] }
    puts "\[BD\] board part: $bp"
} else {
    puts "\[BD\] !! KV260 board part not found — building on the bare part $part (PS preset must then be set manually)"
}
set_property ip_repo_paths $ip_dir [current_project]
update_ip_catalog

create_bd_design design_1
# ── Zynq UltraScale+ MPSoC PS with the board preset ──
set ps [create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e zynq_ultra_ps_e_0]
if {$bp ne ""} { apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e -config {apply_board_preset "1"} $ps }
set_property -dict [list \
    CONFIG.PSU__USE__M_AXI_GP0 {1} CONFIG.PSU__MAXIGP0__DATA_WIDTH {128} \
    CONFIG.PSU__USE__M_AXI_GP1 {0} CONFIG.PSU__USE__M_AXI_GP2 {0} \
    CONFIG.PSU__USE__S_AXI_GP0 {0} CONFIG.PSU__USE__S_AXI_GP2 {0} \
    CONFIG.PSU__FPGA_PL0_ENABLE {1} CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ $pl_mhz \
    CONFIG.PSU__USE__IRQ0 {0} ] $ps
set pl_act [get_property CONFIG.PSU__CRL_APB__PL0_REF_CTRL__ACT_FREQMHZ $ps]
puts "\[BD\] PL clock 0: requested $pl_mhz MHz, actual (constraint) $pl_act MHz"

# ── the datapath IP ──
set ip [create_bd_cell -type ip -vlnv xilinx.com:hls:moe_top:1.0 moe_top_0]
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config [list Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} \
    Master {/zynq_ultra_ps_e_0/M_AXI_HPM0_FPD} Slave {/moe_top_0/s_axi_ctrl} ddr_seg {Auto} intc_ip {New AXI SmartConnect} master_apm {0}] \
    [get_bd_intf_pins moe_top_0/s_axi_ctrl]
# fixed base address for the PS software: 0xA000_0000, 64 KiB
set seg [get_bd_addr_segs -of_objects [get_bd_addr_spaces zynq_ultra_ps_e_0/Data] -filter {NAME =~ "*moe_top*"}]
if {$seg ne ""} { set_property offset 0xA0000000 $seg; set_property range 64K $seg }
validate_bd_design
save_bd_design
set wrapper [make_wrapper -files [get_files design_1.bd] -top]
add_files -norecurse $wrapper
set_property top design_1_wrapper [current_fileset]
update_compile_order -fileset sources_1

# ── synthesis + implementation + bitstream ──
launch_runs impl_1 -to_step write_bitstream -jobs $jobs
wait_on_run impl_1
open_run impl_1
file mkdir reports
report_utilization -hierarchical -hierarchical_depth 3 -file reports/utilization_hier.rpt
report_utilization -file reports/utilization.rpt
report_timing_summary -file reports/timing.rpt
report_power -file reports/power_vectorless.rpt              ;# vectorless estimate (cross-check only; see power_saif.tcl)
set bit [file normalize moe_vivado/moe_vivado.runs/impl_1/design_1_wrapper.bit]
write_hw_platform -fixed -include_bit -force design_1.xsa

# ── timing verdict (the bitstream is only usable for the measurement when this says MET) ──
set run [get_runs impl_1]
set wns [get_property STATS.WNS $run]; set tns [get_property STATS.TNS $run]; set whs [get_property STATS.WHS $run]
if {![string is double -strict $wns] || ![string is double -strict $whs]} {   ;# run stats missing -> ask the timing engine directly
    set wns [get_property SLACK [lindex [get_timing_paths -setup -max_paths 1] 0]]
    set whs [get_property SLACK [lindex [get_timing_paths -hold  -max_paths 1] 0]]
    if {![string is double -strict $tns]} { set tns "n/a" }
}
set met [expr {$wns >= 0 && $whs >= 0}]
set verdict [expr {$met ? "MET" : "NOT MET"}]
set f [open reports/timing_status.txt w]
puts $f "PL clock: $pl_act MHz (requested $pl_mhz)\nWNS = $wns ns\nTNS = $tns ns\nWHS = $whs ns\nTIMING $verdict"
close $f

# ── files for the board: bootgen input and the overlay carrying the PL clock that was actually constrained ──
set f [open moe.bif w]
puts $f "all:\n{\n  \[destination_device = pl\] $bit\n}"
close $f
# assigned-clock-rates a hair below the constraint so both round-to-nearest and round-down divider selection give <= pl_act
set pl_hz [expr {int(floor($pl_act * 1e6)) - 1000}]
set fw [file normalize ../board/firmware]
if {[file isdirectory $fw]} {
    set f [open $fw/pl_clk_hz.txt w]; fconfigure $f -translation lf; puts $f $pl_hz; close $f
    set f [open $fw/pl.dtsi w]; fconfigure $f -translation lf   ;# LF endings: these files go to the board
    puts $f "/dts-v1/;
/plugin/;
/* Kria firmware overlay for the moe_top measurement design (written by vivado/build_bd.tcl): loads moe.bit.bin through the
   FPGA manager and sets PL clock 0 (zynqmp_clk 71 = pl0_ref) to $pl_hz Hz = the $pl_act MHz the bitstream was constrained at.
   The IP is reached via /dev/mem at 0xA000_0000 (M_AXI_HPM0_FPD, 128-bit -> AXI-Lite through SmartConnect); no driver node. */
&fpga_full {
    firmware-name = \"moe.bit.bin\";
    resets = <&zynqmp_reset 116>, <&zynqmp_reset 117>, <&zynqmp_reset 118>, <&zynqmp_reset 119>;
};
&amba {
    afi0: afi0 {
        compatible = \"xlnx,afi-fpga\";
        config-afi = <0 0>, <1 0>, <2 0>, <3 0>, <4 0>, <5 0>, <6 0>, <7 0>, <8 0>, <9 0>, <10 0>, <11 0>, <12 0>, <13 0>, <14 0xa00>, <15 0x000>;
    };
    clocking0: clocking0 {
        #clock-cells = <0>;
        assigned-clock-rates = <$pl_hz>;
        assigned-clocks = <&zynqmp_clk 71>;
        clock-output-names = \"fabric_clk\";
        clocks = <&zynqmp_clk 71>;
        compatible = \"xlnx,fclk\";
    };
};"
    close $f
    puts "\[BD\] overlay: $fw/pl.dtsi (assigned-clock-rates $pl_hz Hz) — install_firmware.sh recompiles moe.dtbo on the board"
}
puts "\n\[BD\] ================================================================"
puts "\[BD\] PL clock $pl_act MHz   WNS $wns ns   TNS $tns ns   WHS $whs ns   ->  TIMING $verdict"
if {!$met} { puts "\[BD\] !! timing NOT met: do not measure with this bitstream — lower MOE_PL_MHZ (e.g. 80) and rerun" }
puts "\[BD\] bitstream: $bit"
puts "\[BD\] next: board\\firmware\\make_firmware.bat  (bootgen -image moe.bif -arch zynqmp -o moe.bit.bin -w)"
puts "\[BD\] ================================================================"
exit
