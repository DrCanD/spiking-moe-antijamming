#!/bin/bash
# On the KV260 (Ubuntu 22.04 for Kria). Run from this directory:  sudo ./install_firmware.sh
# Compiles the overlay, installs the firmware as app "moe", loads it with xmutil (fallback: fpga-manager + configfs),
# then checks that the PL clock the kernel set does not exceed the clock the bitstream was constrained at (pl_clk_hz.txt).
set -e
APP=moe
# pl.dtsi is (re)written by vivado/build_bd.tcl with the constrained PL clock; recompile whenever dtc is available,
# otherwise fall back to the shipped moe.dtbo (compiled from the default 83.332 MHz pl.dtsi).
if command -v dtc >/dev/null; then
    dtc -@ -O dtb -o $APP.dtbo pl.dtsi
    echo "moe.dtbo compiled from pl.dtsi (assigned-clock-rates $(grep -o 'assigned-clock-rates = <[0-9]*>' pl.dtsi | grep -o '[0-9]*') Hz)"
else
    echo "dtc missing -> using the shipped moe.dtbo (check that its clock matches pl_clk_hz.txt)"
fi
mkdir -p /lib/firmware/xilinx/$APP
cp $APP.bit.bin $APP.dtbo shell.json /lib/firmware/xilinx/$APP/
if command -v xmutil >/dev/null; then
    xmutil unloadapp || true
    xmutil listapps
    xmutil loadapp $APP
else
    echo "xmutil not found -> fpga-manager + configfs overlay"
    cp $APP.bit.bin /lib/firmware/
    mkdir -p /sys/kernel/config/device-tree/overlays/$APP
    cat $APP.dtbo > /sys/kernel/config/device-tree/overlays/$APP/dtbo
fi
sleep 1
echo "fpga state: $(cat /sys/class/fpga_manager/fpga0/state)"
mountpoint -q /sys/kernel/debug || mount -t debugfs none /sys/kernel/debug 2>/dev/null || true
ACT=$(cat /sys/kernel/debug/clk/pl0_ref/clk_rate 2>/dev/null || echo "")
if [ -n "$ACT" ]; then
    echo "PL clock  : $ACT Hz (pl0_ref)"
    if [ -f pl_clk_hz.txt ]; then
        LIM=$(tr -d "\r\n[:space:]" < pl_clk_hz.txt)      # file may carry CRLF (written by Vivado on Windows)
        # allow +0.1 % for divider rounding (99999001-style requests); anything above means the overlay and the bitstream disagree
        if [ "$ACT" -le $((LIM + LIM / 1000 + 2000)) ]; then
            echo "PL clock check: OK ($ACT Hz <= constraint $LIM Hz + rounding)"
        else
            echo "!! PL clock check FAILED: kernel set $ACT Hz but the bitstream was constrained at $LIM Hz — fix pl.dtsi, rerun"
            exit 2
        fi
    fi
    echo "$ACT" > pl_clk_actual_hz.txt
else
    echo "PL clock  : n/a (debugfs unavailable) — cannot verify against pl_clk_hz.txt"
fi
