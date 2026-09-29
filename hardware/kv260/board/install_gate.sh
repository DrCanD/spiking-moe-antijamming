#!/usr/bin/env bash
set -euo pipefail
gate_board_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$gate_board_dir"
if [ "$(id -u)" -ne 0 ]; then echo 'Run with sudo.'; exit 1; fi
command -v dtc >/dev/null || { echo 'device-tree-compiler (dtc) is required.'; exit 1; }
command -v xmutil >/dev/null || { echo 'This loader targets the existing Kria Ubuntu/xmutil setup.'; exit 1; }
python3 - <<'PY'
import json,hashlib
from pathlib import Path
r=json.loads(Path('build_receipt.json').read_text())
assert r['completed'] and 'TIMING MET' in r['timing_status']
for p,k in [('firmware/moe_gate.bit.bin','bitstream_sha256'),('regmap.json','regmap_sha256'),('firmware/pl.dtsi','overlay_source_sha256')]:
 assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==r[k],p+' hash mismatch'
PY
dtc -@ -O dtb -o firmware/moe_gate.dtbo firmware/pl.dtsi
mkdir -p /lib/firmware/xilinx/moe_gate
cp firmware/moe_gate.bit.bin firmware/moe_gate.dtbo firmware/shell.json /lib/firmware/xilinx/moe_gate/
if ! xmutil unloadapp; then
    echo "unloadapp did not complete (possibly no active app); loadapp must succeed before continuing."
fi
xmutil loadapp moe_gate
mountpoint -q /sys/kernel/debug || mount -t debugfs none /sys/kernel/debug
python3 - <<'PY'
import json
from pathlib import Path
r=json.loads(Path('build_receipt.json').read_text())
actual=int(Path('/sys/kernel/debug/clk/pl0_ref/clk_rate').read_text())
limit=r['clock_request_hz']
assert actual<=limit*1.001+2000,(actual,limit)
assert Path('/sys/class/fpga_manager/fpga0/state').read_text().strip()=='operating'
Path('firmware/pl_clk_actual_hz.txt').write_text(str(actual)+'\n')
print('Loaded moe_gate, PL clock:',actual,'Hz')
PY
