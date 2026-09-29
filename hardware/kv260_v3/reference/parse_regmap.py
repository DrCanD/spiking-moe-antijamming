#!/usr/bin/env python3
"""HLS driver header (moe_hls/sol1/impl/ip/drivers/moe_top_v1_0/src/xmoe_top_hw.h) -> regmap.json for moe_ctl.py
   python3 parse_regmap.py <path/to/xmoe_top_hw.h> [regmap.json]"""
import re, sys, json
src = open(sys.argv[1]).read(); out = sys.argv[2] if len(sys.argv) > 2 else 'regmap.json'
m = dict(re.findall(r'#define\s+X[A-Z0-9_]+?_CTRL_ADDR_(\w+)\s+(0x[0-9A-Fa-f]+)', src))     # XMOE_TOP_ / XRF_TOP_ ...
need = ['AP_CTRL', 'CMD_DATA', 'ARG0_DATA', 'ARG1_DATA', 'PACE_DATA', 'N_SAMPLES_DATA', 'WINDOW_BASE', 'WINDOW_HIGH', 'RES_BASE', 'RES_HIGH']
need += ['BLK_EN_DATA'] if 'BLK_EN_DATA' in m else ['MODE_DATA']     # moe_top has blk_en, rf_top has mode
missing = [k for k in need if k not in m]
if missing: sys.exit(f'missing in header: {missing}; found {sorted(m)}')
json.dump({k: m[k] for k in need}, open(out, 'w'), indent=1)
print({k: m[k] for k in need}); print('->', out)
