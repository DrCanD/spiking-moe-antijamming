#!/usr/bin/env python3
"""Fallback if bootgen is not available: strip the Xilinx .bit header -> raw configuration data (.bit.bin) for the
ZynqMP FPGA manager (equivalent to bootgen -arch zynqmp with [destination_device = pl]; no byte swapping on ZynqMP).
   python3 bit2bin.py design_1_wrapper.bit moe.bit.bin"""
import struct, sys
src, dst = sys.argv[1], sys.argv[2]
b = open(src, 'rb').read(); pos = 0
n = struct.unpack('>H', b[pos:pos + 2])[0]; pos += 2 + n            # initial 9-byte magic (length-prefixed)
pos += 2                                                            # 0x0001
while True:
    key = b[pos:pos + 1]; pos += 1
    if key == b'e':
        length = struct.unpack('>I', b[pos:pos + 4])[0]; pos += 4; data = b[pos:pos + length]; break
    n = struct.unpack('>H', b[pos:pos + 2])[0]; val = b[pos + 2:pos + 2 + n]; pos += 2 + n
    txt = val.rstrip(b'\0').decode(errors='replace'); print(key.decode() + ': ' + txt)
open(dst, 'wb').write(data)
print(f'{dst}: {len(data)} bytes (raw bitstream, {len(b)} in .bit)')
