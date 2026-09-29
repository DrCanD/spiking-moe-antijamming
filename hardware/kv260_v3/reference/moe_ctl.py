#!/usr/bin/env python3
"""PS-side driver for the moe_top HLS IP (AXI-Lite at 0xA000_0000) via /dev/mem — run as root (sudo).
Register offsets come from regmap.json, generated from the HLS driver header by parse_regmap.py, so nothing is
hard-coded except the base address chosen in vivado/build_bd.tcl."""
import json, mmap, os, struct, time
import numpy as np

BASE = 0xA0000000; SPAN = 0x10000
WIN_WORDS = 1024; RES_WORDS = 1024; N_MAX = 100000
CMD_NOP, CMD_LOAD, CMD_RUN, CMD_VERIFY, CMD_READ = 0, 1, 2, 3, 4
EN_FE, EN_CONV, EN_ALE_NB, EN_ALE_SW, EN_BLANK = 1, 2, 4, 8, 16
CAP_NONE, CAP_SPIKES, CAP_E_NB, CAP_E_SW, CAP_MASK, CAP_SS_LO, CAP_SS_MID, CAP_SS_HI = range(8)
DESIGNS = {  # the paper's power table: name -> blk_en
    'D0_empty': 0, 'D1_frontend_B': EN_FE, 'D2_conv_fft1024': EN_CONV, 'D3_ale128_nb': EN_ALE_NB,
    'D3b_ale128_nb+sw': EN_ALE_NB | EN_ALE_SW, 'D4_blanker': EN_BLANK, 'D5_full_moe': EN_FE | EN_ALE_NB | EN_ALE_SW | EN_BLANK}
# result-word layout (must match hls/src/moe_types.h)
R = dict(COUNT=0, ISI_N=1, ISI_SUM=2, ISI_SQ=3, SF_CNT=5, SF_ISI_SUM=25, SF_ISI_CNT=45, RF_CNT=65, SUM_X2=385, SUM_E2_NB=387, SUM_E2_SW=389,
         BLANK_FLAGGED=391, E_NB_LAST=392, E_SW_LAST=393, CONV_SX=394, CONV_SX2=396, CONV_SX3=398, CONV_SX4=400, CONV_SE=403, CONV_ZC=443,
         CONV_NSEG=444, CONV_P=445, CHK=958, PACE_ACC=974, FRAMES_DONE=975, MAGIC=976, N_REPL=977, BLK_EN=978)
RES_MAGIC = 0x4D4F4531


class MoE:
    def __init__(self, base=BASE, regmap=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'regmap.json')):
        rm = json.load(open(regmap)); self.o = {k: int(v, 0) if isinstance(v, str) else v for k, v in rm.items()}
        for k in ('AP_CTRL', 'CMD_DATA', 'ARG0_DATA', 'ARG1_DATA', 'BLK_EN_DATA', 'PACE_DATA', 'N_SAMPLES_DATA', 'WINDOW_BASE', 'RES_BASE'):
            assert k in self.o, f'regmap.json lacks {k}'
        self.fd = os.open('/dev/mem', os.O_RDWR | os.O_SYNC)
        self.mm = mmap.mmap(self.fd, SPAN, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=base)
        self.mv = memoryview(self.mm).cast('I')            # 32-bit accesses only (AXI-Lite)

    # ── raw ──
    def rd(self, off): return self.mv[off // 4]
    def wr(self, off, v): self.mv[off // 4] = v & 0xFFFFFFFF
    def rd_block(self, off, n): i = off // 4; return [self.mv[i + k] for k in range(n)]
    def wr_block(self, off, vals):
        i = off // 4
        for k, v in enumerate(vals): self.mv[i + k] = v & 0xFFFFFFFF

    # ── control ──
    def idle(self): return bool(self.rd(self.o['AP_CTRL']) & 0x4)
    def start(self, cmd, arg0=0, arg1=0, blk_en=0, pace=0, n=N_MAX):
        assert self.idle(), 'IP busy'
        self.wr(self.o['CMD_DATA'], cmd); self.wr(self.o['ARG0_DATA'], arg0); self.wr(self.o['ARG1_DATA'], arg1)
        self.wr(self.o['BLK_EN_DATA'], blk_en); self.wr(self.o['PACE_DATA'], pace); self.wr(self.o['N_SAMPLES_DATA'], n)
        self.wr(self.o['AP_CTRL'], 0x1)                     # ap_start
        self.t_start = time.time()
    def wait(self, timeout=600.0):
        t0 = time.time()
        while not self.idle():
            if time.time() - t0 > timeout: raise TimeoutError('IP did not finish')
            time.sleep(0.001 if time.time() - t0 < 1 else 0.05)
        return time.time() - self.t_start
    def call(self, *a, **k): self.start(*a, **k); return self.wait()

    # ── data ──
    def load(self, x):
        x = np.asarray(x, dtype=np.int16); n = len(x); assert n <= N_MAX
        for c in range((n + WIN_WORDS - 1) // WIN_WORDS):
            chunk = np.zeros(WIN_WORDS, dtype=np.int64); seg = x[c * WIN_WORDS:(c + 1) * WIN_WORDS]; chunk[:len(seg)] = seg
            self.wr_block(self.o['WINDOW_BASE'], [int(v) & 0xFFFFFFFF for v in chunk])
            self.call(CMD_LOAD, arg0=c, n=n)
        self.n = n
    def read_capture(self, n):
        out = np.zeros(n, dtype=np.int16)
        for c in range((n + WIN_WORDS - 1) // WIN_WORDS):
            self.call(CMD_READ, arg0=c, n=self.n)
            w = np.array(self.rd_block(self.o['WINDOW_BASE'], WIN_WORDS), dtype=np.uint32).astype(np.int32).astype(np.int16)
            seg = out[c * WIN_WORDS:(c + 1) * WIN_WORDS]; seg[:] = w[:len(seg)]
        return out
    def verify(self, cap, blk_en):
        dt = self.call(CMD_VERIFY, arg1=cap, blk_en=blk_en, n=self.n); return dt
    def results(self):
        r = self.rd_block(self.o['RES_BASE'], RES_WORDS); assert r[R['MAGIC']] == RES_MAGIC, 'bad result magic (IP not programmed?)'; return r
    def frames_done(self): return self.rd(self.o['RES_BASE'] + 4 * R['FRAMES_DONE'])

    @staticmethod
    def u64(r, at): return r[at] | (r[at + 1] << 32)
    @staticmethod
    def s64(r, at):
        v = r[at] | (r[at + 1] << 32); return v - (1 << 64) if v >= (1 << 63) else v


if __name__ == '__main__':
    m = MoE()
    print('AP_CTRL = 0x%08x  idle=%s' % (m.rd(m.o['AP_CTRL']), m.idle()))
    m.call(CMD_NOP)
    r = m.results()
    print('magic ok, N_REPL=%d, frames_done=%d, blk_en(last)=%d' % (r[R['N_REPL']], r[R['FRAMES_DONE']], r[R['BLK_EN']]))
