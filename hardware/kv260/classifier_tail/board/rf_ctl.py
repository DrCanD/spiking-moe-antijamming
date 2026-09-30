#!/usr/bin/env python3
"""PS-side driver for the rf_top (D6) HLS IP — router tail: feature finalisation + Random Forest — at 0xA000_0000 via
/dev/mem (sudo). Register offsets from regmap_rf.json (parse_regmap.py on xrf_top_hw.h)."""
import json, mmap, os, struct, time
import numpy as np

BASE = 0xA0000000; SPAN = 0x10000
WIN_WORDS = 1024; RES_WORDS = 256
RC_NOP, RC_LOAD, RC_RUN, RC_VERIFY = 0, 1, 2, 3
M_OVERHEAD, M_FEATURES, M_FOREST_FIXED, M_FOREST_RANDOM = 0, 1, 2, 3
MODES = {'R0_overhead': M_OVERHEAD, 'R1_features': M_FEATURES, 'R2_features+forest_fixed': M_FOREST_FIXED, 'R3_features+forest_random': M_FOREST_RANDOM}
Q = dict(MAGIC=0, N_REPL=1, MODE=2, INFER_DONE=3, PACE_ACC=4, N=5, CHK=8, VERDICT=16, ROT=17, VOTES=20, FEATS=32, REPL_LOADED=52)
RES_MAGIC = 0x52464F31
CLASSES = ['broadband', 'narrowband', 'none', 'pulse', 'sweep']
N_SUB, K = 20, 16
# moe_top result-word offsets used to build the frame block from *_expected.json
R = dict(COUNT=0, ISI_N=1, ISI_SUM=2, ISI_SQ=3, SF_CNT=5, SF_ISI_SUM=25, SF_ISI_CNT=45, RF_CNT=65)


def words_from_expected(e):
    st = e['spike_stats']; w = [0] * WIN_WORDS
    w[R['COUNT']] = st['count']; w[R['ISI_N']] = st['isi_n']; w[R['ISI_SUM']] = st['isi_sum']
    w[R['ISI_SQ']] = st['isi_sq_sum'] & 0xFFFFFFFF; w[R['ISI_SQ'] + 1] = st['isi_sq_sum'] >> 32
    w[R['SF_CNT']:R['SF_CNT'] + N_SUB] = st['sf_cnt']; w[R['SF_ISI_SUM']:R['SF_ISI_SUM'] + N_SUB] = st['sf_isi_sum']; w[R['SF_ISI_CNT']:R['SF_ISI_CNT'] + N_SUB] = st['sf_isi_cnt']
    rc = np.array(e['rf_counts'], dtype=np.int64).reshape(-1); w[R['RF_CNT']:R['RF_CNT'] + N_SUB * K] = [int(v) for v in rc]
    return w, int(st['n'])


def rotate_words(w, rot):
    w2 = list(w)
    for base in (R['SF_CNT'], R['SF_ISI_SUM'], R['SF_ISI_CNT']):
        for k in range(N_SUB): w2[base + k] = w[base + (k + rot) % N_SUB]
    for k in range(N_SUB):
        for b in range(K): w2[R['RF_CNT'] + k * K + b] = w[R['RF_CNT'] + ((k + rot) % N_SUB) * K + b]
    return w2


class RF:
    def __init__(self, base=BASE, regmap=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'regmap_rf.json')):
        rm = json.load(open(regmap)); self.o = {k: int(v, 0) if isinstance(v, str) else v for k, v in rm.items()}
        for k in ('AP_CTRL', 'CMD_DATA', 'ARG0_DATA', 'ARG1_DATA', 'MODE_DATA', 'PACE_DATA', 'N_SAMPLES_DATA', 'WINDOW_BASE', 'RES_BASE'):
            assert k in self.o, f'regmap_rf.json lacks {k}'
        self.fd = os.open('/dev/mem', os.O_RDWR | os.O_SYNC)
        self.mm = mmap.mmap(self.fd, SPAN, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=base)
        self.mv = memoryview(self.mm).cast('I')
        self.n = 100000

    def rd(self, off): return self.mv[off // 4]
    def wr(self, off, v): self.mv[off // 4] = v & 0xFFFFFFFF
    def rd_block(self, off, n): i = off // 4; return [self.mv[i + k] for k in range(n)]
    def wr_block(self, off, vals):
        i = off // 4
        for k, v in enumerate(vals): self.mv[i + k] = v & 0xFFFFFFFF
    def idle(self): return bool(self.rd(self.o['AP_CTRL']) & 0x4)
    def start(self, cmd, arg0=0, arg1=0, mode=0, pace=0, n=None):
        assert self.idle(), 'IP busy'
        self.wr(self.o['CMD_DATA'], cmd); self.wr(self.o['ARG0_DATA'], arg0); self.wr(self.o['ARG1_DATA'], arg1)
        self.wr(self.o['MODE_DATA'], mode); self.wr(self.o['PACE_DATA'], pace); self.wr(self.o['N_SAMPLES_DATA'], self.n if n is None else n)
        self.wr(self.o['AP_CTRL'], 0x1); self.t_start = time.time()
    def wait(self, timeout=600.0):
        t0 = time.time()
        while not self.idle():
            if time.time() - t0 > timeout: raise TimeoutError('IP did not finish')
            time.sleep(0.001 if time.time() - t0 < 1 else 0.05)
        return time.time() - self.t_start
    def call(self, *a, **k): self.start(*a, **k); return self.wait()

    def load_words(self, w, n, replica=-1):
        """frame block (1024 words) -> replica (or all replicas with -1)"""
        self.n = n; self.wr_block(self.o['WINDOW_BASE'], [int(v) & 0xFFFFFFFF for v in w])
        self.call(RC_LOAD, arg0=replica & 0xFFFFFFFF, n=n)
    def verify(self, rot, mode):
        self.call(RC_VERIFY, arg0=rot, mode=mode); r = self.results()
        feats = [self.d(r, Q['FEATS'] + 2 * i) for i in range(9)]; votes = [self.d(r, Q['VOTES'] + 2 * i) for i in range(5)]
        return r[Q['VERDICT']], feats, votes, r[Q['CHK']]
    def results(self):
        r = self.rd_block(self.o['RES_BASE'], RES_WORDS); assert r[Q['MAGIC']] == RES_MAGIC, 'bad result magic (IP not programmed?)'; return r
    @staticmethod
    def d(r, at): return struct.unpack('<d', struct.pack('<II', r[at], r[at + 1]))[0]


if __name__ == '__main__':
    m = RF()
    print('AP_CTRL = 0x%08x  idle=%s' % (m.rd(m.o['AP_CTRL']), m.idle()))
    m.call(RC_NOP); r = m.results()
    print('magic ok, N_REPL=%d, infer_done=%d, mode(last)=%d, replicas loaded mask=0x%x' % (r[Q['N_REPL']], r[Q['INFER_DONE']], r[Q['MODE']], r[Q['REPL_LOADED']]))
