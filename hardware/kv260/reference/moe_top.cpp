// Top level: one HLS IP holding N_REPL replicas of the final-MoE datapath (D1 front-end B, D2 conventional FFT
// front-end, D3 ALE-128 x2 step sizes, D4 blanker) fed by an N_REPL-stage sample delay chain from an on-chip
// replay buffer (URAM). Per-block enables select the design under measurement (D0..D5) at run time, so ONE
// bitstream serves the whole power table. Replica 0 can capture its streams for bit-exact verification against
// the Exp-5 test vectors; every replica folds all of its accumulators into a checksum so nothing is optimised away.
#include "moe_top.h"
#include "moe_blocks.h"

struct RepOut {
    sint<2>  spike;  sint<32> e_nb, e_sw;
    bool     blank_valid, blank_mask;  uint_<40> blank_ss;  uint_<12> blank_grp;
};

static inline void put64(unsigned int res[RES_WORDS], int at, uint_<64> v) {
#pragma HLS INLINE
    res[at] = (unsigned int)(v & 0xFFFFFFFFu); res[at + 1] = (unsigned int)((v >> 32) & 0xFFFFFFFFu);
}

// ─────────────────────────────────────────── one replica (template = distinct hardware + distinct statics) ───
template <int R>
void rep_step(bool first, bool last, idx_t i, x_t x, uint_<8> en, uint_<20> sf_len,
              unsigned int res[RES_WORDS], uint_<32> &chk, RepOut &o) {
#pragma HLS INLINE off
    // D1 state
    static FeState fe;
    static uint_<16> sf_cnt[N_SUBFRAMES];  static uint_<32> sf_isi_sum[N_SUBFRAMES];  static uint_<16> sf_isi_cnt[N_SUBFRAMES];
    static sint<28> zr[RF_BANDS], zi[RF_BANDS];  static bool prev[RF_BANDS];
    static uint_<16> rf_cnt[N_SUBFRAMES][RF_BANDS];
#pragma HLS ARRAY_PARTITION variable=zr complete
#pragma HLS ARRAY_PARTITION variable=zi complete
#pragma HLS ARRAY_PARTITION variable=prev complete
#pragma HLS ARRAY_PARTITION variable=rf_cnt complete dim=2
#pragma HLS BIND_STORAGE variable=rf_cnt type=ram_s2p impl=lutram
    // D3 state (history shared by both step sizes)
    static x_t xd[ALE_HIST];
#pragma HLS ARRAY_PARTITION variable=xd complete
    static AleShared ash;  static AleState anb, asw;
    static sint<32> w_nb[ALE_TAPS], w_sw[ALE_TAPS];
#pragma HLS ARRAY_PARTITION variable=w_nb cyclic factor=ALE_LANES
#pragma HLS ARRAY_PARTITION variable=w_sw cyclic factor=ALE_LANES
#pragma HLS BIND_STORAGE variable=w_nb type=ram_s2p impl=lutram
#pragma HLS BIND_STORAGE variable=w_sw type=ram_s2p impl=lutram
    static uint_<64> sum_x2;
    // D4 state
    static BlankState bl;
    // D2 state
    static ConvState cv;
    static x_t inbuf[FFT_N];  static sint<27> fre[FFT_N], fim[FFT_N];  static uint_<32> P[FFT_BINS];  static uint_<48> se[N_SUBFRAMES];
#pragma HLS BIND_STORAGE variable=inbuf type=ram_s2p impl=bram
#pragma HLS BIND_STORAGE variable=fre type=ram_t2p impl=bram
#pragma HLS BIND_STORAGE variable=fim type=ram_t2p impl=bram
#pragma HLS BIND_STORAGE variable=P type=ram_s2p impl=bram

    o.spike = 0; o.e_nb = 0; o.e_sw = 0; o.blank_valid = false; o.blank_mask = false; o.blank_ss = 0; o.blank_grp = 0;
    if (first) sum_x2 = 0;

    if (en & EN_FE)   o.spike = fe_step(fe, sf_cnt, sf_isi_sum, sf_isi_cnt, zr, zi, prev, rf_cnt, first, i, x, sf_len);
    if (en & (EN_ALE_NB | EN_ALE_SW)) {
        ale_shared_step(ash, xd, i);
        sum_x2 = sum_x2 + (uint_<64>)((sint<32>)x * (sint<32>)x);
    }
    if (en & EN_ALE_NB) o.e_nb = ale_step(anb, w_nb, ash, xd, first, x, (uint_<13>)ALE_MU_NB_Q16);
    if (en & EN_ALE_SW) o.e_sw = ale_step(asw, w_sw, ash, xd, first, x, (uint_<13>)ALE_MU_SW_Q16);
    if (en & (EN_ALE_NB | EN_ALE_SW)) ale_hist_push(xd, x);
    if (en & EN_BLANK) { blank_step(bl, first, x); o.blank_valid = bl.valid; o.blank_mask = bl.mask_out; o.blank_ss = bl.ss_out; o.blank_grp = bl.grp - 1; }
    if (en & EN_CONV)  conv_step(cv, inbuf, fre, fim, P, se, first, i, x, sf_len);

    if (last) {
        // ── checksum over EVERY accumulator (keeps all replicas observable for synthesis) ──
        uint_<32> c = (uint_<32>)fe.count ^ ((uint_<32>)fe.isi_n << 8) ^ (uint_<32>)fe.isi_sum ^ (uint_<32>)(fe.isi_sq_sum & 0xFFFFFFFFu) ^ (uint_<32>)(fe.isi_sq_sum >> 32);
    CHK_SF: for (int k = 0; k < N_SUBFRAMES; k++) {
#pragma HLS PIPELINE II=1
            c = (c * 31u) ^ (uint_<32>)sf_cnt[k] ^ ((uint_<32>)sf_isi_cnt[k] << 16) ^ (uint_<32>)sf_isi_sum[k] ^ (uint_<32>)(se[k] & 0xFFFFFFFFu) ^ (uint_<32>)(se[k] >> 32);
        CHK_RF: for (int b = 0; b < RF_BANDS; b++) {
#pragma HLS UNROLL
                c ^= (uint_<32>)rf_cnt[k][b] << (b & 15);
            }
        }
    CHK_P: for (int k = 0; k < FFT_BINS; k++) {
#pragma HLS PIPELINE II=1
            c = (c * 31u) ^ (uint_<32>)P[k];
        }
        c ^= (uint_<32>)(anb.sum_e2 & 0xFFFFFFFFu) ^ (uint_<32>)(anb.sum_e2 >> 32) ^ (uint_<32>)(asw.sum_e2 & 0xFFFFFFFFu) ^ (uint_<32>)(asw.sum_e2 >> 32);
        c ^= (uint_<32>)anb.e_last ^ ((uint_<32>)asw.e_last << 3) ^ (uint_<32>)bl.flagged ^ (uint_<32>)(sum_x2 & 0xFFFFFFFFu) ^ (uint_<32>)(sum_x2 >> 32);
        c ^= (uint_<32>)(cv.sx & 0xFFFFFFFFu) ^ (uint_<32>)(cv.sx2 & 0xFFFFFFFFu) ^ (uint_<32>)(cv.sx2 >> 32) ^ (uint_<32>)(cv.sx3 & 0xFFFFFFFFu) ^ (uint_<32>)(cv.sx3 >> 32)
           ^ (uint_<32>)(cv.sx4 & 0xFFFFFFFFu) ^ (uint_<32>)((cv.sx4 >> 32) & 0xFFFFFFFFu) ^ (uint_<32>)(cv.sx4 >> 64) ^ (uint_<32>)cv.zc ^ ((uint_<32>)cv.n_seg << 20);
        chk = c;
        // ── replica 0 dumps its accumulators for the PS (feature finalisation + router) and for verification ──
        if (R == 0) {
            res[R_COUNT] = (unsigned int)fe.count; res[R_ISI_N] = (unsigned int)fe.isi_n; res[R_ISI_SUM] = (unsigned int)fe.isi_sum;
            put64(res, R_ISI_SQ_LO, (uint_<64>)fe.isi_sq_sum);
        DUMP_SF: for (int k = 0; k < N_SUBFRAMES; k++) {
#pragma HLS PIPELINE II=1
                res[R_SF_CNT + k] = (unsigned int)sf_cnt[k]; res[R_SF_ISI_SUM + k] = (unsigned int)sf_isi_sum[k]; res[R_SF_ISI_CNT + k] = (unsigned int)sf_isi_cnt[k];
                put64(res, R_CONV_SE + 2 * k, (uint_<64>)se[k]);
            }
        DUMP_RF: for (int k = 0; k < N_SUBFRAMES; k++) {
            DUMP_RF_B: for (int b = 0; b < RF_BANDS; b++) {
#pragma HLS PIPELINE II=1
                    res[R_RF_CNT + k * RF_BANDS + b] = (unsigned int)rf_cnt[k][b];
                }
            }
            put64(res, R_SUM_X2, sum_x2); put64(res, R_SUM_E2_NB, anb.sum_e2); put64(res, R_SUM_E2_SW, asw.sum_e2);
            res[R_BLANK_FLAGGED] = (unsigned int)bl.flagged; res[R_E_NB_LAST] = (unsigned int)anb.e_last; res[R_E_SW_LAST] = (unsigned int)asw.e_last;
            put64(res, R_CONV_SX, (uint_<64>)(sint<64>)cv.sx); put64(res, R_CONV_SX2, (uint_<64>)cv.sx2); put64(res, R_CONV_SX3, (uint_<64>)cv.sx3);
            res[R_CONV_SX4] = (unsigned int)(cv.sx4 & 0xFFFFFFFFu); res[R_CONV_SX4 + 1] = (unsigned int)((cv.sx4 >> 32) & 0xFFFFFFFFu); res[R_CONV_SX4 + 2] = (unsigned int)(cv.sx4 >> 64);
            res[R_CONV_ZC] = (unsigned int)cv.zc; res[R_CONV_NSEG] = (unsigned int)cv.n_seg;
        DUMP_P: for (int k = 0; k < FFT_BINS; k++) {
#pragma HLS PIPELINE II=1
                res[R_CONV_P + k] = (unsigned int)P[k];
            }
        }
    }
}

// compile-time recursion over the replicas (each R is a distinct function => distinct hardware instance)
template <int R>
inline void rep_all(bool first, bool last, idx_t i, const x_t chain[N_REPL], uint_<8> en, uint_<20> sf_len,
                    unsigned int res[RES_WORDS], uint_<32> chk[N_REPL], RepOut &o0) {
#pragma HLS INLINE
    RepOut o;
    rep_step<R>(first, last, i, chain[R], en, sf_len, res, chk[R], o);
    if (R == 0) o0 = o;
    rep_all<R + 1>(first, last, i, chain, en, sf_len, res, chk, o0);
}
template <>
inline void rep_all<N_REPL>(bool, bool, idx_t, const x_t[N_REPL], uint_<8>, uint_<20>, unsigned int[RES_WORDS], uint_<32>[N_REPL], RepOut &) {}

// ─────────────────────────────────────────── top ─────────────────────────────────────────────────────────────
void moe_top(int cmd, int arg0, int arg1, int blk_en, int pace, int n_samples, int window[WIN_WORDS], unsigned int res[RES_WORDS]) {
#pragma HLS INTERFACE s_axilite port=cmd       bundle=ctrl
#pragma HLS INTERFACE s_axilite port=arg0      bundle=ctrl
#pragma HLS INTERFACE s_axilite port=arg1      bundle=ctrl
#pragma HLS INTERFACE s_axilite port=blk_en    bundle=ctrl
#pragma HLS INTERFACE s_axilite port=pace      bundle=ctrl
#pragma HLS INTERFACE s_axilite port=n_samples bundle=ctrl
#pragma HLS INTERFACE s_axilite port=window    bundle=ctrl
#pragma HLS INTERFACE s_axilite port=res       bundle=ctrl
#pragma HLS INTERFACE s_axilite port=return    bundle=ctrl

    static x_t replay[N_MAX];
    static x_t capture[N_MAX];
#pragma HLS BIND_STORAGE variable=replay  type=ram_1p impl=uram
#pragma HLS BIND_STORAGE variable=capture type=ram_1p impl=uram
    static x_t chain[N_REPL];
#pragma HLS ARRAY_PARTITION variable=chain complete
    static uint_<32> pace_acc = 0;
    static uint_<32> chk[N_REPL];
#pragma HLS ARRAY_PARTITION variable=chk complete

    int n = n_samples;
    if (n > N_MAX) n = N_MAX;
    if (n < 2 * FFT_N) n = 2 * FFT_N;
    const uint_<20> sf_len = (uint_<20>)(n / N_SUBFRAMES);
    res[R_MAGIC] = RES_MAGIC; res[R_N_REPL] = N_REPL; res[R_BLK_EN] = (unsigned int)blk_en;

    if (cmd == CMD_LOAD) {
        const int base = arg0 * WIN_WORDS;
    LOAD: for (int j = 0; j < WIN_WORDS; j++) {
#pragma HLS PIPELINE II=1
            const int a = base + j;
            if (a < N_MAX) replay[a] = (x_t)(int16_t)window[j];
        }
    } else if (cmd == CMD_READ) {
        const int base = arg0 * WIN_WORDS;
    READ: for (int j = 0; j < WIN_WORDS; j++) {
#pragma HLS PIPELINE II=1
            const int a = base + j;
            window[j] = (a < N_MAX) ? (int)capture[a] : 0;
        }
    } else if (cmd == CMD_RUN || cmd == CMD_VERIFY) {
        const int n_frames = (cmd == CMD_VERIFY) ? 1 : arg0;
        const int cap = (cmd == CMD_VERIFY) ? arg1 : CAP_NONE;
        const uint_<8> en = (uint_<8>)blk_en;
    FRAMES: for (int f = 0; f < n_frames; f++) {
        SAMPLES: for (int i = 0; i < n; i++) {
                const x_t x = replay[i];
            CHAIN: for (int r = N_REPL - 1; r > 0; r--) {
#pragma HLS UNROLL
                    chain[r] = chain[r - 1];
                }
                chain[0] = x;
                const bool first = (i == 0), last = (i == n - 1);
                RepOut o0;
                rep_all<0>(first, last, (idx_t)i, chain, en, sf_len, res, chk, o0);
                if (cap == CAP_SPIKES)      capture[i] = (x_t)o0.spike;
                else if (cap == CAP_E_NB)   capture[i] = sat16(o0.e_nb);
                else if (cap == CAP_E_SW)   capture[i] = sat16(o0.e_sw);
                else if (cap != CAP_NONE && o0.blank_valid) {
                    x_t v = 0;
                    if (cap == CAP_MASK)        v = o0.blank_mask ? (x_t)1 : (x_t)0;
                    else if (cap == CAP_SS_LO)  v = (x_t)(int16_t)(uint16_t)(o0.blank_ss & 0xFFFF);
                    else if (cap == CAP_SS_MID) v = (x_t)(int16_t)(uint16_t)((o0.blank_ss >> 16) & 0xFFFF);
                    else if (cap == CAP_SS_HI)  v = (x_t)(int16_t)(uint16_t)((o0.blank_ss >> 32) & 0xFF);
                    capture[o0.blank_grp] = v;
                }
                if (last) {
                CHK_OUT: for (int r = 0; r < N_REPL; r++) {
#pragma HLS PIPELINE II=1
                        res[R_CHK + r] = (unsigned int)chk[r];
                    }
                }
            PACE: for (int p = 0; p < pace; p++) {
#pragma HLS PIPELINE II=1
                    pace_acc = pace_acc + ((pace_acc >> 1) ^ (uint_<32>)p);
                }
            }
            res[R_FRAMES_DONE] = (unsigned int)(f + 1);
        }
    }
    res[R_PACE_ACC] = (unsigned int)pace_acc;
}
