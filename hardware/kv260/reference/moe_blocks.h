// Streaming (one sample per call) fixed-point blocks of the final MoE datapath — a 1:1 transcription of the
// Exp-5 golden models (exp5_kv260_package.py: fx_encoder, fx_spike_stats, fx_rf_bank, fx_ale, fx_blanker_mask)
// and of golden/conv_golden.py (design D2, conventional Welch-1024 front-end).
//   D1  fe_step     : delta encoder (theta_q=2048, refractory 2) + integer spike statistics + 16 resonate-and-fire
//                     neurons (Q2.30 rotation, threshold-crossing counters per sub-frame)
//   D2  conv_step   : Hann-1024 / unscaled radix-2 FFT-1024 (Q15 twiddles) / |X| Welch accumulation + moments
//   D3  ale_step    : NLMS adaptive line enhancer, 128 taps, delay 20, Q6.26 weights, LUT reciprocal
//   D4  blank_step  : energy blanker, 50-sample group sum of squares vs the calibrated threshold
// All shifts on signed values are arithmetic (floor), exactly as numpy int64 '>>' in the golden model.
// Large arrays are passed as parameters (declared static, with their pragmas, in replica_step / moe_top.cpp).
#ifndef MOE_BLOCKS_H
#define MOE_BLOCKS_H
#include "moe_types.h"
#include "moe_params.h"
#ifndef __SYNTHESIS__
#include <cassert>
#define MOE_ASSERT(c) assert(c)
#else
#define MOE_ASSERT(c)
#endif

typedef sint<16>  x_t;          // input sample, Q6.10
typedef uint_<20> idx_t;        // sample index within the frame (< 2^20)

// ─────────────────────────────────────────── D1: encoder + spike statistics + RF bank ───────────────────────
struct FeState {
    sint<18>  ref;  uint_<2> refr;
    uint_<20> count, isi_n, last_idx;  uint_<32> isi_sum;  uint_<40> isi_sq_sum;
    uint_<5>  sf_idx;  uint_<20> sf_pos;
};

// returns the spike (+1/-1/0) of this sample
inline sint<2> fe_step(FeState &s, uint_<16> sf_cnt[N_SUBFRAMES], uint_<32> sf_isi_sum[N_SUBFRAMES], uint_<16> sf_isi_cnt[N_SUBFRAMES],
                       sint<28> zr[RF_BANDS], sint<28> zi[RF_BANDS], bool prev[RF_BANDS], uint_<16> rf_cnt[N_SUBFRAMES][RF_BANDS],
                       bool first, idx_t i, x_t x, uint_<20> sf_len) {
#pragma HLS INLINE off
    if (first) {
        s.ref = x; s.refr = 0; s.count = 0; s.isi_n = 0; s.last_idx = 0; s.isi_sum = 0; s.isi_sq_sum = 0; s.sf_idx = 0; s.sf_pos = 0;
    FE_RST_SF: for (int k = 0; k < N_SUBFRAMES; k++) {
#pragma HLS PIPELINE II=1
            sf_cnt[k] = 0; sf_isi_sum[k] = 0; sf_isi_cnt[k] = 0;
        FE_RST_RF: for (int b = 0; b < RF_BANDS; b++) {
#pragma HLS UNROLL
                rf_cnt[k][b] = 0;
            }
        }
    FE_RST_Z: for (int b = 0; b < RF_BANDS; b++) {
#pragma HLS UNROLL
            zr[b] = 0; zi[b] = 0; prev[b] = false;
        }
    } else {                                                  // sub-frame index = min(i / sf_len, N_SUBFRAMES-1)
        s.sf_pos = s.sf_pos + 1;
        if (s.sf_pos == sf_len) { s.sf_pos = 0; if (s.sf_idx < N_SUBFRAMES - 1) s.sf_idx = s.sf_idx + 1; }
    }
    const uint_<5> sf = s.sf_idx;
    // ── delta encoder (reference DeltaEncoder.encode control flow; refractory = skip the next 2 samples) ──
    sint<2> sp = 0;
    if (s.refr != 0) { s.refr = s.refr - 1; }
    else {
        sint<19> d = (sint<19>)x - (sint<19>)s.ref;
        if (d >= (sint<19>)THETA_Q)       { sp = 1;  s.ref = s.ref + THETA_Q; s.refr = ENC_REFRACTORY; }
        else if (d <= (sint<19>)-THETA_Q) { sp = -1; s.ref = s.ref - THETA_Q; s.refr = ENC_REFRACTORY; }
    }
    // ── integer sufficient statistics (fx_spike_stats) ──
    if (sp != 0) {
        if (s.count != 0) {
            uint_<20> isi = i - s.last_idx;
            s.isi_sum = s.isi_sum + isi;
            s.isi_sq_sum = s.isi_sq_sum + (uint_<40>)((uint_<40>)isi * (uint_<40>)isi);
            s.isi_n = s.isi_n + 1;
            sf_isi_sum[sf] = sf_isi_sum[sf] + isi;
            sf_isi_cnt[sf] = sf_isi_cnt[sf] + 1;
        }
        s.count = s.count + 1; s.last_idx = i; sf_cnt[sf] = sf_cnt[sf] + 1;
    }
    // ── resonate-and-fire bank (fx_rf_bank): z <- z*rot + s/tau; count rising crossings of Im(z) >= th ──
RF_BANK: for (int b = 0; b < RF_BANDS; b++) {
#pragma HLS PIPELINE II=1
        sint<32> cr = RF_ROT_RE_Q[b], ci = RF_ROT_IM_Q[b];
        sint<28> zr0 = zr[b], zi0 = zi[b];
        sint<61> pr = (sint<61>)((sint<61>)zr0 * (sint<61>)cr) - (sint<61>)((sint<61>)zi0 * (sint<61>)ci);
        sint<61> pi = (sint<61>)((sint<61>)zr0 * (sint<61>)ci) + (sint<61>)((sint<61>)zi0 * (sint<61>)cr);
        sint<31> nr = (sint<31>)(pr >> RF_COEF_FRAC);
        sint<31> ni = (sint<31>)(pi >> RF_COEF_FRAC);
        sint<31> zr_new = nr + (sint<31>)((sint<20>)sp * (sint<20>)RF_S_IN_Q);
        MOE_ASSERT(zr_new < (1 << 27) && zr_new >= -(1 << 27) && ni < (1 << 27) && ni >= -(1 << 27));
        zr[b] = (sint<28>)zr_new; zi[b] = (sint<28>)ni;
        bool a = (ni >= (sint<31>)RF_TH_Q);
        if (a && !prev[b]) rf_cnt[sf][b] = rf_cnt[sf][b] + 1;
        prev[b] = a;
    }
    return sp;
}

// ─────────────────────────────────────────── D3: ALE-128 (NLMS) ─────────────────────────────────────────────
#define ALE_HIST (ALE_DELAY + ALE_TAPS - 1)      // 147: xd[j] = x[i-1-j] -> window x[i-delay-k] = xd[delay-1+k]
#ifndef ALE_LANES
#define ALE_LANES 8                              // parallel MAC lanes per ALE (128/8 = 16 cycles per pass at II=1)
#endif
struct AleShared { uint_<40> xx; uint_<6> k; uint_<16> recip; bool active; };
struct AleState  { uint_<64> sum_e2; sint<32> e_last; };

// per-sample part common to both step sizes: window power, normalisation exponent, LUT reciprocal (fx_ale)
inline void ale_shared_step(AleShared &h, const x_t xd[ALE_HIST], idx_t i) {
#pragma HLS INLINE off
    h.active = (i >= (idx_t)(ALE_DELAY + ALE_TAPS));
    if (h.active) {
        uint_<40> pp[ALE_LANES];
#pragma HLS ARRAY_PARTITION variable=pp complete
    ALE_POWER_INIT: for (int l = 0; l < ALE_LANES; l++) {
#pragma HLS UNROLL
            pp[l] = 0;
        }
    ALE_POWER: for (int k = 0; k < ALE_TAPS; k += ALE_LANES) {
#pragma HLS PIPELINE II=1
        ALE_POWER_LANE: for (int l = 0; l < ALE_LANES; l++) {
#pragma HLS UNROLL
                sint<16> a = xd[ALE_DELAY - 1 + k + l];
                pp[l] += (uint_<40>)((sint<32>)a * (sint<32>)a);
            }
        }
        uint_<40> xx = 0;
    ALE_POWER_SUM: for (int l = 0; l < ALE_LANES; l++) {
#pragma HLS UNROLL
            xx += pp[l];
        }
        uint_<41> v = (uint_<41>)xx + RECIP_EPS_Q;
        uint_<6> kk = 0;                                        // kk = bit_length(v) - 1 (highest set bit)
    ALE_BITLEN: for (int b = 0; b <= 40; b++) {
#pragma HLS UNROLL
            if (((v >> b) & 1) != 0) kk = b;
        }
        uint_<46> rem = (uint_<46>)v - ((uint_<46>)1 << kk);    // v - 2^k  (< 2^k)
        uint_<5> m_idx;
        if (kk >= RECIP_B)      m_idx = (uint_<5>)((rem << RECIP_B) >> kk);
        else if (kk > 0)        m_idx = (uint_<5>)(rem << (RECIP_B - kk));
        else                    m_idx = 0;
        h.xx = xx; h.k = kk; h.recip = RECIP_TABLE[m_idx];
    }
}

// history push, called once per sample AFTER both ALE steps (so that xd[j] = x[i-1-j] during the step)
inline void ale_hist_push(x_t xd[ALE_HIST], x_t x) {
#pragma HLS INLINE
ALE_SHIFT: for (int j = ALE_HIST - 1; j > 0; j--) {
#pragma HLS UNROLL
        xd[j] = xd[j - 1];
    }
    xd[0] = x;
}

// one NLMS step; returns e[i] (unclipped)
inline sint<32> ale_step(AleState &s, sint<32> w[ALE_TAPS], const AleShared &h, const x_t xd[ALE_HIST], bool first, x_t x, uint_<13> mu_q) {
#pragma HLS INLINE off
    if (first) {
    ALE_RST: for (int k = 0; k < ALE_TAPS; k++) {
#pragma HLS PIPELINE II=1
            w[k] = 0;
        }
        s.sum_e2 = 0; s.e_last = 0;
    }
    sint<32> e = x;
    if (h.active) {
        sint<56> pa[ALE_LANES];
#pragma HLS ARRAY_PARTITION variable=pa complete
    ALE_FILTER_INIT: for (int l = 0; l < ALE_LANES; l++) {
#pragma HLS UNROLL
            pa[l] = 0;
        }
    ALE_FILTER: for (int k = 0; k < ALE_TAPS; k += ALE_LANES) {
#pragma HLS PIPELINE II=1
        ALE_FILTER_LANE: for (int l = 0; l < ALE_LANES; l++) {
#pragma HLS UNROLL
                pa[l] += (sint<56>)((sint<56>)w[k + l] * (sint<56>)xd[ALE_DELAY - 1 + k + l]);
            }
        }
        sint<56> acc = 0;
    ALE_FILTER_SUM: for (int l = 0; l < ALE_LANES; l++) {
#pragma HLS UNROLL
            acc += pa[l];
        }
        sint<32> y_q = (sint<32>)(acc >> ALE_W_FRAC);
        e = (sint<32>)x - y_q;
        // gain: num = mu_q*err*recip;  gq = num * 2^(26-R-k)  (16 extra fractional bits kept; dw = gq*x >> 16)
        sint<72> num = (sint<72>)((sint<72>)mu_q * (sint<72>)e) * (sint<72>)h.recip;
        int sh = (int)RECIP_R + (int)h.k - ALE_W_FRAC;          // in [-11, 29]
        sint<72> gq72 = (sh >= 0) ? (sint<72>)(num >> sh) : (sint<72>)(num << (-sh));
        MOE_ASSERT(gq72 < ((sint<72>)1 << 47) && gq72 >= -((sint<72>)1 << 47));   // must fit the 48-bit hardware gain
        sint<48> gq = (sint<48>)gq72;
    ALE_UPDATE: for (int k = 0; k < ALE_TAPS; k += ALE_LANES) {
#pragma HLS PIPELINE II=1
        ALE_UPD_LANE: for (int l = 0; l < ALE_LANES; l++) {
#pragma HLS UNROLL
                sint<64> dw = (sint<64>)((sint<64>)gq * (sint<64>)xd[ALE_DELAY - 1 + k + l]) >> 16;
                sint<64> wn = (sint<64>)w[k + l] + dw;
                if (wn > (sint<64>)0x7FFFFFFFLL) wn = (sint<64>)0x7FFFFFFFLL;
                if (wn < -(sint<64>)0x80000000LL) wn = -(sint<64>)0x80000000LL;
                w[k + l] = (sint<32>)wn;
            }
        }
    }
    s.sum_e2 = s.sum_e2 + (uint_<64>)((sint<64>)e * (sint<64>)e);
    s.e_last = e;
    return e;
}

// ─────────────────────────────────────────── D4: energy blanker ─────────────────────────────────────────────
struct BlankState { uint_<40> ss; uint_<6> pos; uint_<12> grp; uint_<12> flagged; uint_<40> ss_out; bool mask_out; bool valid; };

inline void blank_step(BlankState &s, bool first, x_t x) {
#pragma HLS INLINE off
    if (first) { s.ss = 0; s.pos = 0; s.grp = 0; s.flagged = 0; }
    s.valid = false;
    uint_<40> ss = s.ss + (uint_<40>)((sint<32>)x * (sint<32>)x);
    uint_<6> pos = s.pos + 1;
    if (pos == BLANK_WINDOW) {
        bool m = (ss > (uint_<40>)BLANK_THR_SQ_Q);
        s.mask_out = m; s.ss_out = ss; s.valid = true;
        if (m) s.flagged = s.flagged + 1;
        s.grp = s.grp + 1; ss = 0; pos = 0;
    }
    s.ss = ss; s.pos = pos;
}

// ─────────────────────────────────────────── D2: conventional Welch-1024 front-end ─────────────────────────
struct ConvState {
    uint_<11> pos;
    sint<40> sx;  uint_<48> sx2;  sint<64> sx3;  uint_<80> sx4;
    uint_<20> zc;  sint<2> prev_sgn;  uint_<8> n_seg;
    uint_<5> sf_idx;  uint_<20> sf_pos;
};

inline uint_<10> bitrev10(uint_<10> n) {
#pragma HLS INLINE
    uint_<10> r = 0;
BITREV: for (int b = 0; b < 10; b++) {
#pragma HLS UNROLL
        r = (r << 1) | ((n >> b) & 1);
    }
    return r;
}

inline uint_<26> isqrt52(uint_<52> v) {                        // floor(sqrt(v)), restoring, 26 result bits
#pragma HLS INLINE
    uint_<26> res = 0; uint_<54> rem = 0;
ISQRT: for (int b = 25; b >= 0; b--) {
#pragma HLS UNROLL
        rem = (rem << 2) | ((v >> (2 * b)) & 3);
        uint_<54> trial = ((uint_<54>)res << 2) | 1;
        if (rem >= trial) { rem = rem - trial; res = (res << 1) | 1; }
        else res = res << 1;
    }
    return res;
}

// window + bit-reversed load, 10 radix-2 DIT stages (no scaling, Q15 twiddles), magnitudes accumulated into P
inline void conv_fft(x_t inbuf[FFT_N], sint<27> re[FFT_N], sint<27> im[FFT_N], uint_<32> P[FFT_BINS]) {
#pragma HLS INLINE off
FFT_LOAD: for (int n = 0; n < FFT_N; n++) {
#pragma HLS PIPELINE II=1
        sint<32> w = (sint<32>)((sint<32>)inbuf[n] * (sint<32>)HANN_Q15[n]) >> 15;
        uint_<10> d = bitrev10((uint_<10>)n);
        re[d] = (sint<27>)w; im[d] = 0;
    }
FFT_STAGES: for (int st = 1; st <= FFT_LOG2N; st++) {
        const int half = 1 << (st - 1);
        const int shift = FFT_LOG2N - st;
    FFT_BFLY: for (int idx = 0; idx < FFT_N / 2; idx++) {
#pragma HLS PIPELINE II=2
#pragma HLS DEPENDENCE variable=re inter false
#pragma HLS DEPENDENCE variable=im inter false
            const int j = idx & (half - 1);
            const int ia = ((idx >> (st - 1)) << st) + j;
            const int ib = ia + half;
            sint<16> wr = TW_RE_Q15[j << shift], wi = TW_IM_Q15[j << shift];
            sint<27> br = re[ib], bi = im[ib], ar = re[ia], ai = im[ia];
            sint<44> tr = (sint<44>)((sint<44>)((sint<44>)br * (sint<44>)wr) - (sint<44>)((sint<44>)bi * (sint<44>)wi)) >> 15;
            sint<44> ti = (sint<44>)((sint<44>)((sint<44>)br * (sint<44>)wi) + (sint<44>)((sint<44>)bi * (sint<44>)wr)) >> 15;
            sint<28> nr_a = (sint<28>)ar + (sint<28>)tr, ni_a = (sint<28>)ai + (sint<28>)ti;
            sint<28> nr_b = (sint<28>)ar - (sint<28>)tr, ni_b = (sint<28>)ai - (sint<28>)ti;
            MOE_ASSERT(nr_a < (1 << 26) && nr_a >= -(1 << 26) && nr_b < (1 << 26) && nr_b >= -(1 << 26));
            MOE_ASSERT(ni_a < (1 << 26) && ni_a >= -(1 << 26) && ni_b < (1 << 26) && ni_b >= -(1 << 26));
            re[ia] = (sint<27>)nr_a; im[ia] = (sint<27>)ni_a;
            re[ib] = (sint<27>)nr_b; im[ib] = (sint<27>)ni_b;
        }
    }
FFT_MAG: for (int k = 0; k < FFT_BINS; k++) {
#pragma HLS PIPELINE II=1
        sint<27> r = re[k], i = im[k];
        uint_<52> m2 = (uint_<52>)((sint<54>)r * (sint<54>)r) + (uint_<52>)((sint<54>)i * (sint<54>)i);
        P[k] = P[k] + (uint_<32>)isqrt52(m2);
    }
}

inline void conv_step(ConvState &s, x_t inbuf[FFT_N], sint<27> re[FFT_N], sint<27> im[FFT_N], uint_<32> P[FFT_BINS], uint_<48> se[N_SUBFRAMES],
                      bool first, idx_t i, x_t x, uint_<20> sf_len) {
#pragma HLS INLINE off
    if (first) {
        s.pos = 0; s.sx = 0; s.sx2 = 0; s.sx3 = 0; s.sx4 = 0; s.zc = 0; s.prev_sgn = 0; s.n_seg = 0; s.sf_idx = 0; s.sf_pos = 0;
    CONV_RST_P: for (int k = 0; k < FFT_BINS; k++) {
#pragma HLS PIPELINE II=1
            P[k] = 0;
        }
    CONV_RST_SE: for (int k = 0; k < N_SUBFRAMES; k++) {
#pragma HLS UNROLL
            se[k] = 0;
        }
    } else {
        s.sf_pos = s.sf_pos + 1;
        if (s.sf_pos == sf_len) { s.sf_pos = 0; if (s.sf_idx < N_SUBFRAMES - 1) s.sf_idx = s.sf_idx + 1; }
    }
    sint<32> x2 = (sint<32>)x * (sint<32>)x;
    s.sx = s.sx + (sint<40>)x;
    s.sx2 = s.sx2 + (uint_<48>)x2;
    s.sx3 = s.sx3 + (sint<64>)((sint<64>)x2 * (sint<64>)x);
    s.sx4 = s.sx4 + (uint_<80>)((uint_<64>)x2 * (uint_<64>)x2);
    se[s.sf_idx] = se[s.sf_idx] + (uint_<48>)x2;
    sint<2> sgn = (x > 0) ? (sint<2>)1 : ((x < 0) ? (sint<2>)-1 : (sint<2>)0);
    if (!first && sgn != s.prev_sgn) s.zc = s.zc + 1;
    s.prev_sgn = sgn;
    inbuf[s.pos] = x;
    uint_<11> pos = s.pos + 1;
    if (pos == FFT_N) { conv_fft(inbuf, re, im, P); s.n_seg = s.n_seg + 1; pos = 0; }
    s.pos = pos;
}

inline sint<16> sat16(sint<32> v) {
#pragma HLS INLINE
    if (v > (sint<32>)32767) return (sint<16>)32767;
    if (v < (sint<32>)-32768) return (sint<16>)-32768;
    return (sint<16>)v;
}
#endif
