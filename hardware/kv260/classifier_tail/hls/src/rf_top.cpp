// D6 top: the router's per-frame classification tail — feature finalisation (float64) + Random Forest — fed from a
// frame-word buffer loaded over AXI-Lite. ONE hardware instance of the tail (rep_infer) serves every command:
//   RC_LOAD   : window[] -> fw[replica]  (arg0 = replica, -1 = all)
//   RC_VERIFY : one inference on replica 0 with sub-frame rotation arg0; features / votes / verdict exported
//   RC_RUN    : arg0 inferences on every replica in turn (rotation advances every iteration so the inputs differ),
//               with a pace loop between inferences for rate control; checksums keep every replica observable.
// Modes: 0 = overhead only (loop + pace + checksum), 1 = features only, 2 = features + forest 'fixed', 3 = + forest 'random'.
// N_REPL_RF defaults to 1: the tail is not tied to a sample rate, so the measured power is raised by running the single
// instance at a higher inference rate rather than by replicating hardware.
#include "rf_top.h"
#include "rf_feat.h"

static void rep_infer(int mode, int n, int rot, const uint32_t fw[RF_WIN_WORDS], double feats[N_FEAT], double votes[N_CLASSES],
                      int &verdict, uint32_t &chk) {
#pragma HLS INLINE off
#pragma HLS ALLOCATION operation instances=ddiv limit=1
    double f[N_FEAT];
#pragma HLS ARRAY_PARTITION variable=f complete
    double vs[N_CLASSES], v[N_CLASSES];
#pragma HLS ARRAY_PARTITION variable=vs complete
#pragma HLS ARRAY_PARTITION variable=v complete
    int best = -1, ntrees = 1;
    if (mode >= M_FEATURES) {
        spike_features(fw, n, rot, f);
        rf_features(fw, n, rot, f);
    } else {
    FZ: for (int i = 0; i < N_FEAT; i++) f[i] = (double)(rot + i);
    }
    if (mode == M_FOREST_FIXED)       { best = forest_infer<0>(f, vs); ntrees = RF0_TREES; }
    else if (mode == M_FOREST_RANDOM) { best = forest_infer<1>(f, vs); ntrees = RF1_TREES; }
    else { VZ: for (int c = 0; c < N_CLASSES; c++) vs[c] = 0.0; }
VN: for (int c = 0; c < N_CLASSES; c++) v[c] = vs[c] / (double)ntrees;     // numpy: votes / len(trees)
    // fold everything into the checksum (bit patterns of the doubles)
    uint32_t c = chk ^ (uint32_t)(best + 1);
CHK_F: for (int i = 0; i < N_FEAT; i++) {
        union { double d; uint64_t u; } x; x.d = f[i];
        c = (c * 31u) ^ (uint32_t)(x.u & 0xFFFFFFFFu) ^ (uint32_t)(x.u >> 32);
    }
CHK_V: for (int i = 0; i < N_CLASSES; i++) {
        union { double d; uint64_t u; } x; x.d = v[i];
        c = (c * 31u) ^ (uint32_t)(x.u & 0xFFFFFFFFu) ^ (uint32_t)(x.u >> 32);
    }
    chk = c;
OUT_F: for (int i = 0; i < N_FEAT; i++) feats[i] = f[i];
OUT_V: for (int i = 0; i < N_CLASSES; i++) votes[i] = v[i];
    verdict = best;
}

static inline void put_d(unsigned int res[RF_RES_WORDS], int at, double d) {
#pragma HLS INLINE
    union { double d; uint64_t u; } x; x.d = d;
    res[at] = (unsigned int)(x.u & 0xFFFFFFFFu); res[at + 1] = (unsigned int)(x.u >> 32);
}

void rf_top(int cmd, int arg0, int arg1, int mode, int pace, int n_samples, unsigned int window[RF_WIN_WORDS], unsigned int res[RF_RES_WORDS]) {
#pragma HLS INTERFACE s_axilite port=cmd       bundle=ctrl
#pragma HLS INTERFACE s_axilite port=arg0      bundle=ctrl
#pragma HLS INTERFACE s_axilite port=arg1      bundle=ctrl
#pragma HLS INTERFACE s_axilite port=mode      bundle=ctrl
#pragma HLS INTERFACE s_axilite port=pace      bundle=ctrl
#pragma HLS INTERFACE s_axilite port=n_samples bundle=ctrl
#pragma HLS INTERFACE s_axilite port=window    bundle=ctrl
#pragma HLS INTERFACE s_axilite port=res       bundle=ctrl
#pragma HLS INTERFACE s_axilite port=return    bundle=ctrl

    static uint32_t fw[N_REPL_RF][RF_WIN_WORDS];
#pragma HLS ARRAY_PARTITION variable=fw complete dim=1
#pragma HLS BIND_STORAGE variable=fw type=ram_1p impl=bram
    static uint32_t chk[N_REPL_RF];
#pragma HLS ARRAY_PARTITION variable=chk complete
    static uint32_t pace_acc = 0;
    static uint32_t loaded = 0;

    int n = n_samples; if (n < N_SUB) n = N_SUB;
    res[Q_MAGIC] = RF_RES_MAGIC; res[Q_N_REPL] = N_REPL_RF; res[Q_MODE] = (unsigned int)mode; res[Q_N] = (unsigned int)n;

    if (cmd == RC_LOAD) {
        const int r = arg0;
    LOAD: for (int j = 0; j < RF_WIN_WORDS; j++) {
#pragma HLS PIPELINE II=1
            const uint32_t w = (uint32_t)window[j];
        LOAD_R: for (int q = 0; q < N_REPL_RF; q++) {
#pragma HLS UNROLL
                if (q == r || r < 0) fw[q][j] = w;
            }
        }
        loaded |= (r < 0) ? ((1u << N_REPL_RF) - 1u) : (1u << r);
    } else if (cmd == RC_RUN || cmd == RC_VERIFY) {
        const bool verify = (cmd == RC_VERIFY);
        const int n_infer = verify ? 1 : arg0;
        int rot = verify ? arg0 : 0;                        // VERIFY: rotation requested by the host (0..N_SUB-1)
        if (rot < 0 || rot >= N_SUB) rot = 0;
        res[Q_ROT] = (unsigned int)rot;
    RZ: for (int q = 0; q < N_REPL_RF; q++) chk[q] = verify ? 0u : 0x9E3779B9u * (uint32_t)(q + 1);
    INFER: for (int it = 0; it < n_infer; it++) {
        REPL: for (int q = 0; q < N_REPL_RF; q++) {          // rolled: every replica goes through the same rep_infer instance
                int rq = rot + 5 * q; if (rq >= N_SUB) rq -= N_SUB;
                double feats[N_FEAT], votes[N_CLASSES]; int verdict = -1;
                rep_infer(mode, n, rq, fw[q], feats, votes, verdict, chk[q]);
                res[Q_CHK + q] = chk[q];
                if (q == 0) {
                    res[Q_VERDICT] = (unsigned int)verdict;
                VF: for (int i = 0; i < N_FEAT; i++) put_d(res, Q_FEATS + 2 * i, feats[i]);
                VV: for (int i = 0; i < N_CLASSES; i++) put_d(res, Q_VOTES + 2 * i, votes[i]);
                }
            }
            rot = rot + 1; if (rot >= N_SUB) rot = 0;
        PACE: for (int p = 0; p < pace; p++) {
#pragma HLS PIPELINE II=1
                pace_acc = pace_acc + ((pace_acc >> 1) ^ (uint32_t)p);
            }
            res[Q_INFER_DONE] = (unsigned int)(it + 1);
        }
    }
    res[Q_PACE_ACC] = pace_acc; res[Q_REPL_LOADED] = loaded;
}
