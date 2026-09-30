// Feature finalisation in IEEE-754 double, replicating board/features_ps.py (== the Colab reference) operation by
// operation, including numpy's pairwise summation order for contiguous 1-D reductions, so the results are bit-identical.
// Resource discipline (the design is synthesised with syn.compile.pipeline_loops=0, i.e. no automatic loop pipelining):
// only COLS (one pipelined divider, 320 divisions per inference) and the forest INIT/TREES loops are pipelined; every
// other loop stays rolled so that the double-precision operators are shared, and ALLOCATION limits pin the number of
// ddiv / dsqrt / dmul / dadd instances per function (a pipelined loop becomes its own module with its own operators).
#pragma once
#include <math.h>
#include "rf_types.h"
#include "rf_forest.h"

#define FS_HZ 1000000.0
#define F_LO_HZ 25000.0
#define F_STEP_HZ 30000.0          // numpy linspace(25e3, 475e3, 16): step = 450e3/15 exactly
#define TWO_PI 6.283185307179586   // 2*np.pi as evaluated by Python

static inline double rf_sqrt(double v) { return sqrt(v); }

// numpy pairwise_sum for n <= 128 (loops_utils.h.src); n < 8 -> plain loop from 0.0
static inline double pw_sum(const double *a, int n) {
#pragma HLS INLINE
    if (n < 8) {
        double res = 0.0;
    PW_SMALL: for (int i = 0; i < 8; i++) { if (i < n) res += a[i]; }
        return res;
    }
    double r[8];
#pragma HLS ARRAY_PARTITION variable=r complete
PW_INIT: for (int j = 0; j < 8; j++) r[j] = a[j];
    int i = 8;
    const int lim = n - (n % 8);
PW_BLK: for (; i < lim; i += 8) {
    PW_ACC: for (int j = 0; j < 8; j++) r[j] += a[i + j];
    }
    double res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
PW_REM: for (; i < n; i++) res += a[i];
    return res;
}

static inline double pw_mean(const double *a, int n) {
#pragma HLS INLINE
    return pw_sum(a, n) / (double)n;
}

// np.var / np.std (ddof 0): mean via pairwise sum, then pairwise sum of squared deviations
static inline double np_var(const double *a, int n, double *scratch) {
#pragma HLS INLINE
    const double m = pw_mean(a, n);
VAR_SQ: for (int i = 0; i < n; i++) { const double d = a[i] - m; scratch[i] = d * d; }
    return pw_sum(scratch, n) / (double)n;
}
static inline double np_std(const double *a, int n, double *scratch) { return rf_sqrt(np_var(a, n, scratch)); }

// ── spike statistics -> 5 features; sub-frame arrays read with rotation rot ──
static inline void spike_features(const uint32_t *w, int n, int rot, double *f) {
#pragma HLS INLINE off
#pragma HLS ALLOCATION operation instances=ddiv  limit=1
#pragma HLS ALLOCATION operation instances=dsqrt limit=1
#pragma HLS ALLOCATION operation instances=dmul  limit=1
#pragma HLS ALLOCATION operation instances=dadd  limit=2
    const uint32_t cnt = w[W_COUNT], isi_n = w[W_ISI_N], isi_sum = w[W_ISI_SUM];
    const uint64_t isi_sq = (uint64_t)w[W_ISI_SQ] | ((uint64_t)w[W_ISI_SQ + 1] << 32);
    double sf_cnt[N_SUB], sf_isi_sum[N_SUB], sf_isi_cnt[N_SUB], means[N_SUB], scratch[N_SUB];
RD_SF: for (int k = 0; k < N_SUB; k++) {
        int kk = k + rot; if (kk >= N_SUB) kk -= N_SUB;
        sf_cnt[k] = (double)w[W_SF_CNT + kk]; sf_isi_sum[k] = (double)w[W_SF_ISI_SUM + kk]; sf_isi_cnt[k] = (double)w[W_SF_ISI_CNT + kk];
    }
    f[0] = (double)cnt / (double)n;
    if (isi_n > 2) {
        const double m = (double)isi_sum / (double)isi_n;
        double v = (double)isi_sq / (double)isi_n - m * m;
        if (v < 0.0) v = 0.0;
        f[1] = rf_sqrt(v) / (m + 1e-10);
    } else f[1] = 0.0;
    const double mu = pw_mean(sf_cnt, N_SUB);
    f[2] = (cnt > 0) ? np_var(sf_cnt, N_SUB, scratch) / (mu + 1e-10) : 0.0;
    double mx = sf_cnt[0], mn = sf_cnt[0];
MINMAX: for (int k = 1; k < N_SUB; k++) { if (sf_cnt[k] > mx) mx = sf_cnt[k]; if (sf_cnt[k] < mn) mn = sf_cnt[k]; }
    f[3] = (cnt > 0) ? mx / (mn + 1.0) : 1.0;
    int nm = 0;
MEANS: for (int k = 0; k < N_SUB; k++) { if (sf_isi_cnt[k] > 2.0) { means[nm] = sf_isi_sum[k] / sf_isi_cnt[k]; nm++; } }
    const uint32_t dn = (isi_n > 1u) ? isi_n : 1u;
    const double g = (double)isi_sum / (double)dn;
    f[4] = (cnt >= 4 && nm > 1) ? np_std(means, nm, scratch) / (g + 1e-10) : 0.0;
}

// ── RF-bank counters -> 4 features ──
static inline void rf_features(const uint32_t *w, int n, int rot, double *f) {
#pragma HLS INLINE off
#pragma HLS ALLOCATION operation instances=ddiv  limit=1
#pragma HLS ALLOCATION operation instances=dsqrt limit=1
#pragma HLS ALLOCATION operation instances=dmul  limit=1
#pragma HLS ALLOCATION operation instances=dadd  limit=2
    double den[RF_K], band[RF_K], tot[N_SUB], cen[N_SUB], am[N_SUB], scratch[N_SUB], row[RF_K], rowc[RF_K];
    const double nsub = (double)(n / N_SUB);
DEN: for (int b = 0; b < RF_K; b++) {
        const double fk = (double)b * F_STEP_HZ + F_LO_HZ;      // linspace: arange*step + start
        const double om = (TWO_PI * fk) / FS_HZ;
        den[b] = (nsub * om) / TWO_PI;
        band[b] = 0.0;
    }
    int nact = 0;
ROWS: for (int k = 0; k < N_SUB; k++) {
        int kk = k + rot; if (kk >= N_SUB) kk -= N_SUB;
        double mxv = -1.0; int arg = 0;
    COLS: for (int b = 0; b < RF_K; b++) {
#pragma HLS PIPELINE II=1
            const double r = (double)w[W_RF_CNT + kk * RF_K + b] / den[b];
            row[b] = r; rowc[b] = r * (double)b;
            band[b] = (k == 0) ? r : band[b] + r;            // axis-0 reduction: sequential over rows
            if (r > mxv) { mxv = r; arg = b; }                // np.argmax: first maximum
        }
        const double t = pw_sum(row, RF_K);
        tot[k] = t;
        if (t > 0.0) { cen[nact] = pw_sum(rowc, RF_K) / t; am[nact] = (double)arg; nact++; }
    }
BAND: for (int b = 0; b < RF_K; b++) band[b] = band[b] / (double)N_SUB;
    double bmax = band[0];
BMAX: for (int b = 1; b < RF_K; b++) if (band[b] > bmax) bmax = band[b];
    f[5] = bmax / (pw_mean(band, RF_K) + 1e-10);
    f[6] = pw_sum(band + RF_K / 2, RF_K / 2) / (pw_sum(band, RF_K) + 1e-10);
    if (nact > 1) {
        f[7] = np_std(cen, nact, scratch) / (double)RF_K;
        f[8] = np_std(am, nact, scratch) / (double)RF_K;
    } else { f[7] = 0.0; f[8] = 0.0; }
}

// ── forest inference: level-synchronous traversal of all trees, leaf votes pre-normalised, argmax first-max ──
// vsum[] = raw vote sums (sequential over trees); the caller divides by the tree count (one shared divider)
template <int FID>
static inline int forest_infer(const double *feats, double *vsum) {
#pragma HLS INLINE off
    const int T = (FID == 0) ? RF0_TREES : RF1_TREES;
    uint16_t cur[RF_MAX_TREES];
INIT: for (int t = 0; t < RF_MAX_TREES; t++) {
#pragma HLS PIPELINE II=1
        if (t < T) cur[t] = (FID == 0) ? RF0_ROOT[t] : RF1_ROOT[t];
    }
LEVELS: for (int d = 0; d < RF_MAX_DEPTH; d++) {
    TREES: for (int t = 0; t < RF_MAX_TREES; t++) {
#pragma HLS PIPELINE II=1
            if (t < T) {
                const int i = cur[t];
                const uint16_t leaf = (FID == 0) ? RF0_LEAF[i] : RF1_LEAF[i];
                const int fi = (FID == 0) ? RF0_F[i] : RF1_F[i];
                const double thr = (FID == 0) ? RF0_THR[i] : RF1_THR[i];
                const uint16_t l = (FID == 0) ? RF0_L[i] : RF1_L[i];
                const uint16_t r = (FID == 0) ? RF0_R[i] : RF1_R[i];
                if (!leaf) cur[t] = (feats[fi] <= thr) ? l : r;
            }
        }
    }
    double v[N_CLASSES];
#pragma HLS ARRAY_PARTITION variable=v complete
VZERO: for (int c = 0; c < N_CLASSES; c++) v[c] = 0.0;
VOTES: for (int t = 0; t < RF_MAX_TREES; t++) {           // rolled: sequential over trees (== numpy axis-0 order)
        if (t < T) {
            const int i = cur[t];
            const int li = (int)((FID == 0) ? RF0_LEAF[i] : RF1_LEAF[i]) - 1;
        VADD: for (int c = 0; c < N_CLASSES; c++) {
#pragma HLS UNROLL
                v[c] += (FID == 0) ? RF0_LN[li][c] : RF1_LN[li][c];
            }
        }
    }
    int best = 0;
ARGMAX: for (int c = 1; c < N_CLASSES; c++) if (v[c] > v[best]) best = c;
VOUT: for (int c = 0; c < N_CLASSES; c++) vsum[c] = v[c];
    return best;
}
