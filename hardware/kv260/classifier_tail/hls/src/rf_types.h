// D6 — per-frame classification tail of the router: feature finalisation (float64, same operation order as the
// Colab/PS reference) + Random Forest inference. Word offsets of the input block = the moe_top result layout.
#pragma once
#include <stdint.h>

#ifndef N_REPL_RF
#define N_REPL_RF 1
#endif
#define RF_WIN_WORDS 1024
#define RF_RES_WORDS 256
#define N_SUB 20
#define RF_K 16
#define N_FEAT 9
#define N_CLASSES 5

// input block layout (moe_top result words)
enum { W_COUNT = 0, W_ISI_N = 1, W_ISI_SUM = 2, W_ISI_SQ = 3, W_SF_CNT = 5, W_SF_ISI_SUM = 25, W_SF_ISI_CNT = 45, W_RF_CNT = 65 };
// commands
enum { RC_NOP = 0, RC_LOAD = 1, RC_RUN = 2, RC_VERIFY = 3 };
// modes (what the inference loop computes)
enum { M_OVERHEAD = 0, M_FEATURES = 1, M_FOREST_FIXED = 2, M_FOREST_RANDOM = 3 };
// result words
enum { Q_MAGIC = 0, Q_N_REPL = 1, Q_MODE = 2, Q_INFER_DONE = 3, Q_PACE_ACC = 4, Q_N = 5, Q_CHK = 8, Q_VERDICT = 16, Q_ROT = 17,
       Q_VOTES = 20 /* 5 doubles = 10 words */, Q_FEATS = 32 /* 9 doubles = 18 words */, Q_REPL_LOADED = 52 };
#define RF_RES_MAGIC 0x52464F31u

static inline double rf_sqrt(double v);
