// Word-length abstraction: ap_int<N> under Vitis HLS (csim/csynth/cosim) or with -DUSE_AP_INT,
// plain C integers (>= the ap_int width) otherwise, so the very same source compiles with g++ for the
// golden comparison. Every product is cast explicitly to its accumulator type, so both builds agree bit for bit
// as long as no ap_int<N> overflows — which the csim run on the test vectors checks (see moe_blocks.cpp asserts).
#ifndef MOE_TYPES_H
#define MOE_TYPES_H
#include <cstdint>

#if defined(__VITIS_HLS__) || defined(USE_AP_INT)
  #include "ap_int.h"
  template <int N> using sint  = ap_int<N>;
  template <int N> using uint_ = ap_uint<N>;
  #define MOE_AP_INT 1
#else
  #include <type_traits>
  template <int N> using sint  = typename std::conditional<(N <= 32), int32_t,  typename std::conditional<(N <= 64), int64_t,  __int128>::type>::type;
  template <int N> using uint_ = typename std::conditional<(N <= 32), uint32_t, typename std::conditional<(N <= 64), uint64_t, unsigned __int128>::type>::type;
  #define MOE_AP_INT 0
#endif

#ifndef N_REPL
#define N_REPL 8             // datapath replicas fed by an N_REPL-stage sample delay chain (power amplification); <= 16
#endif
#define N_MAX 100000         // replay / capture buffer length (samples)
#define WIN_WORDS 1024       // AXI-Lite transfer window (32-bit words)
#define RES_WORDS 1024       // AXI-Lite result block (32-bit words)

// commands (AXI-Lite 'cmd')
enum { CMD_NOP = 0, CMD_LOAD = 1, CMD_RUN = 2, CMD_VERIFY = 3, CMD_READ = 4 };
// block-enable bits (AXI-Lite 'blk_en')
enum { EN_FE = 1, EN_CONV = 2, EN_ALE_NB = 4, EN_ALE_SW = 8, EN_BLANK = 16 };
// capture selectors (CMD_VERIFY arg1): what replica 0 writes into the capture buffer
enum { CAP_NONE = 0, CAP_SPIKES = 1, CAP_E_NB = 2, CAP_E_SW = 3, CAP_MASK = 4, CAP_SS_LO = 5, CAP_SS_MID = 6, CAP_SS_HI = 7 };

// result-word layout (res[])
enum {
    R_COUNT = 0, R_ISI_N = 1, R_ISI_SUM = 2, R_ISI_SQ_LO = 3, R_ISI_SQ_HI = 4,
    R_SF_CNT = 5, R_SF_ISI_SUM = 25, R_SF_ISI_CNT = 45,           // 20 each
    R_RF_CNT = 65,                                               // [20][16] row-major -> 65..384
    R_SUM_X2 = 385, R_SUM_E2_NB = 387, R_SUM_E2_SW = 389,        // 64-bit (lo, hi)
    R_BLANK_FLAGGED = 391, R_E_NB_LAST = 392, R_E_SW_LAST = 393,
    R_CONV_SX = 394, R_CONV_SX2 = 396, R_CONV_SX3 = 398, R_CONV_SX4 = 400,   // sx4: 3 words
    R_CONV_SE = 403,                                             // 20 x (lo, hi) -> 403..442
    R_CONV_ZC = 443, R_CONV_NSEG = 444, R_CONV_P = 445,          // P[513] -> 445..957
    R_CHK = 958,                                                 // N_REPL words (<= 16)
    R_PACE_ACC = 974, R_FRAMES_DONE = 975, R_MAGIC = 976, R_N_REPL = 977, R_BLK_EN = 978
};
#define RES_MAGIC 0x4D4F4531u   // 'MOE1'

struct CapOut {                 // per-sample capture outputs of replica 0
    sint<16> value;             // spike / e_nb / e_sw (saturated int16) / mask or sumsq slice
    bool     valid;             // for group-rate captures (mask, sumsq)
    uint_<12> index;            // group index for group-rate captures
};
#endif
