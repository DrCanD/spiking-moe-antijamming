#ifndef MOE_TOP_H
#define MOE_TOP_H
#include "moe_types.h"
// Single HLS IP, AXI-Lite controlled (bundle 'ctrl'):
//   cmd       1=LOAD (window -> replay[arg0*1024 ..]), 2=RUN (arg0 frames, no capture), 3=VERIFY (one frame, capture arg1),
//             4=READ (capture[arg0*1024 ..] -> window)
//   blk_en    bit0 front-end B (D1), bit1 conventional FFT front-end (D2), bit2 ALE mu_NB (D3), bit3 ALE mu_SW, bit4 blanker (D4)
//   pace      extra busy-loop iterations per sample (throughput knob; the PS measures the achieved sample rate)
//   n_samples frame length (<= N_MAX), 100000 for the paper
//   window    1024 x int32 transfer window (LOAD: samples in; READ: captured stream out)
//   res       1024 x uint32 results of replica 0 (layout: moe_types.h R_*), checksums of all replicas, counters
void moe_top(int cmd, int arg0, int arg1, int blk_en, int pace, int n_samples, int window[WIN_WORDS], unsigned int res[RES_WORDS]);
#endif
