// C testbench for rf_top: replays rf_tests.bin (frame words rebuilt from *_expected.json, float64 reference features,
// forest verdicts for the 20 sub-frame rotations x 2 forests). PASS = features bit-identical (rot 0) and every verdict equal.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "../src/rf_top.h"

static double get_d(const unsigned int *res, int at) { union { double d; uint64_t u; } x; x.u = (uint64_t)res[at] | ((uint64_t)res[at + 1] << 32); return x.d; }

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    const char *path = (argc > 1) ? argv[1] : "rf_tests.bin";
    int max_frames = (argc > 2) ? atoi(argv[2]) : 1000;
    FILE *fp = fopen(path, "rb"); if (!fp) { printf("[TB] cannot open %s\n", path); return 1; }
    uint32_t magic, nrec; if (fread(&magic, 4, 1, fp) != 1 || fread(&nrec, 4, 1, fp) != 1 || magic != 0x52465453u) { printf("[TB] bad test file\n"); return 1; }
    static unsigned int window[RF_WIN_WORDS], res[RF_RES_WORDS];
    int fails = 0, frames = 0, feat_mismatch = 0, verdict_mismatch = 0;
    for (uint32_t k = 0; k < nrec && (int)k < max_frames; k++) {
        char name[32]; uint32_t n; static uint32_t w[1024]; double fref[9]; uint32_t v0[20], v1[20];
        if (fread(name, 32, 1, fp) != 1 || fread(&n, 4, 1, fp) != 1 || fread(w, 4, 1024, fp) != 1024 || fread(fref, 8, 9, fp) != 9 || fread(v0, 4, 20, fp) != 20 || fread(v1, 4, 20, fp) != 20) { printf("[TB] short read\n"); return 1; }
        frames++;
        for (int j = 0; j < RF_WIN_WORDS; j++) window[j] = w[j];
        rf_top(RC_LOAD, -1, 0, 0, 0, (int)n, window, res);          // all replicas
        int bad = 0;
        for (int rot = 0; rot < 20; rot++) {
            for (int mode = M_FOREST_FIXED; mode <= M_FOREST_RANDOM; mode++) {
                rf_top(RC_VERIFY, rot, 0, mode, 0, (int)n, window, res);
                const uint32_t exp = (mode == M_FOREST_FIXED) ? v0[rot] : v1[rot];
                if (res[Q_VERDICT] != exp) { verdict_mismatch++; bad = 1; printf("   %s rot %d forest %d: verdict %u != %u\n", name, rot, mode - 2, res[Q_VERDICT], exp); }
                if (rot == 0 && mode == M_FOREST_FIXED) {
                    for (int i = 0; i < 9; i++) {
                        const double got = get_d(res, Q_FEATS + 2 * i);
                        if (memcmp(&got, &fref[i], 8) != 0) { feat_mismatch++; bad = 1; printf("   %s feature %d: %.17g != %.17g (rel %.2e)\n", name, i, got, fref[i], (got - fref[i]) / (fref[i] != 0 ? fref[i] : 1)); }
                    }
                }
            }
        }
        // RUN path: 20 iterations, replica 0 ends at rotation 19 (forest 'random'); checksums must differ across replicas
        rf_top(RC_RUN, 20, 0, M_FOREST_RANDOM, 3, (int)n, window, res);
        if (res[Q_VERDICT] != v1[19]) { bad = 1; printf("   %s RUN: final verdict %u != %u\n", name, res[Q_VERDICT], v1[19]); }
        if (res[Q_INFER_DONE] != 20) { bad = 1; printf("   %s RUN: infer_done %u\n", name, res[Q_INFER_DONE]); }
        for (int q = 1; q < N_REPL_RF; q++) if (res[Q_CHK + q] == res[Q_CHK]) { bad = 1; printf("   %s RUN: replica %d checksum equals replica 0\n", name, q); }
        fails += bad;
        printf("[TB] %-24s n=%u  verdict(fixed,random,rot0)=%u,%u  %s\n", name, n, v0[0], v1[0], bad ? "FAIL" : "ok");
    }
    fclose(fp);
    printf("[TB] %d frames, %d feature mismatches, %d verdict mismatches -> %s\n", frames, feat_mismatch, verdict_mismatch, fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
