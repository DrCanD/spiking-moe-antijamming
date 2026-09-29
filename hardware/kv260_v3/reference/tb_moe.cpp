// C testbench for moe_top: drives the IP exactly as the PS will (LOAD chunks -> VERIFY captures -> READ chunks)
// and compares every stream and accumulator BIT FOR BIT with the fixed-point golden vectors
// (<name>_in_q6_10.bin, _spikes_i8.bin, _ale_nb_q6_10.bin, _ale_sw_q6_10.bin, _expected_words.bin).
//   tb_moe <vectors_dir> [--quick|--medium] [name ...]   (default: every *_in_q6_10.bin in the directory)
//   --quick  = LOAD + 1 VERIFY (spikes, statistics, RF counters, conventional front-end)  -> RTL co-simulation / SAIF
//   --medium = quick + both ALE streams + blanker mask (4 passes)                          -> Vitis csim on a slow host
//   default  = medium + sums of squares (3 passes) + enable-isolation checks + 2-frame RUN (11 passes)
// Exit code 0 = all frames bit-exact. Used by Vitis HLS csim (see run_hls.tcl) and by the g++ build (Makefile).
#include "moe_top.h"
#include "moe_params.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <dirent.h>

static const int N_WORDS = 4929;
enum { W_MAGIC = 0, W_VER = 1, W_N = 2, W_FLAGS = 3, W_COUNT = 4, W_ISI_N = 5, W_ISI_SUM = 6, W_ISI_SQ = 7, W_SF_CNT = 8, W_SF_ISI_SUM = 28,
       W_SF_ISI_CNT = 48, W_RF = 68, W_NG = 388, W_MASK = 389, W_SUMSQ = 2389, W_SX = 4389, W_SX2 = 4390, W_SX3 = 4391, W_SX4_LO = 4392,
       W_SX4_HI = 4393, W_SE = 4394, W_ZC = 4414, W_NSEG = 4415, W_P = 4416 };

static bool read_file(const std::string &p, void *dst, size_t bytes) {
    FILE *f = fopen(p.c_str(), "rb"); if (!f) return false;
    size_t got = fread(dst, 1, bytes, f); fclose(f); return got == bytes;
}

static int window[WIN_WORDS];
static unsigned int res[RES_WORDS];

static void ip_load(const std::vector<int16_t> &x) {
    int n = (int)x.size(); int chunks = (n + WIN_WORDS - 1) / WIN_WORDS;
    for (int c = 0; c < chunks; c++) {
        for (int j = 0; j < WIN_WORDS; j++) { int a = c * WIN_WORDS + j; window[j] = (a < n) ? x[a] : 0; }
        moe_top(CMD_LOAD, c, 0, 0, 0, n, window, res);
    }
}
static void ip_verify(int cap, int blk_en, int n) { moe_top(CMD_VERIFY, 0, cap, blk_en, 0, n, window, res); }
static void ip_read(std::vector<int16_t> &out, int n) {
    out.assign(n, 0); int chunks = (n + WIN_WORDS - 1) / WIN_WORDS;
    for (int c = 0; c < chunks; c++) {
        moe_top(CMD_READ, c, 0, 0, 0, n, window, res);
        for (int j = 0; j < WIN_WORDS; j++) { int a = c * WIN_WORDS + j; if (a < n) out[a] = (int16_t)window[j]; }
    }
}
static unsigned long long get64(int at) { return (unsigned long long)res[at] | ((unsigned long long)res[at + 1] << 32); }

template <class T>
static int cmp_stream(const char *what, const std::vector<T> &got, const std::vector<T> &exp, int n) {
    int bad = 0, first = -1;
    for (int i = 0; i < n; i++) if (got[i] != exp[i]) { if (first < 0) first = i; bad++; }
    if (bad) printf("    !! %-10s %d/%d mismatches (first at %d: got %lld exp %lld)\n", what, bad, n, first, (long long)got[first], (long long)exp[first]);
    else     printf("    ok %-10s %d samples bit-exact\n", what, n);
    return bad;
}
static int cmp_word(const char *what, long long got, long long exp) {
    if (got != exp) { printf("    !! %-14s got %lld exp %lld\n", what, got, exp); return 1; }
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) { printf("usage: tb_moe <vectors_dir> [name ...]\n"); return 2; }
    std::string dir = argv[1]; if (dir.back() != '/' && dir.back() != '\\') dir += '/';
    setvbuf(stdout, NULL, _IONBF, 0);                          // csim captures stdout through a pipe: show progress immediately
    std::vector<std::string> names; bool quick = false, medium = false;
    for (int a = 2; a < argc; a++) {
        std::string s = argv[a];
        if (s == "--quick") quick = true; else if (s == "--medium") medium = true; else names.push_back(s);
    }
    if (names.empty()) {
        DIR *d = opendir(dir.c_str()); if (!d) { printf("cannot open %s\n", dir.c_str()); return 2; }
        while (dirent *e = readdir(d)) { std::string f = e->d_name; const std::string suf = "_in_q6_10.bin";
            if (f.size() > suf.size() && f.compare(f.size() - suf.size(), suf.size(), suf) == 0) names.push_back(f.substr(0, f.size() - suf.size())); }
        closedir(d); std::sort(names.begin(), names.end());
    }
    if (names.empty()) { printf("no *_in_q6_10.bin in %s\n", dir.c_str()); return 2; }
    printf("[TB] moe_top C testbench: N_REPL=%d, %d frames from %s (ap_int build: %d)\n", N_REPL, (int)names.size(), dir.c_str(), MOE_AP_INT);
    int total_bad = 0, frames_ok = 0;
    for (const std::string &nm : names) {
        std::vector<long long> W(N_WORDS);
        if (!read_file(dir + nm + "_expected_words.bin", W.data(), N_WORDS * 8)) { printf("[%s] missing _expected_words.bin (run golden/convert_expected.py)\n", nm.c_str()); total_bad++; continue; }
        if (W[W_MAGIC] != 0x4D4F4556LL) { printf("[%s] bad expected_words magic\n", nm.c_str()); total_bad++; continue; }
        const int n = (int)W[W_N]; const long long flags = W[W_FLAGS];
        std::vector<int16_t> x(n), e_nb(n), e_sw(n); std::vector<int8_t> sp8(n);
        if (!read_file(dir + nm + "_in_q6_10.bin", x.data(), n * 2) || !read_file(dir + nm + "_spikes_i8.bin", sp8.data(), n) ||
            !read_file(dir + nm + "_ale_nb_q6_10.bin", e_nb.data(), n * 2) || !read_file(dir + nm + "_ale_sw_q6_10.bin", e_sw.data(), n * 2)) {
            printf("[%s] missing stream files\n", nm.c_str()); total_bad++; continue; }
        std::vector<int16_t> sp(n); for (int i = 0; i < n; i++) sp[i] = sp8[i];
        printf("[%s] n=%d\n", nm.c_str(), n);
        int bad = 0;
        ip_load(x);
        // ── all blocks on: streams + accumulators ──
        const int ALL = EN_FE | EN_CONV | EN_ALE_NB | EN_ALE_SW | EN_BLANK;
        std::vector<int16_t> got;
        ip_verify(CAP_SPIKES, ALL, n); ip_read(got, n); bad += cmp_stream("spikes", got, sp, n);
        if (res[R_MAGIC] != RES_MAGIC) { printf("    !! result magic\n"); bad++; }
        unsigned chk0 = res[R_CHK];
        bad += cmp_word("count", res[R_COUNT], W[W_COUNT]); bad += cmp_word("isi_n", res[R_ISI_N], W[W_ISI_N]);
        bad += cmp_word("isi_sum", res[R_ISI_SUM], W[W_ISI_SUM]); bad += cmp_word("isi_sq_sum", (long long)get64(R_ISI_SQ_LO), W[W_ISI_SQ]);
        int b2 = 0;
        for (int k = 0; k < N_SUBFRAMES; k++) { b2 += (res[R_SF_CNT + k] != (unsigned)W[W_SF_CNT + k]) + (res[R_SF_ISI_SUM + k] != (unsigned)W[W_SF_ISI_SUM + k]) + (res[R_SF_ISI_CNT + k] != (unsigned)W[W_SF_ISI_CNT + k]); }
        if (b2) printf("    !! sub-frame stats: %d mismatching words\n", b2);
        bad += b2;
        int b3 = 0; for (int k = 0; k < N_SUBFRAMES * RF_BANDS; k++) b3 += (res[R_RF_CNT + k] != (unsigned)W[W_RF + k]);
        if (b3) printf("    !! rf counters: %d/%d mismatching\n", b3, N_SUBFRAMES * RF_BANDS); else printf("    ok rf_cnt     %d counters bit-exact\n", N_SUBFRAMES * RF_BANDS);
        bad += b3;
        if (flags & 4) {
            int b4 = 0;
            b4 += cmp_word("conv sx", (long long)get64(R_CONV_SX), W[W_SX]); b4 += cmp_word("conv sx2", (long long)get64(R_CONV_SX2), W[W_SX2]);
            b4 += cmp_word("conv sx3", (long long)get64(R_CONV_SX3), W[W_SX3]);
            unsigned long long lo = get64(R_CONV_SX4); long long hi = res[R_CONV_SX4 + 2];
            b4 += cmp_word("conv sx4 lo", (long long)lo, W[W_SX4_LO]); b4 += cmp_word("conv sx4 hi", hi, W[W_SX4_HI]);
            for (int k = 0; k < N_SUBFRAMES; k++) b4 += ((long long)get64(R_CONV_SE + 2 * k) != W[W_SE + k]);
            b4 += cmp_word("conv zc", res[R_CONV_ZC], W[W_ZC]); b4 += cmp_word("conv n_seg", res[R_CONV_NSEG], W[W_NSEG]);
            int bp = 0; for (int k = 0; k < FFT_BINS; k++) bp += (res[R_CONV_P + k] != (unsigned)W[W_P + k]);
            if (bp) printf("    !! conv spectrum: %d/%d bins mismatch (bin0 got %u exp %lld)\n", bp, FFT_BINS, res[R_CONV_P], W[W_P]);
            if (b4 + bp == 0) printf("    ok conv       moments, 20 energies, zc, %d Welch bins bit-exact\n", FFT_BINS);
            bad += b4 + bp;
        }
        if (quick) { if (bad == 0) { frames_ok++; printf("  => PASS (quick)\n"); } else printf("  => FAIL (%d)\n", bad); total_bad += bad; continue; }
        ip_verify(CAP_E_NB, ALL, n); ip_read(got, n); bad += cmp_stream("ale_nb", got, e_nb, n);
        ip_verify(CAP_E_SW, ALL, n); ip_read(got, n); bad += cmp_stream("ale_sw", got, e_sw, n);
        if (flags & 2) {
            const int ng = (int)W[W_NG];
            std::vector<int16_t> m, lo, mid, hi;
            ip_verify(CAP_MASK, ALL, n); ip_read(m, ng);
            if (!medium) { ip_verify(CAP_SS_LO, ALL, n); ip_read(lo, ng); ip_verify(CAP_SS_MID, ALL, n); ip_read(mid, ng); ip_verify(CAP_SS_HI, ALL, n); ip_read(hi, ng); }
            int bm = 0, bs = 0, flagged = 0;
            for (int g = 0; g < ng; g++) {
                if (m[g] != (int16_t)W[W_MASK + g]) bm++;
                if (!medium) {
                    long long ss = (long long)(uint16_t)lo[g] | ((long long)(uint16_t)mid[g] << 16) | ((long long)(uint16_t)hi[g] << 32);
                    if (ss != W[W_SUMSQ + g]) bs++;
                }
                flagged += (int)W[W_MASK + g];
            }
            if (bm) printf("    !! blanker mask: %d/%d groups mismatch\n", bm, ng);
            if (bs) printf("    !! blanker sumsq: %d/%d groups mismatch\n", bs, ng);
            bad += bm + bs + cmp_word("blank flagged", res[R_BLANK_FLAGGED], flagged);
            if (bm + bs == 0) printf("    ok blanker    %d groups (mask%s) bit-exact\n", ng, medium ? "" : " + sum of squares");
        }
        if (medium) { if (bad == 0) { frames_ok++; printf("  => PASS (medium)\n"); } else printf("  => FAIL (%d)\n", bad); total_bad += bad; continue; }
        // ── block enables must not interact: D1 alone and D3(NB) alone reproduce the same results ──
        ip_verify(CAP_SPIKES, EN_FE, n); ip_read(got, n);
        int b5 = cmp_stream("spikes(D1)", got, sp, n) + cmp_word("count(D1)", res[R_COUNT], W[W_COUNT]);
        ip_verify(CAP_E_NB, EN_ALE_NB, n); ip_read(got, n); b5 += cmp_stream("ale_nb(D3)", got, e_nb, n);
        bad += b5;
        // ── RUN mode: 2 frames, all blocks, pace 3: checksum of replica 0 must equal the VERIFY one, replicas must differ ──
        moe_top(CMD_RUN, 2, 0, ALL, 3, n, window, res);
        int b6 = 0;
        if (res[R_FRAMES_DONE] != 2u) { printf("    !! frames_done %u\n", res[R_FRAMES_DONE]); b6++; }
        if (res[R_CHK] != chk0) { printf("    !! replica-0 checksum differs between VERIFY (%08x) and RUN (%08x)\n", chk0, res[R_CHK]); b6++; }
        int same = 0; for (int r = 1; r < N_REPL; r++) same += (res[R_CHK + r] == res[R_CHK]);
        if (N_REPL > 1 && same == N_REPL - 1 && W[W_COUNT] != 0) { printf("    !! all replica checksums identical (delay chain not effective?)\n"); b6++; }
        if (!b6) printf("    ok run        2 frames, %d replicas, chk[0]=%08x, %d/%d other replicas differ\n", N_REPL, res[R_CHK], N_REPL - 1 - same, N_REPL - 1);
        bad += b6;
        if (bad == 0) { frames_ok++; printf("  => PASS\n"); } else printf("  => FAIL (%d)\n", bad);
        total_bad += bad;
    }
    printf("[TB] %d/%d frames bit-exact; total mismatches %d -> %s\n", frames_ok, (int)names.size(), total_bad, total_bad ? "FAIL" : "PASS");
    return total_bad ? 1 : 0;
}
