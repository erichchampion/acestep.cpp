// test-audio-layout: the channel-layout contract at the pipeline boundary (#28).
//
// The VAE encoder consumes time-major INTERLEAVED stereo ([t0L, t0R, t1L,
// t1R, ...] -- vae_enc_compute indexes audio[t * 2 + c]); the decoder
// produces PLANAR stereo ([L: T][R: T] -- its output is written channel at
// audio_out + ch * T). Opposite layouts across one pipeline, documented in
// four places (pipeline-synth.h's input docs and AceAudio, audio-io.h's
// blanket planar claim, vae-enc.h's [T, 2] comments), and exactly the
// confusion that produced the previous attempt's "L/R correlation = 0.057"
// noise-that-looked-like-audio.
//
// This test pins the contract with a round trip whose input makes the
// channels maximally distinguishable: L = 220 Hz, R = 880 Hz, fed
// interleaved through encode then decode. The planar output must keep the
// channels distinct -- each output channel correlating with its own input
// clearly above its correlation with the other, and the two output channels
// differing substantially from each other. A swapped or smeared layout
// collapses those separations.
//
//   ./test-audio-layout --vae <gguf>
//
// Needs only the VAE GGUF (the same file carries the encoder and decoder);
// no text encoder, no DiT.

#include "audio-io.h"
#include "vae-enc.h"
#include "vae.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static int failures = 0;

#define CHECK(cond)                                                                      \
    do {                                                                                 \
        if (!(cond)) {                                                                   \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);              \
            failures++;                                                                  \
        }                                                                                \
    } while (0)

// Pearson correlation of two same-length sample ranges. The layout metric:
// a channel that survives the round trip correlates with its own source;
// under a layout confusion it correlates with everything equally (badly).
static double correlation(const float * a, const float * b, int n) {
    double sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0;
    for (int i = 0; i < n; i++) {
        sa += a[i];
        sb += b[i];
        saa += (double) a[i] * a[i];
        sbb += (double) b[i] * b[i];
        sab += (double) a[i] * b[i];
    }
    double cov = sab / n - (sa / n) * (sb / n);
    double va  = saa / n - (sa / n) * (sa / n);
    double vb  = sbb / n - (sb / n) * (sb / n);
    if (va <= 0 || vb <= 0) {
        return 0;
    }
    return cov / std::sqrt(va * vb);
}

static void usage(const char * prog) {
    fprintf(stderr, "Usage: %s --vae <gguf>\n", prog);
}

int main(int argc, char ** argv) {
    const char * vae_path = nullptr;

    if (argc < 2) {
        usage(argv[0]);
        return 1;
    }
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--vae") && i + 1 < argc) {
            vae_path = argv[++i];
        } else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) {
            usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            usage(argv[0]);
            return 1;
        }
    }
    if (!vae_path) {
        fprintf(stderr, "[CLI] ERROR: --vae required\n");
        usage(argv[0]);
        return 1;
    }

    // One second of stereo at 48 kHz: L = 220 Hz, R = 880 Hz, full scale but
    // clipped to a safe peak. Distinct frequencies, because distinctness is
    // what the layout contract has to preserve.
    const int T      = 48000;
    const int max_T  = (T / 1920) + 64;
    std::vector<float> planar_in(2 * (size_t) T);
    for (int t = 0; t < T; t++) {
        float        ts   = (float) t / 48000.0f;
        planar_in[(size_t) t]      = 0.8f * std::sinf(2.0f * 3.14159265f * 220.0f * ts);
        planar_in[(size_t) T + t]  = 0.8f * std::sinf(2.0f * 3.14159265f * 880.0f * ts);
    }

    // The documented INPUT layout: time-major interleaved, exactly what
    // ace-server feeds and what pipeline-synth.h promises. Built here from
    // the planar test signal by the same interleave ace-synth.cpp performs.
    std::vector<float> interleaved(2 * (size_t) T);
    for (int t = 0; t < T; t++) {
        interleaved[(size_t) t * 2 + 0] = planar_in[(size_t) t];
        interleaved[(size_t) t * 2 + 1] = planar_in[(size_t) T + t];
    }

    // Encode: interleaved in, latents out.
    VAEEncoder * enc = new VAEEncoder();
    vae_enc_load(enc, vae_path);
    std::vector<float> latents((size_t) max_T * 64);
    int T_latent = vae_enc_encode_tiled(enc, interleaved.data(), T, latents.data(), max_T, 256, 64);
    CHECK(T_latent > 0);
    fprintf(stderr, "[Test] Encoded: T_latent=%d\n", T_latent);

    // Decode: latents in, PLANAR audio out ([L: T][R: T] flat).
    VAEGGML * dec = new VAEGGML();
    vae_ggml_load(dec, vae_path);
    int max_T_audio = T_latent * 1920;
    std::vector<float> audio_out(2 * (size_t) max_T_audio);
    int T_out = vae_ggml_decode_tiled(dec, latents.data(), T_latent, audio_out.data(), max_T_audio, 256, 64);
    CHECK(T_out > 0);
    fprintf(stderr, "[Test] Decoded: T_audio=%d (%.2fs)\n", T_out, (float) T_out / 48000.0f);

    // The output is planar: left occupies [0, T), right [T, 2T).
    const float * out_l = audio_out.data();
    const float * out_r = audio_out.data() + T_out;
    const float * in_l  = planar_in.data();
    const float * in_r  = planar_in.data() + T;

    int n = T_out < T ? T_out : T;

    // The channel distinctness the whole contract exists to keep.
    double mean_diff = 0;
    for (int i = 0; i < n; i++) {
        mean_diff += std::fabs(out_l[i] - out_r[i]);
    }
    mean_diff /= n;
    fprintf(stderr, "[Test] mean |outL - outR| = %.4f (a channel swap or a planar/interleaved smear drives this to ~0)\n",
            mean_diff);
    CHECK(mean_diff > 0.05f);

    double r_ll = correlation(out_l, in_l, n);
    double r_rr = correlation(out_r, in_r, n);
    double r_lr = correlation(out_l, in_r, n);
    double r_rl = correlation(out_r, in_l, n);
    fprintf(stderr, "[Test] corr: L~L=%.3f R~R=%.3f (own) | L~R=%.3f R~L=%.3f (cross)\n", r_ll, r_rr, r_lr, r_rl);

    // Own-channel identity survives the lossy round trip; cross-correlation
    // must be clearly lower. (The failed attempt measured 0.057 here.)
    CHECK(r_ll > 0.3);
    CHECK(r_rr > 0.3);
    CHECK(r_ll > r_lr + 0.1);
    CHECK(r_rr > r_rl + 0.1);

    vae_enc_free(enc);
    vae_ggml_free(dec);

    if (failures) {
        fprintf(stderr, "test-audio-layout: %d check(s) FAILED\n", failures);
        return 1;
    }
    fprintf(stderr, "test-audio-layout: all checks passed\n");
    return 0;
}
