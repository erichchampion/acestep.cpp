// test-vae-tiling: the VAE's tiles join bit-exactly at the default overlap
// (cadenza-audio #91, #684). A tile decoded -- or encoded -- with at least
// the network's receptive field of context either side computes its kept
// range from exactly the inputs the untiled pass uses, so tiled and untiled
// agree to the bit. The synth default overlap (16) relies on that for the
// decoder (measured: exact at 9 or more) and for the encoder that reads a
// cover's or repaint's source and the reference audio. This checks both at
// 16 against an untiled pass, at a small chunk so there are many joins, and
// reports the smallest exact overlap of each.
//
//   ./test-vae-tiling --vae <vae gguf>

#include "vae-enc.h"
#include "vae.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "test-util.h"

static const int kChunk    = 128;  // latent frames: many joins in 30 s
static const int kDefault  = 16;   // ace_synth_default_params' vae_overlap

int main(int argc, char ** argv) {
    const char * vae_path = nullptr;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--vae") && i + 1 < argc) {
            vae_path = argv[++i];
        }
    }
    if (!vae_path) {
        fprintf(stderr, "Usage: %s --vae <gguf>\n", argv[0]);
        return 2;
    }

    // 30 s of stereo with structure at every scale: a few tones, a slow sweep
    // and seeded noise (an LCG, so every run encodes the same samples).
    const int T = 30 * 48000;
    std::vector<float> audio(2 * (size_t) T);
    uint32_t           lcg = 12345;
    for (int t = 0; t < T; t++) {
        float ts = (float) t / 48000.0f;
        lcg      = lcg * 1664525u + 1013904223u;
        float n  = ((float) (lcg >> 8) / (float) (1u << 24) - 0.5f) * 0.1f;
        float s  = 0.3f * sinf(2.0f * 3.14159265f * (110.0f + 20.0f * ts) * ts) +
                  0.2f * sinf(2.0f * 3.14159265f * 523.25f * ts);
        audio[(size_t) t * 2 + 0] = s + n;
        audio[(size_t) t * 2 + 1] = 0.7f * s - n;
    }

    // Encoder: untiled (a chunk past the whole input) against tiled.
    VAEEncoder * enc = new VAEEncoder();
    vae_enc_load(enc, vae_path);
    const int          max_T = T / 1920 + 64;
    std::vector<float> ref((size_t) max_T * 64), tiled((size_t) max_T * 64);
    int                T_ref = vae_enc_encode_tiled(enc, audio.data(), T, ref.data(), max_T, 1 << 16, 0);
    CHECK(T_ref > 0);
    auto enc_exact = [&](int overlap) {
        std::fill(tiled.begin(), tiled.end(), 0.0f);
        int got = vae_enc_encode_tiled(enc, audio.data(), T, tiled.data(), max_T, kChunk, overlap);
        return got == T_ref && memcmp(tiled.data(), ref.data(), (size_t) T_ref * 64 * sizeof(float)) == 0;
    };
    CHECK(enc_exact(kDefault));
    int enc_min = -1;
    for (int o = kDefault; o >= 0; o--) {
        if (!enc_exact(o)) {
            break;
        }
        enc_min = o;
    }
    fprintf(stderr, "[Test] encoder: tiled at chunk %d is bit-exact down to overlap %d\n", kChunk, enc_min);

    // Decoder: the untiled latents back to audio, untiled against tiled.
    VAEGGML * dec = new VAEGGML();
    vae_ggml_load(dec, vae_path);
    const int          max_A = T_ref * 1920;
    std::vector<float> aref(2 * (size_t) max_A), atiled(2 * (size_t) max_A);
    int                A_ref = vae_ggml_decode_tiled(dec, ref.data(), T_ref, aref.data(), max_A, 1 << 16, 0);
    CHECK(A_ref > 0);
    auto dec_exact = [&](int overlap) {
        std::fill(atiled.begin(), atiled.end(), 0.0f);
        int got = vae_ggml_decode_tiled(dec, ref.data(), T_ref, atiled.data(), max_A, kChunk, overlap);
        return got == A_ref && memcmp(atiled.data(), aref.data(), 2 * (size_t) A_ref * sizeof(float)) == 0;
    };
    CHECK(dec_exact(kDefault));
    int dec_min = -1;
    for (int o = kDefault; o >= 0; o--) {
        if (!dec_exact(o)) {
            break;
        }
        dec_min = o;
    }
    fprintf(stderr, "[Test] decoder: tiled at chunk %d is bit-exact down to overlap %d\n", kChunk, dec_min);

    vae_enc_free(enc);
    vae_ggml_free(dec);
    return test_summary("test-vae-tiling");
}
