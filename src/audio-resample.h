#pragma once
// audio-resample.h: polyphase sample rate conversion via pre-computed
// Kaiser-windowed sinc filter. Only the engine's audio clock (audio-frames.h).
// Part of acestep.cpp. MIT license.

#include "audio-frames.h"

#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#ifndef M_PI
#    define M_PI 3.14159265358979323846
#endif

// Polyphase resampler with pre-computed Kaiser-windowed sinc filter.
//
// Instead of computing bessel_i0 + sqrt + sin per output sample (O(N_out * N_taps)),
// we build a polyphase table once: table[phase][tap] = sinc(d) * kaiser(d).
// The hot loop is then just a table lookup + dot product -- no transcendentals.
//
// Table size: 256 phases * 64 taps * 4 bytes = 64 KB (fits L1 cache).

#define RESAMPLE_N_TAPS   64
#define RESAMPLE_N_PHASES 256
#define RESAMPLE_HALF_LEN (RESAMPLE_N_TAPS / 2)

// The samples per channel `n_in` at `sr_in` resamples to at `sr_out`, in 64
// bits: the one formula audio_resample sizes its output by, and every reader
// bounds a resample with before it runs (cadenza-audio #1266). In double, as
// the resampler always measured it -- an integer n_in * sr_out / sr_in is one
// sample longer for some lengths at 5512 Hz, and would move the samples.
inline long long audio_resampled_samples(long long n_in, int sr_in, int sr_out) {
    if (sr_in <= 0 || sr_out <= 0 || n_in <= 0) {
        return 0;
    }
    const double ratio = (double) sr_out / (double) sr_in;
    return (long long) ((double) n_in * ratio);
}

// Bessel I0 via Taylor series. Only used during table construction
// (called RESAMPLE_N_PHASES * RESAMPLE_N_TAPS = 16384 times total, not millions).
static double audio_resample_bessel_i0(double x) {
    double sum  = 1.0;
    double term = 1.0;
    double y    = x * x * 0.25;
    for (int k = 1; k < 30; k++) {
        term *= y / ((double) k * (double) k);
        sum += term;
        if (term < sum * 1e-15) {
            break;
        }
    }
    return sum;
}

// Build polyphase filter bank.
//
// For output sample i at position center = i / ratio in input space:
//   center_int = floor(center), frac = center - center_int
//   phase = frac * N_PHASES
//   base  = center_int - HALF_LEN + 1
//   for tap 0..N_TAPS-1: h = table[phase][tap], input = src[base + tap]
//
// d (distance from center to tap) = frac + HALF_LEN - 1 - tap
// This depends only on phase and tap, so we can pre-compute everything.
static void audio_resample_build_table(float table[][RESAMPLE_N_TAPS], double fc, double beta) {
    double inv_i0b = 1.0 / audio_resample_bessel_i0(beta);

    for (int p = 0; p < RESAMPLE_N_PHASES; p++) {
        double frac = (double) p / (double) RESAMPLE_N_PHASES;

        for (int tap = 0; tap < RESAMPLE_N_TAPS; tap++) {
            double d = frac + (double) (RESAMPLE_HALF_LEN - 1 - tap);

            // windowed sinc
            double sinc_val;
            if (fabs(d) < 1e-9) {
                sinc_val = 2.0 * fc;
            } else {
                sinc_val = sin(2.0 * M_PI * fc * d) / (M_PI * d);
            }

            // Kaiser window
            double t = d / (double) RESAMPLE_HALF_LEN;
            double win;
            if (t < -1.0 || t > 1.0) {
                win = 0.0;
            } else {
                win = audio_resample_bessel_i0(beta * sqrt(1.0 - t * t)) * inv_i0b;
            }

            table[p][tap] = (float) (sinc_val * win);
        }
    }
}

// Resample a planar float audio buffer from sr_in to sr_out.
//
// in:     planar float [ch0: n_in samples][ch1: n_in samples][...]
// n_in:   samples per channel in the input
// sr_in:  input sample rate (e.g. 44100)
// sr_out: output sample rate (e.g. 48000)
// nch:    number of channels (1 or 2)
// n_out:  receives the output sample count per channel
//
// Returns a malloc'd planar buffer [ch0: n_out][ch1: n_out][...].
// Caller must free() the result.
// Returns NULL on error (bad sr, alloc failure).
static float * audio_resample(const float * in, int n_in, int sr_in, int sr_out, int nch, int * n_out) {
    if (!in || n_in <= 0 || sr_in <= 0 || sr_out <= 0 || nch <= 0) {
        *n_out = 0;
        return NULL;
    }

    // passthrough: no conversion needed
    if (sr_in == sr_out) {
        size_t  sz  = (size_t) n_in * (size_t) nch * sizeof(float);
        float * out = (float *) malloc(sz);
        if (!out) {
            fprintf(stderr, "[Audio-Resample] OOM passthrough buffer (%zu bytes)\n", sz);
            *n_out = 0;
            return NULL;
        }
        *n_out = n_in;
        memcpy(out, in, sz);
        return out;
    }

    double          ratio    = (double) sr_out / (double) sr_in;
    // The output's length, bounded before it is an int: past
    // ACE_MAX_AUDIO_SAMPLES is refused, not cast (cadenza-audio #1266) -- the
    // readers bound it before this, and any other caller gets here.
    const long long n_out_ll = audio_resampled_samples(n_in, sr_in, sr_out);
    if (n_out_ll > ACE_MAX_AUDIO_SAMPLES) {
        fprintf(stderr, "[Audio-Resample] Refused: %d samples at %d Hz is past %d samples at %d Hz\n", n_in, sr_in,
                ACE_MAX_AUDIO_SAMPLES, sr_out);
        *n_out = 0;
        return NULL;
    }
    *n_out = (int) n_out_ll;
    if (*n_out <= 0) {
        *n_out = 0;
        return NULL;
    }

    float * out = (float *) malloc((size_t) (*n_out) * (size_t) nch * sizeof(float));
    if (!out) {
        *n_out = 0;
        return NULL;
    }

    // Kaiser window parameter (beta=9.0 gives ~80 dB stopband)
    double beta = 9.0;

    // cutoff: lowpass at the lower of the two rates to prevent aliasing
    double fc = 0.5 * ((ratio < 1.0) ? ratio : 1.0);

    // build polyphase filter table (one-time cost: ~16K coeff, microseconds)
    float (*table)[RESAMPLE_N_TAPS] =
        (float (*)[RESAMPLE_N_TAPS]) malloc(RESAMPLE_N_PHASES * RESAMPLE_N_TAPS * sizeof(float));
    if (!table) {
        free(out);
        *n_out = 0;
        return NULL;
    }
    audio_resample_build_table(table, fc, beta);

    for (int ch = 0; ch < nch; ch++) {
        const float * src = in + ch * n_in;
        float *       dst = out + ch * (*n_out);

        for (int i = 0; i < *n_out; i++) {
            // Position in input sample space, i * sr_in / sr_out, as an exact
            // rational: its whole part and its remainder in 64-bit integers,
            // so the fraction is as fine at the end of a 600 s source as at
            // its start. A float position lost its fraction from 2^23 samples
            // on (~190 s at 44.1 kHz), and the tail resampled nearest-sample
            // (cadenza-audio #1310).
            const long long num      = (long long) i * sr_in;
            const int       center_i = (int) (num / sr_out);
            const double    frac     = (double) (num % sr_out) / (double) sr_out;

            // phase index + interpolation fraction between adjacent phases
            const double phase_d   = frac * (double) RESAMPLE_N_PHASES;
            int          phase     = (int) phase_d;
            float        phase_mix = (float) (phase_d - (double) phase);
            if (phase >= RESAMPLE_N_PHASES - 1) {
                phase     = RESAMPLE_N_PHASES - 2;
                phase_mix = 1.0f;
            }

            int base = center_i - RESAMPLE_HALF_LEN + 1;

            // dot product with linear interpolation between adjacent phases
            // (avoids quantization artifacts from 256 discrete phases)
            const float * h0 = table[phase];
            const float * h1 = table[phase + 1];

            float sum = 0.0f;
            float wgt = 0.0f;

            for (int tap = 0; tap < RESAMPLE_N_TAPS; tap++) {
                float h = h0[tap] + phase_mix * (h1[tap] - h0[tap]);

                // clamp to input bounds (repeat edge samples)
                int idx = base + tap;
                if (idx < 0) {
                    idx = 0;
                }
                if (idx >= n_in) {
                    idx = n_in - 1;
                }

                sum += src[idx] * h;
                wgt += h;
            }

            // normalize to compensate for edge effects
            dst[i] = (wgt > 1e-12f) ? sum / wgt : 0.0f;
        }
    }

    free(table);
    return out;
}
