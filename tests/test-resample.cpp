// test-resample: the resampler interpolates a long source's tail as it does
// its head (cadenza-audio #1310). Its read position was a float, which keeps
// no fraction past 2^23 samples (~190 s at 44.1 kHz): from there every output
// sample read the nearest input sample, a zero-order hold. This resamples a
// 997 Hz sine of 2^23 samples and two seconds more from 44.1 kHz to 48 kHz,
// and holds the output to the analytic sine at 48 kHz both near the start and
// past 2^23 input samples, to the same small error. No model.

#include "audio-resample.h"
#include "test-util.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static const int    kRateIn    = 44100;
static const int    kRateOut   = 48000;
static const double kHz        = 997.0;
// The most a sample of the resampled sine may differ from the analytic one.
// The exact position leaves ~5e-6 (the filter's passband, 256 interpolated
// phases); the float position left 3e-4 in the head and 4e-2 past 2^23
// samples, where it read the nearest sample.
static const double kMostError = 1e-4;

static double tone(double t) {
    return 0.5 * sin(2.0 * M_PI * kHz * t);
}

// The largest difference from the analytic sine over output samples [a, b).
static double worst(const float * out, int a, int b) {
    double w = 0.0;
    for (int j = a; j < b; j++) {
        const double d = fabs((double) out[j] - tone((double) j / kRateOut));
        if (d > w) {
            w = d;
        }
    }
    return w;
}

int main() {
    const int          n_in = (1 << 23) + 2 * kRateIn;
    std::vector<float> in((size_t) n_in);
    for (int i = 0; i < n_in; i++) {
        in[(size_t) i] = (float) tone((double) i / kRateIn);
    }

    int     n_out = 0;
    float * out   = audio_resample(in.data(), n_in, kRateIn, kRateOut, 1, &n_out);
    CHECK(out != nullptr);
    if (!out) {
        return test_summary("test-resample");
    }
    CHECK(n_out == (int) audio_resampled_samples(n_in, kRateIn, kRateOut));

    // A second of output from 1 s in, and a second from past 2^23 input
    // samples to a filter's length short of the end (its edge repeats).
    const int head_a = kRateOut, head_b = 2 * kRateOut;
    const int tail_a = (int) ((long long) ((1 << 23) + kRateIn / 2) * kRateOut / kRateIn);
    const int tail_b = n_out - RESAMPLE_N_TAPS;
    CHECK(tail_a < tail_b);
    const double head = worst(out, head_a, head_b);
    const double tail = worst(out, tail_a, tail_b);
    fprintf(stderr, "[Test] resample 44.1 -> 48 kHz: worst error %.2e in the head, %.2e past 2^23 samples\n", head,
            tail);
    CHECK(head < kMostError);
    CHECK(tail < kMostError);

    free(out);
    return test_summary("test-resample");
}
