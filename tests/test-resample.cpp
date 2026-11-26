// test-resample: the resampler interpolates a long source's tail as it does
// its head, and every phase between its table's rows (cadenza-audio #1310).
// Its read position was a float, which keeps no fraction past 2^23 samples
// (~190 s at 44.1 kHz): from there every output sample read the nearest input
// sample, a zero-order hold. And its table stopped at phase 255/256, so a
// fraction past it was held there: 22.05 kHz's 319/320, 11.025 kHz's 639/640.
// This resamples a 997 Hz sine to 48 kHz and holds the output to the analytic
// sine -- from 44.1 kHz near the start and past 2^23 input samples, and from
// 22.05 and 11.025 kHz near the start -- and a stereo resample to two mono
// ones. No model.

#include "audio-resample.h"
#include "test-util.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static const int    kRateOut   = 48000;
static const double kHz        = 997.0;
// The most a sample of the resampled sine may differ from the analytic one.
// The exact position and the full table leave ~5e-6 (the filter's passband,
// 256 interpolated phases); the float position left 3e-4 in the head and
// 4e-2 past 2^23 samples, and the held last phase 1.1e-4 at 22.05 kHz and
// 6.7e-4 at 11.025 kHz.
static const double kMostError = 5e-5;

static double tone(double t) {
    return 0.5 * sin(2.0 * M_PI * kHz * t);
}

static std::vector<float> tone_at(int rate, int n) {
    std::vector<float> v((size_t) n);
    for (int i = 0; i < n; i++) {
        v[(size_t) i] = (float) tone((double) i / rate);
    }
    return v;
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

// A sine of `n_in` samples at `rate` resampled to 48 kHz: its worst error over
// the second from 1 s in, and -- for a source past 2^23 samples -- over the
// output past them, to a filter's length short of the end (its edge repeats).
static void check_rate(int rate, int n_in) {
    const std::vector<float> in    = tone_at(rate, n_in);
    int                      n_out = 0;
    float *                  out   = audio_resample(in.data(), n_in, rate, kRateOut, 1, &n_out);
    CHECK(out != nullptr);
    if (!out) {
        return;
    }
    CHECK(n_out == (int) audio_resampled_samples(n_in, rate, kRateOut));
    const double head = worst(out, kRateOut, 2 * kRateOut);
    CHECK(head < kMostError);
    if (n_in > (1 << 23)) {
        const int tail_a = (int) ((long long) ((1 << 23) + rate / 2) * kRateOut / rate);
        const int tail_b = n_out - RESAMPLE_N_TAPS;
        CHECK(tail_a < tail_b);
        const double tail = worst(out, tail_a, tail_b);
        fprintf(stderr, "[Test] resample %d -> %d Hz: worst error %.2e in the head, %.2e past 2^23 samples\n", rate,
                kRateOut, head, tail);
        CHECK(tail < kMostError);
    } else {
        fprintf(stderr, "[Test] resample %d -> %d Hz: worst error %.2e in the head\n", rate, kRateOut, head);
    }
    free(out);
}

int main() {
    check_rate(44100, (1 << 23) + 2 * 44100);
    check_rate(22050, 3 * 22050);
    check_rate(11025, 3 * 11025);

    // A stereo resample is two mono ones: each channel by the same position.
    {
        const int          n    = 2 * 44100;
        std::vector<float> left = tone_at(44100, n), right = tone_at(22050, n), both(2 * (size_t) n);
        memcpy(both.data(), left.data(), (size_t) n * sizeof(float));
        memcpy(both.data() + n, right.data(), (size_t) n * sizeof(float));
        int     n_l = 0, n_r = 0, n_b = 0;
        float * l = audio_resample(left.data(), n, 44100, kRateOut, 1, &n_l);
        float * r = audio_resample(right.data(), n, 44100, kRateOut, 1, &n_r);
        float * b = audio_resample(both.data(), n, 44100, kRateOut, 2, &n_b);
        CHECK(l && r && b && n_l == n_b && n_r == n_b);
        if (l && r && b) {
            CHECK(memcmp(b, l, (size_t) n_b * sizeof(float)) == 0);
            CHECK(memcmp(b + n_b, r, (size_t) n_b * sizeof(float)) == 0);
        }
        free(l);
        free(r);
        free(b);
    }
    return test_summary("test-resample");
}
