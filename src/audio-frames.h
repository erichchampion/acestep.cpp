#pragma once
// audio-frames.h: the engine's audio clock (cadenza-audio #1093) -- the one
// sample rate every stage runs at, and the VAE's hop: the samples per latent
// frame, 25 of them a second. The conversions between samples, latent frames
// and seconds are written here once, not at each log line and buffer size.

#include <limits>

// Samples per second, per channel: what the VAE decodes to and encodes from.
inline constexpr int ACE_SAMPLE_RATE = 48000;

// Samples per latent frame: the VAE's down- and upsampling (2x4x4x6x10).
inline constexpr int ACE_LATENT_HOP = 1920;

// Latent frames per second (25, exactly).
inline constexpr float ACE_LATENT_FPS = (float) ACE_SAMPLE_RATE / ACE_LATENT_HOP;

static_assert(ACE_SAMPLE_RATE % ACE_LATENT_HOP == 0, "a whole number of latent frames a second");

// A latent frame count's length in seconds.
inline constexpr float latent_frames_sec(int frames) {
    return (float) frames * ACE_LATENT_HOP / ACE_SAMPLE_RATE;
}

// A sample count's (per channel) length in seconds.
inline constexpr float samples_sec(int samples) {
    return (float) samples / ACE_SAMPLE_RATE;
}

// The frames of slack a seconds-to-frame step gives a time meant to land on a
// frame start. A float k/25 s can sit just below it -- 2.12f is 2.1199999, and
// at 1440 s a float is off by up to ~0.0015 frames -- and a plain truncation
// then drops to frame k-1: 2991 of the 36000 starts to 1440 s did (#1129).
// 0.01 frames is 0.4 ms.
inline constexpr double ACE_FRAME_SLACK = 0.01;

// A frame count in double as an int, held to what one carries: a NaN is 0, a
// count past +-1e9 frames that bound -- so a request's repainting_end of 1e10 s
// is a far frame, not an undefined cast (#1129).
inline constexpr int frame_index(double frames) {
    if (frames != frames) {
        return 0;
    }
    if (frames < -1e9) {
        return -1000000000;
    }
    if (frames > 1e9) {
        return 1000000000;
    }
    return (int) frames;
}

// The latent frame a time in seconds falls in: truncated, a frame start
// counted as its own frame. A run's length, and a region's first frame (#1129).
inline constexpr int sec_latent_frame(float sec) {
    return frame_index((double) sec * ACE_LATENT_FPS + ACE_FRAME_SLACK);
}

// The frame a span ending at a time in seconds ends before: rounded up, so a
// region holds every frame it reaches into -- a frame end counted as the end,
// not one frame more. With sec_latent_frame for its start, a region's frames
// cover it whole: a sub-frame region, or one ending inside a pad rounded up to
// a frame, is not left short of its last frame (#1129).
inline constexpr int sec_latent_frame_end(float sec) {
    const double frames = (double) sec * ACE_LATENT_FPS - ACE_FRAME_SLACK;
    const int    whole  = frame_index(frames);
    return frames > whole ? whole + 1 : whole;
}

// The whole number of latent frames nearest a span in seconds: a pad's length.
inline constexpr int sec_latent_frames_nearest(float sec) {
    return frame_index((double) sec * ACE_LATENT_FPS + 0.5);
}

// The conversions, held at compile time: a second is 25 frames and 48000
// samples, and a frame start is its own frame -- 2.12 s frame 53, not 52.
static_assert(ACE_LATENT_FPS == 25.0f, "25 latent frames a second");
static_assert(latent_frames_sec(25) == 1.0f && latent_frames_sec(50) == 2.0f, "frames to seconds");
static_assert(samples_sec(ACE_SAMPLE_RATE) == 1.0f, "samples to seconds");
static_assert(sec_latent_frame(1.0f) == 25 && sec_latent_frame(0.039f) == 0 && sec_latent_frame(0.04f) == 1 &&
                  sec_latent_frame(2.12f) == 53 && sec_latent_frame(2.6f) == 65 && sec_latent_frame(1439.96f) == 35999,
              "seconds to the frame they fall in");
static_assert(sec_latent_frame_end(20.0f) == 500 && sec_latent_frame_end(10.03f) == 251 &&
                  sec_latent_frame_end(0.03f) == 1 && sec_latent_frame_end(0.0f) == 0 &&
                  sec_latent_frame_end(-0.5f) == -12,
              "a span's end, rounded up to the frame it reaches into");
static_assert(sec_latent_frames_nearest(0.03f) == 1 && sec_latent_frames_nearest(0.01f) == 0, "a pad's frames");
static_assert(frame_index(1e12) == 1000000000 && frame_index(-1e12) == -1000000000 &&
                  frame_index(std::numeric_limits<double>::quiet_NaN()) == 0,
              "a frame count held to an int");
