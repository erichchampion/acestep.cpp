#pragma once
// audio-frames.h: the engine's audio clock (cadenza-audio #1093) -- the one
// sample rate every stage runs at, and the VAE's hop: the samples per latent
// frame, 25 of them a second. The conversions between samples, latent frames
// and seconds are written here once, not at each log line and buffer size.

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

// The latent frame a time in seconds falls in: truncated, a frame start
// counted as its own frame. The one seconds-to-frame step for a region's
// bounds and a run's length alike (#1129).
inline constexpr int sec_latent_frame(float sec) {
    return (int) ((double) sec * ACE_LATENT_FPS + ACE_FRAME_SLACK);
}

// The whole number of latent frames nearest a span in seconds: a pad's length.
inline constexpr int sec_latent_frames_nearest(float sec) {
    return (int) (sec * ACE_LATENT_FPS + 0.5f);
}

// The conversions, held at compile time: a second is 25 frames and 48000
// samples, and a frame start is its own frame -- 2.12 s frame 53, not 52.
static_assert(ACE_LATENT_FPS == 25.0f, "25 latent frames a second");
static_assert(latent_frames_sec(25) == 1.0f && latent_frames_sec(50) == 2.0f, "frames to seconds");
static_assert(samples_sec(ACE_SAMPLE_RATE) == 1.0f, "samples to seconds");
static_assert(sec_latent_frame(1.0f) == 25 && sec_latent_frame(0.039f) == 0 && sec_latent_frame(0.04f) == 1 &&
                  sec_latent_frame(2.12f) == 53 && sec_latent_frame(2.6f) == 65 && sec_latent_frame(1439.96f) == 35999,
              "seconds to the frame they fall in");
static_assert(sec_latent_frames_nearest(0.03f) == 1 && sec_latent_frames_nearest(0.01f) == 0, "a pad's frames");
