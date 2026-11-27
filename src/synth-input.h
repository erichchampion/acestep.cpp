#pragma once
// synth-input.h: a run's source as its caller gave it -- audio, latents, or
// both -- and the one reading of "given" every pipeline asks: the synth's
// tasks, the tools' checks and understand alike (cadenza-audio #1131). Beside
// audio-frames.h, whose lengths it reads, so no pipeline's own header is
// needed for it.

#include "audio-frames.h"

// One of a run's inputs as its caller gave it: audio (len samples a channel)
// or latents (T_latent frames), either, both or neither (#1057). The ops
// that encode a source or a reference take it whole (#1093).
struct SynthInput {
    const float * audio    = nullptr;
    int           len      = 0;
    const float * latents  = nullptr;
    int           T_latent = 0;

    bool has_audio() const { return audio && len > 0; }

    bool has_latents() const { return latents && T_latent > 0; }

    bool given() const { return has_audio() || has_latents(); }

    // Its length in seconds: the latents' when given, else the audio's.
    float duration_sec() const { return has_latents() ? latent_frames_sec(T_latent) : samples_sec(len); }

    // Its length in the frames an encode keeps: the latents' when given,
    // else the audio's whole frames -- the encoder keeps no part one.
    long long frames() const {
        return has_latents() ? (long long) T_latent : has_audio() ? (long long) samples_frames_floor(len) : 0;
    }
};
