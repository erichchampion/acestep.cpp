#pragma once
// pipeline-synth-impl.h: private state of the synthesis pipeline
//
// Holds the types that the public API exposes as opaque handles (AceSynth,
// AceSynthJob) plus SynthState, the transient state that flows between the
// primitive ops during a single job.
//
// Included by the two implementation files:
//   pipeline-synth.cpp      orchestrator: load, phases, free
//   pipeline-synth-ops.cpp  primitives:   encode, context, noise, dit, vae

#include "audio-frames.h"
#include "debug.h"
#include "model-store.h"
#include "pipeline-synth.h"
#include "request.h"
#include "timer.h"

#include <algorithm>
#include <atomic>
#include <memory>
#include <string>
#include <vector>

// AceSynth is a thin handle over a ModelStore. It carries the ModelKeys the
// ops use to acquire modules and a pointer to the DiT metadata owned by the
// store (silence_latent, null_cond, config, is_turbo). No GPU module is ever
// owned here: each op performs its own require/release against the store.
struct AceSynth {
    ModelStore *   store;
    AceSynthParams params;

    // References: the owner's, plus one per job decoding through this
    // context (ace_synth_retain). The last one frees it: a reload must not
    // free the context under a parked take.
    std::atomic<int> refs{ 1 };
    // Whether the owner's store_hold_key on every key is still in force.
    std::atomic<bool> keys_held{ false };

    // The paths `params` points at, owned here: a job can keep this context
    // past the caller's strings (ace_synth_retain).
    std::string own_text_encoder, own_dit, own_vae, own_adapter, own_dump_dir;

    // CPU metadata, shared with the store's entry, and `meta` points at it.
    // Gives ops access to silence_full, null_cond_cpu, is_turbo and the DiT
    // config without loading the DiT. Shared rather than borrowed so the
    // store can free its entry (the DiT file was replaced) while this
    // context lives.
    std::shared_ptr<const DiTMeta> meta_own;
    const DiTMeta *                meta;

    // Derived constants mirrored for inline use in ops.
    int Oc;      // out_channels (64)
    int ctx_ch;  // in_channels - Oc (128)

    // ModelKeys for the seven GPU modules the pipeline touches, each stamped
    // at load with the identity of the file it names (ModelKey::file_id):
    // every op asks for the bytes this context's metadata came from.
    ModelKey text_enc_key;   // Qwen3 text encoder, from text_encoder_path
    ModelKey cond_enc_key;   // condition encoder, from dit_path
    ModelKey fsq_tok_key;    // FSQ tokenizer, from dit_path
    ModelKey fsq_detok_key;  // FSQ detokenizer, from dit_path
    ModelKey dit_key;        // DiT, from dit_path (+ adapter_path, adapter_scale)
    ModelKey vae_enc_key;    // VAE encoder, from vae_path
    ModelKey vae_dec_key;    // VAE decoder, from vae_path

    // Where the running ace_synth_job_run_dit moves the VAE encodes it made
    // from samples when it fails: its caller's AceSynthEncodes, or null when
    // the caller asked for none. Set only for the call's duration
    // (cadenza-audio #975): nothing outlives the run on the synth. Like the
    // run, not thread-safe.
    AceSynthEncodes * encode_sink = nullptr;
};

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

// Transient state for a single job, shared by reference across the primitive
// ops that make up phase 1 and phase 2.
struct SynthState {
    // model dimensions (from ctx)
    int Oc;
    int ctx_ch;

    // src audio encoding
    bool               have_cover;
    std::vector<float> cover_latents;        // [T_cover * 64] time-major VAE latents
    std::vector<float>
        fsq_clean_latents;  // the clean VAE latents, moved aside by the FSQ roundtrip (clean_src_latents)
    // cover_latents came from encoding src audio that the run does not read
    // again, so a failed run hands out its clean_src_latents to be run again
    // (cadenza-audio #875, #1035).
    bool               src_encoded = false;
    int                T_cover;

    // outpainting: silence padding for repaint beyond source bounds
    float              left_pad_sec;    // seconds of silence prepended (coordinate shift)
    std::vector<float> padded_src;      // interleaved stereo buffer with silence padding (audio path)
    std::vector<float> padded_latents;  // [T_padded, 64] f32 with silence_full padding (latent path)

    // mode flags
    bool  is_repaint;
    bool  is_lego_region;
    float rs;
    float re;
    bool  use_source_context;
    bool  have_codes;
    int   max_codes_len;

    // audio_codes parsed once per request in ops_resolve_params, reused by
    // ops_build_context. Indexed [0..batch_n). Empty entries mean no codes.
    std::vector<std::vector<int>> per_codes;

    // shared params (from reqs[0])
    AceRequest rr;
    float      duration;
    int        num_steps;
    float      guidance_scale;
    float      shift;

    // diffusion schedule
    std::vector<float> schedule;

    // per-batch seeds for the SDE solver (Philox renoising at each step)
    std::vector<int64_t> seeds;

    // latent dimensions
    int T;
    int S;
    int enc_S;

    // region
    int repaint_t0;
    int repaint_t1;

    // DiT instruction
    std::string instruction_str;  // main DiT instruction (set by orchestrator)

    // conditioning tensors
    std::vector<float> timbre_feats;
    int                S_ref_timbre;  // actual timbre sequence length (from VAE encode)
    bool               timbre_encoded = false;  // timbre_feats came from encoding ref_audio

    // text encoding (per-batch)
    std::vector<std::vector<float>> per_enc;
    std::vector<int>                per_enc_S;
    std::vector<std::vector<float>> per_enc_nc;
    std::vector<int>                per_enc_S_nc;
    bool                            need_enc_switch;

    // stacked encoder hidden states
    int                max_enc_S;
    std::vector<float> enc_hidden;
    std::vector<float> enc_hidden_nc;
    std::vector<int>   per_enc_S_nc_final;
    std::vector<float> null_cond_vec;

    // DiT context
    std::vector<float> context;
    std::vector<float> context_silence;
    int                cover_steps;

    // noise + output
    std::vector<float> noise;
    std::vector<float> output;
    std::vector<int>   per_S;

    // debug / timing
    DebugDumper dbg;
    Timer       timer;
};

// Whether the run splices its source back in after the decode -- a repaint,
// or a lego region -- so its encode is not what a run again needs: one rule
// for the splices (ops_vae_decode_track, ops_vae_decode), the failed run's
// hand-out (cadenza-audio #1041), and the region's frames, mask and context
// (#1130). The task sets both flags before it encodes.
inline bool splices_source_after_decode(const SynthState & s) {
    return s.is_repaint || s.is_lego_region;
}

// Whether a lego's bounds ask for a region, read as a repaint reads them
// (cadenza-audio #1134): an end of its own, after the start; or the -1
// sentinel (any negative end, as asked_region_end reads it) -- the source's
// end, or for a start before the source the source's start -- with a start
// other than 0 -- by more than the frame slack, so float noise about 0 is 0,
// not an outpaint of no frame before the source. The defaults, 0 and -1, ask
// the whole song, which a lego runs as no region: a region of all of it would
// be the same song, spliced.
inline bool lego_asks_region(float rs, float re) {
    return re >= 0.0f ? re > rs : sec_latent_frame_end(rs < 0.0f ? -rs : rs) > 0;
}

// The region's latent frames, [repaint_t0, repaint_t1) with
// 0 <= repaint_t0 <= repaint_t1 <= T -- its first frame the one its start falls
// in, its end rounded up to cover the last frame it reaches into (#1129); an
// inverted or wholly-outside region is empty, never a negative or reversed
// range -- for
// a run that splices its source after the decode -- a repaint or a lego region,
// by the one rule above -- and none for any other. A lego region once got none
// (only a repaint was asked), so it masked no frame and gave back its source
// (cadenza-audio #1130). Reads s.rs, s.re (seconds, shifted into the padded
// frame) and s.T.
inline void resolve_region_frames(SynthState & s) {
    s.repaint_t0 = 0;
    s.repaint_t1 = 0;
    if (!splices_source_after_decode(s)) {
        return;
    }
    s.repaint_t0 = std::clamp(sec_latent_frame(s.rs), 0, s.T);
    s.repaint_t1 = std::clamp(sec_latent_frame_end(s.re), s.repaint_t0, s.T);
}

// The source's clean VAE latents: fsq_clean_latents holds them once the FSQ
// roundtrip has moved them there (its degraded latents in cover_latents, #1042),
// cover_latents before (or without one). The noise blend and a failed run's
// hand-out read the same (cadenza-audio #1035). A failed run's hand-out reads
// it after run_tail, so it holds only while nothing there changes cover_latents
// or timbre_feats in place.
inline std::vector<float> & clean_src_latents(SynthState & s) {
    return s.fsq_clean_latents.empty() ? s.cover_latents : s.fsq_clean_latents;
}

// Job bridges phase 1 (DiT) and phase 2 (VAE). The latents live in state.output
// as time-major [batch_n, T, Oc=64] f32 until ace_synth_job_run_vae consumes
// them.
struct AceSynthJob {
    SynthState state;
    int        batch_n;
};
