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
#include "synth-input.h"
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
    // The take's length in samples, for a run whose source is spliced back
    // after its decode as audio: the source with its pads, as asked -- not
    // the whole frames the decode gives, nor the pad's run-on to a frame
    // (cadenza-audio #1142). 0 for any other run.
    int                take_samples = 0;
    // The source's latents decoded about a latent region's edge that falls
    // inside a frame -- [0] its start's, [1] its end's -- for the crossfade
    // back to the source at the edge's own sample (cadenza-audio #1140):
    // planar, edge_src_T samples from sample edge_src_at. Decoded once, for
    // every take; edge_src_at -1 until then, -2 for a decode that failed
    // (not asked again). edge_src_frame: the frame each is about.
    std::vector<float> edge_src[2];
    int                edge_src_at[2]    = { -1, -1 };
    int                edge_src_T[2]     = { 0, 0 };
    int                edge_src_frame[2] = { -1, -1 };

    // mode flags
    bool             is_repaint;
    bool             is_lego_region;
    float            rs;
    float            re;
    bool             use_source_context;
    // The run's task, as the task table holds it (set by run_task): what an
    // op asks of the task -- whether it reads the cover knobs (a cover's
    // alone), the region, the track -- rather than a flag copied for each
    // (cadenza-audio #1187). The ops that read the knobs ask its
    // cover_knobs, not use_source_context, which a lego, an extract, a
    // complete and a text2music from codes set too (#1135, #1154).
    AceSynthTaskInfo task{};
    bool             have_codes;
    int              max_codes_len;

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
    // ops_encode_text's decision (cover_switches_context); false until it
    // runs, so a run that fails before the encode records no switch (#1187).
    bool                            need_enc_switch = false;

    // Why the run refused its request partway, before its job ran: a static
    // reason run_task hands out through ace_synth_job_run_dit's `refused`
    // (cadenza-audio #1193). NULL for a run that failed otherwise.
    const char * refusal = nullptr;
    // Why the run failed otherwise, where an op could tell (synth_failed):
    // the first failure said, which run_task hands out through `refused`'s
    // failure and failed (cadenza-audio #1092). none for a run that has not.
    AceSynthFailure failure = AceSynthFailure::none;
    const char *    failed  = nullptr;
    // The take a decode failed at for decoding to no audio, -1 for none: a
    // decode hands it out as `refused`'s request (cadenza-audio #1286).
    int             failed_take = -1;

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

// The run's progress, passed through to the caller's, seeing whether the
// callback ever asked to cancel: so run_task, and a decode, can say a run that
// stopped for it was cancelled, with no poll of its own -- a poll a caller would see, and
// one a callback that cancels once would answer otherwise (cadenza-audio
// #1092). PCM passes through too, though phase 1 streams none.
struct CancelSeen {
    AceProgress inner;
    bool        cancelled = false;

    static bool poll(void * data, AceStage stage, int step, int total) {
        CancelSeen * self = static_cast<CancelSeen *>(data);
        const bool   stop = self->inner.fn(self->inner.data, stage, step, total);
        self->cancelled   = self->cancelled || stop;
        return stop;
    }

    static void pcm(void * data, const float * left, const float * right, int n_frames) {
        const CancelSeen * self = static_cast<const CancelSeen *>(data);
        self->inner.pcm(self->inner.data, left, right, n_frames);
    }

    // No callback stays none, as a caller's empty progress never cancels.
    AceProgress progress() {
        AceProgress p;
        p.fn   = inner.fn ? &CancelSeen::poll : nullptr;
        p.pcm  = inner.pcm ? &CancelSeen::pcm : nullptr;
        p.data = this;
        return p;
    }
};

// Say why the run failed, `why` a static string, and return -1 for the op to
// return: the first failure said holds, so an op failing in turn for an
// earlier one's cause does not overwrite it (cadenza-audio #1092).
inline int synth_failed(SynthState & s, AceSynthFailure kind, const char * why) {
    if (s.failure == AceSynthFailure::none) {
        s.failure = kind;
        s.failed  = why;
    }
    return -1;
}

// Whether the run splices its source back in after the decode -- a repaint,
// or a lego region -- so its encode is not what a run again needs: one rule
// for the splices (ops_vae_decode_track, ops_vae_decode), the failed run's
// hand-out (cadenza-audio #1041), and the region's frames, mask and context
// (#1130). The task sets both flags before it encodes.
inline bool splices_source_after_decode(const SynthState & s) {
    return s.is_repaint || s.is_lego_region;
}

// Whether the run switches its context to silence at audio_cover_strength's
// step, and so encodes the text a second time for the steps after the switch:
// a cover's, by the task table (s.task.cover_knobs, not use_source_context,
// which other tasks set too, and every cover, #1187), with a strength below
// 1. No task that reads the cover knobs splices its source back -- the table
// test holds every row to that (#1187) -- so no splice term. The one rule
// for ops_encode_text's need_enc_switch and ops_build_context_silence
// (cadenza-audio #1154).
inline bool cover_switches_context(const SynthState & s) {
    return s.task.cover_knobs && s.rr.audio_cover_strength < 1.0f;
}

// Whether the run starts from its source's latents blended into the noise,
// at cover_noise_strength: a cover's, by the task table, with a source
// encoded and a strength past 0 -- ops_init_noise's rule (cadenza-audio
// #1154).
inline bool cover_blends_noise(const SynthState & s) {
    return s.task.cover_knobs && s.have_cover && s.rr.cover_noise_strength > 0.0f;
}

// Whether a lego's bounds ask for a region, read as a repaint reads them
// (cadenza-audio #1134): an end of its own, after the start; or the -1
// sentinel (any negative end, as asked_region_end reads it) -- the source's
// end, or for a start before the source the source's start -- with a start
// whose region is not the whole song in frames: a start past 0 whose first
// frame is past frame 0 (sec_latent_frame, as resolve_region_frames reads
// it), or a start before 0 whose pad is a frame at least (as the latent pad
// rounds it). So float noise about 0 is 0, and a start inside the first frame
// is the whole song too. The defaults, 0 and -1, ask the whole song, which a
// lego runs as no region: a region of all of it would be the same song,
// spliced.
inline bool lego_asks_region(float rs, float re) {
    if (re >= 0.0f) {
        return re > rs;
    }
    return rs > 0.0f ? sec_latent_frame(rs) > 0 : sec_latent_frame_end(-rs) > 0;
}

// A run's frames for the DiT: `frames` rounded up to its patch (a patch of
// none read as 1), in 64 bits -- the one rounding ops_resolve_T, the source's
// bound and the region's frame check read, so they agree for any patch_size
// (cadenza-audio #1137, #1144).
inline long long frames_to_patch(long long frames, long long patch) {
    patch = patch > 0 ? patch : 1;
    return (frames + patch - 1) / patch * patch;
}

// The frames [t0, t1) a region from rs to re seconds (in the padded frame)
// covers in a run of T: its first frame the one rs falls in, its end rounded
// up, both held to the run, and never reversed. The one rule
// resolve_region_frames and the refusal before the encode read (#1137).
inline void region_frames_in(float rs, float re, int T, int & t0, int & t1) {
    t0 = std::clamp(sec_latent_frame(rs), 0, T);
    t1 = std::clamp(sec_latent_frame_end(re), t0, T);
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
    region_frames_in(s.rs, s.re, s.T, s.repaint_t0, s.repaint_t1);
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
