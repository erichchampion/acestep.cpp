#pragma once
// pipeline-synth.h: ACE-Step synthesis pipeline
//
// A lightweight context bound to a ModelStore. The pipeline holds no GPU
// modules: each op (text encoding, cond encoding, FSQ roundtrip, DiT denoise,
// VAE decode) acquires its own modules from the store with RAII release.
// STRICT policy keeps at most one module resident at a time. NEVER policy
// caches everything across calls. DiT weight swap between phases is an
// invisible consequence of the store, not an orchestration concern anymore.

#include "progress.h"
#include "request.h"

#include <cstdlib>
#include <vector>

struct AceSynth;
struct AceSynthJob;
struct ModelStore;

struct AceSynthParams {
    const char * text_encoder_path;  // Qwen3 text encoder GGUF (required)
    const char * dit_path;           // DiT GGUF (required)
    const char * vae_path;           // VAE GGUF (required)
    const char * adapter_path;       // adapter safetensors or directory (NULL to disable)
    float        adapter_scale;      // user scale multiplier
    bool         use_fa;             // flash attention
    bool         clamp_fp16;         // clamp hidden states to FP16 range
    bool         use_batch_cfg;      // batch cond+uncond in one DiT forward
    int          vae_chunk;          // latent frames per tile
    int          vae_overlap;        // overlap frames per side
    const char * dump_dir;           // intermediate tensor dump dir (NULL = disabled)
};

// Output audio buffer. Caller must free with ace_audio_free().
struct AceAudio {
    float * samples;      // planar stereo [L0..LN, R0..RN]
    int     n_samples;    // per channel
    int     sample_rate;  // always 48000
};

// The VAE encodes a failed ace_synth_job_run_dit made (failed_encodes there):
// each [T_latent * 64] f32 time-major, empty for a side it did not keep.
struct AceSynthEncodes {
    std::vector<float> src_latents;
    std::vector<float> ref_latents;
};

void ace_synth_default_params(AceSynthParams * p);

// Build a lightweight synth context bound to a ModelStore. Reads DiT metadata
// (config, silence, null_cond, is_turbo) through the store so text encoding
// and T resolution can run before the DiT itself is ever loaded. All GPU
// modules are acquired per op, never owned by the context. NULL on failure.
AceSynth * ace_synth_load(ModelStore * store, const AceSynthParams * params);

// Phase 1: encode sources, build context, run all DiT denoising steps.
// Modules are acquired as needed: VAE encoder for source and timbre, FSQ
// tokenizer and detokenizer for cover-mode roundtrip, text encoder + cond
// encoder for prompt embedding, DiT for denoising. In STRICT each module is
// released before the next acquires, so VRAM hosts one at a time.
// Produces a job that carries DiT latents in RAM for later VAE decoding.
// reqs[batch_n]: each request has its own caption, lyrics, metadata, audio_codes, and seed.
//   The first request (reqs[0]) is used for shared params (mode, duration, DiT settings);
//   a batch whose requests differ in them is refused (ace_synth_batch_refused).
//   seed must be resolved (non-negative) before calling this function.
// src_audio / ref_audio: interleaved stereo 48kHz buffers, NULL when not applicable.
// src_latents / ref_latents: pre-encoded latents [T_latent * 64] f32 alternative
//   to the matching audio buffer. When non-NULL, the corresponding VAE encoder
//   pass is skipped. Mutually exclusive per side: when both audio and latents
//   are provided for the same side, latents win and audio is ignored.
// batch_n: number of requests (1..9).
// progress: abort + progress callback. Reports ACE_STAGE_DIT between DiT steps,
// and ACE_STAGE_VAE_ENCODE while encoding the source and/or timbre reference in
// Cover-family tasks. Default {} = never cancel, no progress.
// failed_encodes: when non-NULL, the VAE encodes this run made from samples,
//   each [T_latent * 64] f32 time-major -- what that side's *_latents takes --
//   moved there when the run fails (returns NULL or throws: a GPU refusal of
//   its DiT leaves a sourced run's encodes done), so a run again passing them
//   skips the encodes (cadenza-audio #875). Moved, not copied, and only on
//   failure, so a run that succeeds pays nothing (cadenza-audio #1035). Empty
//   after a run that succeeded, and for a side the run was given latents for,
//   whose encode did not finish (or whose timbre fell back to silence), or --
//   the source of a repaint or a lego region -- whose samples the run reads
//   again after the encode, so latents would not make the same takes. NULL:
//   none are kept. The synth keeps nothing of them past the call
//   (cadenza-audio #975).
// Returns NULL on error or cancellation.
AceSynthJob * ace_synth_job_run_dit(AceSynth *         ctx,
                                    const AceRequest * reqs,
                                    const float *      src_audio,
                                    int                src_len,
                                    const float *      src_latents,
                                    int                src_T_latent,
                                    const float *      ref_audio,
                                    int                ref_len,
                                    const float *      ref_latents,
                                    int                ref_T_latent,
                                    int                batch_n,
                                    AceProgress        progress       = {},
                                    AceSynthEncodes *  failed_encodes = nullptr);

// Access the post-DiT denoised latent for one track of the job, owned by the
// job and valid until ace_synth_job_free. Layout is flat [T_latent * 64] f32
// time-major, identical to the /vae wire format: feeding it to /vae decode
// reproduces the track's output audio. *T_out is set to T_latent.
const float * ace_synth_job_get_latent(const AceSynthJob * job, int track_idx, int * T_out);

// Phase 2: latent splice (for repaint/lego) + VAE decode. Acquires the VAE
// decoder from the store; in STRICT this evicts the DiT from phase 1
// transparently. The splice happens in latent space using s.cover_latents
// captured during phase 1, no source audio is needed.
// out[batch_n] allocated by caller, filled with audio buffers.
// Returns 0 on success, -1 on error or cancellation.
int ace_synth_job_run_vae(AceSynth * ctx, AceSynthJob * job, AceAudio * out, AceProgress progress = {});

// Phase 2 for ONE take: splice + VAE decode of just track `take`'s latent
// into *out. Decode-on-demand for embedders: first audio for take 0 does not
// wait on the VAE decoding takes 1..n; call again for the other takes, in
// any order. The job stays valid and the remaining takes stay decodable
// after a failed or cancelled call. Returns 0 on success, -1 on a bad take
// index, error, or cancellation.
int ace_synth_job_run_vae_take(AceSynth * ctx, AceSynthJob * job, int take, AceAudio * out,
                               AceProgress progress = {});

void ace_synth_job_free(AceSynthJob * job);

void ace_audio_free(AceAudio * audio);

// Whether the files this pipeline was loaded from are still the files at
// their paths (#309). The pipeline keeps the modules it loaded while it
// lives, so it keeps working; but one freed meanwhile (STRICT eviction) is
// not read again from a replaced file -- load the pipeline again.
bool ace_synth_is_current(const AceSynth * ctx);

// The keys of every module this pipeline uses, valid while it lives: what
// store_vram_bytes_except leaves out to weigh what is kept beyond a run's
// own set (cadenza-audio #764). A module not yet loaded is simply not
// resident.
struct ModelKey;
std::vector<const ModelKey *> ace_synth_keys(const AceSynth * ctx);

// A job that decodes through `ctx` after its owner may have freed it (a
// reload while takes are parked) retains it, and releases it when done; the
// context goes with the last of them. A retained context keeps only its VAE
// decoder's module (all a parked take's decode reads): the owner's free lets
// the DiT, encoders and FSQ go, so a parked take does not keep a replaced
// model's weights resident.
void ace_synth_retain(AceSynth * ctx);
void ace_synth_release(AceSynth * ctx);
// The owner's free: its keys go at once, the context with its last job.
void ace_synth_free(AceSynth * ctx);

// What the engine's task table says of a task, by name (cadenza-audio #1094),
// so a tool reads it rather than naming tasks itself: whether the task works
// from a source, and whether it needs a caption -- lego, extract and complete
// are told what to make by their track instead. False, *info untouched, for a
// task the engine does not know.
struct AceSynthTaskInfo {
    bool needs_source  = false;
    bool needs_caption = false;
    // Whether the task reads audio_cover_strength and cover_noise_strength;
    // every other runs at their defaults (cadenza-audio #1135).
    bool cover_knobs   = false;
};

bool ace_synth_task_info(const char * task, AceSynthTaskInfo * info);

// Whether a request (non-NULL) is short of the caption its task needs: by
// the table (a caption of white space alone being none), and for a task the
// engine does not know, as if it needed one. ace_synth_batch_refused asks it
// of every request (#1094, #1123).
bool ace_synth_request_missing_caption(const AceRequest * req);

// Why a batch is refused before any run: the request it is refused for, and
// the reason, a static string a tool can answer with (cadenza-audio #1123).
struct AceSynthBatchRefusal {
    int          request = -1;
    const char * reason  = nullptr;
};

// Whether a batch of batch_n requests (reqs non-NULL) is one the engine
// refuses up front, by the task table: a task it does not know, requests that
// mix tasks (a batch runs as its first request's), a request short of the
// caption its task needs -- a caption of whitespace alone is none -- or
// requests that differ in a setting the batch shares, read from its first
// request alone: the steps, schedule, solver, models, output and the like,
// and the region, track, cover knobs and duration where the task reads them
// (#1159). A take's own may differ: seed, caption, lyrics, metas, codes,
// synth_batch_size, mp3_bitrate, and the LM's fields. run_task
// asks it first, and the tools before a run, so each says the same refusal.
// `why`, when given, says which and why; it is left as it was for a batch
// that passes.
bool ace_synth_batch_refused(const AceRequest * reqs, int batch_n, AceSynthBatchRefusal * why);

// How many of a batch's requests (reqs non-NULL) the tools run: each expands
// to its synth_batch_size takes (1..9), and the batch is clamped to the DiT's
// 9, so the requests past the 9th take never run. The tools ask the batch
// rule of these alone -- a request that never runs refuses nothing.
int ace_synth_requests_run(const AceRequest * reqs, int batch_n);

// Tests only: how many runs run_task has refused by ace_synth_batch_refused.
int ace_test_batch_refusals();

// Tests only: with `armed`, the next Cover's FSQ roundtrip fails as its
// tokenizer could not be had (store_require_fsq_tok returning NULL), once --
// a path no input reaches, which a test drives to see the Cover fail and
// still hand out its clean encode (cadenza-audio #1056); without, the hook
// is disarmed, fired or not. ace_test_fsq_require_failures counts the
// requires it has failed, so a test can see it fired.
void ace_test_fail_next_fsq_require(bool armed);
int  ace_test_fsq_require_failures();

// Tests only: how many runs the task table has refused for a task that needs
// a source given none -- so a test can tell that refusal from a failure
// further in, which a sourceless task also reaches (cadenza-audio #1057).
int ace_test_source_refusals();

// Tests only: how many runs the task table has warned of a source given to
// a task that takes none, run without it (cadenza-audio #1095).
int ace_test_source_ignores();

// Tests only: how many runs have been refused for a source, with its pads if
// any, past a run's most frames -- before its pad is allocated or its encode
// run -- so a test can tell that refusal from ops_resolve_T's, which comes
// after the encode (cadenza-audio #1139, #1144).
int ace_test_source_too_long();

// Tests only: how many runs have been refused for a region bound that is no
// number, before its pad or its source's encode (cadenza-audio #1144).
int ace_test_region_nonfinite();

// Tests only: how many runs have been refused for a region that ends at or
// before its start -- by the check before its pad and its source's encode, or
// by adjust_region_coords' after the pad's shift (cadenza-audio #1134).
int ace_test_region_empty();

// Tests only: what the last run made of the request's cover knobs -- the
// strength and noise it ran with, whether it built the switch to a non-cover
// text encode, the step its context switches to silence at (-1: none), and
// its steps -- which a noise blend truncates. Reset when a run starts
// (strength and noise -1, cover_steps -2), written however it ends
// (cadenza-audio #1135).
struct AceTestCoverRecord {
    float strength;
    float noise;
    bool  enc_switch;
    int   cover_steps;
    int   num_steps;
};

AceTestCoverRecord ace_test_last_cover();

// Tests only: how many cover knobs run_task has clamped to [0, 1], or run at
// their default for a NaN, each with a warning (cadenza-audio #1152).
int ace_test_cover_knob_fixes();
