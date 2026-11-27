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
    int     sample_rate;  // always ACE_SAMPLE_RATE (48000, audio-frames.h)
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

// The batch rule's reason for a region bound that is no number, a tool or a
// test can compare against (cadenza-audio #1185).
inline constexpr const char * kAceRefusalRegionNonfinite = "a region bound is not a finite number";

// The reasons a run refuses its request partway, handed out through
// ace_synth_job_run_dit's `refused` (cadenza-audio #1193).
inline constexpr const char * kAceRefusalRegionEmpty   = "the region ends at or before its start";
inline constexpr const char * kAceRefusalRegionNoFrame = "the region covers no latent frame";
inline constexpr const char * kAceRefusalSourceTooLong =
    "the source, with its pads if any, is past a run's most frames";
// The run's length -- its duration, its audio codes or its source, whichever
// sets it -- past a run's most frames.
inline constexpr const char * kAceRefusalLengthTooLong =
    "the run's length (its duration, codes or source) is past a run's most frames";

// The most takes the DiT runs in one batch: a run (ace_synth_job_run_dit)
// takes 1 to this many requests, one take each -- it never reads
// synth_batch_size. The tools expand each request to its synth_batch_size
// takes, clamped to this, and clamp a batch's total to it too.
// cadenza-audio's shim holds its request cap to it (cadenza-audio #1165).
inline constexpr int kAceSynthMaxTakes = 9;

// The takes a request's synth_batch_size asks for: the size, a size below 1
// being one take. What a batch's `asked` sums, uncapped (cadenza-audio #1269).
inline int ace_synth_request_asked(int synth_batch_size) {
    return synth_batch_size < 1 ? 1 : synth_batch_size;
}

// The takes a request's synth_batch_size expands to: what it asks for, held
// to 1..kAceSynthMaxTakes. ace_synth_batch_takes counts a request's takes by
// it alone, for the tools and ace_synth_requests_run.
inline int ace_synth_request_takes(int synth_batch_size) {
    const int asked = ace_synth_request_asked(synth_batch_size);
    return asked > kAceSynthMaxTakes ? kAceSynthMaxTakes : asked;
}

// Why a batch is refused before any run, or a run before its job: the
// request it is refused for (-1: the batch as a whole), and the reason, a
// static string a tool can answer with (cadenza-audio #1123, #1160).
// For a setting the batch shares, `field` is the JSON name of the one that
// differs -- "audio_codes" for a mix of codes and none -- the name a reason's
// "requests differ in <field>" leads with too, so a tool can word
// it its own way without reading `reason` (cadenza-audio #1165); NULL for
// any other refusal.
// A run that failed without refusing its request says why in `failure` and
// `failed` instead (cadenza-audio #1092): `reason` stays NULL for it, so a
// caller reading `reason` as "refused" -- an invalid request -- reads only
// refusals. ace_synth_batch_refused and ace_synth_inputs_refused never set
// them.
enum class AceSynthFailure {
    none,               // not failed, or refused (`reason` says why)
    model_unavailable,  // a model the run needs could not be had from the store
    encode_failed,      // an encode, a detokenize -- the FSQ roundtrip's or the
                        // audio codes' -- made nothing: one kind for one fault
    cancelled,          // the progress callback asked to cancel
    out_of_memory,      // a buffer the engine allocates itself could not be
                        // (a decode's output, cadenza-audio #1286)
    other,              // any other failure: the engine's diagnostic is on stderr
};

// The engine's words for a VAE failure, shared by its failures and the tools
// that say them, so each is worded once (cadenza-audio #1286).
inline constexpr const char * kAceVaeEncoderUnavailable = "the VAE encoder could not be loaded";
inline constexpr const char * kAceVaeDecoderUnavailable = "the VAE decoder could not be loaded";
inline constexpr const char * kAceVaeDecodeFailed       = "the VAE decode failed";
inline constexpr const char * kAceVaeEncodeFailed       = "the VAE encode failed";

struct AceSynthBatchRefusal {
    int             request = -1;
    const char *    reason  = nullptr;
    const char *    field   = nullptr;
    // Why a run failed that was not refused: a static string -- which model,
    // which encode -- set with every failure but `none`.
    AceSynthFailure failure = AceSynthFailure::none;
    const char *    failed  = nullptr;
};

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
//   A reference longer than 30 s is cropped first, audio or latents, to
//   upstream's three 10 s segments, one centred in each third (750 latent
//   frames); what failed_encodes hands back of it is that crop
//   (cadenza-audio #1147).
// batch_n: number of requests (1..kAceSynthMaxTakes).
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
// refused: when non-NULL, why the call refused its request: before its job
//   was made -- its arguments (request -1), the batch rule
//   (ace_synth_batch_refused: the request and its reason), or a task that
//   works from a source given none (request -1) -- or partway through the
//   run, for the request itself (request -1, the kAceRefusal* reasons above:
//   its region, its length), which may come after the source's and the
//   text's encodes. So a caller can answer an invalid request as that, not
//   as the pipeline failing (cadenza-audio #1160, #1193). Its reason is left
//   NULL by a call that was not refused: one that ran, failed for another
//   reason, or was cancelled. One that failed so says which in `failure` and
//   `failed` -- a model the store could not give, an encode that made
//   nothing, a cancel, or another failure -- and one that ran leaves them
//   unset (cadenza-audio #1092). A throw (a GPU error, bad_alloc) sets none.
// Returns NULL on error or cancellation.
AceSynthJob * ace_synth_job_run_dit(AceSynth *             ctx,
                                    const AceRequest *     reqs,
                                    const float *          src_audio,
                                    int                    src_len,
                                    const float *          src_latents,
                                    int                    src_T_latent,
                                    const float *          ref_audio,
                                    int                    ref_len,
                                    const float *          ref_latents,
                                    int                    ref_T_latent,
                                    int                    batch_n,
                                    AceProgress            progress       = {},
                                    AceSynthEncodes *      failed_encodes = nullptr,
                                    AceSynthBatchRefusal * refused        = nullptr);

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
// Returns 0 on success, -1 on error or cancellation. `failed`, when given,
// says why a decode that returned -1 failed, as a run's failure is said
// (cadenza-audio #1286): its failure and failed -- the VAE decoder the store
// could not give (model_unavailable), a cancel, a bad argument (other) --
// and never its reason, as a decode refuses no request. Unset by a decode
// that ran. A take whose own decode made no audio fails the decode, said
// why -- out_of_memory for its output, other for the decode -- the takes
// before it keeping theirs: never a success with an empty take
// (cadenza-audio #1286).
int ace_synth_job_run_vae(AceSynth *             ctx,
                          AceSynthJob *          job,
                          AceAudio *             out,
                          AceProgress            progress = {},
                          AceSynthBatchRefusal * failed   = nullptr);

// Phase 2 for ONE take: splice + VAE decode of just track `take`'s latent
// into *out. Decode-on-demand for embedders: first audio for take 0 does not
// wait on the VAE decoding takes 1..n; call again for the other takes, in
// any order. The job stays valid and the remaining takes stay decodable
// after a failed or cancelled call. Returns 0 on success, -1 on a bad take
// index, error, a decode that made no audio (said why, as for
// ace_synth_job_run_vae), or cancellation; `failed` as for
// ace_synth_job_run_vae.
int ace_synth_job_run_vae_take(AceSynth *             ctx,
                               AceSynthJob *          job,
                               int                    take,
                               AceAudio *             out,
                               AceProgress            progress = {},
                               AceSynthBatchRefusal * failed   = nullptr);

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
    // Whether the task reads the region (repainting_start/_end): the batch
    // rule refuses such a task's request whose bound is no number, before
    // its pad or encode (cadenza-audio #1150), and holds a batch's requests to
    // one region (#1159; a lego asking none on both sides agrees).
    bool reads_region  = false;
    // Whether the task reads the track: the batch rule holds a batch's
    // requests to one (#1159, cadenza-audio #1185).
    bool reads_track   = false;
    // Whether a run of the task can splice its source back over its decode
    // after the decode ends: a repaint always, a lego when its bounds ask a
    // region (ace_synth_request_splices_source says of a request). Only a
    // task that reads a region splices one (cadenza-audio #1190).
    bool splices       = false;
};

bool ace_synth_task_info(const char * task, AceSynthTaskInfo * info);

// How many tasks the table holds, and the name of its `index`th, NULL out of
// [0, count): so a tool or a test walks the engine's own tasks, not a list of
// its own (#1185, cadenza-audio #1198).
int          ace_synth_task_count();
const char * ace_synth_task_name(int index);

// Whether a known task takes no source, so a tool can skip decoding one it
// was sent (cadenza-audio #1157): the engine would drop it with a warning.
// False for a task the engine does not know, which its run refuses.
bool ace_synth_task_ignores_source(const char * task);

// Whether a run of `task` is refused for its inputs: a task that works from a
// source (the table's needs_source) given none -- the rule run_task holds a
// run to, asked by a tool before it loads a model so it says the same
// (cadenza-audio #1193). The source is the run's own arguments, read as the
// run reads them: given when its audio holds a sample (src_len > 0) or its
// latents a frame (src_T_latent > 0), so a file that decoded to nothing is
// none here as in the run (cadenza-audio #1194). *why, when non-NULL, gets
// {-1, the reason}. False for a task the engine does not know: the batch
// rule refuses that by name.
bool ace_synth_inputs_refused(const char *           task,
                              const float *          src_audio,
                              int                    src_len,
                              const float *          src_latents,
                              int                    src_T_latent,
                              AceSynthBatchRefusal * why);

// Whether a run of the request (its task_type and region bounds) splices its
// source back over the decode after it ends -- a repaint, or a lego whose
// bounds ask a region (lego_asks_region) -- so the decode's tiles are not
// what the take plays, and a tool cannot stream them (cadenza-audio #1156).
// It reads the task table's splices column: run_lego sets its flag from it,
// and run_repaint, whose row says always, always splices -- so the run and a
// tool read one rule (cadenza-audio #1190). False for any other task, one the
// engine does not know, a NULL request, and a bound that is no finite number,
// which the batch rule refuses.
bool ace_synth_request_splices_source(const AceRequest * req);

// Whether a request (non-NULL) is short of the caption its task needs: by
// the table (a caption of white space alone being none), and for a task the
// engine does not know, as if it needed one. ace_synth_batch_refused asks it
// of every request, or of a planned batch's first alone (#1094, #1123,
// cadenza-audio #1240).
bool ace_synth_request_missing_caption(const AceRequest * req);

// Whether a batch of batch_n requests (reqs non-NULL) is one the engine
// refuses up front, by the task table: a task it does not know, requests that
// mix tasks (a batch runs as its first request's), a request short of the
// caption its task needs -- a caption of whitespace alone is none -- or
// requests that differ in a setting the batch shares, read from its first
// request alone: the steps, schedule, solver, models, output and the like,
// and the region, track and cover knobs where the task reads them, and for
// a text2music whether it has codes (#1159); a region bound that is no
// number, in any request of a task that reads a region (the table's
// reads_region), is said as that (kAceRefusalRegionNonfinite). A take's own
// may differ: seed, caption, lyrics, metas, its codes, synth_batch_size,
// mp3_bitrate, and the LM's fields; duration is #1162's. run_task asks it
// first, and the tools before a run, so each says the same refusal.
// `why`, when given, says which and why; it is left as it was for a batch
// that passes. `planned`: the batch is still to be planned, and its later
// requests take their caption from the plan of its first, so only the
// first's caption is asked, and refused as the batch's (request -1)
// (cadenza-audio #1165, #1240).
bool ace_synth_batch_refused(const AceRequest * reqs, int batch_n, AceSynthBatchRefusal * why, bool planned = false);

// A batch's takes as the tools expand it: each request to its
// synth_batch_size takes (ace_synth_request_takes), the batch clamped to the
// DiT's kAceSynthMaxTakes. `takes` holds one count per request that runs, in
// order -- the last trimmed to fit, the requests past the cap left out -- and
// sums to `total`. `asked` is the takes the batch asked for, each request's
// raw size (ace_synth_request_asked) summed in 64 bits, so no batch
// overflows it: more than `total` whenever a take is dropped -- a lone
// request asking 50 included, and {50, 1} asks 51 -- for the tools' clamp
// log (cadenza-audio #1269). A NULL reqs is an empty batch, whatever
// batch_n says.
// The tools allocate and expand by this, and ace_synth_requests_run counts
// its requests, so "which requests run" is one rule (cadenza-audio #1243).
// Each tool computes it once per job and expands by it (synth_batch_expand,
// tools/synth-batch-runner.h).
struct AceSynthBatchTakes {
    std::vector<int> takes;
    int              total = 0;
    long long        asked = 0;
};

AceSynthBatchTakes ace_synth_batch_takes(const AceRequest * reqs, int batch_n);

// How many of a batch's requests the tools run (0 for a NULL reqs):
// ace_synth_batch_takes's requests, so the requests past the DiT's
// kAceSynthMaxTakes take never run. The tools ask the batch
// rule of these alone -- a request that never runs refuses nothing.
int ace_synth_requests_run(const AceRequest * reqs, int batch_n);

// Tests only: how many runs run_task has refused by ace_synth_batch_refused.
int ace_test_batch_refusals();

// Tests only: with `armed`, the next Cover's FSQ roundtrip fails as its
// tokenizer could not be had (store_require_fsq_tok returning NULL), once --
// a path no input reaches, which a test drives to see the Cover fail and
// still hand out its clean encode (cadenza-audio #1056); without, the hook
// is disarmed, fired or not. A test sees it fired by the run's failure,
// model_unavailable for the FSQ tokenizer (cadenza-audio #1092), as it
// tells the task table's refusal of a sourceless task by its reason.
void ace_test_fail_next_fsq_require(bool armed);

// Tests only: with `armed`, the next decode's VAE decoder is not had
// (store_require_vae_dec returning NULL), once -- model_unavailable, said as
// kAceVaeDecoderUnavailable; without, disarmed, fired or not (cadenza-audio
// #1286).
void ace_test_fail_next_vae_dec_require(bool armed);

// Tests only: with `armed`, the next take's tiled decode fails as a graph it
// could not run would, once -- a take of no audio, which the decode says as
// kAceVaeDecodeFailed; without, disarmed (cadenza-audio #1286).
void ace_test_fail_next_vae_decode(bool armed);

// Tests only: how many runs the task table has warned of a source given to
// a task that takes none, run without it (cadenza-audio #1095).
int ace_test_source_ignores();

// Tests only: how many runs have been refused for a source, with its pads if
// any, past a run's most frames -- before its pad is allocated or its encode
// run -- so a test can tell that refusal from ops_resolve_T's, which comes
// after the encode (cadenza-audio #1139, #1144).
int ace_test_source_too_long();

// Tests only: how many runs have been refused for a region that ends at or
// before its start -- by the check before its pad and its source's encode, or
// by refuse_region's after the pad's shift (cadenza-audio #1134, #1282).
int ace_test_region_empty();

// Tests only: how many runs have been refused for a region that covers no
// latent frame by the check before its source's encode and its text encode,
// not ops_build_context's after them (cadenza-audio #1137).
int ace_test_region_no_frame();

// Tests only: how many runs have had their source copied for their region
// (padded, or for the audio path's splice) -- after the region's check, never
// for a region it refuses (cadenza-audio #1282).
int ace_test_region_pads();

// Tests only: how many runs have reached their encodes -- the source's and
// the timbre's, which come before the text's -- so a test can tell a run
// refused before them from one refused after (cadenza-audio #1137).
int ace_test_encode_passes();

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
