#pragma once
// pipeline-synth-ops.h: primitive operations of the synthesis pipeline
//
// Each op is a single step of phase 1 or phase 2 (VAE encode, FSQ roundtrip,
// text encoding, context build, noise init, DiT generate, VAE decode). The
// orchestrator in pipeline-synth.cpp calls them in sequence. SynthState and
// AceSynth are defined in pipeline-synth-impl.h.

#include "audio-frames.h"
#include "pipeline-synth.h"
#include "progress.h"

#include <vector>

struct AceSynth;
struct SynthInput;
struct SynthState;

// Phase 1 primitives.

// Encode the source into cover_latents: its latents when given, else its
// audio. Sets s.have_cover, s.cover_latents, s.T_cover.
int ops_encode_src(const AceSynth * ctx, const SynthInput & src, SynthState & s, AceProgress progress = {});

// FSQ roundtrip on cover_latents (cover mode only).
// 0, or -1 when it could not run: a cover is not run on clean latents in its
// place, unsaid (cadenza-audio #1042).
int ops_fsq_roundtrip(const AceSynth * ctx, SynthState & s);

// Whether `codes` parse to any code, as ops_resolve_params reads them: a
// string of commas and spaces alone is none. The batch rule refuses a
// text2music's codes that are not empty yet hold none (cadenza-audio #1159).
bool ops_codes_given(const std::string & codes);

// Resolve shared DiT params (steps, guidance, shift) and scan audio_codes.
int ops_resolve_params(const AceSynth * ctx, const AceRequest * reqs, int batch_n, SynthState & s);

// Build flow-matching timestep schedule.
void ops_build_schedule(SynthState & s);

// Resolve T (latent frame count) and S (patch count).
int ops_resolve_T(const AceSynth * ctx, SynthState & s);

// The part of a reference the timbre path reads, as upstream ACE-Step 1.5
// samples it (process_reference_audio): three segments of `seg` frames, one
// from each third of the reference -- upstream's 3 x 10 s. Upstream places
// each at random in its third; here each is centred, so a take is the same
// from its seed (cadenza-audio #1147). `in` holds `n` frames of `width`
// floats (2 for interleaved stereo samples, 64 for latents). A reference
// longer than 3 * seg frames is cropped into `out`, and true returned; a
// shorter one is read whole: false, `out` untouched.
bool timbre_crop(const float * in, int n, int width, int seg, std::vector<float> & out);

// Upstream's segment: 10 s, in samples and in latent frames.
inline constexpr int ACE_TIMBRE_SEGMENT_SAMPLES = 10 * ACE_SAMPLE_RATE;
inline constexpr int ACE_TIMBRE_SEGMENT_FRAMES  = ACE_TIMBRE_SEGMENT_SAMPLES / ACE_LATENT_HOP;

// Encode timbre from the reference: its latents when given, else its audio
// via VAE, else silence, either cropped by timbre_crop first. Sets
// s.timbre_feats and s.S_ref_timbre.
int ops_encode_timbre(const AceSynth * ctx, const SynthInput & ref, SynthState & s, AceProgress progress = {});

// Per-batch text + lyric encoding (main pass + optional non-cover pass).
// Stacks results into s.enc_hidden / s.enc_hidden_nc.
int ops_encode_text(const AceSynth * ctx, const AceRequest * reqs, int batch_n, SynthState & s);

// Build DiT context tensor [batch_n, T, ctx_ch] = src_latents(64) | mask(64).
int ops_build_context(const AceSynth * ctx, const AceRequest * reqs, int batch_n, SynthState & s);

// Build silence context + cover_steps for audio_cover_strength switching.
void ops_build_context_silence(const AceSynth * ctx, int batch_n, SynthState & s);

// Init noise tensor (Philox) + cover noise blend + per_S.
void ops_init_noise(const AceSynth * ctx, const AceRequest * reqs, int batch_n, SynthState & s);

// Run the DiT denoising loop.
int ops_dit_generate(const AceSynth * ctx, int batch_n, SynthState & s, AceProgress progress);

// Phase 2 primitive.

// Latent splice for repaint/lego (kept generated frames inside [t0, t1),
// source latents elsewhere) followed by VAE decode for every batch item.
// Returns 0 on success, -1 on error/cancel.
int ops_vae_decode(const AceSynth * ctx, int batch_n, AceAudio * out, SynthState & s, AceProgress progress);

// Decode exactly one track of the job into *out. Same splice and tiled decode
// as ops_vae_decode, scoped to `track`; the other tracks' latents stay parked
// in the job state and remain decodable. `track` must be within the batch.
int ops_vae_decode_one(const AceSynth * ctx, int track, AceAudio * out, SynthState & s, AceProgress progress);
