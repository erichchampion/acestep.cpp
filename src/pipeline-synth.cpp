// pipeline-synth.cpp: ACE-Step synthesis pipeline implementation
//
// Thin orchestrator over a ModelStore. Holds no GPU module, no CPU-cached
// DiT state of its own: the store exposes DiTMeta (silence, null_cond, cfg,
// is_turbo) and each op acquires the GPU modules it needs on the fly.
//
// One function per task. Each task reads its inputs, poses its flags on
// SynthState, then calls the ops in a linear sequence. The dispatcher at the
// bottom picks the right task function from reqs[0].task_type.

#include "pipeline-synth.h"

#include "pipeline-synth-impl.h"
#include "pipeline-synth-ops.h"
#include "task-types.h"

#include <array>
#include <atomic>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

void ace_synth_default_params(AceSynthParams * p) {
    p->text_encoder_path = NULL;
    p->dit_path          = NULL;
    p->vae_path          = NULL;
    p->adapter_path      = NULL;
    p->adapter_scale     = 1.0f;
    p->use_fa            = true;
    p->clamp_fp16        = false;
    p->use_batch_cfg     = true;
    p->vae_chunk         = 1024;
    // 16, not 64 (cadenza-audio #91): the decoder's tiles join bit-exactly
    // -- identical to an untiled decode -- at any overlap of 9 latent frames
    // or more (measured at chunk 256 and 1024 on an 88 s song); 8 differs at
    // -67 dB, 4 seams audibly. 16 keeps a margin over 9, and each tile
    // decodes 96 frames less waste than at 64.
    p->vae_overlap       = 16;
    p->dump_dir          = NULL;
}

// Every key the pipeline requires with, as the store holds them for it.
static std::array<const ModelKey *, 7> synth_keys(const AceSynth * ctx) {
    return { &ctx->text_enc_key, &ctx->cond_enc_key, &ctx->fsq_tok_key, &ctx->fsq_detok_key,
             &ctx->dit_key,      &ctx->vae_enc_key,  &ctx->vae_dec_key };
}

AceSynth * ace_synth_load(ModelStore * store, const AceSynthParams * params) {
    if (!store || !params) {
        fprintf(stderr, "[Synth-Load] ERROR: store and params are required\n");
        return NULL;
    }
    if (!params->dit_path) {
        fprintf(stderr, "[Synth-Load] ERROR: dit_path is NULL\n");
        return NULL;
    }
    if (!params->text_encoder_path) {
        fprintf(stderr, "[Synth-Load] ERROR: text_encoder_path is NULL\n");
        return NULL;
    }
    if (!params->vae_path) {
        fprintf(stderr, "[Synth-Load] ERROR: vae_path is NULL\n");
        return NULL;
    }

    AceSynth * ctx = new AceSynth();
    ctx->store     = store;
    ctx->params    = *params;
    // Own the strings: a job may keep this context after the caller's go.
    auto own = [](std::string & slot, const char *& p) {
        if (p) {
            slot = p;
            p    = slot.c_str();
        }
    };
    own(ctx->own_text_encoder, ctx->params.text_encoder_path);
    own(ctx->own_dit, ctx->params.dit_path);
    own(ctx->own_vae, ctx->params.vae_path);
    own(ctx->own_adapter, ctx->params.adapter_path);
    own(ctx->own_dump_dir, ctx->params.dump_dir);

    // Which files this context reads, fixed now (#309): the metadata below
    // and every module an op requires come from these, and a file replaced
    // later is not read under them.
    const std::string dit_id     = store_file_identity(params->dit_path);
    const std::string te_id      = store_file_identity(params->text_encoder_path);
    const std::string vae_id     = store_file_identity(params->vae_path);
    const bool        has_adapter = params->adapter_path && params->adapter_path[0];

    // DiTMeta: config + silence_latent + null_condition_emb + is_turbo,
    // read through the store's cache and shared with it, so the store may
    // free its entry while this context is alive. Avoids loading the DiT
    // itself just to read a few CPU-side tensors.
    ctx->meta_own = store_dit_meta_shared(store, params->dit_path, dit_id);
    if (!ctx->meta_own) {
        fprintf(stderr, "[Synth-Load] FATAL: DiT metadata unavailable for %s\n", params->dit_path);
        delete ctx;
        return NULL;
    }
    ctx->meta = ctx->meta_own.get();
    ctx->Oc     = ctx->meta->cfg.out_channels;           // 64
    ctx->ctx_ch = ctx->meta->cfg.in_channels - ctx->Oc;  // 128

    // ModelKeys. Each path identifies its GGUF; adapter info rides with the
    // DiT key because two DiTs with different adapters are distinct modules.
    ctx->text_enc_key.kind = MODEL_TEXT_ENC;
    ctx->text_enc_key.path = params->text_encoder_path;
    ctx->text_enc_key.file_id = te_id;

    ctx->cond_enc_key.kind = MODEL_COND_ENC;
    ctx->cond_enc_key.path = params->dit_path;
    ctx->cond_enc_key.file_id = dit_id;

    ctx->fsq_tok_key.kind = MODEL_FSQ_TOK;
    ctx->fsq_tok_key.path = params->dit_path;
    ctx->fsq_tok_key.file_id = dit_id;

    ctx->fsq_detok_key.kind = MODEL_FSQ_DETOK;
    ctx->fsq_detok_key.path = params->dit_path;
    ctx->fsq_detok_key.file_id = dit_id;

    ctx->dit_key.kind          = MODEL_DIT;
    ctx->dit_key.path          = params->dit_path;
    ctx->dit_key.adapter_path  = params->adapter_path ? params->adapter_path : "";
    ctx->dit_key.adapter_scale = params->adapter_scale;
    // As store_key_identity builds it, from the ids fixed above.
    ctx->dit_key.file_id = has_adapter ? dit_id + "|" + store_file_identity(params->adapter_path) : dit_id;

    ctx->vae_enc_key.kind = MODEL_VAE_ENC;
    ctx->vae_enc_key.path = params->vae_path;
    ctx->vae_enc_key.file_id = vae_id;

    ctx->vae_dec_key.kind = MODEL_VAE_DEC;
    ctx->vae_dec_key.path = params->vae_path;
    ctx->vae_dec_key.file_id = vae_id;

    fprintf(stderr, "[Synth-Load] Ready: turbo=%s, fa=%s, batch_cfg=%s\n", ctx->meta->is_turbo ? "yes" : "no",
            params->use_fa ? "yes" : "no", params->use_batch_cfg ? "yes" : "no");
    if (params->clamp_fp16) {
        fprintf(stderr, "[Synth-Load] FP16 clamp enabled\n");
    }
    if (params->adapter_path) {
        fprintf(stderr, "[Synth-Load] Adapter: %s (scale=%.2f)\n", params->adapter_path, params->adapter_scale);
    }

    // The store keeps these files' modules while this pipeline lives, even
    // once the files are replaced (#309): its jobs decode with them.
    for (const ModelKey * k : synth_keys(ctx)) {
        store_hold_key(store, *k);
    }
    ctx->keys_held = true;
    return ctx;
}

// Allocate job and init the SynthState fields every task poses the same way.
static AceSynthJob * alloc_job(AceSynth * ctx, const AceRequest * reqs, int batch_n) {
    AceSynthJob * job = new AceSynthJob();
    job->batch_n      = batch_n;
    SynthState & s    = job->state;
    s.Oc              = ctx->Oc;
    s.ctx_ch          = ctx->ctx_ch;
    s.left_pad_sec    = 0.0f;
    s.rr              = reqs[0];
    s.rs              = s.rr.repainting_start;
    s.re              = s.rr.repainting_end;
    s.is_repaint      = false;
    s.is_lego_region  = false;
    s.have_cover      = false;
    s.T_cover         = 0;
    debug_init(&s.dbg, ctx->params.dump_dir);
    return job;
}

// A run's source and reference, one argument for every task.
struct SynthInputs {
    SynthInput src;
    SynthInput ref;
};

// Outpainting: pads the source when the region extends beyond its bounds, and
// returns the input to encode -- the source itself, or a SynthInput pointing
// into s.padded_src or s.padded_latents, valid while those are not resized.
// Audio path: always copies the source into s.padded_src (the post-decode
// splice reads it), zero-padded in samples; the VAE encoder turns the silence
// into ~silence latents.
// Latent path: pads into s.padded_latents with the precomputed silence_full
// latent (the canonical VAE encoding of audio silence), only when there is
// padding to add. No VAE encode needed.
// Sets s.left_pad_sec for adjust_region_coords.
static SynthInput apply_outpainting_padding(const AceSynth *   ctx,
                                            const AceRequest & r,
                                            const SynthInput & src,
                                            SynthState &       s) {
    SynthInput enc = src;

    float src_dur  = src.duration_sec();
    float rs_raw   = r.repainting_start;
    float re_raw   = r.repainting_end;
    float end_time = (re_raw < 0.0f) ? src_dur : re_raw;
    float lpad     = (rs_raw < 0.0f) ? -rs_raw : 0.0f;
    float rpad     = (end_time > src_dur) ? end_time - src_dur : 0.0f;
    s.left_pad_sec = lpad;

    if (src.has_latents()) {
        if (lpad <= 0.0f && rpad <= 0.0f) {
            return enc;
        }
        // Latent pad: round to 25Hz frames, splice silence_full at the edges.
        int           lpad_T  = (int) (lpad * ACE_LATENT_FPS + 0.5f);
        int           rpad_T  = (int) (rpad * ACE_LATENT_FPS + 0.5f);
        int           total_T = lpad_T + src.T_latent + rpad_T;
        const float * sil     = ctx->meta->silence_full.data();
        s.padded_latents.resize((size_t) total_T * 64);
        memcpy(s.padded_latents.data(), sil, (size_t) lpad_T * 64 * sizeof(float));
        memcpy(s.padded_latents.data() + (size_t) lpad_T * 64, src.latents, (size_t) src.T_latent * 64 * sizeof(float));
        memcpy(s.padded_latents.data() + (size_t) (lpad_T + src.T_latent) * 64, sil,
               (size_t) rpad_T * 64 * sizeof(float));
        enc.latents  = s.padded_latents.data();
        enc.T_latent = total_T;
        fprintf(stderr, "[Outpaint] latent pad left=%.1fs (%d) right=%.1fs (%d) total=%.1fs\n", lpad, lpad_T, rpad,
                rpad_T, latent_frames_sec(total_T));
        return enc;
    }

    // Audio path: always populate s.padded_src so the post-decode waveform
    // splice always has the source PCM to read from, with or without outpaint.
    int lpad_s       = (int) (lpad * ACE_SAMPLE_RATE);
    int rpad_s       = (int) (rpad * ACE_SAMPLE_RATE);
    int padded_total = src.len + lpad_s + rpad_s;
    s.padded_src.resize((size_t) padded_total * 2);
    if (lpad_s > 0 || rpad_s > 0) {
        memset(s.padded_src.data(), 0, s.padded_src.size() * sizeof(float));
    }
    memcpy(s.padded_src.data() + (size_t) lpad_s * 2, src.audio, (size_t) src.len * 2 * sizeof(float));
    enc.audio = s.padded_src.data();
    enc.len   = padded_total;
    if (lpad > 0.0f || rpad > 0.0f) {
        fprintf(stderr, "[Outpaint] audio pad left=%.1fs right=%.1fs total=%.1fs\n", lpad, rpad,
                samples_sec(padded_total));
    }
    return enc;
}

// Shift region coords into the padded reference frame, resolve sentinel end
// (-1) to either left pad boundary (outpaint) or source end (inpaint).
// src_dur is the unpadded source duration in seconds, agnostic of audio vs
// latent input. Returns false when the resolved range is empty or inverted.
static bool adjust_region_coords(SynthState & s, float src_dur) {
    s.rs += s.left_pad_sec;
    if (s.re < 0.0f) {
        s.re = (s.rr.repainting_start < 0.0f) ? s.left_pad_sec : src_dur + s.left_pad_sec;
    } else {
        s.re += s.left_pad_sec;
    }
    if (s.re <= s.rs) {
        fprintf(stderr, "[Region] ERROR: end (%.1f) <= start (%.1f)\n", s.re, s.rs);
        return false;
    }
    fprintf(stderr, "[Region] %.1fs..%.1fs (canvas=%.1fs)\n", s.rs, s.re, latent_frames_sec(s.T_cover));
    return true;
}

// Uppercase track name for instruction templates, warn on unknown names.
static std::string prepare_track(const std::string & track, const char * label) {
    std::string upper = track;
    for (char & ch : upper) {
        ch = (char) toupper((unsigned char) ch);
    }
    validate_track_names(track, label);
    return upper;
}

// Warn when a stem task runs on a turbo model: the training objective does
// not cover stem isolation, output degrades to incoherent noise.
static void warn_if_turbo_stem(const AceSynth * ctx, const char * task_name) {
    if (ctx->meta->is_turbo) {
        fprintf(stderr, "[Synth-Run] WARNING: %s requires base model, turbo output incoherent\n", task_name);
    }
}

// Owns a task's job until run_task returns it. A run that ends without it --
// a task that failed, or a throw (#403) -- frees it, first moving the encodes it made from
// samples into the call's sink, when it has one, to be run again (cadenza-audio
// #875, #975). Moved, never copied, and only on failure (#1035): a run that
// succeeds hands out nothing and pays nothing.
namespace {
class JobOwner {
  public:
    JobOwner(AceSynth * ctx, AceSynthJob * job) : ctx_(ctx), job_(job) {}

    JobOwner(const JobOwner &)             = delete;
    JobOwner & operator=(const JobOwner &) = delete;

    ~JobOwner() {
        if (!job_ || !ctx_->encode_sink) {
            return;
        }
        SynthState & s = job_->state;
        if (s.src_encoded) {
            ctx_->encode_sink->src_latents = std::move(clean_src_latents(s));
        }
        if (s.timbre_encoded) {
            ctx_->encode_sink->ref_latents = std::move(s.timbre_feats);
        }
    }

    AceSynthJob * get() const { return job_.get(); }

    AceSynthJob * release() { return job_.release(); }

  private:
    AceSynth *                   ctx_;
    std::unique_ptr<AceSynthJob> job_;
};
}  // namespace

// Pin VAE-Enc across two back-to-back encodes (source then timbre) so STRICT
// does not unload and reload the 160 MB weights between them. On return the
// pin is released, VAE-Enc can be evicted by the next require. The pin is
// only justified when at least one side actually goes through the encoder:
// when both src and ref arrive as pre-encoded latents, no pin is needed.
static bool pinned_encode_src_and_timbre(AceSynth *         ctx,
                                         const SynthInput & src,
                                         const SynthInput & ref,
                                         SynthState &       s,
                                         AceProgress        progress) {
    // Neither input given needs no pin either: the timbre takes its silence
    // path, the source no encode.
    const bool  have_src_audio   = src.has_audio();
    const bool  have_src_latents = src.has_latents();
    const bool  have_src         = src.given();
    const bool  have_ref_audio   = ref.has_audio();
    bool        need_vae_pin     = (have_src_audio || have_ref_audio);
    ModelHandle vae_pin(ctx->store, need_vae_pin ? store_require_vae_enc(ctx->store, ctx->vae_enc_key) : NULL);
    if (need_vae_pin && !vae_pin.ptr) {
        fprintf(stderr, "[Pipeline-Synth] FATAL: store_require_vae_enc (pin) failed\n");
        return false;
    }
    if (have_src) {
        if (ops_encode_src(ctx, src, s, progress) != 0) {
            return false;
        }
        // Handed out by a failed run (#875, #1035) unless the run reads the
        // source's samples again: a repaint and a lego region splice the
        // source back in after the decode, so their encode is not the run
        // again's. Said by the task (#1041), not read from padded_src, which
        // only the outpainting padding happens to leave.
        s.src_encoded = have_src_audio && !have_src_latents && !splices_source_after_decode(s);
    }
    if (ops_encode_timbre(ctx, ref, s, progress) != 0) {
        return false;
    }
    return true;
}

// Common tail every task ends with once its inputs are encoded and flags are
// posed: resolve params, resolve T, build schedule, encode text, build
// context, init noise, run DiT. Returns 0 on success, -1 on error/cancel.
static int run_tail(AceSynth * ctx, const AceRequest * reqs, int batch_n, SynthState & s, AceProgress progress) {
    if (ops_resolve_params(ctx, reqs, batch_n, s) != 0) {
        return -1;
    }
    if (ops_resolve_T(ctx, s) != 0) {
        return -1;
    }
    ops_build_schedule(s);
    if (ops_encode_text(ctx, reqs, batch_n, s) != 0) {
        return -1;
    }
    if (ops_build_context(ctx, reqs, batch_n, s) != 0) {
        return -1;
    }
    ops_build_context_silence(ctx, batch_n, s);
    ops_init_noise(ctx, reqs, batch_n, s);
    if (ops_dit_generate(ctx, batch_n, s, progress) != 0) {
        return -1;
    }
    return 0;
}

// text2music: pure generation. No src audio. Optional timbre reference.
// Audio codes from LM (or absent) condition the DiT context.
static int run_text2music(AceSynth *          ctx,
                          const AceRequest *  reqs,
                          const SynthInputs & in,
                          int                 batch_n,
                          SynthState &        s,
                          AceProgress         progress) {
    // audio_codes from the LM produce a latent context; empty codes fall back
    // to silence (DiT-only). The DiT was trained with the cover instruction on
    // latent context, so the two flags cascade from the same condition.
    s.use_source_context = !reqs[0].audio_codes.empty();
    s.instruction_str    = s.use_source_context ? DIT_INSTR_COVER : DIT_INSTR_TEXT2MUSIC;

    if (!pinned_encode_src_and_timbre(ctx, SynthInput{}, in.ref, s, progress)) {
        return -1;
    }
    if (run_tail(ctx, reqs, batch_n, s, progress) != 0) {
        return -1;
    }
    return 0;
}

// cover: src audio recomposed with FSQ roundtrip degrading the context, so
// the DiT diverges from the original while staying thematically aligned.
static int run_cover(AceSynth *          ctx,
                     const AceRequest *  reqs,
                     const SynthInputs & in,
                     int                 batch_n,
                     SynthState &        s,
                     AceProgress         progress) {
    s.use_source_context = true;
    s.instruction_str    = DIT_INSTR_COVER;

    if (!pinned_encode_src_and_timbre(ctx, in.src, in.ref, s, progress)) {
        return -1;
    }

    // The FSQ roundtrip keeps the clean latents in fsq_clean_latents, for
    // cover_noise_strength's blend and a failed run's hand-out
    // (clean_src_latents), and conditions on the degraded ones (#1042).
    if (ops_fsq_roundtrip(ctx, s) != 0) {
        return -1;
    }

    if (run_tail(ctx, reqs, batch_n, s, progress) != 0) {
        return -1;
    }
    return 0;
}

// cover-nofsq: cover without the FSQ roundtrip. DiT works on clean 25Hz VAE
// latents, output stays close to source structure and timbre. Pass
// ref_audio = src_audio for best results.
static int run_cover_nofsq(AceSynth *          ctx,
                           const AceRequest *  reqs,
                           const SynthInputs & in,
                           int                 batch_n,
                           SynthState &        s,
                           AceProgress         progress) {
    s.use_source_context = true;
    s.instruction_str    = DIT_INSTR_COVER;

    if (!pinned_encode_src_and_timbre(ctx, in.src, in.ref, s, progress)) {
        return -1;
    }
    if (run_tail(ctx, reqs, batch_n, s, progress) != 0) {
        return -1;
    }
    return 0;
}

// repaint: region-bounded inpaint/outpaint. Source padded with silence at the
// edges when the region extends beyond bounds, audio-side or latent-side
// depending on what the caller provided. DiT regenerates inside the region,
// phase 2 splices the result with the source latents and decodes once.
static int run_repaint(AceSynth *          ctx,
                       const AceRequest *  reqs,
                       const SynthInputs & in,
                       int                 batch_n,
                       SynthState &        s,
                       AceProgress         progress) {
    s.is_repaint         = true;
    s.use_source_context = true;
    s.instruction_str    = DIT_INSTR_REPAINT;

    const SynthInput enc = apply_outpainting_padding(ctx, reqs[0], in.src, s);

    if (!pinned_encode_src_and_timbre(ctx, enc, in.ref, s, progress)) {
        return -1;
    }
    if (!adjust_region_coords(s, in.src.duration_sec())) {
        return -1;
    }
    if (run_tail(ctx, reqs, batch_n, s, progress) != 0) {
        return -1;
    }
    return 0;
}

// lego: stem generation. With valid rs/re: region-constrained, DiT generates
// only in the zone with full audio context. Without: whole-song generation.
// audio_cover_strength forced to 1.0 so all DiT steps hear the backing track.
// Both in.src.audio and in.src.latents are accepted; in region mode the source is
// padded with silence at the edges when the region extends beyond bounds.
static int run_lego(AceSynth *          ctx,
                    const AceRequest *  reqs,
                    const SynthInputs & in,
                    int                 batch_n,
                    SynthState &        s,
                    AceProgress         progress) {
    s.is_lego_region        = (s.rr.repainting_end > s.rr.repainting_start);
    s.use_source_context    = true;
    std::string track_upper = prepare_track(s.rr.track, "Lego");
    s.instruction_str       = dit_instr_lego(track_upper);
    fprintf(stderr, "[Synth-Run] task=%s\n", reqs[0].task_type.c_str());
    warn_if_turbo_stem(ctx, "lego");

    // In region mode, padded at the edges the region passes; else as given.
    const SynthInput enc = s.is_lego_region ? apply_outpainting_padding(ctx, reqs[0], in.src, s) : in.src;

    if (!pinned_encode_src_and_timbre(ctx, enc, in.ref, s, progress)) {
        return -1;
    }
    if (s.is_lego_region && !adjust_region_coords(s, in.src.duration_sec())) {
        return -1;
    }
    if (run_tail(ctx, reqs, batch_n, s, progress) != 0) {
        return -1;
    }
    return 0;
}

// extract: stem isolation from a full mix.
static int run_extract(AceSynth *          ctx,
                       const AceRequest *  reqs,
                       const SynthInputs & in,
                       int                 batch_n,
                       SynthState &        s,
                       AceProgress         progress) {
    s.use_source_context    = true;
    std::string track_upper = prepare_track(s.rr.track, "Extract");
    s.instruction_str       = dit_instr_extract(track_upper);
    fprintf(stderr, "[Synth-Run] task=%s\n", reqs[0].task_type.c_str());
    warn_if_turbo_stem(ctx, "extract");

    if (!pinned_encode_src_and_timbre(ctx, in.src, in.ref, s, progress)) {
        return -1;
    }
    if (run_tail(ctx, reqs, batch_n, s, progress) != 0) {
        return -1;
    }
    return 0;
}

// complete: extend an isolated stem with more content.
static int run_complete(AceSynth *          ctx,
                        const AceRequest *  reqs,
                        const SynthInputs & in,
                        int                 batch_n,
                        SynthState &        s,
                        AceProgress         progress) {
    s.use_source_context    = true;
    std::string track_upper = prepare_track(s.rr.track, "Complete");
    s.instruction_str       = dit_instr_complete(track_upper);
    fprintf(stderr, "[Synth-Run] task=%s\n", reqs[0].task_type.c_str());
    warn_if_turbo_stem(ctx, "complete");

    if (!pinned_encode_src_and_timbre(ctx, in.src, in.ref, s, progress)) {
        return -1;
    }
    if (run_tail(ctx, reqs, batch_n, s, progress) != 0) {
        return -1;
    }
    return 0;
}

// The tasks, by name: what run_task dispatches on, and the list of the ones
// it knows (#1057). needs_source: a task that works from a source refuses a
// run given none; a task that takes none runs without one it is given, and
// says so (cadenza-audio #1095). needs_caption: a task told what to make by
// its caption, not by its track (lego, extract, complete) -- the rule the
// tools hold a request to, read through ace_synth_task_info (#1094).
using SynthTaskFn = int (*)(AceSynth *, const AceRequest *, const SynthInputs &, int, SynthState &, AceProgress);

struct SynthTask {
    const char * name;
    SynthTaskFn  run;
    bool         needs_source;
    bool         needs_caption;
};

static const SynthTask kSynthTasks[] = {
    { TASK_TEXT2MUSIC,  run_text2music,  false, true  },
    { TASK_COVER,       run_cover,       true,  true  },
    { TASK_COVER_NOFSQ, run_cover_nofsq, true,  true  },
    { TASK_REPAINT,     run_repaint,     true,  true  },
    { TASK_LEGO,        run_lego,        true,  false },
    { TASK_EXTRACT,     run_extract,     true,  false },
    { TASK_COMPLETE,    run_complete,    true,  false },
};

// The table's row for a task, by name; NULL for one it does not hold.
static const SynthTask * find_synth_task(std::string_view task) {
    for (const SynthTask & t : kSynthTasks) {
        if (task == t.name) {
            return &t;
        }
    }
    return nullptr;
}

bool ace_synth_task_info(const char * task, AceSynthTaskInfo * info) {
    const SynthTask * entry = task ? find_synth_task(task) : nullptr;
    if (!entry) {
        return false;
    }
    if (info) {
        info->needs_source  = entry->needs_source;
        info->needs_caption = entry->needs_caption;
    }
    return true;
}

bool ace_synth_request_missing_caption(const AceRequest * req) {
    if (!req->caption.empty()) {
        return false;
    }
    const SynthTask * entry = find_synth_task(req->task_type);
    return !entry || entry->needs_caption;
}

// The runs run_task has refused for want of a source (ace_test_source_refusals).
static std::atomic<int> g_test_source_refusals{ 0 };

int ace_test_source_refusals() {
    return g_test_source_refusals.load();
}

// The runs run_task has warned of a source their task ignores
// (ace_test_source_ignores).
static std::atomic<int> g_test_source_ignores{ 0 };

int ace_test_source_ignores() {
    return g_test_source_ignores.load();
}

// Dispatches on reqs[0].task_type to the right task function. task_type is
// always set: request_init defaults it to text2music, the JSON parser ignores
// empty strings.
static AceSynthJob * run_task(AceSynth *          ctx,
                              const AceRequest *  reqs,
                              const SynthInputs & in,
                              int                 batch_n,
                              AceProgress         progress) {
    // An unknown task, and a task that works from a source given none (the
    // table's needs_source), are refused before the job is made -- the name
    // first, so a misspelt task is said as that. A task that takes no source
    // given one is warned of.
    const std::string & task  = reqs[0].task_type;
    const SynthTask *   entry = find_synth_task(task);
    if (!entry) {
        fprintf(stderr, "[Synth-Run] ERROR: unknown task_type '%s'\n", task.c_str());
        return NULL;
    }
    if (entry->needs_source && !in.src.given()) {
        g_test_source_refusals.fetch_add(1);
        fprintf(stderr, "[Synth-Run] ERROR: task '%s' requires source audio or latents\n", task.c_str());
        return NULL;
    }
    // A task that takes no source given one runs as asked, without it: a
    // warning, not a refusal, so a caller that sends one along still gets
    // its run -- but not silently, as a cover it may have meant (#1095).
    if (!entry->needs_source && in.src.given()) {
        g_test_source_ignores.fetch_add(1);
        fprintf(stderr, "[Synth-Run] WARNING: task '%s' ignores the given source\n", task.c_str());
    }
    // The one owner of the task's job (#1041): each task fills the job's
    // state and says whether it ran; the job is returned only if it did. A
    // run that ends without it -- -1, or a throw (#403) --
    // frees it, handing out its encodes first (JobOwner).
    JobOwner     job_owner(ctx, alloc_job(ctx, reqs, batch_n));
    SynthState & s  = job_owner.get()->state;
    const int    rc = entry->run(ctx, reqs, in, batch_n, s, progress);
    return rc == 0 ? job_owner.release() : NULL;
}

// Phase 1 entry point.
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
                                    AceProgress        progress,
                                    AceSynthEncodes *  failed_encodes) {
    if (failed_encodes) {
        // Emptied and let go: a caller's sink reused after a failed call holds
        // nothing of it through this one.
        std::vector<float>().swap(failed_encodes->src_latents);
        std::vector<float>().swap(failed_encodes->ref_latents);
    }
    if (!ctx || !reqs || batch_n < 1 || batch_n > 9) {
        return NULL;
    }
    // The sink is this call's only (#975): unset however the call ends, a
    // throw included, so the synth holds no pointer, and no encode, past it.
    struct SinkScope {
        AceSynth * ctx;
        ~SinkScope() { ctx->encode_sink = nullptr; }
    } scope{ ctx };
    ctx->encode_sink  = failed_encodes;
    const SynthInputs in{
        { src_audio, src_len, src_latents, src_T_latent },
        { ref_audio, ref_len, ref_latents, ref_T_latent }
    };
    return run_task(ctx, reqs, in, batch_n, progress);
}

// Pointer to the post-DiT latent for one track in s.output, time-major
// [T*64] f32. Lives until ace_synth_job_free.
const float * ace_synth_job_get_latent(const AceSynthJob * job, int track_idx, int * T_out) {
    const SynthState & s = job->state;
    *T_out               = s.T;
    return s.output.data() + (size_t) track_idx * s.T * s.Oc;
}

// Phase 2: latent splice (for repaint/lego) + VAE decode for every batch item.
int ace_synth_job_run_vae(AceSynth * ctx, AceSynthJob * job, AceAudio * out, AceProgress progress) {
    if (!ctx || !job || !out) {
        return -1;
    }
    return ops_vae_decode(ctx, job->batch_n, out, job->state, progress);
}

int ace_synth_job_run_vae_take(AceSynth * ctx, AceSynthJob * job, int take, AceAudio * out, AceProgress progress) {
    if (!ctx || !job || !out || take < 0 || take >= job->batch_n) {
        return -1;
    }
    return ops_vae_decode_one(ctx, take, out, job->state, progress);
}

void ace_synth_job_free(AceSynthJob * job) {
    delete job;
}

void ace_audio_free(AceAudio * audio) {
    if (audio && audio->samples) {
        free(audio->samples);
        audio->samples   = NULL;
        audio->n_samples = 0;
    }
}

// The owner's holds on every key, dropped once.
static void drop_owner_keys(AceSynth * ctx) {
    if (!ctx->keys_held.exchange(false)) {
        return;
    }
    // Its files' modules are no longer needed by it: a stale one goes now.
    for (const ModelKey * k : synth_keys(ctx)) {
        store_drop_key(ctx->store, *k);
    }
}

static void unref(AceSynth * ctx) {
    if (ctx->refs.fetch_sub(1, std::memory_order_acq_rel) != 1) {
        return;
    }
    drop_owner_keys(ctx);
    delete ctx;
}

void ace_synth_free(AceSynth * ctx) {
    if (!ctx) {
        return;
    }
    drop_owner_keys(ctx);
    unref(ctx);
}

bool ace_synth_is_current(const AceSynth * ctx) {
    if (!ctx) {
        return false;
    }
    for (const ModelKey * k : { &ctx->text_enc_key, &ctx->dit_key, &ctx->vae_dec_key }) {
        if (store_key_identity(*k) != k->file_id) {
            return false;
        }
    }
    return true;
}

std::vector<const ModelKey *> ace_synth_keys(const AceSynth * ctx) {
    if (!ctx) {
        return {};
    }
    const auto keys = synth_keys(ctx);
    return { keys.begin(), keys.end() };
}

void ace_synth_retain(AceSynth * ctx) {
    if (ctx) {
        store_hold_key(ctx->store, ctx->vae_dec_key);  // may throw: nothing taken yet
        ctx->refs.fetch_add(1, std::memory_order_relaxed);
    }
}

void ace_synth_release(AceSynth * ctx) {
    if (!ctx) {
        return;
    }
    // A failure freeing a stale module must not keep the context alive.
    try {
        store_drop_key(ctx->store, ctx->vae_dec_key);
    } catch (...) {
    }
    unref(ctx);
}

