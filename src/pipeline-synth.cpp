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

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
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

// Outpainting: pad src_audio with silence when the region extends beyond
// source bounds. Returns the (possibly padded) buffer and length via out
// Pad source for outpainting when the region extends beyond the source bounds.
// Audio path: zero-pads s.padded_src in samples, the VAE encoder will turn
// silence audio into ~silence latents.
// Latent path: pads s.cover_latents directly with the precomputed silence_full
// latent (the canonical VAE encoding of audio silence). No VAE encode needed.
// Sets s.left_pad_sec for adjust_region_coords.
static void apply_outpainting_padding(const AceSynth *   ctx,
                                      const AceRequest & r,
                                      const float *      src_audio,
                                      int                src_len,
                                      const float *      src_latents,
                                      int                src_T_latent,
                                      SynthState &       s,
                                      const float *&     enc_audio,
                                      int &              enc_len,
                                      const float *&     enc_latents,
                                      int &              enc_T_latent) {
    enc_audio    = src_audio;
    enc_len      = src_len;
    enc_latents  = src_latents;
    enc_T_latent = src_T_latent;

    float src_dur =
        (src_latents && src_T_latent > 0) ? (float) src_T_latent * 1920.0f / 48000.0f : (float) src_len / 48000.0f;
    float rs_raw   = r.repainting_start;
    float re_raw   = r.repainting_end;
    float end_time = (re_raw < 0.0f) ? src_dur : re_raw;
    float lpad     = (rs_raw < 0.0f) ? -rs_raw : 0.0f;
    float rpad     = (end_time > src_dur) ? end_time - src_dur : 0.0f;
    s.left_pad_sec = lpad;

    if (src_latents && src_T_latent > 0) {
        if (lpad <= 0.0f && rpad <= 0.0f) {
            return;
        }
        // Latent pad: round to 25Hz frames, splice silence_full at the edges.
        int           lpad_T  = (int) (lpad * 25.0f + 0.5f);
        int           rpad_T  = (int) (rpad * 25.0f + 0.5f);
        int           total_T = lpad_T + src_T_latent + rpad_T;
        const float * sil     = ctx->meta->silence_full.data();
        s.padded_latents.resize((size_t) total_T * 64);
        memcpy(s.padded_latents.data(), sil, (size_t) lpad_T * 64 * sizeof(float));
        memcpy(s.padded_latents.data() + (size_t) lpad_T * 64, src_latents, (size_t) src_T_latent * 64 * sizeof(float));
        memcpy(s.padded_latents.data() + (size_t) (lpad_T + src_T_latent) * 64, sil,
               (size_t) rpad_T * 64 * sizeof(float));
        enc_latents  = s.padded_latents.data();
        enc_T_latent = total_T;
        fprintf(stderr, "[Outpaint] latent pad left=%.1fs (%d) right=%.1fs (%d) total=%.1fs\n", lpad, lpad_T, rpad,
                rpad_T, (float) total_T * 1920.0f / 48000.0f);
        return;
    }

    // Audio path: always populate s.padded_src so the post-decode waveform
    // splice always has the source PCM to read from, with or without outpaint.
    int lpad_s       = (int) (lpad * 48000.0f);
    int rpad_s       = (int) (rpad * 48000.0f);
    int padded_total = src_len + lpad_s + rpad_s;
    s.padded_src.resize((size_t) padded_total * 2);
    if (lpad_s > 0 || rpad_s > 0) {
        memset(s.padded_src.data(), 0, s.padded_src.size() * sizeof(float));
    }
    memcpy(s.padded_src.data() + (size_t) lpad_s * 2, src_audio, (size_t) src_len * 2 * sizeof(float));
    enc_audio = s.padded_src.data();
    enc_len   = padded_total;
    if (lpad > 0.0f || rpad > 0.0f) {
        fprintf(stderr, "[Outpaint] audio pad left=%.1fs right=%.1fs total=%.1fs\n", lpad, rpad,
                (float) padded_total / 48000.0f);
    }
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
    fprintf(stderr, "[Region] %.1fs..%.1fs (canvas=%.1fs)\n", s.rs, s.re, (float) s.T_cover * 1920.0f / 48000.0f);
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

// One of a run's inputs as its caller gave it: audio (len samples a channel)
// or latents (T_latent frames), either, both or neither (#1057).
struct SynthInput {
    const float * audio    = nullptr;
    int           len      = 0;
    const float * latents  = nullptr;
    int           T_latent = 0;

    bool given() const { return (audio && len > 0) || (latents && T_latent > 0); }
};

// A run's source and reference, one argument for every task.
struct SynthInputs {
    SynthInput src;
    SynthInput ref;
};

// Pin VAE-Enc across two back-to-back encodes (source then timbre) so STRICT
// does not unload and reload the 160 MB weights between them. On return the
// pin is released, VAE-Enc can be evicted by the next require. The pin is
// only justified when at least one side actually goes through the encoder:
// when both src and ref arrive as pre-encoded latents, no pin is needed.
static bool pinned_encode_src_and_timbre(AceSynth *    ctx,
                                         const float * src_audio,
                                         int           src_len,
                                         const float * src_latents,
                                         int           src_T_latent,
                                         const float * ref_audio,
                                         int           ref_len,
                                         const float * ref_latents,
                                         int           ref_T_latent,
                                         SynthState &  s,
                                         AceProgress   progress) {
    bool have_src_audio   = src_audio && src_len > 0;
    bool have_src_latents = src_latents && src_T_latent > 0;
    bool have_src         = have_src_audio || have_src_latents;
    bool have_ref_audio   = ref_audio && ref_len > 0;
    bool have_ref_latents = ref_latents && ref_T_latent > 0;
    bool have_ref         = have_ref_audio || have_ref_latents;
    if (!have_src && !have_ref) {
        // Neither encode touches the GPU: timbre takes the silence path.
        if (ops_encode_timbre(ctx, NULL, 0, NULL, 0, s, progress) != 0) {
            return false;
        }
        return true;
    }
    bool        need_vae_pin = (have_src_audio || have_ref_audio);
    ModelHandle vae_pin(ctx->store, need_vae_pin ? store_require_vae_enc(ctx->store, ctx->vae_enc_key) : NULL);
    if (need_vae_pin && !vae_pin.ptr) {
        fprintf(stderr, "[Pipeline-Synth] FATAL: store_require_vae_enc (pin) failed\n");
        return false;
    }
    if (have_src) {
        if (ops_encode_src(ctx, src_audio, src_len, src_latents, src_T_latent, s, progress) != 0) {
            return false;
        }
        // Handed out by a failed run (#875, #1035) unless the run reads the
        // source's samples again: a repaint and a lego region splice the
        // source back in after the decode, so their encode is not the run
        // again's. Said by the task (#1041), not read from padded_src, which
        // only the outpainting padding happens to leave.
        s.src_encoded = have_src_audio && !have_src_latents && !splices_source_after_decode(s);
    }
    if (ops_encode_timbre(ctx, ref_audio, ref_len, ref_latents, ref_T_latent, s, progress) != 0) {
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

    if (!pinned_encode_src_and_timbre(ctx, NULL, 0, NULL, 0, in.ref.audio, in.ref.len, in.ref.latents, in.ref.T_latent,
                                      s, progress)) {
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

    if (!pinned_encode_src_and_timbre(ctx, in.src.audio, in.src.len, in.src.latents, in.src.T_latent, in.ref.audio,
                                      in.ref.len, in.ref.latents, in.ref.T_latent, s, progress)) {
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
// in.ref.audio = in.src.audio for best results.
static int run_cover_nofsq(AceSynth *          ctx,
                           const AceRequest *  reqs,
                           const SynthInputs & in,
                           int                 batch_n,
                           SynthState &        s,
                           AceProgress         progress) {
    s.use_source_context = true;
    s.instruction_str    = DIT_INSTR_COVER;

    if (!pinned_encode_src_and_timbre(ctx, in.src.audio, in.src.len, in.src.latents, in.src.T_latent, in.ref.audio,
                                      in.ref.len, in.ref.latents, in.ref.T_latent, s, progress)) {
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
    bool have_latents    = in.src.latents && in.src.T_latent > 0;
    s.is_repaint         = true;
    s.use_source_context = true;
    s.instruction_str    = DIT_INSTR_REPAINT;

    const float * enc_audio    = NULL;
    int           enc_len      = 0;
    const float * enc_latents  = NULL;
    int           enc_T_latent = 0;
    apply_outpainting_padding(ctx, reqs[0], in.src.audio, in.src.len, in.src.latents, in.src.T_latent, s, enc_audio,
                              enc_len, enc_latents, enc_T_latent);

    if (!pinned_encode_src_and_timbre(ctx, enc_audio, enc_len, enc_latents, enc_T_latent, in.ref.audio, in.ref.len,
                                      in.ref.latents, in.ref.T_latent, s, progress)) {
        return -1;
    }
    float src_dur = have_latents ? (float) in.src.T_latent * 1920.0f / 48000.0f : (float) in.src.len / 48000.0f;
    if (!adjust_region_coords(s, src_dur)) {
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
    bool have_latents       = in.src.latents && in.src.T_latent > 0;
    s.is_lego_region        = (s.rr.repainting_end > s.rr.repainting_start);
    s.use_source_context    = true;
    std::string track_upper = prepare_track(s.rr.track, "Lego");
    s.instruction_str       = dit_instr_lego(track_upper);
    fprintf(stderr, "[Synth-Run] task=%s\n", reqs[0].task_type.c_str());
    warn_if_turbo_stem(ctx, "lego");

    const float * enc_audio    = in.src.audio;
    int           enc_len      = in.src.len;
    const float * enc_latents  = in.src.latents;
    int           enc_T_latent = in.src.T_latent;
    if (s.is_lego_region) {
        apply_outpainting_padding(ctx, reqs[0], in.src.audio, in.src.len, in.src.latents, in.src.T_latent, s, enc_audio,
                                  enc_len, enc_latents, enc_T_latent);
    }

    if (!pinned_encode_src_and_timbre(ctx, enc_audio, enc_len, enc_latents, enc_T_latent, in.ref.audio, in.ref.len,
                                      in.ref.latents, in.ref.T_latent, s, progress)) {
        return -1;
    }
    float src_dur = have_latents ? (float) in.src.T_latent * 1920.0f / 48000.0f : (float) in.src.len / 48000.0f;
    if (s.is_lego_region && !adjust_region_coords(s, src_dur)) {
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

    if (!pinned_encode_src_and_timbre(ctx, in.src.audio, in.src.len, in.src.latents, in.src.T_latent, in.ref.audio,
                                      in.ref.len, in.ref.latents, in.ref.T_latent, s, progress)) {
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

    if (!pinned_encode_src_and_timbre(ctx, in.src.audio, in.src.len, in.src.latents, in.src.T_latent, in.ref.audio,
                                      in.ref.len, in.ref.latents, in.ref.T_latent, s, progress)) {
        return -1;
    }
    if (run_tail(ctx, reqs, batch_n, s, progress) != 0) {
        return -1;
    }
    return 0;
}

// The tasks, by name: what run_task dispatches on, and the list of the ones
// it knows (#1057). needs_source: a task that works from a source refuses a
// run given none.
using SynthTaskFn = int (*)(AceSynth *, const AceRequest *, const SynthInputs &, int, SynthState &, AceProgress);

struct SynthTask {
    const char * name;
    SynthTaskFn  run;
    bool         needs_source;
};

static const SynthTask kSynthTasks[] = {
    { TASK_TEXT2MUSIC,  run_text2music,  false },
    { TASK_COVER,       run_cover,       true  },
    { TASK_COVER_NOFSQ, run_cover_nofsq, true  },
    { TASK_REPAINT,     run_repaint,     true  },
    { TASK_LEGO,        run_lego,        true  },
    { TASK_EXTRACT,     run_extract,     true  },
    { TASK_COMPLETE,    run_complete,    true  },
};

// Dispatches on reqs[0].task_type to the right task function. task_type is
// always set: request_init defaults it to text2music, the JSON parser ignores
// empty strings.
static AceSynthJob * run_task(AceSynth *          ctx,
                              const AceRequest *  reqs,
                              const SynthInputs & in,
                              int                 batch_n,
                              AceProgress         progress) {
    // An unknown task, and a task that works from a source given none (all
    // but text2music), are refused before the job is made -- the name first,
    // so a misspelt task is said as that.
    const std::string & task  = reqs[0].task_type;
    const SynthTask *   entry = nullptr;
    for (const SynthTask & t : kSynthTasks) {
        if (task == t.name) {
            entry = &t;
            break;
        }
    }
    if (!entry) {
        fprintf(stderr, "[Synth-Run] ERROR: unknown task_type '%s'\n", task.c_str());
        return NULL;
    }
    if (entry->needs_source && !in.src.given()) {
        fprintf(stderr, "[Synth-Run] ERROR: task '%s' requires source audio or latents\n", task.c_str());
        return NULL;
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

