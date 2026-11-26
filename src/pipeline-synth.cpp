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
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
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
    // Tiles the VAE can run, refused at the load rather than by a run's
    // division by a stride of 0 (cadenza-audio #1311).
    if (vae_tiling_refused_said("Synth-Load", params->vae_chunk, params->vae_overlap)) {
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
    s.cover_steps     = -1;  // no switch to silence until the context says one
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

// Where a region asked for ends, in the source's own seconds (before any pad):
// its repainting_end, or for the -1 sentinel the left pad's end (an outpaint
// before the source) or the source's end. The one reading of the sentinel, for
// the pads and for the shift (shifted_region, #1138).
static float asked_region_end(float rs_raw, float re_raw, float src_dur) {
    return re_raw >= 0.0f ? re_raw : (rs_raw < 0.0f ? 0.0f : src_dur);
}

// A region asked from rs_raw to re_raw, its -1 end read, shifted into the
// source padded by left_pad_sec: the one shift, which the pad's run-on and the
// run's own region both read (#1138, cadenza-audio #1282).
static void shifted_region(float rs_raw, float re_raw, float src_dur, float left_pad_sec, float & rs, float & re) {
    rs = rs_raw + left_pad_sec;
    re = asked_region_end(rs_raw, re_raw, src_dur) + left_pad_sec;
}

// The frames [t0, t1) a region (shifted) covers in a run that ends nowhere:
// by region_frames_in, the frame rule the run reads -- so the pad's run-on
// decides as the run will (#1138, cadenza-audio #1282).
static void region_frames_unbounded(float rs, float re, int & t0, int & t1) {
    region_frames_in(rs, re, std::numeric_limits<int>::max(), t0, t1);
}

// The runs refused for a region that ends at or before its start, once its
// -1 end is read (ace_test_region_empty).
static std::atomic<int> g_test_region_empty{ 0 };

int ace_test_region_empty() {
    return g_test_region_empty.load();
}

// True, said and counted, for a region that ends at or before its start: the
// one rule prepare_region asks before the pad and the encode, and
// refuse_region after the pad's shift (#1134).
static bool refuse_empty_region(SynthState & s, float end, float start) {
    if (end > start) {
        return false;
    }
    g_test_region_empty.fetch_add(1);
    s.refusal = kAceRefusalRegionEmpty;
    fprintf(stderr, "[Region] ERROR: end (%.1f) <= start (%.1f)\n", end, start);
    return true;
}

// The runs that reached their encodes: the source's, the timbre's, and after
// them the text's (ace_test_encode_passes).
static std::atomic<int> g_test_encode_passes{ 0 };

int ace_test_encode_passes() {
    return g_test_encode_passes.load();
}

// The runs refused for a region of no frame before the encodes
// (ace_test_region_no_frame).
static std::atomic<int> g_test_region_no_frame{ 0 };

int ace_test_region_no_frame() {
    return g_test_region_no_frame.load();
}

// The runs whose source was copied for its region: into s.padded_latents, or
// s.padded_src (ace_test_region_pads).
static std::atomic<int> g_test_region_pads{ 0 };

int ace_test_region_pads() {
    return g_test_region_pads.load();
}

// The region's one check after its shift, once the pad is sized and before it
// is allocated or any encode run: the region shifted into the source padded
// by s.left_pad_sec (into s.rs, s.re), refused if the shift leaves it empty
// -- float noise at an edge -- or if it covers no frame of the `frames` the
// encoder will keep, rounded up to the patch as ops_resolve_T rounds them, by
// the frame rule ops_build_context reads. A region of none would run the
// source's encode and the text encode for nothing (cadenza-audio #1137), and
// copy the whole source first (#1282). A run past its most was refused before
// this, for that is what is wrong with it (#1144). ops_build_context's check
// is a backstop, uncounted. True, said and counted, for a refusal.
static bool refuse_region(const AceSynth * ctx, const AceRequest & r, SynthState & s, float src_dur,
                          long long frames) {
    shifted_region(r.repainting_start, r.repainting_end, src_dur, s.left_pad_sec, s.rs, s.re);
    if (refuse_empty_region(s, s.re, s.rs)) {
        return true;
    }
    // A run its patch rounds past its most is left to the source's own
    // refusal, after (#1144).
    const long long T = frames_to_patch(frames, ctx->meta->cfg.patch_size);
    if (!frames_past_run_most(T)) {
        int t0 = 0, t1 = 0;
        region_frames_in(s.rs, s.re, (int) T, t0, t1);
        if (t1 <= t0) {
            g_test_region_no_frame.fetch_add(1);
            s.refusal = kAceRefusalRegionNoFrame;
            fprintf(stderr, "[Region] ERROR: the region %.3fs..%.3fs covers no latent frame\n", s.rs, s.re);
            return true;
        }
    }
    fprintf(stderr, "[Region] %.1fs..%.1fs\n", s.rs, s.re);
    return false;
}

// The runs refused for a source, with its pads if any, past a run's most
// frames -- before its pad is allocated or its encode run
// (ace_test_source_too_long).
static std::atomic<int> g_test_source_too_long{ 0 };

int ace_test_source_too_long() {
    return g_test_source_too_long.load();
}

// True, said and counted, for a run of more frames than a run's most: the one
// refusal the outpaint's pads and every task's source share (#1139, #1144).
static bool refuse_past_run_most(SynthState & s, long long frames, const char * what) {
    if (!frames_past_run_most(frames)) {
        return false;
    }
    g_test_source_too_long.fetch_add(1);
    s.refusal = kAceRefusalSourceTooLong;
    fprintf(stderr, "[Synth-Run] ERROR: %s, %lld frames, is past a run's most, %d\n", what, frames,
            ACE_MAX_LATENT_FRAMES);
    return true;
}

// Outpainting: pads the source when the region extends beyond its bounds, and
// returns the input to encode -- the source itself, or a SynthInput pointing
// into s.padded_src or s.padded_latents, valid while those are not resized.
// Audio path: always copies the source into s.padded_src (the post-decode
// splice reads it), zero-padded in samples; the VAE encoder turns the silence
// into ~silence latents.
// Latent path: pads into s.padded_latents with the precomputed silence_full
// latent (the canonical VAE encoding of audio silence), only when there is
// padding to add. No VAE encode needed.
// Either path pads on to one more frame only for a region that would else lie
// wholly past the source's last whole frame -- a region of none, refused --
// so a region with frames inside the source pads as it always did (#1138).
// Sets s.left_pad_sec, and the region shifted into the padded source (s.rs,
// s.re). False, said, nothing allocated, for a pad that would take the run
// past its most frames (ACE_MAX_LATENT_FRAMES) -- a far region bound, a
// repainting_end of 1e10 s, is refused before its pad is, not after gigabytes
// of one (#1139) -- or a region refuse_region refuses: before the pad, the
// source's encode and the text encode (#1137, cadenza-audio #1282).
static bool prepare_region(const AceSynth *   ctx,
                           const AceRequest & r,
                           const SynthInput & src,
                           SynthState &       s,
                           SynthInput &       enc) {
    // Both bounds are finite: run_task's batch rule refused a request whose
    // task reads a region and a bound of it is no number (#1150).
    enc = src;

    float src_dur  = src.duration_sec();
    float rs_raw   = r.repainting_start;
    float re_raw   = r.repainting_end;
    float end_time = asked_region_end(rs_raw, re_raw, src_dur);
    // A region that ends at or before its start -- an inverted repaint, or a
    // start at or past the source's end with the -1 end, which reads as the
    // source's end -- is refused here, before its pad or its source's encode,
    // as refuse_region would refuse it after (#1134).
    if (refuse_empty_region(s, end_time, rs_raw)) {
        return false;
    }
    float lpad     = (rs_raw < 0.0f) ? -rs_raw : 0.0f;
    float rpad     = (end_time > src_dur) ? end_time - src_dur : 0.0f;
    s.left_pad_sec = lpad;
    // The padded frames, in 64 bits, against the run's most.
    auto too_long  = [&s](int64_t frames) {
        return refuse_past_run_most(s, frames, "the source with its pads");
    };

    if (src.has_latents()) {
        if (lpad <= 0.0f && rpad <= 0.0f) {
            return !refuse_region(ctx, r, s, src_dur, src.T_latent);
        }
        // Latent pad: whole 25Hz frames, the nearest, silence_full spliced at
        // the edges. The left one places the source as near as a frame can to
        // where it was asked, and is a frame, not none, for a pad past the
        // slack (#1138).
        int           lpad_T = std::max(sec_latent_frames_nearest(lpad), sec_latent_frame_end(lpad) > 0 ? 1 : 0);
        int           rpad_T = sec_latent_frames_nearest(rpad);
        // The region is shifted by the pad as padded, not as asked: a 0.03 s
        // pad is one frame, 0.04 s (cadenza-audio #1129).
        float         rs = 0.0f, re = 0.0f;
        shifted_region(rs_raw, re_raw, src_dur, latent_frames_sec(lpad_T), rs, re);
        // A region wholly past the frames padded has the frames it reaches.
        const int64_t held   = (int64_t) lpad_T + src.T_latent + rpad_T;
        int           t0 = 0, t1 = 0;
        region_frames_unbounded(rs, re, t0, t1);
        const int past = (t0 >= held && t1 > held) ? t1 : 0;
        // The frames the run will hold, in 64 bits -- the region's end for one
        // lying wholly past the padded frames -- against the run's most,
        // before any is added or allocated: each pad is held to 1e9 frames,
        // and two can pass an int.
        const int64_t total  = past > 0 ? (int64_t) past : held;
        if (too_long(total)) {
            return false;
        }
        rpad_T += (int) (total - held);
        // Pads within the frame slack of none pad nothing: the source as
        // given, the region not shifted (#1129).
        if (lpad_T == 0 && rpad_T == 0) {
            s.left_pad_sec = 0.0f;
            return !refuse_region(ctx, r, s, src_dur, src.T_latent);
        }
        int total_T    = lpad_T + src.T_latent + rpad_T;
        s.left_pad_sec = latent_frames_sec(lpad_T);
        if (refuse_region(ctx, r, s, src_dur, total_T)) {
            return false;
        }
        const float * sil = ctx->meta->silence_full.data();
        g_test_region_pads.fetch_add(1);
        s.padded_latents.resize((size_t) total_T * 64);
        memcpy(s.padded_latents.data(), sil, (size_t) lpad_T * 64 * sizeof(float));
        memcpy(s.padded_latents.data() + (size_t) lpad_T * 64, src.latents, (size_t) src.T_latent * 64 * sizeof(float));
        memcpy(s.padded_latents.data() + (size_t) (lpad_T + src.T_latent) * 64, sil,
               (size_t) rpad_T * 64 * sizeof(float));
        enc.latents  = s.padded_latents.data();
        enc.T_latent = total_T;
        fprintf(stderr, "[Outpaint] latent pad left=%.2fs (%d) right=%.2fs (%d) total=%.1fs\n",
                latent_frames_sec(lpad_T), lpad_T, latent_frames_sec(rpad_T), rpad_T, latent_frames_sec(total_T));
        return true;
    }

    // Audio path: always populate s.padded_src so the post-decode waveform
    // splice always has the source PCM to read from, with or without outpaint.
    // Pads in whole samples, the nearest; the region shifted by the pad as
    // padded, as the latent path's is (#1129).
    int lpad_s                = sec_samples_nearest(lpad);
    int rpad_s                = sec_samples_nearest(rpad);
    s.left_pad_sec            = samples_sec(lpad_s);
    // The encoder keeps whole frames only, so a region reaching past the last
    // whole frame -- to a source's end inside one, or into a pad of less
    // than a frame -- would not be generated there: wholly past it, it would
    // have no frame and be refused as covering none (#1138); else its part
    // frame would be the source's, the take cut back to it (#1142). Then the
    // pad runs on to that frame's end, and the take is held to the source's
    // own length after its decode.
    const int64_t padded = (int64_t) lpad_s + src.len + rpad_s;
    const int64_t whole  = padded / ACE_LATENT_HOP;
    float         rs = 0.0f, re = 0.0f;
    shifted_region(rs_raw, re_raw, src_dur, s.left_pad_sec, rs, re);
    int t0 = 0, t1 = 0;
    region_frames_unbounded(rs, re, t0, t1);
    const bool run_on = t1 > whole;
    // The frames the encoder will keep -- whole ones, or the one the pad runs
    // on to -- against the run's most, before any is allocated: a 600 s
    // take is the most, though a float pad lands a sample past it (#1139).
    if (too_long(run_on ? padded_samples_frames(lpad_s, src.len, rpad_s) : whole)) {
        return false;
    }
    int padded_total = (int) padded;
    s.take_samples   = padded_total;
    if (run_on) {
        const int frame_end = samples_frames_ceil(padded_total) * ACE_LATENT_HOP;
        rpad_s += frame_end - padded_total;
        padded_total = frame_end;
    }
    // The frames the encoder keeps: whole ones (SynthInput::frames).
    if (refuse_region(ctx, r, s, src_dur, samples_frames_floor(padded_total))) {
        return false;
    }
    g_test_region_pads.fetch_add(1);
    s.padded_src.resize((size_t) padded_total * 2);
    if (lpad_s > 0 || rpad_s > 0) {
        memset(s.padded_src.data(), 0, s.padded_src.size() * sizeof(float));
    }
    memcpy(s.padded_src.data() + (size_t) lpad_s * 2, src.audio, (size_t) src.len * 2 * sizeof(float));
    enc.audio = s.padded_src.data();
    enc.len   = padded_total;
    if (lpad_s > 0 || rpad_s > 0) {
        fprintf(stderr, "[Outpaint] audio pad left=%.3fs (%d) right=%.3fs (%d) total=%.1fs\n", samples_sec(lpad_s),
                lpad_s, samples_sec(rpad_s), rpad_s, samples_sec(padded_total));
    }
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
    g_test_encode_passes.fetch_add(1);
    // Neither input given needs no pin either: the timbre takes its silence
    // path, the source no encode.
    const bool  have_src_audio   = src.has_audio();
    const bool  have_src_latents = src.has_latents();
    const bool  have_src         = src.given();
    const bool  have_ref_audio   = ref.has_audio();
    // A source past a run's most frames is refused here, before the pin
    // loads the encoder or the encode spends minutes on it -- for every synth
    // task, as ops_resolve_T would refuse its T after (cadenza-audio #1144).
    // An outpaint's pads were bounded before they were allocated (#1139);
    // this is the source a task encodes, padded or not. Its frames are
    // rounded up to the DiT's patch, as ops_resolve_T rounds T, so the two
    // bounds agree for any patch_size, not only one that divides the most.
    if (refuse_past_run_most(s, frames_to_patch(src.frames(), ctx->meta->cfg.patch_size), "the source")) {
        return false;
    }
    bool        need_vae_pin     = (have_src_audio || have_ref_audio);
    ModelHandle vae_pin(ctx->store, need_vae_pin ? store_require_vae_enc(ctx->store, ctx->vae_enc_key) : NULL);
    if (need_vae_pin && !vae_pin.ptr) {
        fprintf(stderr, "[Pipeline-Synth] FATAL: store_require_vae_enc (pin) failed\n");
        synth_failed(s, AceSynthFailure::model_unavailable, "the VAE encoder could not be loaded");
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
    // A repaint always splices its source back: its table row's
    // SynthSplice::always, which ace_synth_request_splices_source says of
    // one whose bounds the batch rule let run and the table scenario pins
    // (#1156, #1190), held true here whatever reaches this.
    s.is_repaint         = true;
    s.use_source_context = true;
    s.instruction_str    = DIT_INSTR_REPAINT;

    SynthInput enc;
    if (!prepare_region(ctx, reqs[0], in.src, s, enc)) {
        return -1;
    }

    if (!pinned_encode_src_and_timbre(ctx, enc, in.ref, s, progress)) {
        return -1;
    }
    if (run_tail(ctx, reqs, batch_n, s, progress) != 0) {
        return -1;
    }
    return 0;
}

// lego: stem generation. With a region (lego_asks_region): region-constrained, DiT generates
// only in the zone with full audio context. Without: whole-song generation.
// audio_cover_strength forced to 1.0 so all DiT steps hear the backing track:
// run_task holds every task but a cover to the cover knobs' defaults, by the
// table's cover_knobs (cadenza-audio #1135).
// Both in.src.audio and in.src.latents are accepted; in region mode the source is
// padded with silence at the edges when the region extends beyond bounds.
static int run_lego(AceSynth *          ctx,
                    const AceRequest *  reqs,
                    const SynthInputs & in,
                    int                 batch_n,
                    SynthState &        s,
                    AceProgress         progress) {
    // Whether the lego asks a region: the query a tool reads to tell a
    // spliced take, so the two cannot drift (cadenza-audio #1156). A bound
    // that is no number was refused by run_task's batch rule, not read as a
    // region or none (#1144, #1150).
    s.is_lego_region = ace_synth_request_splices_source(&reqs[0]);
    // An end of its own at or before the start asks no region a lego can
    // run -- a repaint refuses it -- so the lego runs the whole song, said.
    // (0, 0) is a client's unset range -- the webui resets to it -- not one.
    if (s.rr.repainting_end >= 0.0f && s.rr.repainting_end <= s.rr.repainting_start &&
        !(s.rr.repainting_start == 0.0f && s.rr.repainting_end == 0.0f)) {
        fprintf(stderr, "[Region] WARNING: lego region %.1f..%.1f s ends at or before its start: the whole song\n",
                s.rr.repainting_start, s.rr.repainting_end);
    }
    s.use_source_context    = true;
    std::string track_upper = prepare_track(s.rr.track, "Lego");
    s.instruction_str       = dit_instr_lego(track_upper);
    fprintf(stderr, "[Synth-Run] task=%s\n", reqs[0].task_type.c_str());
    warn_if_turbo_stem(ctx, "lego");

    // In region mode, padded at the edges the region passes; else as given.
    SynthInput enc = in.src;
    if (s.is_lego_region && !prepare_region(ctx, reqs[0], in.src, s, enc)) {
        return -1;
    }

    if (!pinned_encode_src_and_timbre(ctx, enc, in.ref, s, progress)) {
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

// A task's splice: never; always (a repaint, whose bounds are its region,
// the whole song among them); or when its bounds ask a region
// (lego_asks_region: a lego, whose defaults ask the whole song, no region).
enum class SynthSplice { never, always, asked_region };

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
    // Whether audio_cover_strength and cover_noise_strength are the task's
    // to read: a cover's alone (docs/ARCHITECTURE.md). Every other task runs
    // at their defaults -- a lego, an extract, a complete at full source
    // context, never switched to silence or started from the source (#1135).
    bool         cover_knobs;
    // Whether the task reads the region (repainting_start/_end) and the
    // track: a batch's requests must agree in those it reads (#1159).
    bool         reads_region;
    bool         reads_track;
    // Whether, and when, a run of the task splices its source back over its
    // decode after the decode ends -- what ace_synth_request_splices_source
    // reads and run_lego sets its flag from; run_repaint always splices, as
    // its row says (cadenza-audio #1156, #1190). The batch rule reads
    // asked_region to let takes that ask no region differ in their unset
    // bounds. A splice is of the region, so only a task that reads one
    // splices: the table test holds every row to that.
    SynthSplice  splices;
};

static const SynthTask kSynthTasks[] = {
    { TASK_TEXT2MUSIC,  run_text2music,  false, true,  false, false, false, SynthSplice::never        },
    { TASK_COVER,       run_cover,       true,  true,  true,  false, false, SynthSplice::never        },
    { TASK_COVER_NOFSQ, run_cover_nofsq, true,  true,  true,  false, false, SynthSplice::never        },
    { TASK_REPAINT,     run_repaint,     true,  true,  false, true,  false, SynthSplice::always       },
    { TASK_LEGO,        run_lego,        true,  false, false, true,  true,  SynthSplice::asked_region },
    { TASK_EXTRACT,     run_extract,     true,  false, false, false, true,  SynthSplice::never        },
    { TASK_COMPLETE,    run_complete,    true,  false, false, false, true,  SynthSplice::never        },
};

// Why a task the table does not hold is refused: by the batch rule, and by
// run_task should a call reach it past the rule.
static constexpr const char * kRefusalUnknownTask = "unknown task_type";

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
        info->cover_knobs   = entry->cover_knobs;
        info->reads_region  = entry->reads_region;
        info->reads_track   = entry->reads_track;
        info->splices       = entry->splices != SynthSplice::never;
    }
    return true;
}

int ace_synth_task_count() {
    return (int) (sizeof(kSynthTasks) / sizeof(kSynthTasks[0]));
}

const char * ace_synth_task_name(int index) {
    return index >= 0 && index < ace_synth_task_count() ? kSynthTasks[index].name : nullptr;
}

bool ace_synth_task_ignores_source(const char * task) {
    AceSynthTaskInfo info;
    return ace_synth_task_info(task, &info) && !info.needs_source;
}

// The input rule for a task's row, the source read as the run reads it
// (SynthInput::given): run_task and ace_synth_inputs_refused alike, so a
// tool and the run say one rule of one reading (cadenza-audio #1194).
static bool inputs_refused(const SynthTask & entry, const SynthInput & src, AceSynthBatchRefusal * why) {
    if (!entry.needs_source || src.given()) {
        return false;
    }
    if (why) {
        *why = { -1, "the task requires source audio or latents, and none was given" };
    }
    return true;
}

bool ace_synth_inputs_refused(const char *           task,
                              const float *          src_audio,
                              int                    src_len,
                              const float *          src_latents,
                              int                    src_T_latent,
                              AceSynthBatchRefusal * why) {
    const SynthTask * entry = task ? find_synth_task(task) : nullptr;
    return entry && inputs_refused(*entry, { src_audio, src_len, src_latents, src_T_latent }, why);
}

bool ace_synth_request_splices_source(const AceRequest * req) {
    const SynthTask * entry = req ? find_synth_task(req->task_type) : nullptr;
    if (!entry || !std::isfinite(req->repainting_start) || !std::isfinite(req->repainting_end)) {
        return false;
    }
    // run_repaint's is_repaint, and run_lego's is_lego_region, which both
    // set from this: what splices_source_after_decode reads of the run.
    switch (entry->splices) {
        case SynthSplice::never:
            return false;
        case SynthSplice::always:
            return true;
        case SynthSplice::asked_region:
            return lego_asks_region(req->repainting_start, req->repainting_end);
    }
    return false;
}

bool ace_synth_request_missing_caption(const AceRequest * req) {
    if (request_caption_given(req->caption)) {
        return false;
    }
    const SynthTask * entry = find_synth_task(req->task_type);
    return !entry || entry->needs_caption;
}

// Whether two floats are the same setting: NaN is NaN, so knobs asked alike
// differ in nothing.
static bool same_setting(float a, float b) {
    return a == b || (std::isnan(a) && std::isnan(b));
}

template <typename T> static bool same_setting(const T & a, const T & b) {
    return a == b;
}

// The first field the engine or the tools read from a batch's first request
// alone in which `b` differs from `a`, said as the reason a batch with them
// is refused; NULL when they agree (#1159). The task's own -- the region, the
// track and the cover knobs -- only for a task that reads them. A lego that
// asks no region asks the whole song, whichever unset bounds it sends.
// duration is not held here: a batch's takes of differing durations is
// #1162's.
// `field` gets the JSON name of the field that differs.
static const char * shared_field_differs(const AceRequest & a,
                                         const AceRequest & b,
                                         const SynthTask &  task,
                                         const char *&      field) {
#define ACE_SHARED(name, read)                                                      \
    if ((read) && !same_setting(a.name, b.name)) {                                  \
        field = #name;                                                              \
        return "requests differ in " #name ": a batch runs as its first request's"; \
    }
    // A task whose bounds splice only when they ask a region (the table's
    // asked_region, a lego's) reads none when neither asks one (#1190). Both
    // requests' bounds are finite here: the caller refused any that is not
    // before comparing.
    const bool asked         = task.splices == SynthSplice::asked_region;
    const bool region_unread = asked && !lego_asks_region(a.repainting_start, a.repainting_end) &&
                               !lego_asks_region(b.repainting_start, b.repainting_end);
    ACE_SHARED(repainting_start, task.reads_region && !region_unread)
    ACE_SHARED(repainting_end, task.reads_region && !region_unread)
    ACE_SHARED(track, task.reads_track)
    // A text2music's codes are a take's own, but whether it has any is the
    // batch's: the first's set the instruction and the context for all
    // (run_text2music), so a take with codes and one without cannot share.
    // The caller refused codes that hold no code, so emptiness says it.
    if (!task.needs_source && a.audio_codes.empty() != b.audio_codes.empty()) {
        field = "audio_codes";
        return "requests differ in audio_codes: some give codes and some none, and a batch runs as its "
               "first request's";
    }
    ACE_SHARED(audio_cover_strength, task.cover_knobs)
    ACE_SHARED(cover_noise_strength, task.cover_knobs)
    ACE_SHARED(inference_steps, true)
    ACE_SHARED(guidance_scale, true)
    ACE_SHARED(shift, true)
    ACE_SHARED(custom_timesteps, true)
    ACE_SHARED(solver, true)
    ACE_SHARED(stork_substeps, true)
    ACE_SHARED(dcw_scaler, true)
    ACE_SHARED(dcw_high_scaler, true)
    ACE_SHARED(dcw_mode, true)
    ACE_SHARED(latent_shift, true)
    ACE_SHARED(latent_rescale, true)
    ACE_SHARED(synth_model, true)
    ACE_SHARED(vae, true)
    ACE_SHARED(adapter, true)
    ACE_SHARED(adapter_scale, true)
    ACE_SHARED(output_format, true)
    ACE_SHARED(peak_clip, true)
#undef ACE_SHARED
    return nullptr;
}

// A request's own fault in a batch of `task`, said as that rather than as a
// difference: a region bound that is no number, where the task reads a
// region, and a text2music's codes that hold no code, which the LM reads as
// codes given (it skips its own) and the synth would read as none -- and
// shared_field_differs reads as codes held. NULL when it has none. The batch
// rule asks it of each request (cadenza-audio #1165).
static const char * request_own_fault(const AceRequest & req, const SynthTask & task) {
    if (task.reads_region && !(std::isfinite(req.repainting_start) && std::isfinite(req.repainting_end))) {
        return kAceRefusalRegionNonfinite;
    }
    if (!task.needs_source && !req.audio_codes.empty() && !ops_codes_given(req.audio_codes)) {
        return "audio_codes holds no code";
    }
    return nullptr;
}

bool ace_synth_batch_refused(const AceRequest * reqs, int batch_n, AceSynthBatchRefusal * why, bool planned) {
    auto refuse = [&](int request, const char * reason, const char * field = nullptr) {
        if (why) {
            why->request = request;
            why->reason  = reason;
            why->field   = field;
        }
        return true;
    };
    if (!reqs || batch_n < 1) {
        return refuse(-1, "a batch holds one request at least");
    }
    // Each request's task first, so a misspelt one -- at any index -- is said
    // as that, not as a mix or the caption an unknown task is short of.
    for (int i = 0; i < batch_n; i++) {
        if (!find_synth_task(reqs[i].task_type)) {
            return refuse(i, kRefusalUnknownTask);
        }
    }
    for (int i = 0; i < batch_n; i++) {
        // A batch runs as its first request's task: another's would run as
        // it, unsaid -- a text2music as a cover, a cover without its source.
        if (reqs[i].task_type != reqs[0].task_type) {
            return refuse(i, "requests mix tasks: a batch runs as its first request's task");
        }
        // A planned batch's later requests take theirs from its first's plan,
        // so its first's caption is the batch's: refused as the batch's.
        if ((i == 0 || !planned) && ace_synth_request_missing_caption(&reqs[i])) {
            return refuse(planned ? -1 : i, "caption is required");
        }
    }
    // The engine reads a batch's shared settings from its first request alone
    // (s.rr): another's would run as the first's, unsaid -- two legos of two
    // tracks both as the first's, two repaints of two regions both the
    // first's. Refused, by the first field that differs (#1159). A take's own
    // are allowed: its seed, caption, lyrics, metas and codes, its
    // synth_batch_size and mp3_bitrate, and the LM's fields synth never reads.
    const SynthTask & task = *find_synth_task(reqs[0].task_type);
    // Each request on its own first, so a fault of one is said as that, not
    // as a difference -- the first's too, so a tool answers it as it answers
    // a later one's.
    for (int i = 0; i < batch_n; i++) {
        if (const char * fault = request_own_fault(reqs[i], task)) {
            return refuse(i, fault);
        }
    }
    for (int i = 1; i < batch_n; i++) {
        const char * field = nullptr;
        if (const char * reason = shared_field_differs(reqs[0], reqs[i], task, field)) {
            return refuse(i, reason, field);
        }
    }
    return false;
}

AceSynthBatchTakes ace_synth_batch_takes(const AceRequest * reqs, int batch_n) {
    AceSynthBatchTakes batch;
    if (!reqs) {
        return batch;
    }
    for (int i = 0; i < batch_n; i++) {
        const int takes = ace_synth_request_takes(reqs[i].synth_batch_size);
        batch.asked += ace_synth_request_asked(reqs[i].synth_batch_size);
        if (batch.total < kAceSynthMaxTakes) {
            const int room = kAceSynthMaxTakes - batch.total;
            const int runs = takes < room ? takes : room;
            batch.takes.push_back(runs);
            batch.total += runs;
        }
    }
    return batch;
}

int ace_synth_requests_run(const AceRequest * reqs, int batch_n) {
    return (int) ace_synth_batch_takes(reqs, batch_n).takes.size();
}

static std::atomic<int> g_test_batch_refusals{ 0 };

int ace_test_batch_refusals() {
    return g_test_batch_refusals.load();
}

// The runs run_task has warned of a source their task ignores
// (ace_test_source_ignores).
static std::atomic<int> g_test_source_ignores{ 0 };

int ace_test_source_ignores() {
    return g_test_source_ignores.load();
}

// What the last run made of the request's cover knobs (ace_test_last_cover).
// Reset when each run starts and written however it ends, so none is a
// former run's; one snapshot under one lock, so none mixes two runs.
static const AceTestCoverRecord kCoverRecordUnset{ -1.0f, -1.0f, false, -2, -1 };
static std::mutex               g_test_cover_mutex;
static AceTestCoverRecord       g_test_last_cover = kCoverRecordUnset;

static void set_test_last_cover(const AceTestCoverRecord & r) {
    std::lock_guard<std::mutex> lock(g_test_cover_mutex);
    g_test_last_cover = r;
}

AceTestCoverRecord ace_test_last_cover() {
    std::lock_guard<std::mutex> lock(g_test_cover_mutex);
    return g_test_last_cover;
}

// The cover knobs run_task has clamped, or run at their default for a NaN
// (ace_test_cover_knob_fixes).
static std::atomic<int> g_test_cover_knob_fixes{ 0 };

int ace_test_cover_knob_fixes() {
    return g_test_cover_knob_fixes.load();
}

// The run's progress, passed through to the caller's, seeing whether the
// callback ever asked to cancel: so run_task can say a run that stopped for
// it was cancelled, with no poll of its own -- a poll a caller would see, and
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

// Dispatches on reqs[0].task_type to the right task function. task_type is
// always set: request_init defaults it to text2music, the JSON parser ignores
// empty strings.
static AceSynthJob * run_task(AceSynth *             ctx,
                              const AceRequest *     reqs,
                              const SynthInputs &    in,
                              int                    batch_n,
                              AceProgress            progress,
                              AceSynthBatchRefusal & refused) {
    // Refused before the job is made: a batch the batch rule refuses -- an
    // unknown task, mixed tasks, a request short of its caption, a region
    // bound that is no number for a task that reads one (the table's
    // reads_region: the engine's one check of it, #1150); the rule the tools
    // ask too (#1123) -- and a task that works from a source given none
    // (the table's needs_source). A task that takes no source given one is
    // warned of.
    set_test_last_cover(kCoverRecordUnset);
    AceSynthBatchRefusal refusal;
    if (ace_synth_batch_refused(reqs, batch_n, &refusal)) {
        g_test_batch_refusals.fetch_add(1);
        fprintf(stderr, "[Synth-Run] ERROR: request %d (task '%s'): %s\n", refusal.request,
                refusal.request >= 0 ? reqs[refusal.request].task_type.c_str() : "none", refusal.reason);
        refused = refusal;
        return NULL;
    }
    // The task's row, looked up once: the rule above refused a task it does
    // not hold, and this holds the run to that by itself (cadenza-audio
    // #1194), so no row below is read through NULL.
    const std::string & task  = reqs[0].task_type;
    const SynthTask *   entry = find_synth_task(task);
    if (!entry) {
        refused = { 0, kRefusalUnknownTask };
        return NULL;
    }
    if (inputs_refused(*entry, in.src, &refused)) {
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
    SynthState & s = job_owner.get()->state;
    // What the ops that read the cover knobs ask (cadenza-audio #1154): the
    // rewrite below is what the run says and records of them.
    s.cover_knobs  = entry->cover_knobs;
    // The cover knobs' defaults, which both rules below run a knob at.
    AceRequest   def;
    request_init(&def);
    // The cover knobs are a cover's alone: every other task runs at their
    // defaults, said when a task with a source -- one they would have
    // reached -- was asked otherwise; text2music never read them (#1135).
    if (!entry->cover_knobs) {
        if (entry->needs_source && (s.rr.audio_cover_strength != def.audio_cover_strength ||
                                    s.rr.cover_noise_strength != def.cover_noise_strength)) {
            fprintf(stderr,
                    "[Synth-Run] WARNING: task '%s' runs at audio_cover_strength %g, cover_noise_strength %g, "
                    "not the %g, %g asked: they are a cover's\n",
                    task.c_str(), def.audio_cover_strength, def.cover_noise_strength, s.rr.audio_cover_strength,
                    s.rr.cover_noise_strength);
        }
        s.rr.audio_cover_strength = def.audio_cover_strength;
        s.rr.cover_noise_strength = def.cover_noise_strength;
    } else {
        // A cover's knobs are fractions: one out of [0, 1] -- an infinity
        // too -- is clamped to it, and a NaN, which no clamp can place, runs
        // at its default; either said and counted (cadenza-audio #1152, owner
        // decision). A strength below 0 ran as 1 -- its switch step was
        // negative, which no step reaches -- and now runs as 0, hearing the
        // source on no step; a noise past 1 runs as 1, the shortest schedule.
        auto in_range = [&task](float & knob, float fallback, const char * name) {
            const float asked = knob;
            knob              = std::isnan(asked) ? fallback : std::clamp(asked, 0.0f, 1.0f);
            if (!(knob == asked)) {  // true for a NaN too
                g_test_cover_knob_fixes.fetch_add(1);
                fprintf(stderr,
                        "[Synth-Run] WARNING: task '%s' runs at %s %g, not the %g asked: a fraction in [0, 1], "
                        "its default for a NaN\n",
                        task.c_str(), name, knob, asked);
            }
        };
        in_range(s.rr.audio_cover_strength, def.audio_cover_strength, "audio_cover_strength");
        in_range(s.rr.cover_noise_strength, def.cover_noise_strength, "cover_noise_strength");
    }

    // What the run made of the request, written however it ends -- a throw
    // included -- while the job's state still stands (destroyed before it).
    struct CoverRecord {
        const SynthState & s;

        ~CoverRecord() {
            set_test_last_cover({ s.rr.audio_cover_strength, s.rr.cover_noise_strength, s.need_enc_switch,
                                  s.cover_steps, s.num_steps });
        }
    } record{ s };

    CancelSeen seen{ progress };
    const int  rc = entry->run(ctx, reqs, in, batch_n, s, seen.progress());
    // A request the run refused partway -- its region, its length -- is
    // handed out as a refusal, as one refused before the job is (#1193).
    // Any other failure is said as its kind (#1092): a cancel the callback
    // asked for first, over what the op that stopped for it said; then the
    // op's own; else a failure no op named.
    if (rc != 0 && s.refusal) {
        refused = { -1, s.refusal };
    } else if (rc != 0) {
        if (seen.cancelled) {
            refused.failure = AceSynthFailure::cancelled;
            refused.failed  = "the run was cancelled";
        } else if (s.failure != AceSynthFailure::none) {
            refused.failure = s.failure;
            refused.failed  = s.failed;
        } else {
            refused.failure = AceSynthFailure::other;
            refused.failed  = "the run failed";
        }
    }
    return rc == 0 ? job_owner.release() : NULL;
}

// Phase 1 entry point.
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
                                    AceProgress            progress,
                                    AceSynthEncodes *      failed_encodes,
                                    AceSynthBatchRefusal * refused) {
    // Unset, and filled only by a refusal or a failure (#1160, #1092); a
    // caller that asks for none gets one of its own to fill.
    AceSynthBatchRefusal   none;
    AceSynthBatchRefusal & why = refused ? *refused : none;
    why                        = {};
    if (failed_encodes) {
        // Emptied and let go: a caller's sink reused after a failed call holds
        // nothing of it through this one.
        std::vector<float>().swap(failed_encodes->src_latents);
        std::vector<float>().swap(failed_encodes->ref_latents);
    }
    // Reset before any refusal, so a call refused here leaves no former
    // run's record (run_task resets it again for its own).
    set_test_last_cover(kCoverRecordUnset);
    static_assert(kAceSynthMaxTakes == 9, "the refusal below says the cap");
    if (!ctx || !reqs || batch_n < 1 || batch_n > kAceSynthMaxTakes) {
        why = { -1, "a run takes a synth and 1..9 requests" };
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
    return run_task(ctx, reqs, in, batch_n, progress, why);
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

