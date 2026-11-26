#pragma once
// synth-batch-runner.h: two-phase orchestration shared by the synth binaries
//
// Phase 1 (all groups) runs ace_synth_job_run_dit. Each call acquires the DiT
// from the store for the duration of its denoising loop and releases it on
// scope exit. Under EVICT_STRICT this means the VAE encoder, text encoder,
// cond encoder and DiT never coexist in VRAM; under EVICT_NEVER they all
// accumulate across calls. Phase 2 (all jobs) runs ace_synth_job_run_vae,
// which acquires the VAE decoder on entry and releases it on exit.

#include "pipeline-synth.h"

#include <cstdio>
#include <string>
#include <vector>

// A batch expanded to the takes that run: each running request's seed
// resolved once, its takes that seed + 0, 1, ... in order, by the counts
// ace_synth_batch_takes gave -- the one expansion both tools make, from the
// batch computed once per job (cadenza-audio #1243, #1269). `take_request[k]`
// is the request take k came from, `take_index[k]` (when given) its place
// among that request's takes. reqs' seeds are resolved in place.
static void synth_batch_expand(std::vector<AceRequest> &  reqs,
                               const AceSynthBatchTakes & batch,
                               std::vector<AceRequest> &  group,
                               std::vector<int> &         take_request,
                               std::vector<int> *         take_index = nullptr) {
    group.clear();
    take_request.clear();
    group.reserve(batch.total);
    take_request.reserve(batch.total);
    if (take_index) {
        take_index->clear();
        take_index->reserve(batch.total);
    }
    for (int ri = 0; ri < (int) batch.takes.size() && ri < (int) reqs.size(); ri++) {
        request_resolve_seed(&reqs[ri]);
        const long long base_seed = reqs[ri].seed;
        for (int i = 0; i < batch.takes[ri]; i++) {
            AceRequest r = reqs[ri];
            r.seed       = base_seed + i;
            group.push_back(r);
            take_request.push_back(ri);
            if (take_index) {
                take_index->push_back(i);
            }
        }
    }
}

// What a tool logs of a batch of `batch_n` requests (cadenza-audio #1269):
// the tracks and the requests that run, and, when a take asked is dropped,
// the takes asked against the DiT's most and -- when whole requests are --
// the last that runs and the first dropped, each said by `name(i)` (a CLI's
// request file, the server's index). One line each, untagged.
template <class Name>
static std::vector<std::string> synth_batch_notes(const AceSynthBatchTakes & batch, int batch_n, Name name) {
    std::vector<std::string> notes;
    const int                runs = (int) batch.takes.size();
    char                     line[256];
    if (batch.asked > batch.total) {
        snprintf(line, sizeof(line), "Batch %lld exceeds DiT max %d, clamping", batch.asked, kAceSynthMaxTakes);
        notes.push_back(line);
    }
    if (batch.total > 1 || runs < batch_n) {
        snprintf(line, sizeof(line), "Batch: %d track(s) from %d of %d request(s)", batch.total, runs, batch_n);
        notes.push_back(line);
    }
    if (runs > 0 && runs < batch_n) {
        notes.push_back("Runs up to " + std::string(name(runs - 1)) + "; drops " + std::string(name(runs)) +
                        (batch_n - runs > 1 ? " and the " + std::to_string(batch_n - runs - 1) + " after it" : ""));
    }
    return notes;
}

// Run a batch of request groups through the two synthesis phases.
//
// groups[g][i]: request i of group g. All requests in a group must share
//   the same T (same audio_codes or same duration), which the ops assume
//   when they stack per-batch tensors for a single DiT forward.
//   seed must be resolved (non-negative) on every request.
// src_audio / ref_audio: interleaved stereo 48kHz buffers, NULL when not applicable.
// src_latents / ref_latents: pre-encoded latents [T_latent * 64] f32 alternative
//   to the matching audio buffer. When non-NULL, the corresponding VAE encoder
//   pass is skipped for every group. The same buffers are shared across groups,
//   matching how src_audio and ref_audio are shared today.
// audio_out[sum_g(groups[g].size())]: pre-allocated slots filled by phase 2.
//   On error, slots completed before the failure keep their audio; the rest
//   are left at {NULL, 0, 0}. Caller owns ace_audio_free.
// latents_out: optional capture of one post-DiT latent per generated track,
//   indexed identically to audio_out. Each entry is [T_track * 64] f32 time-major,
//   T_track = entry.size() / 64. Pass NULL to skip the capture.
// refused: when non-NULL, why the engine refused a group's run -- before its
//   job or partway, as ace_synth_job_run_dit says it (cadenza-audio #1160,
//   #1193) -- its request counted across every group, as audio_out is; or
//   why a group's run or its VAE decode failed otherwise, its failure and
//   failed (cadenza-audio #1092, #1286), set with every failure.
// Returns 0 on success, -1 on any error or cancellation.
static int synth_batch_run(AceSynth *                             ctx,
                           std::vector<std::vector<AceRequest>> & groups,
                           const float *                          src_audio,
                           int                                    src_len,
                           const float *                          src_latents,
                           int                                    src_T_latent,
                           const float *                          ref_audio,
                           int                                    ref_len,
                           const float *                          ref_latents,
                           int                                    ref_T_latent,
                           AceAudio *                             audio_out,
                           std::vector<std::vector<float>> *      latents_out = nullptr,
                           AceProgress                            progress    = {},
                           AceSynthBatchRefusal *                 refused     = nullptr) {
    const int                  n_groups = (int) groups.size();
    std::vector<AceSynthJob *> jobs(n_groups, nullptr);
    std::vector<int>           audio_off(n_groups, 0);

    if (latents_out) {
        latents_out->clear();
    }

    // Phase 1: denoising loop for each group. The DiT is acquired and released
    // by ops_dit_generate inside ace_synth_job_run_dit.
    int off = 0;
    for (int g = 0; g < n_groups; g++) {
        const int gn = (int) groups[g].size();
        jobs[g] = ace_synth_job_run_dit(ctx, groups[g].data(), src_audio, src_len, src_latents, src_T_latent, ref_audio,
                                        ref_len, ref_latents, ref_T_latent, gn, progress, nullptr, refused);
        if (!jobs[g]) {
            if (refused && refused->reason && refused->request >= 0) {
                refused->request += off;
            }
            for (int j = 0; j < g; j++) {
                ace_synth_job_free(jobs[j]);
            }
            return -1;
        }
        audio_off[g] = off;
        off += gn;
    }

    // Capture one post-DiT latent per track, time-major [T*64], indexed to
    // match audio_out. Latents live in jobs[g]->state.output until run_vae
    // frees the job; capture happens before phase 2.
    if (latents_out) {
        latents_out->resize((size_t) off);
        for (int g = 0; g < n_groups; g++) {
            const int gn = (int) groups[g].size();
            for (int i = 0; i < gn; i++) {
                int           T   = 0;
                const float * src = ace_synth_job_get_latent(jobs[g], i, &T);
                (*latents_out)[audio_off[g] + i].assign(src, src + (size_t) T * 64);
            }
        }
    }

    // Phase 2: latent splice + VAE decode for each job. The decoder is
    // acquired and released by ops_vae_decode inside ace_synth_job_run_vae.
    for (int g = 0; g < n_groups; g++) {
        const int rc = ace_synth_job_run_vae(ctx, jobs[g], audio_out + audio_off[g], progress, refused);
        ace_synth_job_free(jobs[g]);
        jobs[g] = nullptr;
        if (rc != 0) {
            for (int j = g + 1; j < n_groups; j++) {
                ace_synth_job_free(jobs[j]);
            }
            return -1;
        }
    }

    return 0;
}
