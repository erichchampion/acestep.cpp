#pragma once
// Progress + cancel callback for the generation pipelines.
//
// This widens the old `bool (*cancel)(void *)` + `void * cancel_data` pair (#18):
// the same callback now reports where the run is (stage, step, total) AND cancels
// by returning true. Every pipeline that took the cancel pair takes an
// AceProgress by value instead, defaulted to {} (no callback = no progress, never
// cancel), so callers that passed nothing are unchanged.

enum AceStage {
    ACE_STAGE_LM,          // LM planning + code generation
    ACE_STAGE_DIT,         // DiT flow-matching sampler
    ACE_STAGE_VAE_ENCODE,  // VAE tiled encode (reference audio -> latents)
    ACE_STAGE_VAE_DECODE,  // VAE tiled decode (latents -> audio)
    ACE_STAGE_MP3,         // MP3 output encode
};

// fn is called at each pipeline poll with the current (stage, step, total) and
// returns true to cancel the run. step is 0-based; total is the loop's iteration
// count. A report with step == 0 is emitted before a loop's first unit of work so
// a UI can size a segmented bar (and so an empty loop still announces its stage).
// fn == nullptr means no progress and never cancel.
//
// Thread-safety: fn is called sequentially for LM, DiT and VAE, but MP3 encode is
// fork-join, so fn may be called concurrently from worker threads there. A cancel
// implementation that only reads a flag (the common case) is naturally safe; one
// that also records progress must tolerate concurrent calls, or ignore
// ACE_STAGE_MP3.
struct AceProgress {
    bool (*fn)(void * data, AceStage stage, int step, int total) = nullptr;
    void * data                                                  = nullptr;
};

// The single place the pipelines poll: reports (stage, step, total) and returns
// true if the callback asked to cancel.
static inline bool ace_progress(const AceProgress & p, AceStage stage, int step, int total) {
    return p.fn && p.fn(p.data, stage, step, total);
}
