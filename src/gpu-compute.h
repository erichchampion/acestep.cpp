#pragma once
// Every graph the engine runs goes through ace_graph_compute, which fails the
// run when the GPU did not do its work (cadenza-audio #403).
//
// ggml_backend_sched_graph_compute returns the status of *submitting* the
// graph; the command buffers run afterwards, and one the GPU discards (a GPU
// reset on iOS: kIOGPUCommandBufferCallbackErrorInnocentVictim) is only seen
// by the synchronize that follows, which returns nothing and leaves the Metal
// backend in an error state: every later compute on it is refused. Ignored,
// the run carries on over outputs the GPU never wrote -- a DiT "finishing"
// on garbage, then a decode that crashes the app.
//
// So after each compute, each GPU backend of the scheduler is asked for that
// state directly: the patched ggml exports ggml-metal's error flag
// (ggml_backend_metal_has_error, cadenza-audio #405), set by the synchronize
// the compute already ran -- no extra command buffer, and nothing that a
// ggml update moving graph_compute's own check could silently turn off. A
// GPU backend without that export is probed instead with an empty graph,
// which a backend in the error state refuses before encoding anything. The
// CPU backend has no such state and is not asked. A failure raises
// ace_gpu_error (an ace_fatal_error; the app build's C shim reports it as its
// own status), naming the stage. A scheduler that could not allocate the
// graph is out of memory, not a GPU fault: std::bad_alloc in the app build
// (the CLIs' ace_fatal otherwise).
//
// The backend stays broken until it is recreated. Modules share one backend
// (backend.h), so ModelStore, on every lookup -- a miss as much as a hit --
// finds cached modules on a broken backend, stops sharing it
// (ace_backend_invalidate) and drops them: the next load makes a fresh backend,
// and modules still held free the old one as they are released. ace_backend_init
// also refuses to share a broken backend, for loads outside the store.
//
// A failure the system caused -- iOS refusing GPU work from an app in the
// background -- is told apart from a fault by Metal's own error, which the
// patched ggml keeps (ggml_backend_metal_refused, cadenza-audio #817): the
// ace_gpu_error says `refused`, so an embedder runs the step again once the
// app is back instead of failing it.
//
// ace_test_fail_gpu_stage(<stage>) (tests only) makes every compute of that
// stage fail until it is cleared, as a discarded command buffer does, and
// marks the backend broken, so the store sees what Metal's flag would say;
// ace_test_refuse_gpu_stage(<stage>) does the same as the system's refusal.
// Set by a call, not read from the environment on every compute (#405): a
// shipping build pays one relaxed load.
//
// The state all this keeps -- the sets of backends a test broke, the
// injected stage, the per-registry export caches, the probe graph -- is
// defined once, in gpu-compute.cpp (cadenza-audio #1048), not as
// function-local statics in inline functions here, which were one copy only
// while each stayed `inline`: a failure injected in one translation unit is
// seen by every other by construction. What stays inline here holds none.
#include "ace-fatal.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <exception>
#include <string>

// The GPU failed a compute: an ace_fatal_error of its own type, so an
// embedder can tell the user this, not "the engine hit a problem" -- trying
// again works (a fresh backend is loaded), and a shorter or smaller run asks
// less of the GPU. Thrown only under ACESTEP_FATAL_THROWS.
// `refused`: the system refused the GPU (an app in the background), not a
// fault -- the step runs again once the app is back (#817).
struct ace_gpu_error : ace_fatal_error {
    bool refused;

    ace_gpu_error(int c, std::string msg, bool r = false) : ace_fatal_error(c, std::move(msg)), refused(r) {}
};

namespace ace_gpu {

// A function the backend's own registry exports by name -- the patched
// ggml's ggml_backend_metal_has_error, ggml_backend_metal_set_n_cb -- or
// null for a backend without it. Asked by the per-registry cache below and by
// backend.cpp for the n_cb setter; backend.cpp also looks up the Metal ceiling
// and the CPU thread setter on a registry directly, before a backend exists.
inline void * backend_proc(ggml_backend_t b, const char * name) {
    ggml_backend_dev_t d = b ? ggml_backend_get_device(b) : nullptr;
    ggml_backend_reg_t r = d ? ggml_backend_dev_backend_reg(d) : nullptr;
    return r ? ggml_backend_reg_get_proc_address(r, name) : nullptr;
}

}  // namespace ace_gpu

// Whether GPU backend `b` can still run work (a CPU backend always can).
bool ace_backend_healthy(ggml_backend_t b);

// Whether broken backend `b` was broken by the system's refusal of the GPU
// (#817) rather than a fault. A backend that cannot say was not.
bool ace_backend_refused(ggml_backend_t b);

// The first GPU backend of `sched` that can no longer run work, or null.
ggml_backend_t ace_sched_broken_backend(ggml_backend_sched_t sched);

// Forget a backend being freed, so a new one at the same address is not
// taken for broken (tests only ever add to the set).
void ace_backend_forget(ggml_backend_t b);

// Raise the GPU failure for stage `what`; `refused`, the system's refusal.
[[noreturn]] void ace_gpu_fail(const char * what, const char * why, bool refused = false);

// Make every compute of `stage` fail, as a discarded command buffer does, or
// -- with null or "" -- stop (tests only).
void ace_test_fail_gpu_stage(const char * stage, bool refused = false);

// The same, failing as the system's refusal of the GPU does (tests only).
void ace_test_refuse_gpu_stage(const char * stage);

// Whether a test has made this stage fail (tests only).
bool ace_gpu_injected(const char * what);

// Mark `b` broken, as a discarded buffer leaves Metal's (tests only).
void ace_gpu_poison(ggml_backend_t b);

// Fail stage `what` for a compute that returned `st`, or whose backend it
// left broken (`broken`, else null) -- as refused when the system refused it.
void ace_check_compute(ggml_status st, ggml_backend_t broken, const char * what);

// Run `gf` on `sched`, and fail the run if the GPU did not (see above).
// `what` names the stage in the message ("DiT", "VAE-Decode", ...).
void ace_graph_compute(ggml_backend_sched_t sched, ggml_cgraph * gf, const char * what);

// The same for a graph run directly on one backend (the adapter merge).
void ace_backend_compute(ggml_backend_t backend, ggml_cgraph * gf, const char * what);

// Frees a per-call graph context -- and resets its scheduler -- when a failed
// compute throws past the code that frees them; does nothing otherwise, so
// the normal path's own free is untouched.
struct ace_graph_unwind_guard {
    ggml_backend_sched_t sched;
    ggml_context *       ctx;
    int                  pending = std::uncaught_exceptions();

    ~ace_graph_unwind_guard() {
        if (std::uncaught_exceptions() > pending) {
            if (sched) {
                ggml_backend_sched_reset(sched);
            }
            if (ctx) {
                ggml_free(ctx);
            }
        }
    }
};
