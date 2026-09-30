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
// So after each compute, every backend of the scheduler is probed with an
// empty graph: a backend in the error state refuses it at once (Metal checks
// before encoding anything), a healthy one commits nothing. A failure of
// either raises ace_fatal -- thrown as ace_fatal_error in the app build, whose
// C shim reports it as a failed generation -- naming the stage.
//
// The backend stays broken until it is recreated: ModelStore probes a cached
// module with ace_sched_healthy on every hit and replaces a broken one, so
// the next generation loads a fresh backend instead of failing the same way.
//
// ACE_TEST_GPU_FAIL=<stage> (tests only) makes the first compute of that
// stage fail as a discarded command buffer would, once per process, and
// marks the scheduler broken so the store's probe sees it too.
#include "ace-fatal.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_set>

// The GPU failed a compute: an ace_fatal_error of its own type, so an
// embedder can tell the user this, not "the engine hit a problem" -- trying
// again works (the store loads a fresh backend), and a shorter or smaller run
// asks less of the GPU. Thrown only under ACESTEP_FATAL_THROWS.
struct ace_gpu_error : ace_fatal_error {
    using ace_fatal_error::ace_fatal_error;
};

namespace ace_gpu {

// Schedulers ACE_TEST_GPU_FAIL has broken. Never populated outside tests.
inline std::mutex &                          poisoned_mtx() { static std::mutex m; return m; }
inline std::unordered_set<const void *> &    poisoned() { static std::unordered_set<const void *> p; return p; }

// An empty graph to probe backends with; one per thread (built once, never
// freed: a few hundred bytes).
inline ggml_cgraph * empty_graph() {
    static thread_local ggml_cgraph * g = nullptr;
    if (!g) {
        ggml_init_params p = { ggml_graph_overhead_custom(1, false), nullptr, true };
        ggml_context *   c = ggml_init(p);
        g                  = ggml_new_graph_custom(c, 1, false);
    }
    return g;
}

// Whether this stage is the one ACE_TEST_GPU_FAIL names and has not fired yet.
inline bool injected_failure(const char * what) {
    static std::atomic<bool> fired{ false };
    const char *             want = std::getenv("ACE_TEST_GPU_FAIL");
    if (!want || !*want || std::strcmp(want, what) != 0) {
        return false;
    }
    return !fired.exchange(true);
}

}  // namespace ace_gpu

// Whether every backend of `sched` can still run work.
inline bool ace_sched_healthy(ggml_backend_sched_t sched) {
    if (!sched) {
        return true;
    }
    {
        std::lock_guard<std::mutex> lock(ace_gpu::poisoned_mtx());
        if (ace_gpu::poisoned().count(sched)) {
            return false;
        }
    }
    const int n = ggml_backend_sched_get_n_backends(sched);
    for (int i = 0; i < n; i++) {
        ggml_backend_t b = ggml_backend_sched_get_backend(sched, i);
        if (b && ggml_backend_graph_compute(b, ace_gpu::empty_graph()) != GGML_STATUS_SUCCESS) {
            return false;
        }
    }
    return true;
}

// Forget a scheduler the store is freeing, so a new one at the same address
// is not taken for broken (tests only ever add to the set).
inline void ace_sched_forget(ggml_backend_sched_t sched) {
    std::lock_guard<std::mutex> lock(ace_gpu::poisoned_mtx());
    ace_gpu::poisoned().erase(sched);
}

// Run `gf` on `sched`, and fail the run if the GPU did not (see above).
// `what` names the stage in the message ("DiT", "VAE-Decode", ...).
inline void ace_graph_compute(ggml_backend_sched_t sched, ggml_cgraph * gf, const char * what) {
    ggml_status st = ggml_backend_sched_graph_compute(sched, gf);
    if (ace_gpu::injected_failure(what)) {
        std::lock_guard<std::mutex> lock(ace_gpu::poisoned_mtx());
        ace_gpu::poisoned().insert(sched);
        st = GGML_STATUS_FAILED;
    }
    if (st != GGML_STATUS_SUCCESS || !ace_sched_healthy(sched)) {
#ifdef ACESTEP_FATAL_THROWS
        char msg[256];
        snprintf(msg, sizeof(msg),
                 "[%s] FATAL: the GPU did not complete this step (ggml status %d); the device may have reset\n", what,
                 (int) st);
        fputs(msg, stderr);
        throw ace_gpu_error(1, msg);
#else
        ace_fatal(1, "[%s] FATAL: the GPU did not complete this step (ggml status %d); the device may have reset\n",
                  what, (int) st);
#endif
    }
}
