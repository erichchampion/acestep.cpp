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
// So after each compute, each GPU backend of the scheduler is probed with an
// empty graph, submitted asynchronously: a backend in the error state refuses
// it before encoding anything, a healthy one commits an empty command buffer
// and nothing waits for it. The CPU backend has no such state and is not
// probed. A failure raises ace_gpu_error (an ace_fatal_error; the app build's
// C shim reports it as its own status), naming the stage. A scheduler that
// could not allocate the graph is out of memory, not a GPU fault:
// std::bad_alloc.
//
// The backend stays broken until it is recreated. Modules share one backend
// (backend.h), so ModelStore, finding a cached module's backend broken on a
// hit, stops sharing it (backend_invalidate) and drops every module on it:
// the next load makes a fresh backend, and modules still held free the old
// one as they are released.
//
// ACE_TEST_GPU_FAIL=<stage> (tests only) makes every compute of that stage
// fail while it is set, as a discarded command buffer does, and marks the
// backend broken, so the store's probe sees what Metal's would.
#include "ace-fatal.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <new>
#include <unordered_set>

// The GPU failed a compute: an ace_fatal_error of its own type, so an
// embedder can tell the user this, not "the engine hit a problem" -- trying
// again works (a fresh backend is loaded), and a shorter or smaller run asks
// less of the GPU. Thrown only under ACESTEP_FATAL_THROWS.
struct ace_gpu_error : ace_fatal_error {
    using ace_fatal_error::ace_fatal_error;
};

namespace ace_gpu {

// Backends ACE_TEST_GPU_FAIL has broken. Empty outside tests; `any` spares
// the lock on every probe.
inline std::mutex & poisoned_mtx() {
    static std::mutex m;
    return m;
}

inline std::unordered_set<const void *> & poisoned() {
    static std::unordered_set<const void *> p;
    return p;
}

inline std::atomic<bool> & any_poisoned() {
    static std::atomic<bool> a{ false };
    return a;
}

inline bool is_poisoned(ggml_backend_t b) {
    if (!any_poisoned().load(std::memory_order_relaxed)) {
        return false;
    }
    std::lock_guard<std::mutex> lock(poisoned_mtx());
    return poisoned().count(b) > 0;
}

// One empty graph for every probe: computing it reads nothing and writes
// nothing, so sharing it across threads is safe. Built once, never freed.
inline ggml_cgraph * empty_graph() {
    static ggml_cgraph * g = [] {
        ggml_init_params p = { ggml_graph_overhead_custom(1, false), nullptr, true };
        return ggml_new_graph_custom(ggml_init(p), 1, false);
    }();
    return g;
}

inline bool is_cpu(ggml_backend_t b) {
    ggml_backend_dev_t d = ggml_backend_get_device(b);
    return d && ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_CPU;
}

}  // namespace ace_gpu

// Whether GPU backend `b` can still run work (a CPU backend always can).
inline bool ace_backend_healthy(ggml_backend_t b) {
    if (!b || ace_gpu::is_cpu(b)) {
        return true;
    }
    if (ace_gpu::is_poisoned(b)) {
        return false;
    }
    return ggml_backend_graph_compute_async(b, ace_gpu::empty_graph()) == GGML_STATUS_SUCCESS;
}

// The first GPU backend of `sched` that can no longer run work, or null.
inline ggml_backend_t ace_sched_broken_backend(ggml_backend_sched_t sched) {
    if (!sched) {
        return nullptr;
    }
    const int n = ggml_backend_sched_get_n_backends(sched);
    for (int i = 0; i < n; i++) {
        ggml_backend_t b = ggml_backend_sched_get_backend(sched, i);
        if (!ace_backend_healthy(b)) {
            return b;
        }
    }
    return nullptr;
}

// Forget a backend being freed, so a new one at the same address is not
// taken for broken (tests only ever add to the set).
inline void ace_backend_forget(ggml_backend_t b) {
    if (!ace_gpu::any_poisoned().load(std::memory_order_relaxed)) {
        return;
    }
    std::lock_guard<std::mutex> lock(ace_gpu::poisoned_mtx());
    ace_gpu::poisoned().erase(b);
}

// Raise the GPU failure for stage `what`.
[[noreturn]] inline void ace_gpu_fail(const char * what, const char * why) {
    char msg[256];
    snprintf(msg, sizeof(msg), "[%s] FATAL: the GPU did not complete this step (%s); the device may have reset\n",
             what, why);
#ifdef ACESTEP_FATAL_THROWS
    fputs(msg, stderr);
    throw ace_gpu_error(1, msg);
#else
    ace_fatal(1, "%s", msg);
#endif
}

// Whether ACE_TEST_GPU_FAIL names this stage (tests only).
inline bool ace_gpu_injected(const char * what) {
    const char * want = std::getenv("ACE_TEST_GPU_FAIL");
    return want && *want && std::strcmp(want, what) == 0;
}

// Mark `b` broken, as a discarded buffer leaves Metal's (tests only).
inline void ace_gpu_poison(ggml_backend_t b) {
    if (!b || ace_gpu::is_cpu(b)) {
        return;
    }
    std::lock_guard<std::mutex> lock(ace_gpu::poisoned_mtx());
    ace_gpu::poisoned().insert(b);
    ace_gpu::any_poisoned().store(true);
}

// Run `gf` on `sched`, and fail the run if the GPU did not (see above).
// `what` names the stage in the message ("DiT", "VAE-Decode", ...).
inline void ace_graph_compute(ggml_backend_sched_t sched, ggml_cgraph * gf, const char * what) {
    ggml_status st = ggml_backend_sched_graph_compute(sched, gf);
    if (ace_gpu_injected(what)) {
        const int n = ggml_backend_sched_get_n_backends(sched);
        for (int i = 0; i < n; i++) {
            ace_gpu_poison(ggml_backend_sched_get_backend(sched, i));
        }
        st = GGML_STATUS_FAILED;
    }
    if (st == GGML_STATUS_ALLOC_FAILED) {
        fprintf(stderr, "[%s] FATAL: could not allocate the graph\n", what);
        throw std::bad_alloc();
    }
    if (st != GGML_STATUS_SUCCESS) {
        char why[64];
        snprintf(why, sizeof(why), "ggml status %d", (int) st);
        ace_gpu_fail(what, why);
    }
    if (ace_sched_broken_backend(sched)) {
        ace_gpu_fail(what, "the backend is in its error state");
    }
}

// The same for a graph run directly on one backend (the adapter merge).
inline void ace_backend_compute(ggml_backend_t backend, ggml_cgraph * gf, const char * what) {
    ggml_status st = ggml_backend_graph_compute(backend, gf);
    if (ace_gpu_injected(what)) {
        ace_gpu_poison(backend);
        st = GGML_STATUS_FAILED;
    }
    if (st == GGML_STATUS_ALLOC_FAILED) {
        fprintf(stderr, "[%s] FATAL: could not allocate the graph\n", what);
        throw std::bad_alloc();
    }
    if (st != GGML_STATUS_SUCCESS) {
        char why[64];
        snprintf(why, sizeof(why), "ggml status %d", (int) st);
        ace_gpu_fail(what, why);
    }
    if (!ace_backend_healthy(backend)) {
        ace_gpu_fail(what, "the backend is in its error state");
    }
}
