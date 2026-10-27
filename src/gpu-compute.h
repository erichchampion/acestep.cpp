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
// (backend_invalidate) and drops them: the next load makes a fresh backend,
// and modules still held free the old one as they are released. backend_init
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
#include "ace-fatal.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <mutex>
#include <new>
#include <string>
#include <unordered_set>

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

// Backends a test's injected failure has broken. Empty outside tests; `any`
// spares the lock on every check.
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

// Of those, the ones broken as the system's refusal is (tests only).
inline std::unordered_set<const void *> & refused() {
    static std::unordered_set<const void *> r;
    return r;
}

inline bool is_poisoned(ggml_backend_t b) {
    if (!any_poisoned().load(std::memory_order_relaxed)) {
        return false;
    }
    std::lock_guard<std::mutex> lock(poisoned_mtx());
    return poisoned().count(b) > 0;
}

inline bool is_refused_by_test(ggml_backend_t b) {
    if (!any_poisoned().load(std::memory_order_relaxed)) {
        return false;
    }
    std::lock_guard<std::mutex> lock(poisoned_mtx());
    return refused().count(b) > 0;
}

// One empty graph for every probe: computing it reads nothing and writes
// nothing, so sharing it across threads is safe. Built once, never freed, in
// zeroed memory (a backend that peeks at nodes[0] finds null, not garbage).
inline ggml_cgraph * empty_graph() {
    static ggml_cgraph * g = [] {
        const size_t     size = ggml_graph_overhead_custom(1, false);
        ggml_init_params p    = { size, calloc(1, size), true };
        return ggml_new_graph_custom(ggml_init(p), 1, false);
    }();
    return g;
}

inline bool is_cpu(ggml_backend_t b) {
    ggml_backend_dev_t d = ggml_backend_get_device(b);
    return d && ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_CPU;
}

// A function the backend's own registry exports by name -- the patched
// ggml's ggml_backend_metal_has_error, ggml_backend_metal_set_n_cb -- or
// null for a backend without it. The one lookup backend.h shares.
inline void * backend_proc(ggml_backend_t b, const char * name) {
    ggml_backend_dev_t d = b ? ggml_backend_get_device(b) : nullptr;
    ggml_backend_reg_t r = d ? ggml_backend_dev_backend_reg(d) : nullptr;
    return r ? ggml_backend_reg_get_proc_address(r, name) : nullptr;
}

// A backend's export `name`, kept per registry: a registry's exports never
// change, and the error-state reader is asked on every compute. One
// immutable entry behind one atomic pointer, so a registry is never read
// paired with another's export (#613). An entry replaced is leaked: there
// are as many as registries ever asked, a handful.
struct proc_entry {
    ggml_backend_reg_t reg;
    void *             fn;
};
inline void * cached_proc(ggml_backend_t b, const char * name, std::atomic<const proc_entry *> & cached) {
    ggml_backend_dev_t d = ggml_backend_get_device(b);
    ggml_backend_reg_t r = d ? ggml_backend_dev_backend_reg(d) : nullptr;
    if (!r) {
        return nullptr;
    }
    const proc_entry * e = cached.load(std::memory_order_acquire);
    if (e && e->reg == r) {
        return e->fn;
    }
    auto * fresh = new proc_entry{ r, backend_proc(b, name) };
    cached.store(fresh, std::memory_order_release);
    return fresh->fn;
}

// The backend's error-state reader (#405), or null.
using has_error_t = bool (*)(ggml_backend_t);
inline has_error_t has_error_of(ggml_backend_t b) {
    static std::atomic<const proc_entry *> cached{ nullptr };
    return (has_error_t) cached_proc(b, "ggml_backend_metal_has_error", cached);
}

// Whether the backend's error state is the system's refusal (#817), or null.
using refused_t = bool (*)(ggml_backend_t);
inline refused_t refused_of(ggml_backend_t b) {
    static std::atomic<const proc_entry *> cached{ nullptr };
    return (refused_t) cached_proc(b, "ggml_backend_metal_refused", cached);
}

// The stage a test has made fail (tests only): `armed` spares the lock on
// every compute when none is.
inline std::atomic<bool> & injection_armed() {
    static std::atomic<bool> a{ false };
    return a;
}
inline std::string & injected_stage() {
    static std::string s;
    return s;
}
inline bool & injected_refusal() {
    static bool r = false;
    return r;
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
    if (ace_gpu::has_error_t has_error = ace_gpu::has_error_of(b)) {
        return !has_error(b);
    }
    return ggml_backend_graph_compute(b, ace_gpu::empty_graph()) == GGML_STATUS_SUCCESS;
}

// Whether broken backend `b` was broken by the system's refusal of the GPU
// (#817) rather than a fault. A backend that cannot say was not.
inline bool ace_backend_refused(ggml_backend_t b) {
    if (!b || ace_gpu::is_cpu(b)) {
        return false;
    }
    if (ace_gpu::is_refused_by_test(b)) {
        return true;
    }
    ace_gpu::refused_t refused = ace_gpu::refused_of(b);
    return refused && refused(b);
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
    ace_gpu::refused().erase(b);
}

// Raise the GPU failure for stage `what`; `refused`, the system's refusal.
[[noreturn]] inline void ace_gpu_fail(const char * what, const char * why, bool refused = false) {
    char msg[256];
    if (refused) {
        snprintf(msg, sizeof(msg), "[%s] FATAL: the system refused the GPU this step (%s); the app is in the background\n",
                 what, why);
    } else {
        snprintf(msg, sizeof(msg), "[%s] FATAL: the GPU did not complete this step (%s); the device may have reset\n",
                 what, why);
    }
#ifdef ACESTEP_FATAL_THROWS
    fputs(msg, stderr);
    throw ace_gpu_error(1, msg, refused);
#else
    ace_fatal(1, "%s", msg);
#endif
}

// Make every compute of `stage` fail, as a discarded command buffer does, or
// -- with null or "" -- stop (tests only).
inline void ace_test_fail_gpu_stage(const char * stage, bool refused = false) {
    std::lock_guard<std::mutex> lock(ace_gpu::poisoned_mtx());
    ace_gpu::injected_stage()   = stage ? stage : "";
    ace_gpu::injected_refusal() = refused;
    ace_gpu::injection_armed().store(!ace_gpu::injected_stage().empty());
}

// The same, failing as the system's refusal of the GPU does (tests only).
inline void ace_test_refuse_gpu_stage(const char * stage) {
    ace_test_fail_gpu_stage(stage, true);
}

// Whether a test has made this stage fail (tests only).
inline bool ace_gpu_injected(const char * what) {
    if (!ace_gpu::injection_armed().load(std::memory_order_relaxed)) {
        return false;
    }
    std::lock_guard<std::mutex> lock(ace_gpu::poisoned_mtx());
    return ace_gpu::injected_stage() == what;
}

// Mark `b` broken, as a discarded buffer leaves Metal's (tests only).
inline void ace_gpu_poison(ggml_backend_t b) {
    if (!b || ace_gpu::is_cpu(b)) {
        return;
    }
    std::lock_guard<std::mutex> lock(ace_gpu::poisoned_mtx());
    ace_gpu::poisoned().insert(b);
    if (ace_gpu::injected_refusal()) {
        ace_gpu::refused().insert(b);
    }
    ace_gpu::any_poisoned().store(true);
}

// Fail stage `what` for a compute that returned `st`, or whose backend it
// left broken (`broken`, else null) -- as refused when the system refused it.
inline void ace_check_compute(ggml_status st, ggml_backend_t broken, const char * what) {
    if (st == GGML_STATUS_ALLOC_FAILED) {
#ifdef ACESTEP_FATAL_THROWS
        fprintf(stderr, "[%s] FATAL: could not allocate the graph\n", what);
        throw std::bad_alloc();
#else
        ace_fatal(1, "[%s] FATAL: could not allocate the graph\n", what);
#endif
    }
    const bool refused = broken && ace_backend_refused(broken);
    if (st != GGML_STATUS_SUCCESS) {
        char why[64];
        snprintf(why, sizeof(why), "ggml status %d", (int) st);
        ace_gpu_fail(what, why, refused);
    }
    if (broken) {
        ace_gpu_fail(what, "the backend is in its error state", refused);
    }
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
    // Asked after a failed status too: a backend refused before this compute
    // fails it outright, and says whether that was the system's refusal.
    ace_check_compute(st, ace_sched_broken_backend(sched), what);
}

// The same for a graph run directly on one backend (the adapter merge).
inline void ace_backend_compute(ggml_backend_t backend, ggml_cgraph * gf, const char * what) {
    ggml_status st = ggml_backend_graph_compute(backend, gf);
    if (ace_gpu_injected(what)) {
        ace_gpu_poison(backend);
        st = GGML_STATUS_FAILED;
    }
    ace_check_compute(st, ace_backend_healthy(backend) ? nullptr : backend, what);
}

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
