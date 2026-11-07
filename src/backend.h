#pragma once
// backend.h: shared GGML backend initialization
//
// All modules use the same pattern: load all backends, pick best GPU,
// keep CPU as fallback. This avoids duplicating init logic across
// qwen3.h, qwen3-lm.h, cond.h, dit.h, vae.h.

#include "ace-fatal.h"
#include "backend-config.h"
#include "ggml-backend.h"
#include "gpu-compute.h"

#if defined(__APPLE__)
#    include <TargetConditionals.h>
#endif
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>


struct BackendPair {
    ggml_backend_t backend;
    ggml_backend_t cpu_backend;
    bool           has_gpu;
};

// Cached backend state, shared across every model that loads within one binary.
// Defined once, in backend.cpp (cadenza-audio #973; #900 made it inline, which
// held only while every piece was written so): whichever translation unit
// reaches ace_backend_init() -- model-store.cpp's store_require_* for every engine
// load, the neural-codec tool's own VAE load, or any loader added later --
// shares one cache, one refcount and one record of the threads a load ran
// with, by construction. The state is file-local there (cadenza-audio #1048):
// only the functions below, each under its lock, can change it, and this
// header holds none of it.
//
// Each backend pair counts its own holders (#403): a GPU backend left in the
// error state by a failed command buffer is taken out of sharing
// (ace_backend_invalidate), so the next load makes a fresh pair, while the
// modules still on the old one free it as they are released.

// The sharing state as one locked read (tests): the live pairs, and the index
// of the one new loads share, or -1.
struct AceBackendSharing {
    size_t live;
    int    current;
};
AceBackendSharing ace_backend_sharing(void);

// GGML CPU thread count: an embedder override (ace_backend_configure),
// otherwise one thread per useful physical core (backend.cpp says why).
int ace_backend_cpu_n_threads(void);

// ggml's backends, loaded once per process: every device lookup -- an
// ace_backend_init, the enumeration below -- reads the same registry. Defined
// in backend.cpp, with its once-flag (cadenza-audio #1033).
void ace_backend_load_all_once(void);

// Split each graph across more Metal command buffers (cadenza-audio #403).
// iOS's GPU watchdog resets the GPU when one command buffer runs too long
// ("progress timeout"), and ggml-metal's default puts most of a graph in one
// buffer. The setter is reached through the Metal registry's proc address, so
// a build without Metal (or a ggml without the export) is unaffected.
// ACE_METAL_N_CB overrides the default, clamped to 1 through the patched ggml's
// ceiling (so the logged count is the one in effect). Only iOS splits, where the
// app runs no LM, so the LM's per-token graphs keep n_cb 1 on macOS. ggml logs a
// "n_cb > 2 is not recommended" warning when it is set; that is expected here.
//
// iOS splits 128 ways (cadenza-audio #474). 8 was enough for an M1 iPad but not
// for phone GPUs: on an A14 the longest of 9 buffers ran ~5 s in a 2-take 180 s
// DiT step and ~4 s in a 256-frame VAE tile, past the watchdog. At 129 the
// longest was ~1.2 s -- one kernel -- and the generation took no longer.
//
// ace_backend_metal_max_n_cb is that ceiling: the patched ggml's own, read
// from its Metal registry (ggml_backend_metal_max_n_cb, 128 today), not a copy
// kept here (#408), and cached once per process in backend.cpp (#1033), which
// says when it is read and what it falls back to.
int ace_backend_metal_max_n_cb(void);

// The n_cb for an ACE_METAL_N_CB value `request` (null: unset), without
// reading ACE_METAL_N_CB: an embedder's tests ask it for each value rather
// than setenv the variable under threads that may be reading it
// (cadenza-audio #759). Reading the ceiling still loads the registry, as
// ace_backend_metal_max_n_cb says, and ggml's setenv with it.
inline int ace_backend_metal_n_cb_for(const char * request) {
    if (request) {
        // strtol, not atoi: a value past long's range saturates (ERANGE)
        // rather than wrapping, so it is held to the ceiling like any other.
        const long n = std::strtol(request, nullptr, 10);
        if (n >= 1) {
            const int ceiling = ace_backend_metal_max_n_cb();
            return n < ceiling ? (int) n : ceiling;
        }
    }
#if defined(__APPLE__) && TARGET_OS_IPHONE
    return ace_backend_metal_max_n_cb();
#else
    return 1;
#endif
}

inline int ace_backend_metal_n_cb(void) {
    return ace_backend_metal_n_cb_for(std::getenv("ACE_METAL_N_CB"));
}

// The devices an embedder may name in ace_backend_configure() /
// ace_backend_set_device(), so a name can be offered and checked when it is
// configured, not discovered wrong at the first model load (#135, #19): the
// GPUs (integrated ones too) and the CPU. An accelerator (BLAS) runs beside
// a backend, never as one, so it is not offered.
static inline bool backend_device_selectable(ggml_backend_dev_t d) {
    const enum ggml_backend_dev_type t = ggml_backend_dev_type(d);
    return t == GGML_BACKEND_DEVICE_TYPE_GPU || t == GGML_BACKEND_DEVICE_TYPE_IGPU ||
           t == GGML_BACKEND_DEVICE_TYPE_CPU;
}
// The selectable devices' names ("MTL0", "CPU", ...), in registry order: one
// walk, for a picker and for a refusal's "Available:" alike.
static inline std::vector<const char *> ace_backend_device_list(void) {
    ace_backend_load_all_once();
    std::vector<const char *> out;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        if (backend_device_selectable(d)) {
            out.push_back(ggml_backend_dev_name(d));
        }
    }
    return out;
}
static inline size_t ace_backend_device_count(void) {
    return ace_backend_device_list().size();
}
// The name of selectable device `i`, or null past the last.
static inline const char * ace_backend_device_name(size_t i) {
    const auto names = ace_backend_device_list();
    return i < names.size() ? names[i] : nullptr;
}

// Whether `name` is a selectable device of this build -- ace_backend_init's
// lookup, made early (case does not matter, there or here).
static inline bool ace_backend_device_available(const char * name) {
    if (!name || !*name) {
        return false;
    }
    ace_backend_load_all_once();
    ggml_backend_dev_t d = ggml_backend_dev_by_name(name);
    return d && backend_device_selectable(d);
}

// Stop sharing `backend` with new loads (#403): it is in the error state, and
// the next ace_backend_init makes a fresh pair. Its holders keep it until they
// release it.
void ace_backend_invalidate(ggml_backend_t backend);

// Initialize backends: load all available (CUDA, Metal, Vulkan...), pick the
// best one, keep CPU as fallback. label: log prefix, e.g. "DiT", "VAE", "LM".
// Subsequent calls reuse the same backend (single VMM pool). Fatal (ace_fatal)
// when no backend can be made.
BackendPair ace_backend_init(const char * label);

// The CPU thread count the latest load's backend pair ran with, or 0 before
// any load; `inits`, when given, gets the ace_backend_init calls so far. Both read
// under one lock.
int ace_backend_cpu_threads_loaded(unsigned long long * inits);

// Release a backend reference. Frees GPU + CPU backends when refcount hits 0.
// A null backend took no ref and is ignored.
void ace_backend_release(ggml_backend_t backend, ggml_backend_t cpu_backend);

// Create a scheduler from a backend pair.
// max_nodes: graph size hint (4096 for small models, 8192 for large)
// When a GPU is present, use its host buffer type for the CPU backend.
// Pinned memory lets the scheduler keep more ops on GPU instead of
// falling back to CPU with plain malloc.
static ggml_backend_sched_t backend_sched_new(BackendPair bp, int max_nodes) {
    ggml_backend_t             backends[2] = { bp.backend, bp.cpu_backend };
    ggml_backend_buffer_type_t bufts[2]    = { NULL, NULL };
    int                        n           = (bp.backend == bp.cpu_backend) ? 1 : 2;

    bufts[0] = ggml_backend_get_default_buffer_type(bp.backend);
    if (n == 2) {
        ggml_backend_dev_t         gpu_dev   = ggml_backend_get_device(bp.backend);
        ggml_backend_buffer_type_t host_buft = gpu_dev ? ggml_backend_dev_host_buffer_type(gpu_dev) : NULL;
        bufts[1] = host_buft ? host_buft : ggml_backend_get_default_buffer_type(bp.cpu_backend);
    }

    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, bufts, n, max_nodes, false, true);
    if (!sched) {
        ace_fatal(1, "[Load] FATAL: failed to create scheduler\n");
    }
    return sched;
}
