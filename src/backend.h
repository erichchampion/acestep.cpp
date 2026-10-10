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
#include <thread>
#include <vector>

#ifdef __APPLE__
#    include <sys/sysctl.h>
#endif

struct BackendPair {
    ggml_backend_t backend;
    ggml_backend_t cpu_backend;
    bool           has_gpu;
};

// Cached backend state, shared across every model that loads within one binary.
// static-in-header (internal linkage) means one copy per TU, not one program-wide
// like ace_backend_config() -- and that is fine only because exactly one TU per
// binary ever reaches backend_init(): in the engine every load goes through
// model-store.cpp's store_require_*, and the neural-codec tool is a standalone TU
// that loads its own VAE. If a second TU in the same binary ever called a loader
// directly, its loads would get a separate cache and refcount and not share --
// make this a shared singleton (ace_backend_config-style) if that day comes.
//
// Each backend pair counts its own holders (#403): a GPU backend left in the
// error state by a failed command buffer is taken out of sharing
// (backend_invalidate), so the next load makes a fresh pair, while the
// modules still on the old one free it as they are released.
struct BackendSlot {
    BackendPair bp;
    int         refs;
};

static std::vector<BackendSlot> g_backend_slots;        // every live pair
// Guards the slots and the current index: a GPU failure's sweep may run on
// one engine's thread while another engine loads (#613). Recursive, as
// backend_init invalidates; taken after a store's own mutex, never before.
// Leaked, as the Metal statics are (#402).
static std::recursive_mutex & backend_slots_mtx() {
    static auto * m = new std::recursive_mutex();
    return *m;
}
static int                      g_backend_current = -1;  // the one new loads share, or -1

// The auto GGML CPU thread count: one thread per useful physical core. GEMM
// shares SIMD units across hyperthreads, so one-per-physical is optimal.
static int backend_cpu_auto_threads(void) {
#ifdef __APPLE__
    // Apple silicon has no SMT and asymmetric cores, so logical/2 is the wrong
    // count -- it halves as if for hyperthreads. hw.perflevel0 is the
    // performance-core cluster, the cores GEMM should run on: the E-cores are
    // much slower and just drag a shared graph. That is 12 P-cores on an M3 Max
    // (12P+4E, where logical/2 gave 8) and 2 on an iPhone 15 Pro (2P+4E, where
    // logical/2 gave 3) -- fewer than before on a 2P device, but the fast cores
    // only, which is the one-thread-per-useful-physical-core this function wants.
    // (physicalcpu == logicalcpu here anyway, no SMT; physicalcpu just reads as
    // the honest intent.) The perflevel sysctls exist only on Apple silicon: on an
    // Intel Mac this query fails and control falls through to the logical/2 below,
    // which is the right answer there because Intel does have SMT.
    int    perf = 0;
    size_t sz   = sizeof(perf);
    if (sysctlbyname("hw.perflevel0.physicalcpu", &perf, &sz, nullptr, 0) == 0 && perf > 0) {
        return perf;
    }
#endif
    // Non-Apple: x86 with SMT is the target, where logical / 2 approximates the
    // physical core count. (A non-SMT non-Apple host -- e.g. ARM64 Linux -- would
    // be undercounted, but that is not a platform this engine ships on.)
    int n = (int) std::thread::hardware_concurrency() / 2;
    return n > 0 ? n : 1;
}

// GGML CPU thread count: an embedder override (ace_backend_configure), otherwise
// the auto physical-core count. ace_resolve_threads() (backend-config.h) holds the
// shared policy: an explicit override is trusted -- an embedder may deliberately
// spend all cores including the E-cores -- and only clamped to the logical CPU count
// (ace_logical_cpus(), the same source the MP3 path uses) so an absurd value cannot
// ask GGML to spawn that many threads. The one-thread-per-physical-core reasoning is
// the default's (backend_cpu_auto_threads); an explicit request overrides it.
static int backend_cpu_n_threads(void) {
    return ace_resolve_threads(ace_backend_config().n_threads, ace_logical_cpus(), backend_cpu_auto_threads());
}

// Standalone CPU backend via Registry API (DL-safe, no ggml-cpu.h needed).
// Sets thread count via proc address since ggml_backend_cpu_device_init_backend
// ignores its params string and always defaults to GGML_DEFAULT_N_THREADS (4).
// Returns NULL on failure.
static ggml_backend_t cpu_backend_new(int n_threads) {
    ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    ggml_backend_t     cpu     = NULL;
    if (cpu_dev) {
        cpu = ggml_backend_dev_init(cpu_dev, NULL);
    }
    if (!cpu) {
        cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, NULL);
    }
    if (!cpu) {
        return NULL;
    }

    ggml_backend_dev_t dev = ggml_backend_get_device(cpu);
    ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : NULL;
    if (reg) {
        auto set_fn =
            (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
        if (set_fn) {
            set_fn(cpu, n_threads);
        }
    }
    return cpu;
}

// Split each graph across more Metal command buffers (cadenza-audio #403).
// iOS's GPU watchdog resets the GPU when one command buffer runs too long
// ("progress timeout"), and ggml-metal's default puts most of a graph in one
// buffer. The setter is reached through the Metal registry's proc address, so
// a build without Metal (or a ggml without the export) is unaffected.
// ACE_METAL_N_CB overrides the default, clamped to 1-128 (the patched ggml's
// ceiling, so the logged count is the one in effect). Only iOS splits, where the
// app runs no LM, so the LM's per-token graphs keep n_cb 1 on macOS. ggml logs a
// "n_cb > 2 is not recommended" warning when it is set; that is expected here.
//
// iOS splits 128 ways (cadenza-audio #474). 8 was enough for an M1 iPad but not
// for phone GPUs: on an A14 the longest of 9 buffers ran ~5 s in a 2-take 180 s
// DiT step and ~4 s in a 256-frame VAE tile, past the watchdog. At 129 the
// longest was ~1.2 s -- one kernel -- and the generation took no longer.
static constexpr int BACKEND_METAL_MAX_N_CB = 128;

static int backend_metal_n_cb(void) {
    if (const char * v = std::getenv("ACE_METAL_N_CB")) {
        const int n = atoi(v);
        if (n >= 1) {
            return n < BACKEND_METAL_MAX_N_CB ? n : BACKEND_METAL_MAX_N_CB;
        }
    }
#if defined(__APPLE__) && TARGET_OS_IPHONE
    return BACKEND_METAL_MAX_N_CB;
#else
    return 1;
#endif
}

static void backend_split_command_buffers(ggml_backend_t backend) {
    ggml_backend_dev_t dev = backend ? ggml_backend_get_device(backend) : NULL;
    ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : NULL;
    if (!reg) {
        return;
    }
    using set_n_cb_t = void (*)(ggml_backend_t, int);
    auto set_n_cb    = (set_n_cb_t) ace_gpu::backend_proc(backend, "ggml_backend_metal_set_n_cb");
    if (!set_n_cb) {
#if defined(__APPLE__) && TARGET_OS_IPHONE
        // A Metal backend without the export is a ggml without our patch: on
        // iOS 27 that brings the watchdog resets back (#403), so say so.
        // Metal is known by its registry's name, not a device's display name
        // (#408): "MTL" today, "Metal" in the older ggml this warning is for.
        const char * reg_name = ggml_backend_reg_name(reg);
        if (strcmp(reg_name, "MTL") == 0 || strcmp(reg_name, "Metal") == 0) {
            fprintf(stderr, "[Load] WARNING: this ggml does not export ggml_backend_metal_set_n_cb -- "
                            "graphs stay in 2 command buffers, and iOS's GPU watchdog may reset long steps\n");
        }
#endif
        return;
    }
    const int n_cb = backend_metal_n_cb();
    set_n_cb(backend, n_cb);
    fprintf(stderr, "[Load] Metal command buffers per graph: %d\n", n_cb + 1);
}

// ggml's backends, loaded once: every device lookup -- a backend_init, the
// enumeration below -- reads the same registry.
static inline void backend_load_all_once(void) {
    static std::once_flag once;
    std::call_once(once, [] { ggml_backend_load_all(); });
}

// The devices an embedder may name in ace_backend_configure() /
// ace_backend_set_device(), so a name can be offered and checked when it is
// configured, not discovered wrong at the first model load (#135, #19): the
// GPUs and the CPU. An accelerator (BLAS) runs beside a backend, never as
// one, so it is not offered.
static inline bool backend_device_selectable(ggml_backend_dev_t d) {
    const enum ggml_backend_dev_type t = ggml_backend_dev_type(d);
    return t == GGML_BACKEND_DEVICE_TYPE_GPU || t == GGML_BACKEND_DEVICE_TYPE_CPU;
}
static inline std::vector<ggml_backend_dev_t> backend_selectable_devices(void) {
    backend_load_all_once();
    std::vector<ggml_backend_dev_t> out;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        if (backend_device_selectable(ggml_backend_dev_get(i))) {
            out.push_back(ggml_backend_dev_get(i));
        }
    }
    return out;
}
static inline size_t ace_backend_device_count(void) {
    return backend_selectable_devices().size();
}
// The name of selectable device `i` ("MTL0", "CPU", ...), or null past the last.
static inline const char * ace_backend_device_name(size_t i) {
    const auto devs = backend_selectable_devices();
    return i < devs.size() ? ggml_backend_dev_name(devs[i]) : nullptr;
}
// Whether `name` is a selectable device of this build -- backend_init's
// lookup, made early (case does not matter, there or here).
static inline bool ace_backend_device_available(const char * name) {
    if (!name || !*name) {
        return false;
    }
    backend_load_all_once();
    ggml_backend_dev_t d = ggml_backend_dev_by_name(name);
    return d && backend_device_selectable(d);
}

// Stop sharing `backend` with new loads (#403): it is in the error state, and
// the next backend_init makes a fresh pair. Its holders keep it until they
// release it.
static void backend_invalidate(ggml_backend_t backend) {
    std::lock_guard<std::recursive_mutex> lock(backend_slots_mtx());
    if (g_backend_current >= 0 && g_backend_slots[g_backend_current].bp.backend == backend) {
        fprintf(stderr, "[Load] %s backend failed: new loads get a fresh one\n", ggml_backend_name(backend));
        g_backend_current = -1;
    }
}

// Initialize backends: load all available (CUDA, Metal, Vulkan...),
// pick the best one, keep CPU as fallback.
// label: log prefix, e.g. "DiT", "VAE", "LM"
// Subsequent calls reuse the same backend (single VMM pool).
static BackendPair backend_init(const char * label) {
    std::lock_guard<std::recursive_mutex> lock(backend_slots_mtx());
    // A shared backend a failed command buffer left broken (#403) is not
    // handed out again: this load, and every one after, gets a fresh pair.
    if (g_backend_current >= 0 && !ace_backend_healthy(g_backend_slots[g_backend_current].bp.backend)) {
        backend_invalidate(g_backend_slots[g_backend_current].bp.backend);
    }
    if (g_backend_current >= 0) {
        BackendSlot & slot = g_backend_slots[g_backend_current];
        slot.refs++;
        fprintf(stderr, "[Load] %s backend: %s (shared)\n", label, ggml_backend_name(slot.bp.backend));
        return slot.bp;
    }

    backend_load_all_once();
    BackendPair bp = {};

    // Device selection: an explicit ace_backend_configure() wins, then the
    // GGML_BACKEND env var (the CLI fallback), then auto-best below.
    // Device names: CUDA0, Vulkan0, CPU, BLAS (see ggml_backend_dev_name).
    // An *empty* value from either source (config device "" or GGML_BACKEND="")
    // reads as unset and falls through to auto-best -- the `force_backend[0]`
    // guard -- by design; only a non-empty, unknown name is a hard error.
    const std::string & cfg_device    = ace_backend_config().device;
    const bool          from_config   = !cfg_device.empty();
    const char *        force_backend = from_config ? cfg_device.c_str() : std::getenv("GGML_BACKEND");
    if (force_backend && force_backend[0]) {
        bp.backend = ggml_backend_init_by_name(force_backend, nullptr);
        if (!bp.backend) {
            std::string avail;
            for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
                avail += ' ';
                avail += ggml_backend_dev_name(ggml_backend_dev_get(i));
            }
            ace_fatal(1, "[Load] FATAL: backend '%s' (from %s) not found. Available:%s\n", force_backend,
                      from_config ? "ace_backend_configure" : "GGML_BACKEND", avail.c_str());
        }
    } else {
        bp.backend = ggml_backend_init_best();
    }
    if (!bp.backend) {
        ace_fatal(1, "[Load] FATAL: no backend available\n");
    }
    bool best_is_cpu = (strcmp(ggml_backend_name(bp.backend), "CPU") == 0);
    int  n_threads   = backend_cpu_n_threads();
    if (best_is_cpu) {
        ggml_backend_free(bp.backend);
        bp.backend     = cpu_backend_new(n_threads);
        bp.cpu_backend = bp.backend;
    } else {
        bp.cpu_backend = cpu_backend_new(n_threads);
    }
    if (!bp.cpu_backend) {
        // Under ACESTEP_FATAL_THROWS this throws, and bp is a local that never
        // reaches a module or the backend cache -- so free the GPU backend
        // allocated above (the best_is_cpu branch already freed it and left
        // bp.backend null) before unwinding, or it leaks. exit(1) does not care.
        if (bp.backend) {
            ggml_backend_free(bp.backend);
        }
        ace_fatal(1, "[Load] FATAL: failed to init CPU backend\n");
    }
    bp.has_gpu = !best_is_cpu;
    backend_split_command_buffers(bp.backend);
    fprintf(stderr, "[Load] %s backend: %s (CPU threads: %d)\n", label, ggml_backend_name(bp.backend), n_threads);

    try {
        g_backend_slots.push_back({ bp, 1 });
    } catch (...) {
        if (bp.backend && bp.backend != bp.cpu_backend) {
            ggml_backend_free(bp.backend);
        }
        ggml_backend_free(bp.cpu_backend);
        throw;
    }
    g_backend_current = (int) g_backend_slots.size() - 1;
    return bp;
}

// Release a backend reference. Frees GPU + CPU backends when refcount hits 0.
static void backend_release(ggml_backend_t backend, ggml_backend_t cpu_backend) {
    // A caller with no backend (m->backend == null) never took a ref -- e.g. a
    // load that threw before backend_init under ACESTEP_FATAL_THROWS, whose
    // del_* still runs here. Every caller passes m->backend, so null means "no
    // ref"; decrementing would corrupt the shared count and could free a backend
    // another live module still holds. vae/vae-enc open the GGUF before
    // backend_init, so they are the ones that reach here with a null backend.
    if (!backend) {
        return;
    }
    std::lock_guard<std::recursive_mutex> lock(backend_slots_mtx());
    int i = 0;
    while (i < (int) g_backend_slots.size() && g_backend_slots[i].bp.backend != backend) {
        i++;
    }
    if (i == (int) g_backend_slots.size()) {
        return;
    }
    if (--g_backend_slots[i].refs > 0) {
        return;
    }
    if (backend != cpu_backend) {
        ace_backend_forget(backend);
        ggml_backend_free(backend);
    }
    if (cpu_backend) {
        ggml_backend_free(cpu_backend);
    }
    g_backend_slots.erase(g_backend_slots.begin() + i);
    if (g_backend_current == i) {
        g_backend_current = -1;
    } else if (g_backend_current > i) {
        g_backend_current--;
    }
}



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
