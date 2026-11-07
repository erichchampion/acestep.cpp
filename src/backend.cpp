// backend.cpp: the shared GGML backend state and the functions that keep it
// (cadenza-audio #973). One definition per program: backend.h declares them,
// and every module, tool and test that loads a model reaches this copy.

#include "backend.h"

#include <mutex>
#include <thread>
#ifdef __APPLE__
#    include <sys/sysctl.h>
#endif

// Everything below is file-local (cadenza-audio #1048): the functions in
// backend.h are the only way to it, each under the lock. The slot type too,
// in an anonymous namespace, so its vector's instantiations are this file's.
namespace {
struct BackendSlot {
    BackendPair bp;
    int         refs;
    int         n_threads;  // the CPU backend's thread count, as made
};
}  // namespace

// Every live pair. Made on first use and leaked, as the mutex below is:
// constructed before any load reaches it, whichever translation unit's static
// initialisation runs first, and never destroyed under a late release at exit.
static std::vector<BackendSlot> & backend_slots() {
    static auto * slots = new std::vector<BackendSlot>();
    return *slots;
}

// Guards the slots, the current index and the records below: a GPU failure's
// sweep may run on one engine's thread while another engine loads (#613).
// Recursive, as ace_backend_init invalidates; taken after a store's own
// mutex, never before. Leaked, as the Metal statics are (#402).
static std::recursive_mutex & backend_slots_mtx() {
    static auto * m = new std::recursive_mutex();
    return *m;
}

static int                g_backend_current      = -1;  // the one new loads share, or -1
// The CPU thread count of the pair the latest ace_backend_init handed out, made
// or shared, or 0 before any: what a load ran with, which a config set since
// does not change while a pair made before it is still shared
// (cadenza-audio #889). Kept after that pair is freed, so it is read with
// g_backend_inits, the ace_backend_init calls so far: a caller that noted the count
// before its loads tells "none since" from a count left by earlier ones.
// Under backend_slots_mtx; ace_backend_cpu_threads_loaded is their reader.
static int                g_backend_last_threads = 0;
static unsigned long long g_backend_inits        = 0;

// ggml's backends, loaded once per process (backend.h). One once-flag for the
// program: defined here, not inline in the header (cadenza-audio #1033).
void ace_backend_load_all_once(void) {
    static std::once_flag once;
    std::call_once(once, [] { ggml_backend_load_all(); });
}

// The Metal command-buffer ceiling (backend.h). Reading it loads the registry,
// which opens the Metal device, so it is read only when it is needed: on iOS,
// or for an override. The registry is found by name because the shim asks
// before any backend exists. Without the export -- no Metal, or a ggml without
// our patch -- it is upstream's 8, which any ggml-metal's n_cb setter takes,
// and iOS, which needs the finer split, says so. Cached once per process
// (#1033).
static constexpr int BACKEND_METAL_UPSTREAM_MAX_N_CB = 8;

int ace_backend_metal_max_n_cb(void) {
    static const int ceiling = [] {
        ace_backend_load_all_once();
        using max_n_cb_t       = int (*)(void);
        ggml_backend_reg_t reg = ggml_backend_reg_by_name("MTL");
        auto               max_n_cb =
            reg ? (max_n_cb_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_metal_max_n_cb") : nullptr;
        const int n = max_n_cb ? max_n_cb() : 0;
        if (n >= 1) {
            return n;
        }
#if defined(__APPLE__) && TARGET_OS_IPHONE
        if (reg) {
            fprintf(stderr,
                    "[Load] WARNING: this ggml does not export ggml_backend_metal_max_n_cb -- "
                    "graphs split at most %d ways, and iOS's GPU watchdog may reset long steps\n",
                    BACKEND_METAL_UPSTREAM_MAX_N_CB + 1);
        }
#endif
        return BACKEND_METAL_UPSTREAM_MAX_N_CB;
    }();
    return ceiling;
}

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
int ace_backend_cpu_n_threads(void) {
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
            fprintf(stderr,
                    "[Load] WARNING: this ggml does not export ggml_backend_metal_set_n_cb -- "
                    "graphs stay in 2 command buffers, and iOS's GPU watchdog may reset long steps\n");
        }
#endif
        return;
    }
    const int n_cb = ace_backend_metal_n_cb();
    set_n_cb(backend, n_cb);
    fprintf(stderr, "[Load] Metal command buffers per graph: %d\n", n_cb + 1);
}

// Stop sharing `backend` with new loads (#403): it is in the error state, and
// the next ace_backend_init makes a fresh pair. Its holders keep it until they
// release it.
void ace_backend_invalidate(ggml_backend_t backend) {
    std::lock_guard<std::recursive_mutex> lock(backend_slots_mtx());
    auto &                                slots = backend_slots();
    if (g_backend_current >= 0 && slots[g_backend_current].bp.backend == backend) {
        fprintf(stderr, "[Load] %s backend failed: new loads get a fresh one\n", ggml_backend_name(backend));
        g_backend_current = -1;
    }
}

// Initialize backends: load all available (CUDA, Metal, Vulkan...),
// pick the best one, keep CPU as fallback.
// label: log prefix, e.g. "DiT", "VAE", "LM"
// Subsequent calls reuse the same backend (single VMM pool).
BackendPair ace_backend_init(const char * label) {
    std::lock_guard<std::recursive_mutex> lock(backend_slots_mtx());
    auto &                                slots = backend_slots();
    // A shared backend a failed command buffer left broken (#403) is not
    // handed out again: this load, and every one after, gets a fresh pair.
    if (g_backend_current >= 0 && !ace_backend_healthy(slots[g_backend_current].bp.backend)) {
        ace_backend_invalidate(slots[g_backend_current].bp.backend);
    }
    if (g_backend_current >= 0) {
        BackendSlot & slot = slots[g_backend_current];
        slot.refs++;
        g_backend_last_threads = slot.n_threads;
        g_backend_inits++;
        fprintf(stderr, "[Load] %s backend: %s (shared)\n", label, ggml_backend_name(slot.bp.backend));
        return slot.bp;
    }

    ace_backend_load_all_once();
    BackendPair bp = {};

    // Device selection: an explicit ace_backend_configure() wins, then the
    // GGML_BACKEND env var (the CLI fallback), then auto-best below.
    // Device names: CUDA0, Vulkan0, MTL0, CPU (see ggml_backend_dev_name); the
    // selectable ones only (backend_device_selectable).
    // An *empty* value from either source (config device "" or GGML_BACKEND="")
    // reads as unset and falls through to auto-best -- the `force_backend[0]`
    // guard -- by design; only a non-empty, unknown name is a hard error.
    const std::string & cfg_device    = ace_backend_config().device;
    const bool          from_config   = !cfg_device.empty();
    const char *        force_backend = from_config ? cfg_device.c_str() : std::getenv("GGML_BACKEND");
    if (force_backend && force_backend[0]) {
        // The one selectable-device rule (cadenza-audio #616): a GPU, an
        // integrated GPU or the CPU -- what the shim's check and the picker
        // offer. An accelerator (BLAS) runs beside a backend, never as one,
        // so naming it is refused here as there, and the "Available:" list
        // names the same devices.
        if (ace_backend_device_available(force_backend)) {
            bp.backend = ggml_backend_init_by_name(force_backend, nullptr);
        }
        if (!bp.backend) {
            std::string avail;
            for (const char * name : ace_backend_device_list()) {
                avail += ' ';
                avail += name;
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
    int  n_threads   = ace_backend_cpu_n_threads();
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
        slots.push_back({ bp, 1, n_threads });
    } catch (...) {
        if (bp.backend && bp.backend != bp.cpu_backend) {
            ggml_backend_free(bp.backend);
        }
        ggml_backend_free(bp.cpu_backend);
        throw;
    }
    g_backend_current      = (int) slots.size() - 1;
    g_backend_last_threads = n_threads;
    g_backend_inits++;
    return bp;
}

// The sharing state, read under the lock (backend.h).
AceBackendSharing ace_backend_sharing(void) {
    std::lock_guard<std::recursive_mutex> lock(backend_slots_mtx());
    return { backend_slots().size(), g_backend_current };
}

// The CPU thread count the latest load's backend pair ran with, or 0 before
// any load; `inits`, when given, gets the ace_backend_init calls so far (see
// g_backend_last_threads). Both read under one lock.
int ace_backend_cpu_threads_loaded(unsigned long long * inits) {
    std::lock_guard<std::recursive_mutex> lock(backend_slots_mtx());
    if (inits) {
        *inits = g_backend_inits;
    }
    return g_backend_last_threads;
}

// Release a backend reference. Frees GPU + CPU backends when refcount hits 0.
void ace_backend_release(ggml_backend_t backend, ggml_backend_t cpu_backend) {
    // A caller with no backend (m->backend == null) never took a ref -- e.g. a
    // load that threw before ace_backend_init under ACESTEP_FATAL_THROWS, whose
    // del_* still runs here. Every caller passes m->backend, so null means "no
    // ref"; decrementing would corrupt the shared count and could free a backend
    // another live module still holds. vae/vae-enc open the GGUF before
    // ace_backend_init, so they are the ones that reach here with a null backend.
    if (!backend) {
        return;
    }
    std::lock_guard<std::recursive_mutex> lock(backend_slots_mtx());
    auto &                                slots = backend_slots();
    int                                   i     = 0;
    while (i < (int) slots.size() && slots[i].bp.backend != backend) {
        i++;
    }
    if (i == (int) slots.size()) {
        return;
    }
    if (--slots[i].refs > 0) {
        return;
    }
    if (backend != cpu_backend) {
        ace_backend_forget(backend);
        ggml_backend_free(backend);
    }
    if (cpu_backend) {
        ggml_backend_free(cpu_backend);
    }
    slots.erase(slots.begin() + i);
    if (g_backend_current == i) {
        g_backend_current = -1;
    } else if (g_backend_current > i) {
        g_backend_current--;
    }
}
