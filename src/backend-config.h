#pragma once
// Explicit CPU thread count and compute-device selection for embedders (#19).
//
// An app calls ace_backend_configure() once, before its first model load, instead
// of relying on the GGML_BACKEND environment variable -- an app does not configure
// itself with getenv. The env var stays as a fallback for the CLIs: when nothing
// has been configured, backend init still honours GGML_BACKEND, then auto-picks
// the best device.
//
// This header is deliberately free of any ggml dependency so an embedder can
// include just it to configure the engine; backend.h reads the config.
#include <string>

struct AceBackendConfig {
    std::string device;         // backend name (e.g. "Metal", "CUDA0", "CPU"); "" = auto
    int         n_threads = 0;  // CPU worker threads; <= 0 = auto
};

// The one config instance, shared across every translation unit: a function-local
// static inside an inline (external-linkage) function has exactly one instance
// program-wide, so the app's ace_backend_configure() and the engine's backend
// init read the same object even though they compile in different TUs.
inline AceBackendConfig & ace_backend_config() {
    static AceBackendConfig cfg;
    return cfg;
}

// Set the compute device and CPU thread count before the first model load.
//   device:    backend name; NULL or "" means auto (falls back to the
//              GGML_BACKEND env var, then the best available device).
//   n_threads: CPU worker threads; <= 0 means auto (the performance-core count on
//              Apple silicon, hardware_concurrency()/2 elsewhere).
// Not thread-safe: call once during setup, before any generate call.
inline void ace_backend_configure(const char * device, int n_threads) {
    ace_backend_config().device    = device ? device : "";
    ace_backend_config().n_threads = n_threads;
}
