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
// include just it to configure the engine; backend.h reads the config. Like the
// rest of the engine's API it is C++ (not extern "C"): a Swift/Obj-C app calls it
// from its C++ or Objective-C++ bridge (the same shim it already needs for
// ace_synth_* etc.), not from Swift directly.
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

// Set the compute device: backend name, or NULL/"" for auto (falls back to the
// GGML_BACKEND env var, then the best available device).
inline void ace_backend_set_device(const char * device) {
    ace_backend_config().device = device ? device : "";
}

// Set the CPU worker thread count: <= 0 means auto (the performance-core count on
// Apple silicon, hardware_concurrency()/2 elsewhere).
inline void ace_backend_set_threads(int n_threads) {
    ace_backend_config().n_threads = n_threads;
}

// Set BOTH device and thread count at once -- a convenience for the common
// configure-once-at-startup call. It sets the full config, so passing e.g.
// (nullptr, 8) also resets the device to auto; to change only one field later use
// the single-field setters above. Not thread-safe: call during setup, before any
// generate call.
inline void ace_backend_configure(const char * device, int n_threads) {
    ace_backend_set_device(device);
    ace_backend_set_threads(n_threads);
}
