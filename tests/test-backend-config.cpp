// test-backend-config: explicit CPU thread count and device selection (#19).
//
// Verifies the embedder-facing contract: ace_backend_configure() sets the thread
// count and device before first use, backend_cpu_n_threads() honours the override
// and otherwise auto-picks a positive default, and clearing the config restores
// auto.
#include "backend.h"

#include <cstdio>

static int failures = 0;

#define CHECK(cond)                                                         \
    do {                                                                    \
        if (!(cond)) {                                                      \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            failures++;                                                     \
        }                                                                   \
    } while (0)

int main() {
    // Auto by default: a positive thread count, no forced device.
    CHECK(ace_backend_config().device.empty());
    int auto_threads = backend_cpu_n_threads();
    CHECK(auto_threads > 0);

    // An explicit thread count is honoured exactly.
    ace_backend_configure(nullptr, 7);
    CHECK(backend_cpu_n_threads() == 7);
    ace_backend_configure(nullptr, 1);
    CHECK(backend_cpu_n_threads() == 1);

    // <= 0 means auto again -- back to the same positive default.
    ace_backend_configure(nullptr, 0);
    CHECK(backend_cpu_n_threads() == auto_threads);
    ace_backend_configure(nullptr, -4);
    CHECK(backend_cpu_n_threads() == auto_threads);

    // The device is settable and cleared by "" / NULL.
    ace_backend_configure("Metal", 0);
    CHECK(ace_backend_config().device == "Metal");
    ace_backend_configure("CPU", 3);
    CHECK(ace_backend_config().device == "CPU");
    CHECK(backend_cpu_n_threads() == 3);
    ace_backend_configure(nullptr, 0);
    CHECK(ace_backend_config().device.empty());

    // The single-field setters change one field without disturbing the other.
    ace_backend_set_device("Metal");
    ace_backend_set_threads(5);
    CHECK(ace_backend_config().device == "Metal");
    CHECK(backend_cpu_n_threads() == 5);
    ace_backend_set_threads(2);  // device untouched
    CHECK(ace_backend_config().device == "Metal");
    CHECK(backend_cpu_n_threads() == 2);
    ace_backend_set_device(nullptr);  // threads untouched
    CHECK(ace_backend_config().device.empty());
    CHECK(backend_cpu_n_threads() == 2);
    ace_backend_configure(nullptr, 0);  // reset both

    // An absurd thread count is clamped to the logical CPU count, not passed
    // through to spawn that many workers.
    int hw = (int) std::thread::hardware_concurrency();
    if (hw > 0) {
        ace_backend_configure(nullptr, 100000);
        CHECK(backend_cpu_n_threads() == hw);
        ace_backend_configure(nullptr, 0);
    }

    // End-to-end: a configured device actually drives backend_init's selection.
    // "CPU" is always available, so this needs no GPU.
    ace_backend_configure("CPU", 2);
    BackendPair bp = backend_init("test");
    CHECK(bp.backend != nullptr);
    CHECK(strcmp(ggml_backend_name(bp.backend), "CPU") == 0);
    CHECK(!bp.has_gpu);
    backend_release(bp.backend, bp.cpu_backend);
    ace_backend_configure(nullptr, 0);

#ifdef __APPLE__
    // On Apple the auto default is the performance-core count, not logical/2.
    // Sanity: it is positive and does not exceed the total logical CPUs.
    int    total = 0;
    size_t sz    = sizeof(total);
    if (sysctlbyname("hw.logicalcpu", &total, &sz, nullptr, 0) == 0 && total > 0) {
        CHECK(auto_threads <= total);
    }
#endif

    if (failures == 0) {
        printf("test-backend-config: all checks passed (auto threads = %d)\n", auto_threads);
        return 0;
    }
    fprintf(stderr, "test-backend-config: %d check(s) failed\n", failures);
    return 1;
}
