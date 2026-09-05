// test-backend-config: explicit CPU thread count and device selection (#19).
//
// Verifies the embedder-facing contract: ace_backend_configure() sets the thread
// count and device before first use, backend_cpu_n_threads() honours an explicit
// override (clamped to the logical CPU count) and otherwise auto-picks a positive
// default, and clearing the config restores auto. All thread values here are
// relative to this host's logical CPU count so the test passes on any machine.
#include "backend.h"

#include <cstdio>
#include <cstring>           // strcmp
#include <thread>            // hardware_concurrency
#ifdef __APPLE__
#    include <sys/sysctl.h>  // sysctlbyname
#endif

static int failures = 0;

#define CHECK(cond)                                                         \
    do {                                                                    \
        if (!(cond)) {                                                      \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            failures++;                                                     \
        }                                                                   \
    } while (0)

int main() {
    const int hw = (int) std::thread::hardware_concurrency();
    const int t2 = hw >= 2 ? 2 : 1;  // a second in-range thread value

    // Auto by default: a positive thread count, no forced device, never more than
    // the logical CPUs.
    CHECK(ace_backend_config().device.empty());
    const int auto_threads = backend_cpu_n_threads();
    CHECK(auto_threads > 0);
    if (hw > 0) {
        CHECK(auto_threads <= hw);
    }

    // An explicit count within [1, logical CPUs] is honoured exactly.
    ace_backend_configure(nullptr, 1);
    CHECK(backend_cpu_n_threads() == 1);
    ace_backend_configure(nullptr, t2);
    CHECK(backend_cpu_n_threads() == t2);
    if (hw > 0) {
        ace_backend_configure(nullptr, hw);  // exactly the ceiling
        CHECK(backend_cpu_n_threads() == hw);
        // Over the ceiling -- including an absurd value -- is clamped to it, not
        // passed through to spawn that many workers.
        ace_backend_configure(nullptr, hw + 100);
        CHECK(backend_cpu_n_threads() == hw);
        ace_backend_configure(nullptr, 100000);
        CHECK(backend_cpu_n_threads() == hw);
    }

    // <= 0 means auto again -- the positive default.
    ace_backend_configure(nullptr, 0);
    CHECK(backend_cpu_n_threads() == auto_threads);
    ace_backend_configure(nullptr, -4);
    CHECK(backend_cpu_n_threads() == auto_threads);

    // The device is settable and cleared by "" / NULL.
    ace_backend_configure("Metal", 0);
    CHECK(ace_backend_config().device == "Metal");
    ace_backend_configure(nullptr, 0);
    CHECK(ace_backend_config().device.empty());

    // The single-field setters change one field without disturbing the other.
    ace_backend_set_device("Metal");
    ace_backend_set_threads(1);
    CHECK(ace_backend_config().device == "Metal");
    CHECK(backend_cpu_n_threads() == 1);
    ace_backend_set_threads(t2);  // device untouched
    CHECK(ace_backend_config().device == "Metal");
    CHECK(backend_cpu_n_threads() == t2);
    ace_backend_set_device(nullptr);  // threads untouched
    CHECK(ace_backend_config().device.empty());
    CHECK(backend_cpu_n_threads() == t2);
    ace_backend_configure(nullptr, 0);  // reset both

    // End-to-end: a configured device actually drives backend_init's selection.
    // "CPU" is always available, so this needs no GPU.
    ace_backend_configure("CPU", 1);
    BackendPair bp = backend_init("test");
    CHECK(bp.backend != nullptr);
    CHECK(strcmp(ggml_backend_name(bp.backend), "CPU") == 0);
    CHECK(!bp.has_gpu);
    backend_release(bp.backend, bp.cpu_backend);
    ace_backend_configure(nullptr, 0);

    if (failures == 0) {
        printf("test-backend-config: all checks passed (auto threads = %d, logical = %d)\n", auto_threads, hw);
        return 0;
    }
    fprintf(stderr, "test-backend-config: %d check(s) failed\n", failures);
    return 1;
}
