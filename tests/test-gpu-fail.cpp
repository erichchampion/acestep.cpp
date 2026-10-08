// test-gpu-fail: the ggml Metal error paths the engine's GPU failure handling
// leans on (cadenza-audio #102, #402, #405; Phase 10, 10.1).
//
// - A Metal buffer that cannot be allocated comes back NULL, which ggml-alloc
//   already treats as a failed allocation -- not a dereference of the null
//   buffer, which crashed (#102). An allocation past the process's budget is
//   routine on iOS, so this is the difference between an error and a segfault.
// - ggml_backend_metal_has_error is exported through get_proc_address and
//   reads the backend's state on a healthy backend (#405). (A real command
//   buffer failure cannot be arranged from outside; the engine's fault
//   injection poisons a side set, not Metal's state.)
// - This whole binary is the leak test (#402): main returns with a live Metal
//   buffer and its device, buffer type and backend still referenced, and exit
//   must be clean -- with the statics leaked on purpose, there is no Metal
//   teardown to run. (One small shared buffer never reaches the residency
//   assert that fired for real model loads, so this is a smoke test for the
//   leak, not a reproduction of #177.)
//
//   ./test-gpu-fail
//
// Needs a Metal device; on a machine without one it says so and passes.

#include "ggml-backend.h"
#include "ggml.h"

#include <cstdio>
#include <cstring>
#include <cstdlib>

static int failures = 0;

#define CHECK(cond)                                                                      \
    do {                                                                                 \
        if (!(cond)) {                                                                   \
            printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);                        \
            failures++;                                                                  \
        }                                                                                \
    } while (0)

typedef bool (*has_error_fn_t)(ggml_backend_t);

int main(void) {
    ggml_backend_reg_t reg = ggml_backend_reg_by_name("MTL");
    ggml_backend_dev_t dev = reg ? ggml_backend_reg_dev_get(reg, 0) : NULL;
    if (reg == NULL || dev == NULL || strncmp(ggml_backend_dev_name(dev), "MTL", 3) != 0) {
        printf("test-gpu-fail: SKIP: no Metal GPU device\n");
        return 0;
    }

    // The export (#405): present, and false while the backend is healthy.
    auto has_error = (has_error_fn_t)ggml_backend_reg_get_proc_address(reg, "ggml_backend_metal_has_error");
    CHECK(has_error != NULL);
    if (has_error == NULL) {
        printf("%d failure(s)\n", failures);
        return failures > 0;
    }

    ggml_backend_t backend = ggml_backend_dev_init(dev, /*params=*/NULL);
    CHECK(backend != NULL);
    if (backend == NULL) {
        // has_error(NULL) would assert: report and stop instead.
        printf("%d failure(s)\n", failures);
        return 1;
    }
    CHECK(has_error(backend) == false);

    ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(dev);

    // The null check (#102): an allocation no process can host comes back as
    // a failed allocation. Before, this line dereferenced the null buffer.
    ggml_backend_buffer_t impossible = ggml_backend_buft_alloc_buffer(buft, (size_t)1 << 62);
    CHECK(impossible == NULL);

    // The leak test (#402): a buffer that stays alive to the end. `backend`,
    // `small` and `buft` are deliberately still referenced at return.
    ggml_backend_buffer_t small = ggml_backend_buft_alloc_buffer(buft, 1 << 20);
    CHECK(small != NULL);
    CHECK(has_error(backend) == false);

    printf("%s (%d failure(s))\n", failures > 0 ? "FAIL" : "test-gpu-fail: OK", failures);
    return failures > 0;
}
