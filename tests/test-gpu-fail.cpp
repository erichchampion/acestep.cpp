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
// - A failed command buffer's error decides whether the backend was refused
//   (#817, #845, #849): Metal's MTLCommandBufferErrorDomain /
//   MTLCommandBufferErrorNotPermitted (7) and nothing else -- not another
//   code of that domain, nor code 7 of another -- read back through
//   ggml_backend_metal_refused. A hook puts a backend in the error state a
//   failed buffer would, as this test cannot make Metal fail or refuse one.
// - This whole binary is the leak test (#402): main returns with a live Metal
//   buffer and its device, buffer type and backend still referenced, and exit
//   must be clean -- with the statics leaked on purpose, there is no Metal
//   teardown to run. (One small shared buffer never reaches the residency
//   assert that fired for real model loads, so this is a smoke test for the
//   leak, not a reproduction of #177.)
//
//   ./test-gpu-fail
//
// Needs a Metal device; on a machine without one it says so and exits
// TEST_SKIPPED (77), which ctest reports as skipped -- not as a pass.

#include "ggml-backend.h"
#include "ggml.h"

#include <cstdio>
#include <cstring>
#include <cstdlib>

#include "test-util.h"

typedef bool (*has_error_fn_t)(ggml_backend_t);
typedef void (*fail_fn_t)(ggml_backend_t, const char *, long);

// A fresh backend failed as a command buffer with that error would leave it:
// whether it is in its error state, and whether that was a refusal.
static void failed_as(ggml_backend_dev_t dev, fail_fn_t fail, has_error_fn_t has_error,
                      has_error_fn_t refused, const char * domain, long code, bool want_refused) {
    ggml_backend_t backend = ggml_backend_dev_init(dev, /*params=*/NULL);
    CHECK(backend != NULL);
    if (backend == NULL) {
        return;
    }
    CHECK(!has_error(backend) && !refused(backend));
    fail(backend, domain, code);
    CHECK(has_error(backend));
    CHECK(refused(backend) == want_refused);
    ggml_backend_free(backend);
}

int main(void) {
    ggml_backend_reg_t reg = ggml_backend_reg_by_name("MTL");
    ggml_backend_dev_t dev = reg ? ggml_backend_reg_dev_get(reg, 0) : NULL;
    if (reg == NULL || dev == NULL || strncmp(ggml_backend_dev_name(dev), "MTL", 3) != 0) {
        printf("test-gpu-fail: SKIP: no Metal GPU device\n");
        return TEST_SKIPPED;
    }

    // The export (#405): present, and false while the backend is healthy.
    auto has_error = (has_error_fn_t)ggml_backend_reg_get_proc_address(reg, "ggml_backend_metal_has_error");
    CHECK(has_error != NULL);
    if (has_error == NULL) {
        return test_summary("test-gpu-fail");
    }

    // The refusal (#849): each error's backend in its error state, refused
    // only by Metal's NotPermitted.
    auto fail = (fail_fn_t)ggml_backend_reg_get_proc_address(reg, "ggml_backend_metal_fail_for_test");
    auto refused = (has_error_fn_t)ggml_backend_reg_get_proc_address(reg, "ggml_backend_metal_refused");
    CHECK(fail != NULL && refused != NULL);
    if (fail != NULL && refused != NULL) {
        failed_as(dev, fail, has_error, refused, "MTLCommandBufferErrorDomain", 7, true);    // NotPermitted: what an iPad reports
        failed_as(dev, fail, has_error, refused, "MTLCommandBufferErrorDomain", 1, false);   // Internal: a fault
        failed_as(dev, fail, has_error, refused, "MTLCommandBufferErrorDomain", 2, false);   // Timeout: the watchdog
        failed_as(dev, fail, has_error, refused, "IOGPUCommandQueueErrorDomain", 7, false);  // code 7 of another domain
        failed_as(dev, fail, has_error, refused, NULL, 7, false);                            // no error to read
        failed_as(dev, fail, has_error, refused, "\xff", 7, false);                          // not UTF-8: none either
    }

    ggml_backend_t backend = ggml_backend_dev_init(dev, /*params=*/NULL);
    CHECK(backend != NULL);
    if (backend == NULL) {
        // has_error(NULL) would assert: report and stop instead.
        return test_summary("test-gpu-fail");
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

    return test_summary("test-gpu-fail");
}
