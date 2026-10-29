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
//   A backend failed twice, refused once and faulted once, is refused (#817).
// - The error's underlying errors are logged beside it (#849, #857): a chain,
//   a wide set and a cycle, each error once and at most 16 in all; and one
//   that names a background refusal only in an underlying error's text is
//   said, though taken for a fault. A second hook builds those errors.
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
#include <string>

#include "test-util.h"

typedef bool (*has_error_fn_t)(ggml_backend_t);
typedef void (*fail_fn_t)(ggml_backend_t, const char *, long);
// ggml_backend_metal_fail_for_test_chain's, as ggml-metal-context.h declares it.
struct test_error {
    const char * domain;
    long         code;
    const char * text;
};
typedef bool (*fail_chain_fn_t)(ggml_backend_t, const char *, long, const char *, const test_error *, int, int);
enum { CHAIN = 0, WIDE = 1, CYCLE = 2 };

// What ggml logs while a case runs.
static std::string g_log;
static void capture(ggml_log_level level, const char * text, void * user) {
    (void) level;
    (void) user;
    g_log += text;
}

static int count_of(const std::string & haystack, const char * needle) {
    int n = 0;
    for (size_t at = haystack.find(needle); at != std::string::npos; at = haystack.find(needle, at + 1)) {
        n++;
    }
    return n;
}

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

// A fresh backend failed with an error wrapping `underlying` as `shape`:
// what was logged -- echoed, so a failed check shows it -- whether it was
// refused, and whether a cycle closed. Neither, with no backend.
static std::string failed_with(ggml_backend_dev_t dev, fail_chain_fn_t fail, has_error_fn_t refused,
                               const char * domain, long code, const char * text,
                               const test_error * underlying, int n, int shape, bool * was_refused,
                               bool * closed = NULL) {
    *was_refused = false;
    if (closed) {
        *closed = false;
    }
    ggml_backend_t backend = ggml_backend_dev_init(dev, /*params=*/NULL);
    CHECK(backend != NULL);
    if (backend == NULL) {
        return "";
    }
    g_log.clear();
    ggml_log_set(capture, NULL);
    const bool cycle = fail(backend, domain, code, text, underlying, n, shape);
    ggml_log_set(NULL, NULL);
    fputs(g_log.c_str(), stderr);
    *was_refused = refused(backend);
    if (closed) {
        *closed = cycle;
    }
    ggml_backend_free(backend);
    return g_log;
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

        // Failed twice (#817): one refusal is enough, whichever came first.
        for (int refusal_first = 0; refusal_first < 2; refusal_first++) {
            ggml_backend_t backend = ggml_backend_dev_init(dev, /*params=*/NULL);
            CHECK(backend != NULL);
            if (backend != NULL) {
                fail(backend, "MTLCommandBufferErrorDomain", refusal_first ? 7 : 1);
                fail(backend, "MTLCommandBufferErrorDomain", refusal_first ? 1 : 7);
                CHECK(has_error(backend) && refused(backend));
                ggml_backend_free(backend);
            }
        }
    }

    // The underlying errors (#849, #857), logged after the error's own.
    auto fail_chain = (fail_chain_fn_t)ggml_backend_reg_get_proc_address(reg, "ggml_backend_metal_fail_for_test_chain");
    CHECK(fail_chain != NULL && refused != NULL);
    if (fail_chain != NULL && refused != NULL) {
        bool was_refused = true;
        const test_error chain[] = { { "IOGPUCommandQueueErrorDomain", 3, "the queue" },
                                     { "IOGPUDeviceErrorDomain", 4, "the device" },
                                     { "NSPOSIXErrorDomain", 5, "the kernel" } };
        // A chain, each once and in order, after the error's own.
        std::string log = failed_with(dev, fail_chain, refused, "MTLCommandBufferErrorDomain", 1, "a fault",
                                      chain, 3, CHAIN, &was_refused);
        CHECK(!was_refused);
        CHECK(count_of(log, "; underlying ") == 3);
        const size_t queue = log.find("underlying IOGPUCommandQueueErrorDomain 3: the queue");
        const size_t device = log.find("underlying IOGPUDeviceErrorDomain 4: the device");
        const size_t kernel = log.find("underlying NSPOSIXErrorDomain 5: the kernel");
        CHECK(log.find("error: a fault (MTLCommandBufferErrorDomain 1)") != std::string::npos);
        CHECK(queue != std::string::npos && device != std::string::npos && kernel != std::string::npos);
        CHECK(queue < device && device < kernel);
        CHECK(count_of(log, "names a background refusal") == 0);

        // A cycle -- closed, the last wrapping the error again -- each once,
        // and the walk ends (ctest times a hang out).
        bool closed = false;
        log = failed_with(dev, fail_chain, refused, "MTLCommandBufferErrorDomain", 1, "a fault", chain, 3, CYCLE,
                          &was_refused, &closed);
        CHECK(closed);
        CHECK(!was_refused);
        CHECK(count_of(log, "; underlying ") == 3);
        CHECK(count_of(log, "the kernel") == 1);

        // Wide: at most 16 errors in all, the error's own among them.
        test_error many[20];
        for (int i = 0; i < 20; i++) {
            many[i] = { "IOGPUCommandQueueErrorDomain", (long) i, "one of many" };
        }
        log = failed_with(dev, fail_chain, refused, "MTLCommandBufferErrorDomain", 1, "a fault", many, 20, WIDE,
                          &was_refused);
        CHECK(log.find("error: a fault (MTLCommandBufferErrorDomain 1)") != std::string::npos);
        CHECK(!was_refused);
        CHECK(count_of(log, "; underlying ") == 15);
        // The first fifteen, each in order, and none after them.
        size_t at = 0;
        for (int i = 0; i < 20; i++) {
            char entry[64];
            snprintf(entry, sizeof(entry), "underlying IOGPUCommandQueueErrorDomain %d: ", i);
            const size_t found = log.find(entry);
            if (i < 15) {
                CHECK(found != std::string::npos && found >= at);
                at = found == std::string::npos ? at : found;
            } else {
                CHECK(found == std::string::npos);
            }
        }

        // Two errors alike -- domain, code and text -- are two errors, each
        // logged: once by identity, not by isEqual.
        const test_error alike[] = { { "IOGPUCommandQueueErrorDomain", 9, "the same" },
                                     { "IOGPUCommandQueueErrorDomain", 9, "the same" } };
        log = failed_with(dev, fail_chain, refused, "MTLCommandBufferErrorDomain", 1, "a fault", alike, 2, WIDE,
                          &was_refused);
        CHECK(!was_refused);
        CHECK(count_of(log, "; underlying ") == 2);
        CHECK(count_of(log, "underlying IOGPUCommandQueueErrorDomain 9: the same") == 2);

        // A refusal named only underneath: said, and taken for a fault.
        const test_error named[] = { { "IOGPUCommandQueueErrorDomain", 6,
                                       "kIOGPUCommandBufferCallbackErrorBackgroundExecutionNotPermitted" } };
        log = failed_with(dev, fail_chain, refused, "MTLCommandBufferErrorDomain", 1, "a fault", named, 1, CHAIN,
                          &was_refused);
        CHECK(!was_refused);
        CHECK(count_of(log, "names a background refusal") == 1);

        // The refusal's own code with underlying errors: refused, no warning.
        log = failed_with(dev, fail_chain, refused, "MTLCommandBufferErrorDomain", 7,
                          "kIOGPUCommandBufferCallbackErrorBackgroundExecutionNotPermitted", chain, 3, CHAIN, &was_refused);
        CHECK(was_refused);
        CHECK(count_of(log, "names a background refusal") == 0);
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
