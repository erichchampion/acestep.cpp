// test-util.h: the one CHECK scaffold for the engine's C test binaries
// (cadenza-audio #585). Every test reports a failed check the same way -- on
// stderr, with its file and line -- and ends through test_summary, so ctest
// (and a reader) sees one shape of pass and fail.
#pragma once

#include <cstdio>

static int failures = 0;

#define CHECK(cond)                                                         \
    do {                                                                    \
        if (!(cond)) {                                                      \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            failures++;                                                     \
        }                                                                   \
    } while (0)

// The exit status ctest reads as "skipped" (SKIP_RETURN_CODE in
// tests/CMakeLists.txt): the test could not run here -- no Metal device, say.
// A runner that must run it (CI on Metal hardware) treats a skip as a failure.
static const int TEST_SKIPPED = 77;

// The binary's result: a line saying how it went, and its exit status.
static inline int test_summary(const char * name) {
    if (failures == 0) {
        printf("%s: all checks passed\n", name);
        return 0;
    }
    fprintf(stderr, "%s: %d check(s) failed\n", name, failures);
    return 1;
}
