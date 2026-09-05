#pragma once
// One place the engine says "this load cannot continue."
//
// By default ace_fatal() prints the message to stderr and exit(1)s -- byte for
// byte what the ~25 `fprintf(stderr, "[X] FATAL: ...\n", ...); exit(1);` sites
// did inline before. That is correct for the CLIs.
//
// Define ACESTEP_FATAL_THROWS (the app does; the CLIs do not) and the same
// message is thrown as ace_fatal_error instead, so C++ stack unwinding runs:
// every half-built ggml_context and backend buffer is freed by its destructor,
// and ModelStore's std::lock_guard unlocks, rather than the process dying
// mid-load with the mutex still held. The message travels with the exception,
// so an app can show *why* a load failed without scraping stderr.
//
// Why an exception and not the alternatives (see #17):
//   - setjmp/longjmp skips C++ destructors -- it would leak the half-built
//     module's ggml_context and backend buffer and leave the lock_guard held,
//     deadlocking the next call.
//   - Propagating a bool up the chain is a ~2000-line diff across seven headers
//     for the ~1000 straight-line load calls, and the right behaviour on
//     "tensor missing" is to abandon the whole load anyway.
#include <cstdarg>
#include <cstdio>
#include <cstdlib>

#ifdef ACESTEP_FATAL_THROWS
#    include <stdexcept>
#    include <string>
#    include <utility>
#endif

#if defined(__GNUC__) || defined(__clang__)
#    define ACE_FATAL_PRINTF_FMT __attribute__((format(printf, 2, 3)))
#else
#    define ACE_FATAL_PRINTF_FMT
#endif

#ifdef ACESTEP_FATAL_THROWS
// Carries the exit code the CLI would have used and the formatted message.
struct ace_fatal_error : std::runtime_error {
    int code;
    ace_fatal_error(int c, std::string msg) : std::runtime_error(std::move(msg)), code(c) {}
};
#endif

[[noreturn]] ACE_FATAL_PRINTF_FMT inline void ace_fatal(int code, const char * fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
#ifdef ACESTEP_FATAL_THROWS
    // Format into a string so the message rides with the exception, and echo it
    // to stderr so the diagnostic is present whether or not a caller catches.
    va_list ap2;
    va_copy(ap2, ap);
    char        stackbuf[1024];
    int         n = vsnprintf(stackbuf, sizeof(stackbuf), fmt, ap);
    std::string msg;
    if (n < 0) {
        msg = fmt;  // formatting failed; fall back to the raw format string
    } else if (static_cast<size_t>(n) < sizeof(stackbuf)) {
        msg.assign(stackbuf, static_cast<size_t>(n));
    } else {
        msg.resize(static_cast<size_t>(n) + 1);
        vsnprintf(&msg[0], msg.size(), fmt, ap2);
        msg.resize(static_cast<size_t>(n));  // drop the NUL vsnprintf wrote
    }
    va_end(ap2);
    va_end(ap);
    fputs(msg.c_str(), stderr);
    throw ace_fatal_error(code, std::move(msg));
#else
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    exit(code);
#endif
}
