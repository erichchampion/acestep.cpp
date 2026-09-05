#pragma once
// One place the engine says "this load cannot continue."
//
// By default ace_fatal() prints the message to stderr and exit(1)s -- byte for
// byte what the ~25 `fprintf(stderr, "[X] FATAL: ...\n", ...); exit(1);` sites
// did inline before. That is correct for the CLIs.
//
// Define ACESTEP_FATAL_THROWS (the app does; the CLIs do not) and the same
// message is thrown as ace_fatal_error instead, so C++ stack unwinding runs:
// every RAII-owned ggml_context and backend buffer is freed, and ModelStore's
// std::lock_guard unlocks, rather than the process dying mid-load. The message
// is echoed to stderr *and* carried on the exception.
//
// Where the exception ends up depends on who raised it. ModelStore catches
// ace_fatal_error at the load boundary (store_require_*) and turns a failed load
// into the nullptr it already returned on failure -- so load callers are
// unchanged and read the reason from stderr. A fatal raised anywhere without
// such a boundary propagates to the caller carrying the message.
//
// ace_fatal_error is defined even in the default (non-throwing) build so callers
// can `catch (const ace_fatal_error &)` unconditionally; it is simply never
// thrown there. Catching this specific type rather than `...` lets an unrelated
// std::bad_alloc keep propagating instead of being masked as a benign failure.
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
#include <stdexcept>
#include <string>
#include <utility>

#if defined(__GNUC__) || defined(__clang__)
#    define ACE_FATAL_PRINTF_FMT __attribute__((format(printf, 2, 3)))
#else
#    define ACE_FATAL_PRINTF_FMT
#endif

// Carries the exit code the CLI would have used and the formatted message.
// Defined unconditionally; only thrown under ACESTEP_FATAL_THROWS.
struct ace_fatal_error : std::runtime_error {
    int code;
    ace_fatal_error(int c, std::string msg) : std::runtime_error(std::move(msg)), code(c) {}
};

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
