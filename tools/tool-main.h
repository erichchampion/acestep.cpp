#pragma once
// tool-main.h: every CLI's main, under one rule for a thrown fatal
// (cadenza-audio #1049).
//
// A build with ACESTEP_FATAL_THROWS -- one value for every target, inherited
// through acestep-fatal (#1033); build-xcframework.sh and build-ios-libs.sh
// turn it on -- throws a fatal rather than exit()ing, and a graph that could
// not be allocated as std::bad_alloc (gpu-compute's ace_check_compute). A tool
// built in such a configuration would reach std::terminate and abort on
// either; run through these it fails with the status the exit()ing build
// exits with instead. Both said why before they threw. Anything else still
// crashes, as ace-fatal.h wants.
//
// ace_tool_main runs a CLI's main. A thread that computes (ace-server's
// worker) runs each job through ace_tool_run: a throw there would not reach
// main's catch, so it exits the process with the status, as ace_fatal
// does on that thread in the exit()ing build.
//
//   static int synth_main(int argc, char ** argv) { ... }
//   int main(int argc, char ** argv) { return ace_tool_main(synth_main, argc, argv); }

#include "ace-fatal.h"

#include <cstdio>
#include <cstdlib>
#include <new>

inline int ace_tool_main(int (*body)(int, char **), int argc, char ** argv) {
#ifdef ACESTEP_FATAL_THROWS
    try {
        return body(argc, argv);
    } catch (const ace_fatal_error & e) {
        return e.code;
    } catch (const std::bad_alloc &) {
        return 1;
    }
#else
    return body(argc, argv);
#endif
}

template <class F> inline void ace_tool_run(F && job) {
#ifdef ACESTEP_FATAL_THROWS
    try {
        job();
    } catch (const ace_fatal_error & e) {
        fflush(stderr);
        std::exit(e.code);
    } catch (const std::bad_alloc &) {
        fflush(stderr);
        std::exit(1);
    }
#else
    job();
#endif
}
