#pragma once
// tool-main.h: every CLI's main, and ace-server's jobs, under one rule for a
// thrown fatal (cadenza-audio #1049).
//
// A build with ACESTEP_FATAL_THROWS -- one value for every target, inherited
// through acestep-fatal (#1033); build-xcframework.sh and build-ios-libs.sh
// turn it on -- throws a fatal rather than exit()ing, and a graph that could
// not be allocated as ace_graph_alloc_error (gpu-compute.h). Uncaught, either
// reaches std::terminate and aborts. Both said why before they threw.
//
// ace_tool_main runs a CLI's main: either fails it with the status the
// exit()ing build exits with. ace_tool_job runs one of ace-server's jobs on
// its worker thread, where main's catch never reaches: either fails that job
// (the server marks it so) and the server goes on -- a GPU fault leaves the
// backend to be recreated at the next load, as the store does for any
// embedder. Anything else still crashes, as ace-fatal.h wants: another
// std::bad_alloc is not masked.
//
//   static int synth_main(int argc, char ** argv) { ... }
//   int main(int argc, char ** argv) { return ace_tool_main(synth_main, argc, argv); }

#include "ace-fatal.h"
#include "gpu-compute.h"

// Runs `fn`: -1 when it returns, else the status of the fatal or graph
// allocation failure it threw.
template <class F> inline int ace_tool_catch(F && fn) {
#ifdef ACESTEP_FATAL_THROWS
    try {
        fn();
    } catch (const ace_fatal_error & e) {
        return e.code;
    } catch (const ace_graph_alloc_error &) {
        return 1;
    }
#else
    fn();
#endif
    return -1;
}

inline int ace_tool_main(int (*body)(int, char **), int argc, char ** argv) {
    int       rc     = 0;
    const int thrown = ace_tool_catch([&] { rc = body(argc, argv); });
    return thrown < 0 ? rc : thrown;
}

// Whether `job` ran to its end, rather than throwing a fatal.
template <class F> inline bool ace_tool_job(F && job) {
    return ace_tool_catch(job) < 0;
}
