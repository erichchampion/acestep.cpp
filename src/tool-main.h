#pragma once
// tool-main.h: every CLI's main, under one rule for a thrown fatal
// (cadenza-audio #1049).
//
// A build with ACESTEP_FATAL_THROWS -- one value for every target, inherited
// through acestep-fatal (#1033); build-xcframework.sh and build-ios-libs.sh
// turn it on -- throws a fatal rather than exit()ing. A tool built in such a
// configuration would reach std::terminate and abort on one; run through
// ace_tool_main it fails with the fatal's status instead, as the exit()ing
// build does. ace_fatal said why before it threw. Anything else still
// crashes, as ace-fatal.h wants.
//
//   static int synth_main(int argc, char ** argv) { ... }
//   int main(int argc, char ** argv) { return ace_tool_main(synth_main, argc, argv); }

#include "ace-fatal.h"

inline int ace_tool_main(int (*body)(int, char **), int argc, char ** argv) {
#ifdef ACESTEP_FATAL_THROWS
    try {
        return body(argc, argv);
    } catch (const ace_fatal_error & e) {
        return e.code;
    }
#else
    return body(argc, argv);
#endif
}
