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
#include "audio-frames.h"
#include "gpu-compute.h"

#include <cctype>
#include <cerrno>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <string>

// A fatal's words for a job's error: its log line less the "[Tag] FATAL: "
// it is printed with, and the newline after it (cadenza-audio #1194).
inline std::string ace_fatal_words(const char * line) {
    std::string  words = line ? line : "";
    const size_t at    = words.find("FATAL: ");
    if (at != std::string::npos) {
        words.erase(0, at + 7);
    }
    while (!words.empty() && std::isspace((unsigned char) words.back())) {
        words.pop_back();
    }
    return words;
}

// Runs `fn`: -1 when it returns, else the status of the fatal or graph
// allocation failure it threw, its words in `what` when given -- so a server
// job can say why it failed (cadenza-audio #1194).
template <class F> inline int ace_tool_catch(F && fn, std::string * what = nullptr) {
#ifdef ACESTEP_FATAL_THROWS
    try {
        fn();
    } catch (const ace_fatal_error & e) {
        if (what) {
            *what = ace_fatal_words(e.what());
        }
        return e.code;
    } catch (const ace_graph_alloc_error & e) {
        if (what) {
            *what = ace_fatal_words(e.what());
        }
        return 1;
    }
#else
    (void) what;
    fn();
#endif
    return -1;
}

inline int ace_tool_main(int (*body)(int, char **), int argc, char ** argv) {
    int       rc     = 0;
    const int thrown = ace_tool_catch([&] { rc = body(argc, argv); });
    return thrown < 0 ? rc : thrown;
}

// Whether a tool refuses its --vae-chunk and --vae-overlap: the engine's rule
// (vae_tiling_refused), said with the flags as given (cadenza-audio #1266,
// #1311). Every tool asks it where it parses them, before anything loads.
inline bool ace_tool_tiling_refused(const char * tag, int chunk, int overlap) {
    return vae_tiling_refused_said(tag, chunk, overlap, "--vae-chunk", "--vae-overlap");
}

// A tool's numeric flag, its argument read whole (strtol, base 10) and held
// to [lo, hi] into `out`; false, said under `tag`, for anything else -- atoi
// read "80x" as 80 and "x" as 0 (cadenza-audio #1194). Every tool reads its
// numeric flags by it, each range the flag's own meaning.
inline bool ace_tool_int_flag(const char * tag, const char * flag, const char * arg, long lo, long hi, int & out) {
    char * end   = nullptr;
    errno        = 0;
    const long v = strtol(arg, &end, 10);
    if (end == arg || *end || errno || v < lo || v > hi) {
        fprintf(stderr, "[%s] ERROR: %s takes a whole number from %ld to %ld, not '%s'\n", tag, flag, lo, hi, arg);
        return false;
    }
    out = (int) v;
    return true;
}

// Where a tool says the engine explains a failure it gave no words for: one
// phrase for every tool (cadenza-audio #1286).
inline constexpr const char * kAceToolDiagnostic = "the engine's diagnostic is on stderr";

// The max_seq a tool's --max-seq takes: 0 for the model's default, as the
// loads read it, or a length.
inline constexpr long kAceToolMaxSeqDefault = 0;

// Whether `job` ran to its end, rather than throwing a fatal: its words in
// `what` when it threw one.
template <class F> inline bool ace_tool_job(F && job, std::string * what = nullptr) {
    return ace_tool_catch(job, what) < 0;
}
