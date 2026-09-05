// test-progress: the AceProgress report/cancel contract (#18).
//
// Header-only -- exercises ace_progress() and the pre-loop + per-step reporting
// pattern the pipelines use, without loading any model. It checks that a run
// emits ordered (stage, step, total) reports for each stage, that step 0 is
// reported before the first unit of work, and that returning true cancels.
#include "progress.h"

#include <cstdio>
#include <vector>

static int failures = 0;

#define CHECK(cond)                                                         \
    do {                                                                    \
        if (!(cond)) {                                                      \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            failures++;                                                     \
        }                                                                   \
    } while (0)

struct Report {
    AceStage stage;
    int      step;
    int      total;
};

// Records every call; cancels when the level-triggered flag is set or when step
// reaches cancel_at (-1 = never).
struct Recorder {
    std::vector<Report> reports;
    int                 cancel_at = -1;
    bool                flag      = false;
};

static bool record_fn(void * data, AceStage stage, int step, int total) {
    auto * r = static_cast<Recorder *>(data);
    r->reports.push_back({ stage, step, total });
    return r->flag || (r->cancel_at >= 0 && step >= r->cancel_at);
}

// Mirrors what every pipeline loop does: one sizing report, then poll at the top
// of each iteration. Returns the step it cancelled at, or total if it ran fully.
static int run_stage(const AceProgress & p, AceStage stage, int total) {
    (void) ace_progress(p, stage, 0, total);  // size the bar
    for (int step = 0; step < total; step++) {
        if (ace_progress(p, stage, step, total)) {
            return step;  // cancelled
        }
        // ... work ...
    }
    return total;
}

int main() {
    // 1. A default (empty) AceProgress never cancels and never calls anything.
    {
        AceProgress none;
        CHECK(!ace_progress(none, ACE_STAGE_DIT, 3, 10));
        CHECK(run_stage(none, ACE_STAGE_DIT, 5) == 5);  // ran fully, no callback
    }

    // 2. A run reports ordered (stage, step, total) covering all four stages, with
    //    step 0 emitted before the first unit of work.
    {
        Recorder    rec;
        AceProgress p{ record_fn, &rec };

        const struct {
            AceStage stage;
            int      total;
        } stages[] = {
            { ACE_STAGE_LM,         4 },
            { ACE_STAGE_DIT,        3 },
            { ACE_STAGE_VAE_ENCODE, 2 },
            { ACE_STAGE_VAE_DECODE, 3 },
        };

        for (auto & s : stages) {
            CHECK(run_stage(p, s.stage, s.total) == s.total);
        }
        // Each stage emits: sizing(0) + poll(0..total-1) = total + 1 reports.
        size_t expect = 0;
        for (auto & s : stages) {
            expect += static_cast<size_t>(s.total) + 1;
        }
        CHECK(rec.reports.size() == expect);

        // Walk the recorded reports and confirm per stage: first two steps are 0
        // (sizing then poll-0), steps are non-decreasing, total is constant, and
        // the stage matches.
        size_t i = 0;
        for (auto & s : stages) {
            CHECK(rec.reports[i].stage == s.stage && rec.reports[i].step == 0 && rec.reports[i].total == s.total);
            i++;  // sizing report
            for (int step = 0; step < s.total; step++, i++) {
                CHECK(rec.reports[i].stage == s.stage);
                CHECK(rec.reports[i].step == step);
                CHECK(rec.reports[i].total == s.total);
            }
        }
    }

    // 3. Returning true cancels at that step, and the loop stops there.
    {
        Recorder rec;
        rec.cancel_at = 2;
        AceProgress p{ record_fn, &rec };
        CHECK(run_stage(p, ACE_STAGE_DIT, 10) == 2);  // stopped at step 2, not 10
        // sizing(0) + poll(0),poll(1),poll(2)=cancel -> last recorded step is 2.
        CHECK(!rec.reports.empty());
        CHECK(rec.reports.back().step == 2);
    }

    // 4. ace_cancelled is a bare cancel poll: it honours the cancel return but
    //    marks itself with step < 0 so a consumer can skip it for progress.
    {
        Recorder    rec;
        AceProgress p{ record_fn, &rec };
        CHECK(!ace_cancelled(p, ACE_STAGE_MP3));  // not cancelled
        CHECK(rec.reports.size() == 1);
        CHECK(rec.reports.back().stage == ACE_STAGE_MP3);
        CHECK(rec.reports.back().step < 0);  // distinguishable from a real report

        rec.flag = true;                     // now the (level-triggered) callback cancels
        CHECK(ace_cancelled(p, ACE_STAGE_MP3));

        AceProgress none;
        CHECK(!ace_cancelled(none, ACE_STAGE_MP3));  // null fn: never cancels
    }

    if (failures == 0) {
        printf("test-progress: all checks passed\n");
        return 0;
    }
    fprintf(stderr, "test-progress: %d check(s) failed\n", failures);
    return 1;
}
