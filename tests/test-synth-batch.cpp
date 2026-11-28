// test-synth-batch: the tools' batch expansion and its log (cadenza-audio
// #1269). ace-synth and ace-server expand a batch by synth_batch_expand,
// from the ace_synth_batch_takes each computes once per job, and say its
// clamp by synth_batch_notes: these hold what a tool runs and says -- ten
// one-take requests run the first nine, the tenth named as dropped; a lone
// request asking 50 runs 9 and logs its clamp. ace-server refuses a batch
// asking past the cap instead (ace_synth_batch_takes_refused, cadenza-audio
// #1182), which these hold too. lint-batch-expansion.cmake holds the tools
// to these helpers. No model is loaded.

#include "pipeline-synth.h"
#include "request.h"
#include "synth-batch-runner.h"
#include "test-util.h"

#include <string>
#include <vector>

static std::vector<AceRequest> batch_of(const std::vector<int> & sizes) {
    std::vector<AceRequest> reqs(sizes.size());
    for (size_t i = 0; i < sizes.size(); i++) {
        request_init(&reqs[i]);
        reqs[i].synth_batch_size = sizes[i];
        reqs[i].seed             = 1000 * (long long) (i + 1);
    }
    return reqs;
}

static std::vector<std::string> notes_of(const std::vector<AceRequest> & reqs, const AceSynthBatchTakes & batch) {
    return synth_batch_notes(batch, (int) reqs.size(), [](int i) { return "r" + std::to_string(i); });
}

static bool said(const std::vector<std::string> & notes, const std::string & words) {
    for (const std::string & n : notes) {
        if (n.find(words) != std::string::npos) {
            return true;
        }
    }
    return false;
}

int main() {
    // Ten one-take requests: the first nine run, one take each, their own
    // seeds; the tenth is dropped, and named.
    {
        std::vector<AceRequest>  reqs  = batch_of(std::vector<int>(10, 1));
        const AceSynthBatchTakes batch = ace_synth_batch_takes(reqs.data(), (int) reqs.size());
        std::vector<AceRequest>  group;
        std::vector<int>         take_request, take_index;
        synth_batch_expand(reqs, batch, group, take_request, &take_index);
        CHECK(group.size() == (size_t) kAceSynthMaxTakes);
        CHECK(take_request == (std::vector<int>{ 0, 1, 2, 3, 4, 5, 6, 7, 8 }));
        CHECK(take_index == std::vector<int>(kAceSynthMaxTakes, 0));
        CHECK(group.size() == 9 && group[8].seed == 9000);
        const std::vector<std::string> notes = notes_of(reqs, batch);
        CHECK(said(notes, "Batch 10 exceeds DiT max 9, clamping"));
        CHECK(said(notes, "9 track(s) from 9 of 10 request(s)"));
        CHECK(said(notes, "Runs up to r8; drops r9"));
        CHECK(!said(notes, "after it"));
    }
    // A lone request asking 50: 9 takes of it, seeds base + 0..8, and its
    // clamp logged -- it was silent when each request was capped first.
    {
        std::vector<AceRequest>  reqs  = batch_of({ 50 });
        const AceSynthBatchTakes batch = ace_synth_batch_takes(reqs.data(), 1);
        std::vector<AceRequest>  group;
        std::vector<int>         take_request, take_index;
        synth_batch_expand(reqs, batch, group, take_request, &take_index);
        CHECK(group.size() == 9 && group[0].seed == 1000 && group[8].seed == 1008);
        CHECK(take_request == std::vector<int>(9, 0));
        CHECK(take_index == (std::vector<int>{ 0, 1, 2, 3, 4, 5, 6, 7, 8 }));
        const std::vector<std::string> notes = notes_of(reqs, batch);
        CHECK(said(notes, "Batch 50 exceeds DiT max 9, clamping"));
        CHECK(!said(notes, "drops"));
    }
    // {50, 1, 1} asks 52: the first runs 9, the next two are dropped, named.
    {
        std::vector<AceRequest>        reqs  = batch_of({ 50, 1, 1 });
        const AceSynthBatchTakes       batch = ace_synth_batch_takes(reqs.data(), 3);
        const std::vector<std::string> notes = notes_of(reqs, batch);
        CHECK(said(notes, "Batch 52 exceeds DiT max 9, clamping"));
        CHECK(said(notes, "Runs up to r0; drops r1 and the 1 after it"));
    }
    // Under the cap: everything runs, nothing clamped; one take says nothing.
    {
        std::vector<AceRequest>  reqs  = batch_of({ 0, 2 });
        const AceSynthBatchTakes batch = ace_synth_batch_takes(reqs.data(), 2);
        std::vector<AceRequest>  group;
        std::vector<int>         take_request;
        synth_batch_expand(reqs, batch, group, take_request);
        CHECK(take_request == (std::vector<int>{ 0, 1, 1 }));
        CHECK(group.size() == 3 && group[2].seed == 2001);
        const std::vector<std::string> notes = notes_of(reqs, batch);
        CHECK(notes.size() == 1 && said(notes, "3 track(s) from 2 of 2 request(s)"));
        std::vector<AceRequest> one = batch_of({ 1 });
        CHECK(notes_of(one, ace_synth_batch_takes(one.data(), 1)).empty());
    }
    // The server's refusal: a batch asking past the cap, by its requests'
    // sizes summed -- ten of one, a lone 50, {5, 5} -- and none for a batch
    // at it, {4, 5}, or a size below 1, which asks one take.
    for (const std::vector<int> & past : {
             std::vector<int>(10, 1), std::vector<int>{ 50 },
               std::vector<int>{ 5, 5 }
    }) {
        std::vector<AceRequest> reqs = batch_of(past);
        CHECK(ace_synth_batch_takes_refused(ace_synth_batch_takes(reqs.data(), (int) reqs.size())) ==
              kAceRefusalTooManyTakes);
    }
    for (const std::vector<int> & fits : {
             std::vector<int>{ 4, 5 },
             std::vector<int>(9, 0), std::vector<int>{ 9 }
    }) {
        std::vector<AceRequest> reqs = batch_of(fits);
        CHECK(ace_synth_batch_takes_refused(ace_synth_batch_takes(reqs.data(), (int) reqs.size())) == nullptr);
    }
    return test_summary("test-synth-batch");
}
