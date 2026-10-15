// test-understand-cot: what an understand pass keeps of a metadata block its
// decode stopped inside, and where it closes one that runs on (cadenza-audio
// #648). Self-contained: no model is loaded.
#include "pipeline-understand.h"
#include "prompt.h"

#include <string>

#include "test-util.h"

static AcePrompt parsed(const std::string & text) {
    AcePrompt p = {};
    parse_cot_and_lyrics(cot_drop_open_field(text), &p);
    return p;
}

int main() {
    // A closed block is kept as it is, its last field and all.
    const std::string closed = "<think>\nbpm: 120\nkeyscale: C major\n</think>\n\nla la";
    CHECK(cot_drop_open_field(closed) == closed);

    // Cut inside a value: the fields before it are kept, the cut one dropped.
    {
        AcePrompt p = parsed("<think>\nbpm: 120\ncaption: a bright pop song\nduration: 30\nkeyscale: C ma");
        CHECK(p.bpm == 120);
        CHECK(p.caption == "a bright pop song");
        CHECK(p.duration == 30.0f);
        CHECK(p.keyscale.empty());
    }
    // Cut inside the tempo: 12 of 120 is not a tempo.
    {
        AcePrompt p = parsed("<think>\nbpm: 12");
        CHECK(p.bpm == 0);
    }
    // Cut inside a field's name: the field before it is whole.
    {
        AcePrompt p = parsed("<think>\nbpm: 120\nkeysc");
        CHECK(p.bpm == 120);
    }
    // Cut just past a field's line: that field is whole.
    {
        AcePrompt p = parsed("<think>\nbpm: 120\nduration: 30\n");
        CHECK(p.bpm == 120);
        CHECK(p.duration == 30.0f);
    }
    // A caption wraps onto indented lines: one whose line ended may go on, so
    // it is dropped -- and the fields before it kept.
    {
        AcePrompt p = parsed("<think>\nbpm: 96\ncaption: a slow ballad with\n  soft piano and\n");
        CHECK(p.bpm == 96);
        CHECK(p.caption.empty());
    }
    // A caption followed by the next field's name is whole.
    {
        AcePrompt p = parsed("<think>\nbpm: 96\ncaption: a slow ballad with\n  soft piano\nduration: 4");
        CHECK(p.bpm == 96);
        CHECK(p.caption == "a slow ballad with soft piano");
        CHECK(p.duration == 0.0f);
    }
    // Nothing written yet, or no <think>: nothing to keep, and no crash.
    CHECK(cot_drop_open_field("").empty());
    CHECK(cot_drop_open_field("<think>") == "<think>");
    CHECK(cot_drop_open_field("<think>\n") == "<think>\n");
    CHECK(parsed("bpm: 100\nkeyscale: A minor\n").bpm == 100);

    // A block that runs on is closed at the cap, or on the bound's last token
    // when that comes first -- never past the bound.
    CHECK(ace_understand_metadata_limit(4096) == ACE_UNDERSTAND_METADATA_TOKENS);
    CHECK(ace_understand_metadata_limit(ACE_UNDERSTAND_METADATA_TOKENS + 1) == ACE_UNDERSTAND_METADATA_TOKENS);
    CHECK(ace_understand_metadata_limit(256) == 255);
    CHECK(ace_understand_metadata_limit(1) == 0);
    // Far above a block's 100-160 tokens: a whole one is never cut.
    CHECK(ACE_UNDERSTAND_METADATA_TOKENS >= 3 * 160);

    return test_summary("test-understand-cot");
}
