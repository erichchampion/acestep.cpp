// test-tiling-load: the engine's loads refuse a VAE tiling it can't run
// (cadenza-audio #1311), as the tools do where they parse their flags. A
// chunk of 0 divided a run by a stride of 0; an overlap of half the chunk or
// more cut an interior tile's core away. ace_understand_load is asked in its
// tok-only form, which stamps its paths and reads none, so it needs no model;
// ace_synth_load reads the DiT's metadata, so it runs with --models only.
//
//   ./test-tiling-load [--models <dir>]

#include "model-registry.h"
#include "model-store.h"
#include "pipeline-synth.h"
#include "pipeline-understand.h"
#include "test-util.h"

#include <cstdio>
#include <cstring>

// The tilings refused, and the engine's own defaults' -- which every load takes.
struct Tiling {
    int chunk;
    int overlap;
};

static const Tiling kRefused[] = {
    { 0,                     16 },
    { -1,                    16 },
    { ACE_VAE_TILE_MOST + 1, 16 },
    { 1024,                  -1 },
    { 4,                     2  },
    { 4,                     3  },
    { 128,                   64 },
};

int main(int argc, char ** argv) {
    const char * models_dir = nullptr;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--models") && i + 1 < argc) {
            models_dir = argv[++i];
        }
    }

    ModelStore * store = store_create(EVICT_STRICT);

    AceUnderstandParams und;
    ace_understand_default_params(&und);
    und.model_path = nullptr;
    und.dump_dir   = "/nonexistent/dumps";
    und.dit_path   = "/nonexistent/dit.gguf";
    und.vae_path   = "/nonexistent/vae.gguf";
    {
        AceUnderstand * ok = ace_understand_load(store, &und);
        CHECK(ok != nullptr);  // the defaults' tiling
        if (ok) {
            ace_understand_free(ok);
        }
    }
    for (const Tiling & t : kRefused) {
        AceUnderstandParams p = und;
        p.vae_chunk           = t.chunk;
        p.vae_overlap         = t.overlap;
        AceUnderstand * ud    = ace_understand_load(store, &p);
        if (ud) {
            fprintf(stderr, "[Test] FAIL: ace_understand_load took chunk %d overlap %d\n", t.chunk, t.overlap);
            ace_understand_free(ud);
        }
        CHECK(ud == nullptr);
    }

    if (models_dir) {
        ModelRegistry registry;
        CHECK(registry_scan(&registry, models_dir));
        CHECK(!registry.dit.empty() && !registry.text_enc.empty() && !registry.vae.empty());
        if (!registry.dit.empty() && !registry.text_enc.empty() && !registry.vae.empty()) {
            AceSynthParams syn;
            ace_synth_default_params(&syn);
            syn.text_encoder_path = registry.text_enc[0].path.c_str();
            syn.dit_path          = registry.dit[0].path.c_str();
            syn.vae_path          = registry.vae[0].path.c_str();
            AceSynth * ok         = ace_synth_load(store, &syn);
            CHECK(ok != nullptr);  // the defaults' tiling
            if (ok) {
                ace_synth_free(ok);
            }
            for (const Tiling & t : kRefused) {
                AceSynthParams p = syn;
                p.vae_chunk      = t.chunk;
                p.vae_overlap    = t.overlap;
                AceSynth * synth = ace_synth_load(store, &p);
                if (synth) {
                    fprintf(stderr, "[Test] FAIL: ace_synth_load took chunk %d overlap %d\n", t.chunk, t.overlap);
                    ace_synth_free(synth);
                }
                CHECK(synth == nullptr);
            }
        }
    } else {
        fprintf(stderr, "[Test] ace_synth_load not asked: no --models\n");
    }

    store_free(store);
    return test_summary("test-tiling-load");
}
