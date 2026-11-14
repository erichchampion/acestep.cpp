// test-model-store.cpp: exercise ModelStore with real GGUF loads.
//
// Runs the store in both policies and prints what's resident at each step.
// Serves as living documentation of the expected require / release flow
// and catches regressions in eviction or refcounting logic.
//
// Usage:
//   ./test-model-store --models <dir>
//
// Scans the registry exactly like the CLI binaries and picks the first entry
// of each bucket (LM, DiT, VAE).

#include "backend-config.h"
#include "backend.h"
#include "model-registry.h"
#include "model-store.h"
#include "pipeline-lm.h"
#include "pipeline-synth-impl.h"
#include "pipeline-synth.h"
#include "pipeline-understand.h"
#include "prompt.h"
#include "version.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>

static void dump(ModelStore * s, const char * tag) {
    fprintf(stderr, "[Test] %-24s modules=%d, vram=%.1f MB\n", tag, store_gpu_module_count(s),
            (float) store_vram_bytes(s) / (1024.0f * 1024.0f));
}

// Scenario 1: STRICT policy. Load three different modules in sequence,
// each should evict the previous. At no point should more than one GPU
// module be resident.
static int scenario_strict(const char * lm_path, const char * dit_path, const char * vae_path) {
    fprintf(stderr, "[Test] scenario 1: STRICT\n");
    ModelStore * s = store_create(EVICT_STRICT);
    dump(s, "empty");

    ModelKey k_vae_enc = { MODEL_VAE_ENC, vae_path, 0, 0, "", 1.0f };
    auto *   vae_enc   = store_require_vae_enc(s, k_vae_enc);
    if (!vae_enc) {
        fprintf(stderr, "[Test] FAIL: VAE-Enc load\n");
        store_free(s);
        return 1;
    }
    dump(s, "after require VAE-Enc");
    if (store_gpu_module_count(s) != 1) {
        fprintf(stderr, "[Test] FAIL: expected 1 module, got %d\n", store_gpu_module_count(s));
        store_free(s);
        return 1;
    }
    store_release(s, vae_enc);
    dump(s, "after release VAE-Enc");
    if (store_gpu_module_count(s) != 0) {
        fprintf(stderr, "[Test] FAIL: STRICT should unload on release, got %d modules\n", store_gpu_module_count(s));
        store_free(s);
        return 1;
    }

    ModelKey k_fsq_tok = { MODEL_FSQ_TOK, dit_path, 0, 0, "", 1.0f };
    auto *   fsq_tok   = store_require_fsq_tok(s, k_fsq_tok);
    if (!fsq_tok) {
        fprintf(stderr, "[Test] FAIL: FSQ-Tok load\n");
        store_free(s);
        return 1;
    }
    dump(s, "after require FSQ-Tok");
    store_release(s, fsq_tok);
    dump(s, "after release FSQ-Tok");

    ModelKey k_lm = { MODEL_LM, lm_path, 8192, 1, "", 1.0f };
    auto *   lm   = store_require_lm(s, k_lm);
    if (!lm) {
        fprintf(stderr, "[Test] FAIL: LM load\n");
        store_free(s);
        return 1;
    }
    dump(s, "after require LM");
    if (store_gpu_module_count(s) != 1) {
        fprintf(stderr, "[Test] FAIL: expected 1 module, got %d\n", store_gpu_module_count(s));
        store_free(s);
        return 1;
    }
    store_release(s, lm);
    dump(s, "after release LM");

    store_free(s);
    fprintf(stderr, "[Test] scenario 1: PASS\n");
    return 0;
}

// Scenario 2: NEVER policy. Same three modules, should accumulate and stay.
static int scenario_never(const char * lm_path, const char * dit_path, const char * vae_path) {
    fprintf(stderr, "[Test] scenario 2: NEVER\n");
    ModelStore * s = store_create(EVICT_NEVER);

    ModelKey k_vae_enc = { MODEL_VAE_ENC, vae_path, 0, 0, "", 1.0f };
    auto *   vae_enc   = store_require_vae_enc(s, k_vae_enc);
    store_release(s, vae_enc);
    dump(s, "after 1st release VAE-Enc");

    ModelKey k_fsq_tok = { MODEL_FSQ_TOK, dit_path, 0, 0, "", 1.0f };
    auto *   fsq_tok   = store_require_fsq_tok(s, k_fsq_tok);
    store_release(s, fsq_tok);
    dump(s, "after 1st release FSQ-Tok");

    ModelKey k_lm = { MODEL_LM, lm_path, 8192, 1, "", 1.0f };
    auto *   lm   = store_require_lm(s, k_lm);
    store_release(s, lm);
    dump(s, "after 1st release LM");

    if (store_gpu_module_count(s) != 3) {
        fprintf(stderr, "[Test] FAIL: NEVER should keep 3 modules, got %d\n", store_gpu_module_count(s));
        store_free(s);
        return 1;
    }

    // Second require on VAE-Enc: should cache-hit, no reload.
    auto * vae_enc_2 = store_require_vae_enc(s, k_vae_enc);
    if (vae_enc_2 != vae_enc) {
        fprintf(stderr, "[Test] FAIL: cache hit expected, got different pointer\n");
        store_free(s);
        return 1;
    }
    dump(s, "after 2nd require VAE-Enc");
    store_release(s, vae_enc_2);

    store_free(s);
    fprintf(stderr, "[Test] scenario 2: PASS\n");
    return 0;
}

// Scenario 3: CPU-resident modules are shared, never counted as GPU modules.
static int scenario_cpu(const char * lm_path, const char * dit_path) {
    fprintf(stderr, "[Test] scenario 3: CPU-resident\n");
    ModelStore * s = store_create(EVICT_STRICT);

    auto * bpe1 = store_bpe(s, lm_path);
    auto * bpe2 = store_bpe(s, lm_path);
    if (bpe1 != bpe2) {
        fprintf(stderr, "[Test] FAIL: BPE not cached\n");
        store_free(s);
        return 1;
    }

    auto * silence1 = store_silence(s, dit_path);
    auto * silence2 = store_silence(s, dit_path);
    if (silence1 != silence2) {
        fprintf(stderr, "[Test] FAIL: silence not cached\n");
        store_free(s);
        return 1;
    }

    auto * meta = store_dit_meta(s, dit_path);
    if (!meta) {
        fprintf(stderr, "[Test] FAIL: DiT meta load\n");
        store_free(s);
        return 1;
    }
    fprintf(stderr, "[Test] DiT meta: hidden=%d, layers=%d, turbo=%d, null_cond=%zu\n", meta->cfg.hidden_size,
            meta->cfg.n_layers, meta->is_turbo, meta->null_cond_cpu.size());

    if (store_gpu_module_count(s) != 0) {
        fprintf(stderr, "[Test] FAIL: CPU accessors should not allocate GPU modules\n");
        store_free(s);
        return 1;
    }

    store_free(s);
    fprintf(stderr, "[Test] scenario 3: PASS\n");
    return 0;
}

// Scenario 4: LM sharing invariant. The whole point of the refactor is that
// ace-lm and ace-understand share one LM instance through the store. The
// identity of an LM key is (path, max_seq, n_kv_sets); adapter_path and
// adapter_scale are DiT-only extras and must NOT participate in the LM key.
// Regression guard: if a future change reintroduces adapter_* into the LM
// hash/eq, this test fails and the LM silently duplicates in VRAM under
// --keep-loaded.
static int scenario_lm_sharing(const char * lm_path) {
    fprintf(stderr, "[Test] scenario 4: LM sharing invariant\n");
    ModelStore * s = store_create(EVICT_NEVER);

    // Two keys that differ ONLY in adapter_* fields. These are outside the
    // LM key by design. Must return the same pointer, must not reload.
    ModelKey k_a = { MODEL_LM, lm_path, 8192, 2, "", 1.0f };
    ModelKey k_b = { MODEL_LM, lm_path, 8192, 2, "garbage-adapter-path", 99.0f };

    auto * lm_a = store_require_lm(s, k_a);
    if (!lm_a) {
        fprintf(stderr, "[Test] FAIL: LM load with k_a\n");
        store_free(s);
        return 1;
    }
    auto * lm_b = store_require_lm(s, k_b);
    if (lm_b != lm_a) {
        fprintf(stderr, "[Test] FAIL: LM sharing broken: adapter_* leaked into LM key\n");
        store_free(s);
        return 1;
    }
    if (store_gpu_module_count(s) != 1) {
        fprintf(stderr, "[Test] FAIL: expected 1 LM instance, got %d\n", store_gpu_module_count(s));
        store_free(s);
        return 1;
    }

    // A third key with a different max_seq is a different LM (max_seq IS part
    // of the key because the KV cache allocation depends on it).
    ModelKey k_c  = { MODEL_LM, lm_path, 4096, 2, "", 1.0f };
    auto *   lm_c = store_require_lm(s, k_c);
    if (lm_c == lm_a) {
        fprintf(stderr, "[Test] FAIL: max_seq difference must produce distinct LM\n");
        store_free(s);
        return 1;
    }
    if (store_gpu_module_count(s) != 2) {
        fprintf(stderr, "[Test] FAIL: expected 2 LM instances, got %d\n", store_gpu_module_count(s));
        store_free(s);
        return 1;
    }

    store_release(s, lm_a);
    store_release(s, lm_b);
    store_release(s, lm_c);
    store_free(s);
    fprintf(stderr, "[Test] scenario 4: PASS\n");
    return 0;
}

// Scenario 5: kind differentiation. The VAE GGUF holds both the encoder and
// the decoder; they share the same path but are different modules. The store
// must treat (kind, path) as distinct identities even when the path collides.
static int scenario_kind_split(const char * vae_path) {
    fprintf(stderr, "[Test] scenario 5: kind differentiation\n");
    ModelStore * s = store_create(EVICT_NEVER);

    ModelKey k_enc = { MODEL_VAE_ENC, vae_path, 0, 0, "", 1.0f };
    ModelKey k_dec = { MODEL_VAE_DEC, vae_path, 0, 0, "", 1.0f };

    auto * enc = store_require_vae_enc(s, k_enc);
    auto * dec = store_require_vae_dec(s, k_dec);
    if (!enc || !dec) {
        fprintf(stderr, "[Test] FAIL: VAE enc or dec load\n");
        store_free(s);
        return 1;
    }
    if ((void *) enc == (void *) dec) {
        fprintf(stderr, "[Test] FAIL: VAE enc and dec collapsed into one module\n");
        store_free(s);
        return 1;
    }
    if (store_gpu_module_count(s) != 2) {
        fprintf(stderr, "[Test] FAIL: expected 2 modules (enc + dec), got %d\n", store_gpu_module_count(s));
        store_free(s);
        return 1;
    }

    store_release(s, enc);
    store_release(s, dec);
    store_free(s);
    fprintf(stderr, "[Test] scenario 5: PASS\n");
    return 0;
}

// Scenario 6: refcount correctness. Nested require of the same key returns
// the same pointer and bumps the refcount. Under STRICT the module only
// unloads when the refcount drops to zero, not on the first release.
//
// Not tested here: STRICT require of a DIFFERENT kind while another module
// has refcount > 0 is a programming error and the store asserts. abort()
// cannot be safely caught in this binary, so the invariant lives as a
// comment in model-store.cpp (evict_all_except).
static int scenario_refcount(const char * vae_path) {
    fprintf(stderr, "[Test] scenario 6: refcount correctness\n");
    ModelStore * s = store_create(EVICT_STRICT);

    ModelKey k = { MODEL_VAE_ENC, vae_path, 0, 0, "", 1.0f };

    auto * p1 = store_require_vae_enc(s, k);
    auto * p2 = store_require_vae_enc(s, k);
    if (p1 != p2) {
        fprintf(stderr, "[Test] FAIL: nested require must return same pointer\n");
        store_free(s);
        return 1;
    }
    if (store_gpu_module_count(s) != 1) {
        fprintf(stderr, "[Test] FAIL: nested require must not load twice, got %d modules\n", store_gpu_module_count(s));
        store_free(s);
        return 1;
    }

    // First release drops rc from 2 to 1: under STRICT the module must stay
    // resident because someone still holds the second handle.
    store_release(s, p1);
    if (store_gpu_module_count(s) != 1) {
        fprintf(stderr, "[Test] FAIL: STRICT evicted module with rc>0, got %d modules\n", store_gpu_module_count(s));
        store_free(s);
        return 1;
    }

    // Second release drops rc to 0: STRICT unloads now.
    store_release(s, p2);
    if (store_gpu_module_count(s) != 0) {
        fprintf(stderr, "[Test] FAIL: STRICT kept module after last release, got %d modules\n",
                store_gpu_module_count(s));
        store_free(s);
        return 1;
    }

    store_free(s);
    fprintf(stderr, "[Test] scenario 6: PASS\n");
    return 0;
}

// Scenario 7: ace-server integration invariant.
//
// The whole shared-LM guarantee depends on ace_lm and ace_understand building
// byte-identical LM ModelKeys from user params. If a future flag is wired into
// AceLmParams but forgotten in the AceUnderstandParams mirror that ace-server
// maintains, the two contexts will build divergent keys and the store will
// quietly load the LM twice under --keep-loaded.
//
// We simulate the exact propagation block from ace-server main() and compare
// the two keys by hand. The test must fail loudly when keys drift.
static int scenario_integration(const char * lm_path, const char * dit_path, const char * vae_path) {
    fprintf(stderr, "[Test] scenario 7: ace-server integration\n");
    ModelStore * s = store_create(EVICT_NEVER);

    AceLmParams lm_p;
    ace_lm_default_params(&lm_p);
    lm_p.model_path = lm_path;
    lm_p.max_seq    = 4096;  // non-default, exercises the propagation path
    lm_p.max_batch  = 2;

    AceUnderstandParams und_p;
    ace_understand_default_params(&und_p);
    und_p.model_path = lm_path;
    und_p.dit_path   = dit_path;
    und_p.vae_path   = vae_path;
    // Replicate ace-server main() propagation. Every LM-key field must land here.
    und_p.max_seq    = lm_p.max_seq;
    und_p.max_batch  = lm_p.max_batch;

    AceLm * lm = ace_lm_load(s, &lm_p);
    if (!lm) {
        fprintf(stderr, "[Test] FAIL: ace_lm_load\n");
        store_free(s);
        return 1;
    }
    AceUnderstand * und = ace_understand_load(s, &und_p);
    if (!und) {
        fprintf(stderr, "[Test] FAIL: ace_understand_load\n");
        ace_lm_free(lm);
        store_free(s);
        return 1;
    }

    const ModelKey * k_lm  = ace_lm_lm_key(lm);
    const ModelKey * k_und = ace_understand_lm_key(und);
    if (!k_lm || !k_und) {
        fprintf(stderr, "[Test] FAIL: lm_key accessor returned NULL\n");
        ace_understand_free(und);
        ace_lm_free(lm);
        store_free(s);
        return 1;
    }

    bool same = k_lm->kind == k_und->kind && k_lm->path == k_und->path && k_lm->max_seq == k_und->max_seq &&
                k_lm->n_kv_sets == k_und->n_kv_sets;
    if (!same) {
        fprintf(stderr, "[Test] FAIL: LM ModelKey divergence between pipelines\n");
        fprintf(stderr, "  ace-lm:         kind=%d max_seq=%d n_kv_sets=%d\n", (int) k_lm->kind, k_lm->max_seq,
                k_lm->n_kv_sets);
        fprintf(stderr, "  ace-understand: kind=%d max_seq=%d n_kv_sets=%d\n", (int) k_und->kind, k_und->max_seq,
                k_und->n_kv_sets);
        ace_understand_free(und);
        ace_lm_free(lm);
        store_free(s);
        return 1;
    }

    // The codes a call holds whole (cadenza-audio #648): the context less the
    // prompt's framing, its end token and the decode's reserve -- the
    // engine's when none is asked (1024, or 256 for the metadata alone).
    BPETokenizer * bpe = store_bpe(s, lm_path, k_und->file_id);
    const int framing = bpe ? (int) build_understand_prompt(*bpe, nullptr, 0).size() : -1;
    const int full    = ace_understand_code_capacity(und, false, 0);
    const int meta    = ace_understand_code_capacity(und, true, 0);
    const int asked   = ace_understand_code_capacity(und, false, 500);
    if (framing <= 0 || full != 4096 - framing - 1 - 1024 || meta != 4096 - framing - 1 - 256 ||
        asked != 4096 - framing - 1 - 500 || ace_understand_code_capacity(nullptr, false, 0) != -1) {
        fprintf(stderr, "[Test] FAIL: code capacity full=%d meta=%d asked=%d (framing %d)\n", full, meta, asked,
                framing);
        ace_understand_free(und);
        ace_lm_free(lm);
        store_free(s);
        return 1;
    }

    ace_understand_free(und);
    ace_lm_free(lm);
    store_free(s);
    fprintf(stderr, "[Test] scenario 7: PASS\n");
    return 0;
}

// Scenario 8: store_release_unheld (cadenza-audio #750). Under NEVER an idle
// module stays for reuse; release_unheld frees it -- but not one a caller
// still has required, nor one a live context holds (store_hold_key).
static int scenario_release_unheld(const char * vae_path) {
    fprintf(stderr, "[Test] scenario 8: release unheld\n");
    ModelStore * s = store_create(EVICT_NEVER);

    ModelKey k_enc = { MODEL_VAE_ENC, vae_path, 0, 0, "", 1.0f };
    ModelKey k_dec = { MODEL_VAE_DEC, vae_path, 0, 0, "", 1.0f };
    k_enc.file_id = store_key_identity(k_enc);  // stamped, as a pipeline's are
    k_dec.file_id = store_key_identity(k_dec);

    auto * enc = store_require_vae_enc(s, k_enc);
    auto * dec = store_require_vae_dec(s, k_dec);
    if (!enc || !dec) {
        fprintf(stderr, "[Test] FAIL: VAE enc or dec load\n");
        store_free(s);
        return 1;
    }
    store_hold_key(s, k_dec);  // a live context's

    // The encoder is still required: nothing goes.
    store_release_unheld(s);
    if (store_gpu_module_count(s) != 2) {
        fprintf(stderr, "[Test] FAIL: release_unheld freed a required module, %d left\n", store_gpu_module_count(s));
        store_free(s);
        return 1;
    }

    // Both released: the held decoder stays, the idle encoder goes.
    store_release(s, enc);
    store_release(s, dec);
    if (store_gpu_module_count(s) != 2) {
        fprintf(stderr, "[Test] FAIL: NEVER unloaded on release, %d left\n", store_gpu_module_count(s));
        store_free(s);
        return 1;
    }
    store_release_unheld(s);
    dump(s, "after release_unheld");
    if (store_gpu_module_count(s) != 1) {
        fprintf(stderr, "[Test] FAIL: expected the held decoder alone, got %d modules\n", store_gpu_module_count(s));
        store_free(s);
        return 1;
    }
    if (store_require_vae_dec(s, k_dec) != dec) {
        fprintf(stderr, "[Test] FAIL: the held decoder was not the one kept\n");
        store_free(s);
        return 1;
    }
    store_release(s, dec);

    // Its context gone, the decoder is unheld: freed too.
    store_drop_key(s, k_dec);
    store_release_unheld(s);
    if (store_gpu_module_count(s) != 0 || store_vram_bytes(s) != 0) {
        fprintf(stderr, "[Test] FAIL: expected an empty store, got %d modules\n", store_gpu_module_count(s));
        store_free(s);
        return 1;
    }

    store_free(s);
    fprintf(stderr, "[Test] scenario 8: PASS\n");
    return 0;
}

// Scenario 9: store_vram_bytes_except (cadenza-audio #764). The resident sum
// without a run's own keys is the rest's: none of them, one of them, all of
// them left out, and a key not resident (or null) changes nothing.
static int scenario_vram_except(const char * vae_path) {
    fprintf(stderr, "[Test] scenario 9: vram except\n");
    ModelStore * s = store_create(EVICT_NEVER);

    ModelKey k_enc = { MODEL_VAE_ENC, vae_path, 0, 0, "", 1.0f };
    ModelKey k_dec = { MODEL_VAE_DEC, vae_path, 0, 0, "", 1.0f };
    ModelKey k_tok = { MODEL_FSQ_TOK, vae_path, 0, 0, "", 1.0f };  // never loaded
    k_enc.file_id = store_key_identity(k_enc);
    k_dec.file_id = store_key_identity(k_dec);

    auto * enc = store_require_vae_enc(s, k_enc);
    auto * dec = store_require_vae_dec(s, k_dec);
    if (!enc || !dec) {
        fprintf(stderr, "[Test] FAIL: VAE enc or dec load\n");
        store_free(s);
        return 1;
    }
    const size_t total = store_vram_bytes(s);
    const size_t enc_only = store_vram_bytes_except(s, { &k_dec });
    const size_t dec_only = store_vram_bytes_except(s, { &k_enc });
    int          rc = 0;
    if (store_vram_bytes_except(s, {}) != total || store_vram_bytes_except(s, { &k_tok, nullptr }) != total) {
        fprintf(stderr, "[Test] FAIL: leaving out nothing resident changed the sum\n");
        rc = 1;
    }
    if (enc_only == 0 || dec_only == 0 || enc_only + dec_only != total) {
        fprintf(stderr, "[Test] FAIL: enc %zu + dec %zu is not the total %zu\n", enc_only, dec_only, total);
        rc = 1;
    }
    if (store_vram_bytes_except(s, { &k_enc, &k_dec, &k_tok }) != 0) {
        fprintf(stderr, "[Test] FAIL: leaving out every resident key left bytes\n");
        rc = 1;
    }
    store_release(s, enc);
    store_release(s, dec);
    store_free(s);
    if (rc == 0) {
        fprintf(stderr, "[Test] scenario 9: PASS\n");
    }
    return rc;
}

static void usage(const char * prog) {
    fprintf(stderr, "acestep.cpp %s\n\n", ACE_VERSION);
    fprintf(stderr,
            "Usage: %s --models <dir> [--scenarios all|store|encodes] | --scenarios splice|tasks\n"
            "\n"
            "Required:\n"
            "  --models <dir>          Directory of GGUF model files\n"
            "Optional:\n"
            "  --scenarios <which>     store: the store's policies (1-10); encodes: a\n"
            "                          failed run's hand-out (11); splice: the\n"
            "                          splice rule alone (12), no --models; tasks:\n"
            "                          the task table's query alone (13), no --models;\n"
            "                          all (default)\n",
            prog);
}

// Scenario 10: the CPU thread count a load ran with (cadenza-audio #889). A
// load shared from a live pair runs with that pair's count, not the one
// configured since; a pair made once none is alive runs with the new one.
// The counts fit any host: a configured count is capped at its logical CPUs.
// On a one-CPU host both are 1, so a shared pair's count and the configured
// one can't be told apart there: the scenario says so, and checks the rest.
static int scenario_loaded_threads(const char * vae_path) {
    fprintf(stderr, "[Test] scenario 10: loaded CPU threads\n");
    ModelKey  k_enc = { MODEL_VAE_ENC, vae_path, 0, 0, "", 1.0f };
    ModelKey  k_dec = { MODEL_VAE_DEC, vae_path, 0, 0, "", 1.0f };
    const int a     = 1;
    const int b     = ace_logical_cpus() > 1 ? 2 : 1;
    int       rc    = 0;
    unsigned long long before = 0, after = 0;
    if (a == b) {
        fprintf(stderr, "[Test] scenario 10: one logical CPU: a shared pair's count and a new one's are both %d, "
                        "so sharing is not told from the configured count here\n", a);
    }

    store_last_load_cpu_threads(&before);
    ace_backend_set_threads(a);
    ModelStore * s   = store_create(EVICT_NEVER);
    auto *       enc = store_require_vae_enc(s, k_enc);
    int          got = store_last_load_cpu_threads(&after);
    if (!enc || got != a || after != before + 1) {
        // Loaded, and counted, but at another count: likely a pair an
        // earlier scenario left alive, shared at its own count.
        const bool shared = enc && after == before + 1;
        fprintf(stderr, "[Test] FAIL: a pair made at %d threads reads %d after %llu loads%s\n", a, got,
                after - before, shared ? " (a pair an earlier scenario left alive, shared?)" : "");
        rc = 1;
    }
    // This translation unit reads the same backend state model-store.cpp's
    // loads wrote -- the thread record and load count, the live pairs and
    // the one new loads share. Defined once, in backend.cpp (#973), the two
    // views are one by construction; a disagreement now would mean the
    // record was reset or left stale between the store's read and this one
    // (cadenza-audio #900, #1033).
    unsigned long long here    = 0;
    const int          threads = ace_backend_cpu_threads_loaded(&here);
    if (threads != got || here != after) {
        fprintf(stderr,
                "[Test] FAIL: the backend state reads %d threads after %llu loads, model-store's %d after "
                "%llu: reset or stale between the two reads?\n",
                threads, here, got, after);
        rc = 1;
    }
    // Only while the encoder holds a pair: a failed load leaves none, and
    // says so above.
    if (enc) {
        const AceBackendSharing sharing = ace_backend_sharing();
        if (sharing.live == 0 || sharing.current < 0) {
            fprintf(stderr,
                    "[Test] FAIL: %zu live pairs and current %d while the encoder holds one: "
                    "its pair released or taken out of sharing early?\n",
                    sharing.live, sharing.current);
            rc = 1;
        }
    }
    // The encoder holds the pair, so the decoder shares it.
    ace_backend_set_threads(b);
    auto * dec = store_require_vae_dec(s, k_dec);
    got        = store_last_load_cpu_threads(&after);
    if (!dec || got != a || after != before + 2) {
        fprintf(stderr, "[Test] FAIL: a load sharing the %d-thread pair reads %d after %llu loads\n", a, got,
                after - before);
        rc = 1;
    }
    store_release(s, enc);
    store_release(s, dec);
    store_free(s);
    // Freed, the pair's count stays, and no load has happened since.
    got = store_last_load_cpu_threads(&before);
    if (got != a || before != after) {
        fprintf(stderr, "[Test] FAIL: freeing moved the reading to %d threads, %llu loads\n", got, before);
        rc = 1;
    }

    s   = store_create(EVICT_NEVER);
    dec = store_require_vae_dec(s, k_dec);
    got = store_last_load_cpu_threads(&after);
    if (!dec || got != b || after != before + 1) {
        fprintf(stderr, "[Test] FAIL: a pair made at %d threads reads %d\n", b, got);
        rc = 1;
    }
    store_release(s, dec);
    store_free(s);
    ace_backend_set_threads(0);
    if (rc == 0) {
        fprintf(stderr, "[Test] scenario 10: PASS\n");
    }
    return rc;
}

// Scenario 11 (cadenza-audio #1042): what a failed run hands out through its
// sink, for a failure by NULL -- a run cancelled at the DiT's first poll --
// not only by a throw (GPUFailureTests' refusal). A cover keeps its source's
// clean encode, at the source's length, and no reference it was not given; a
// text2music with a reference keeps the reference's only; a repaint, which
// splices its source back in after the decode, keeps no source encode; and a
// cover-nofsq keeps the same clean encode as the cover, whose FSQ roundtrip
// moved the clean latents aside rather than copying them first. And
// (cadenza-audio #1056) a lego region, which splices its source as a repaint
// does, keeps none; a whole-song lego keeps its clean encode; and a cover
// whose FSQ roundtrip fails -- by a test hook, no input reaching it -- fails
// before its DiT and still keeps its clean encode. And (cadenza-audio #1057)
// an extract and a complete keep their clean encode, as a cover does, and
// every task that needs a source refuses a run given none.
// Cancels at the DiT's first poll, counting the polls it cancelled: a run
// that failed before its DiT (an encode, a require) reached none.
static bool cancel_at_dit(void * data, AceStage stage, int, int) {
    if (stage != ACE_STAGE_DIT) {
        return false;
    }
    ++*(int *) data;
    return true;
}

// Scenario 12: what the one predicate the hand-out and the decode share,
// splices_source_after_decode, says of a task's flags (cadenza-audio #1056),
// and the region frames it gives those it holds for (#1130)
// -- the rule scenario 11's repaint can't tell from the padding it also
// leaves. The flags a task sets are scenario 11's lego runs'. No model
// needed: `--scenarios splice` runs it alone, with no --models.
static int scenario_splice_rule() {
    fprintf(stderr, "[Test] scenario 12: the tasks that splice their source after the decode\n");
    int  rc   = 0;
    auto want = [&](const char * what, bool repaint, bool lego_region, bool splices) {
        SynthState st{};
        st.is_repaint     = repaint;
        st.is_lego_region = lego_region;
        if (splices_source_after_decode(st) != splices) {
            fprintf(stderr, "[Test] FAIL: %s %s its source after the decode\n", what,
                    splices ? "does not splice" : "splices");
            rc = 1;
        }
    };
    want("a repaint", true, false, true);
    want("a lego region", false, true, true);
    want("a cover, or any other task", false, false, false);
    // The frames of the region those splice, by the same rule: a repaint's
    // and a lego region's, clamped to the run's T, and none for any other task
    // -- a lego region once got none, and gave back its source (#1130).
    auto frames = [&](const char * what, bool repaint, bool lego_region, float rs, float re, int T, int t0, int t1) {
        SynthState st{};
        st.is_repaint     = repaint;
        st.is_lego_region = lego_region;
        st.rs             = rs;
        st.re             = re;
        st.T              = T;
        st.repaint_t0     = -7;
        st.repaint_t1     = -7;
        resolve_region_frames(st);
        if (st.repaint_t0 != t0 || st.repaint_t1 != t1) {
            fprintf(stderr, "[Test] FAIL: %s %.2fs..%.2fs of %d frames is frames [%d, %d), want [%d, %d)\n", what, rs,
                    re, T, st.repaint_t0, st.repaint_t1, t0, t1);
            rc = 1;
        }
    };
    frames("a repaint", true, false, 10.0f, 20.0f, 750, 250, 500);
    frames("a lego region", false, true, 10.0f, 20.0f, 750, 250, 500);
    frames("a lego region past the end", false, true, 20.0f, 40.0f, 750, 500, 750);
    frames("a lego region before the start", false, true, -1.0f, 2.0f, 750, 0, 50);
    frames("a lego region wholly past the end", false, true, 40.0f, 50.0f, 750, 750, 750);
    frames("a repaint wholly before the start", true, false, -2.0f, -1.0f, 750, 0, 0);
    frames("an inverted repaint", true, false, 20.0f, 10.0f, 750, 500, 500);
    frames("a cover", false, false, 10.0f, 20.0f, 750, 0, 0);
    // A lego reads its bounds as a repaint does: the -1 end (any negative)
    // is the source's end, so a start with it asks a region to there -- or,
    // before the source, an outpaint -- where it once ran the whole song.
    // The defaults, 0 and -1, ask the whole song: no region (#1134).
    {
        struct Case {
            float rs, re;
            bool  want;
        };

        // A start within the frame slack of 0 is 0: float noise about it asks
        // the whole song, not an outpaint of no frame before the source; a
        // start inside the first frame (0.02 s) is the whole song in frames.
        for (const Case & c : {
                 Case{ 10.0f,    20.0f, true  },
                 Case{ 10.0f,    -1.0f, true  },
                 Case{ -5.0f,    -1.0f, true  },
                 Case{ 0.0f,     -1.0f, false },
                 Case{ 20.0f,    10.0f, false },
                 Case{ 0.0f,     0.0f,  false },
                 Case{ -0.0001f, -1.0f, false },
                 Case{ 0.0001f,  -1.0f, false },
                 Case{ -0.001f,  -1.0f, true  },
                 Case{ 0.02f,    -1.0f, false },
                 Case{ 0.05f,    -1.0f, true  }
        }) {
            if (lego_asks_region(c.rs, c.re) != c.want) {
                fprintf(stderr, "[Test] FAIL: a lego from %.4fs to %.1fs asks %s region, want %s\n", c.rs, c.re,
                        c.want ? "no" : "a", c.want ? "one" : "none");
                rc = 1;
            }
        }
    }
    // A region on frame starts gets those frames -- every start to 1440 s
    // its own frame, none a frame early, as 2991 once were (#1129).
    for (int k = 0; k <= 36000; k++) {
        const float sec = (float) k / ACE_LATENT_FPS;
        if (sec_latent_frame(sec) != k) {
            fprintf(stderr, "[Test] FAIL: %.4fs, frame %d's start, is frame %d\n", sec, k, sec_latent_frame(sec));
            rc = 1;
            break;
        }
    }
    frames("a repaint from 2.12s", true, false, 2.12f, 2.6f, 750, 53, 65);
    // A region's end rounds up to the frame it reaches into: a sub-frame
    // region is a frame, a region into a pad rounded up holds that pad's
    // frame, and an audio outpaint's sub-frame pad its one frame (#1129).
    frames("a sub-frame repaint", true, false, 5.01f, 5.03f, 750, 125, 126);
    frames("a region into a pad rounded up", true, false, 0.0f, 10.03f, 251, 0, 251);
    frames("an audio outpaint 0.03s before", true, false, 0.0f, 0.03f, 750, 0, 1);
    frames("a region ending far out", true, false, 10.0f, 1e10f, 750, 250, 750);
    // A latent outpaint of 0.03 s before the source, to the pad's end: one
    // frame padded, the region shifted by it as padded (0.04 s), so the
    // region is that frame -- not [0, 0), a region that repaints nothing.
    {
        const int   lpad_T = sec_latent_frames_nearest(0.03f);
        const float pad    = latent_frames_sec(lpad_T);
        frames("a latent outpaint 0.03s before", true, false, -0.03f + pad, pad, 750, 0, 1);
    }
    // A source's frames, as the bound before its encode counts them, at a
    // run's most: 15000 latent frames run, and 600 s of audio with a part
    // frame more is 15000 whole ones, which runs too -- one frame more of
    // either does not (cadenza-audio #1144). No sample is read.
    {
        const float sample = 0.0f;

        struct Case {
            const char * name;
            SynthInput   in;
            long long    want;
        };

        SynthInput most_latents, past_latents, most_audio, past_audio;
        most_latents.latents = past_latents.latents = &sample;
        most_latents.T_latent                       = ACE_MAX_LATENT_FRAMES;
        past_latents.T_latent                       = ACE_MAX_LATENT_FRAMES + 1;
        most_audio.audio = past_audio.audio = &sample;
        most_audio.len                      = ACE_MAX_LATENT_FRAMES * ACE_LATENT_HOP + ACE_LATENT_HOP - 1;
        past_audio.len                      = (ACE_MAX_LATENT_FRAMES + 1) * ACE_LATENT_HOP;
        for (const Case & c : {
                 Case{ "15000 latent frames",             most_latents, ACE_MAX_LATENT_FRAMES     },
                 Case{ "15001 latent frames",             past_latents, ACE_MAX_LATENT_FRAMES + 1 },
                 Case{ "600 s and a part frame of audio", most_audio,   ACE_MAX_LATENT_FRAMES     },
                 Case{ "15001 frames of audio",           past_audio,   ACE_MAX_LATENT_FRAMES + 1 }
        }) {
            const bool want_past = c.want > ACE_MAX_LATENT_FRAMES;
            if (c.in.frames() != c.want || frames_past_run_most(c.in.frames()) != want_past) {
                fprintf(stderr, "[Test] FAIL: %s is %lld frames (want %lld), %s\n", c.name, c.in.frames(), c.want,
                        frames_past_run_most(c.in.frames()) ? "past a run's most" : "within it");
                rc = 1;
            }
        }
    }
    if (rc == 0) {
        fprintf(stderr, "[Test] scenario 12 PASS\n");
    }
    return rc;
}

// Scenario 13: the task table, as a tool reads it (cadenza-audio #1094):
// what ace_synth_task_info says of each task -- a source needed by all but
// text2music, a caption by all but lego, extract and complete -- that it
// knows no other, and the caption rule the tools hold a request to by it.
// No model needed: `--scenarios tasks` runs it alone, with no --models.
static int scenario_task_table() {
    fprintf(stderr, "[Test] scenario 13: the task table, as a tool reads it\n");
    int rc = 0;

    struct Want {
        const char * task;
        bool         needs_source, needs_caption, cover_knobs;
    };

    for (const Want & w : {
             Want{ "text2music",  false, true,  false },
             Want{ "cover",       true,  true,  true  },
             Want{ "cover-nofsq", true,  true,  true  },
             Want{ "repaint",     true,  true,  false },
             Want{ "lego",        true,  false, false },
             Want{ "extract",     true,  false, false },
             Want{ "complete",    true,  false, false }
    }) {
        AceSynthTaskInfo info{ !w.needs_source, !w.needs_caption, !w.cover_knobs };
        if (!ace_synth_task_info(w.task, &info)) {
            fprintf(stderr, "[Test] FAIL: the task table does not know %s\n", w.task);
            rc = 1;
        } else if (info.needs_source != w.needs_source || info.needs_caption != w.needs_caption ||
                   info.cover_knobs != w.cover_knobs) {
            fprintf(stderr, "[Test] FAIL: %s needs_source %d needs_caption %d cover_knobs %d, want %d %d %d\n", w.task,
                    info.needs_source, info.needs_caption, info.cover_knobs, w.needs_source, w.needs_caption,
                    w.cover_knobs);
            rc = 1;
        }
    }
    // A task it does not know -- misspelt, empty, none -- is said as that,
    // the caller's info left as it was.
    for (const char * task : { "covers", "", (const char *) nullptr }) {
        AceSynthTaskInfo info{ true, false };
        if (ace_synth_task_info(task, &info) || !info.needs_source || info.needs_caption) {
            fprintf(stderr, "[Test] FAIL: the task table knows '%s', or wrote to the info of a task it does not\n",
                    task ? task : "(null)");
            rc = 1;
        }
    }
    // The caption rule: a task told what to make by its caption needs one,
    // and so does a task the engine does not know, refused by name later.
    auto missing = [&](const char * task, const char * caption, bool want) {
        AceRequest r;
        request_init(&r);
        r.task_type = task;
        r.caption   = caption;
        if (ace_synth_request_missing_caption(&r) != want) {
            fprintf(stderr, "[Test] FAIL: a %s request with caption '%s' is %s\n", task, caption,
                    want ? "not short of one" : "short of one");
            rc = 1;
        }
    };
    missing("text2music", "", true);
    missing("cover", "", true);
    missing("covers", "", true);
    missing("lego", "", false);
    missing("extract", "", false);
    missing("complete", "", false);
    missing("text2music", "a song", false);
    missing("covers", "a song", false);
    // A caption of whitespace alone is none (the app's models trim it too).
    missing("text2music", " \t\n", true);
    missing("lego", " ", false);
    // The app's trim is Unicode's: no-break, ideographic and line-separator
    // spaces are white space too; a zero-width space is not (nor is it in
    // Swift's set), so a caption of it is given.
    missing("text2music", "\xC2\xA0", true);
    missing("text2music", "\xE3\x80\x80 \xE2\x80\xA8", true);
    missing("text2music", "\xE2\x80\x8B", false);
    missing("text2music", "\xC2\xA0song", false);
    // Bytes that start no well-formed code point are a character, not white
    // space: a lone 0x85 or 0xA0 (Latin-1), an overlong space (C0 A0), a
    // 4-byte lead cut short (F2 80 80), a stray continuation byte.
    missing("text2music", "\x85", false);
    missing("text2music", "\xA0", false);
    missing("text2music", "\xC0\xA0", false);
    missing("text2music", "\xF2\x80\x80", false);
    missing("text2music", " \x80 ", false);
    // A batch is refused up front, by request and reason, for a task the
    // engine does not know (said as that, before any caption), requests that
    // mix tasks, or a request short of its caption -- any of them, not only
    // the first (cadenza-audio #1123).
    {
        struct Req {
            const char * task;
            const char * caption;
        };

        struct Case {
            const char *     name;
            std::vector<Req> batch;
            int              want_request;  // -1: not refused
        };

        // The reason each refusal is said with: the case's name says which.
        auto want_reason = [](const char * name) {
            return strstr(name, "unknown") ? "unknown" : strstr(name, " then ") ? "mix" : "caption";
        };

        const std::vector<Case> cases = {
            { "one text2music",                   { { "text2music", "a song" } },                              -1 },
            { "two text2music",                   { { "text2music", "a song" }, { "text2music", "another" } }, -1 },
            { "lego with no captions",            { { "lego", "" }, { "lego", "" } },                          -1 },
            { "an unknown task, no caption",      { { "covers", "" } },                                        0  },
            { "a cover then a text2music",        { { "cover", "a song" }, { "text2music", "a song" } },       1  },
            { "a text2music then a cover",        { { "text2music", "a song" }, { "cover", "a song" } },       1  },
            { "a second request with no caption", { { "text2music", "a song" }, { "text2music", "" } },        1  },
            { "a second request of whitespace",   { { "cover", "a song" }, { "cover", "  " } },                1  },
            { "an unknown task second",           { { "text2music", "a song" }, { "covr", "a song" } },        1  },
        };
        for (const Case & c : cases) {
            std::vector<AceRequest> reqs(c.batch.size());
            for (size_t i = 0; i < c.batch.size(); i++) {
                request_init(&reqs[i]);
                reqs[i].task_type = c.batch[i].task;
                reqs[i].caption   = c.batch[i].caption;
            }
            AceSynthBatchRefusal why;
            const bool           refused = ace_synth_batch_refused(reqs.data(), (int) reqs.size(), &why);
            if (refused != (c.want_request >= 0) || (refused && (why.request != c.want_request || !why.reason ||
                                                                 !strstr(why.reason, want_reason(c.name))))) {
                fprintf(stderr, "[Test] FAIL: %s: refused %d for request %d (%s), want %s request %d\n", c.name,
                        refused, why.request, why.reason ? why.reason : "no reason",
                        c.want_request >= 0 ? "for" : "none,", c.want_request);
                rc = 1;
            }
        }
        // The unknown task is said as that, not as its missing caption.
        AceRequest unknown;
        request_init(&unknown);
        unknown.task_type = "covers";
        AceSynthBatchRefusal why;
        AceRequest second[2];
        request_init(&second[0]);
        request_init(&second[1]);
        second[0].caption   = "a song";
        second[1].caption   = "a song";
        second[1].task_type = "covr";
        AceSynthBatchRefusal second_why;
        if (!ace_synth_batch_refused(second, 2, &second_why) || !second_why.reason ||
            !strstr(second_why.reason, "unknown")) {
            fprintf(stderr, "[Test] FAIL: an unknown task second in a batch is refused as '%s', not as unknown\n",
                    second_why.reason ? second_why.reason : "nothing");
            rc = 1;
        }
        // The requests the tools run: each its synth_batch_size takes, the
        // batch clamped to 9 -- so a request past the 9th take refuses nothing.
        {
            std::vector<AceRequest> batch(10);
            for (AceRequest & r : batch) {
                request_init(&r);
            }
            const int ten_of_one      = ace_synth_requests_run(batch.data(), 10);
            batch[0].synth_batch_size = 9;
            const int after_nine      = ace_synth_requests_run(batch.data(), 10);
            batch[0].synth_batch_size = 4;
            batch[1].synth_batch_size = 4;
            const int after_eight     = ace_synth_requests_run(batch.data(), 10);
            if (ten_of_one != 9 || after_nine != 1 || after_eight != 3) {
                fprintf(stderr, "[Test] FAIL: the tools run %d, %d, %d requests, want 9, 1, 3\n", ten_of_one,
                        after_nine, after_eight);
                rc = 1;
            }
        }
        // An empty batch is refused, not read past.
        AceSynthBatchRefusal empty_why;
        if (!ace_synth_batch_refused(&unknown, 0, &empty_why) || empty_why.request != -1) {
            fprintf(stderr, "[Test] FAIL: an empty batch was not refused as one\n");
            rc = 1;
        }
        if (!ace_synth_batch_refused(&unknown, 1, &why) || !why.reason || !strstr(why.reason, "unknown")) {
            fprintf(stderr, "[Test] FAIL: an unknown task with no caption is refused as '%s', not as unknown\n",
                    why.reason ? why.reason : "nothing");
            rc = 1;
        }
    }
    if (rc == 0) {
        fprintf(stderr, "[Test] scenario 13 PASS\n");
    }
    return rc;
}

static int scenario_failed_encodes(const char * text_enc_path, const char * dit_path, const char * vae_path) {
    fprintf(stderr, "[Test] scenario 11: a failed run's encodes, by NULL\n");
    // NEVER: the runs load each module once.
    ModelStore *   s = store_create(EVICT_NEVER);
    AceSynthParams p;
    ace_synth_default_params(&p);
    p.text_encoder_path = text_enc_path;
    p.dit_path          = dit_path;
    p.vae_path          = vae_path;
    AceSynth * synth    = ace_synth_load(s, &p);
    if (!synth) {
        fprintf(stderr, "[Test] FAIL: ace_synth_load\n");
        store_free(s);
        return 1;
    }
    // Two seconds of two different tones, interleaved stereo at 48 kHz.
    const int          seconds = 2, n = seconds * 48000;
    std::vector<float> source((size_t) n * 2), reference((size_t) n * 2);
    for (int i = 0; i < n; i++) {
        source[(size_t) i * 2] = source[(size_t) i * 2 + 1] = 0.3f * sinf(2.0f * 3.14159265f * 220.0f * i / 48000.0f);
        reference[(size_t) i * 2]                           = reference[(size_t) i * 2 + 1] =
            0.3f * sinf(2.0f * 3.14159265f * 523.0f * i / 48000.0f);
    }
    AceProgress cancel;
    int         dit_polls = 0;
    cancel.fn             = cancel_at_dit;
    cancel.data           = &dit_polls;
    // run TASK [SOURCE] [REFERENCE] KEPT [REGION] [POLLS] [SOURCE LATENTS]:
    // the run, failed, and what it kept -- by the cancel at its DiT's first
    // poll (POLLS 1), or before its DiT (POLLS 0). REGION gives a repaint, or
    // a lego, a region; SOURCE LATENTS gives the source as latents.
    auto run = [&](const char * task, const float * src, const float * ref, AceSynthEncodes & kept, bool region = false,
                   int want_polls = 1, const std::vector<float> * src_latents = nullptr, float region_start = 0.5f,
                   float region_end = 1.5f, int src_len = 0, float cover_strength = 1.0f, float cover_noise = 0.0f) {
        AceRequest r;
        request_init(&r);
        r.caption   = "a short test";
        r.duration  = (float) seconds;
        r.seed      = 1;
        r.task_type = task;
        r.audio_cover_strength = cover_strength;
        r.cover_noise_strength = cover_noise;
        if (region) {
            r.repainting_start = region_start;
            r.repainting_end   = region_end;
        }
        const int     polls = dit_polls;
        AceSynthJob * job   = ace_synth_job_run_dit(
            synth, &r, src, src ? (src_len ? src_len : n) : 0, src_latents ? src_latents->data() : NULL,
            src_latents ? (int) (src_latents->size() / 64) : 0, ref, ref ? n : 0, NULL, 0, 1, cancel, &kept);
        if (job) {
            ace_synth_job_free(job);
            return false;
        }
        return dit_polls == polls + want_polls;
    };
    int             rc      = 0;
    const size_t    frames  = (size_t) seconds * 25;
    const int       ignores = ace_test_source_ignores();
    AceSynthEncodes cover, text, repaint, nofsq;
    if (!run("cover", source.data(), NULL, cover) || !run("text2music", NULL, reference.data(), text) ||
        !run("repaint", source.data(), NULL, repaint, true) || !run("cover-nofsq", source.data(), NULL, nofsq)) {
        fprintf(stderr, "[Test] FAIL: a run returned a job, or failed before its DiT's first poll\n");
        rc = 1;
    }
    auto near = [&](const std::vector<float> & v) {
        const size_t got = v.size() / 64;
        return v.size() % 64 == 0 && got + 2 >= frames && got <= frames + 2;
    };
    if (!near(cover.src_latents) || !cover.ref_latents.empty()) {
        fprintf(stderr,
                "[Test] FAIL: a cancelled cover kept %zu source and %zu reference values, want ~%zu frames "
                "of source and none of a reference it was not given\n",
                cover.src_latents.size(), cover.ref_latents.size(), frames);
        rc = 1;
    }
    if (!text.src_latents.empty() || !near(text.ref_latents)) {
        fprintf(stderr,
                "[Test] FAIL: a cancelled text2music with a reference kept %zu source and %zu reference "
                "values, want the reference's only (~%zu frames)\n",
                text.src_latents.size(), text.ref_latents.size(), frames);
        rc = 1;
    }
    if (!repaint.src_latents.empty()) {
        fprintf(stderr, "[Test] FAIL: a cancelled repaint kept %zu source values; it reads its source again\n",
                repaint.src_latents.size());
        rc = 1;
    }
    if (nofsq.src_latents != cover.src_latents) {
        fprintf(stderr,
                "[Test] FAIL: a cancelled cover kept another encode than a cover-nofsq: the FSQ roundtrip's "
                "latents, not the clean ones?\n");
        rc = 1;
    }
    // A lego region splices its source after the decode, as a repaint does:
    // it hands out none (cadenza-audio #1056).
    AceSynthEncodes lego;
    if (!run("lego", source.data(), NULL, lego, true)) {
        fprintf(stderr, "[Test] FAIL: a lego region returned a job, or failed before its DiT's first poll\n");
        rc = 1;
    } else if (!lego.src_latents.empty()) {
        fprintf(stderr, "[Test] FAIL: a cancelled lego region kept %zu source values; it reads its source again\n",
                lego.src_latents.size());
        rc = 1;
    }
    // A whole-song lego (no region) does not splice: it hands out its clean
    // source encode, as a cover does.
    AceSynthEncodes whole;
    if (!run("lego", source.data(), NULL, whole)) {
        fprintf(stderr, "[Test] FAIL: a whole-song lego returned a job, or failed before its DiT's first poll\n");
        rc = 1;
    } else if (whole.src_latents != cover.src_latents) {
        fprintf(stderr,
                "[Test] FAIL: a cancelled whole-song lego kept %zu source values, not its clean encode (the "
                "cover's, %zu)\n",
                whole.src_latents.size(), cover.src_latents.size());
        rc = 1;
    }
    // A region that ends at or before its start once its -1 end is read is
    // refused before its pad and its source's encode, by a repaint and a lego
    // alike: a start past the source's end with -1 (which reads as the
    // source's end), and an inverted repaint (cadenza-audio #1134).
    {
        struct Case {
            const char * task;
            float        rs, re;
        };

        for (const Case & c : {
                 Case{ "repaint", (float) seconds + 1.0f, -1.0f },
                 Case{ "lego",    (float) seconds + 1.0f, -1.0f },
                 Case{ "repaint", 1.5f,                   0.5f  }
        }) {
            AceSynthEncodes empty;
            const int       before = ace_test_region_empty();
            if (!run(c.task, source.data(), NULL, empty, true, 0, nullptr, c.rs, c.re) ||
                ace_test_region_empty() != before + 1 || !empty.src_latents.empty()) {
                fprintf(stderr, "[Test] FAIL: the %s from %.1f s to %.1f s was not refused before its encode\n", c.task,
                        c.rs, c.re);
                rc = 1;
            }
        }
    }
    // run_task refuses a batch the engine's batch rule refuses, before its
    // job or any encode: a cover and a text2music in one batch, which ran as
    // a cover, and a text2music of whitespace for its caption (#1123).
    {
        for (const bool mixed : { true, false }) {
            AceRequest reqs[2];
            for (AceRequest & r : reqs) {
                request_init(&r);
                r.caption  = "a short test";
                r.duration = (float) seconds;
                r.seed     = 1;
            }
            reqs[0].task_type = mixed ? "cover" : "text2music";
            reqs[1].task_type = "text2music";
            if (!mixed) {
                reqs[1].caption = "   ";
            }
            const int       before = ace_test_batch_refusals();
            const int       polls  = dit_polls;
            AceSynthEncodes kept;
            AceSynthJob *   job =
                ace_synth_job_run_dit(synth, reqs, source.data(), n, NULL, 0, NULL, 0, NULL, 0, 2, cancel, &kept);
            if (job) {
                ace_synth_job_free(job);
            }
            if (job || ace_test_batch_refusals() != before + 1 || dit_polls != polls || !kept.src_latents.empty()) {
                fprintf(stderr, "[Test] FAIL: %s was not refused before its run\n",
                        mixed ? "a cover and a text2music in one batch" : "a text2music of whitespace");
                rc = 1;
            }
        }
    }
    // A lego whose own end is at or before its start asks no region a lego
    // can run: it runs the whole song, warned of -- not refused as a
    // repaint's would be -- handing out its clean encode (#1134).
    {
        AceSynthEncodes inverted;
        const int       before = ace_test_region_empty();
        if (!run("lego", source.data(), NULL, inverted, true, 1, nullptr, 1.5f, 0.5f) ||
            ace_test_region_empty() != before || inverted.src_latents != cover.src_latents) {
            fprintf(stderr, "[Test] FAIL: an inverted lego did not run the whole song, handing out its encode\n");
            rc = 1;
        }
    }
    // A lego from 0.5 s with the -1 end is a region to the source's end, as a
    // repaint's: it splices, so it hands out none (#1134). (The whole-song
    // lego above is the defaults, 0 and -1: no region.)
    {
        AceSynthEncodes to_end;
        if (!run("lego", source.data(), NULL, to_end, true, 1, nullptr, 0.5f, -1.0f)) {
            fprintf(stderr, "[Test] FAIL: a lego from 0.5 s to the end returned a job, or failed before its DiT\n");
            rc = 1;
        } else if (!to_end.src_latents.empty()) {
            fprintf(stderr,
                    "[Test] FAIL: a lego from 0.5 s to the end kept %zu source values: it ran the whole song, not "
                    "the region\n",
                    to_end.src_latents.size());
            rc = 1;
        }
    }
    // An extract and a complete, which work from their source but do not
    // splice it, hand out their clean encode too -- the two tasks only this
    // scenario sends through the dispatch table (#1057).
    for (const char * task : { "extract", "complete" }) {
        AceSynthEncodes stem;
        if (!run(task, source.data(), NULL, stem)) {
            fprintf(stderr, "[Test] FAIL: a %s returned a job, or failed before its DiT's first poll\n", task);
            rc = 1;
        } else if (stem.src_latents != cover.src_latents) {
            fprintf(stderr,
                    "[Test] FAIL: a cancelled %s kept %zu source values, not its clean encode (the cover's, %zu)\n",
                    task, stem.src_latents.size(), cover.src_latents.size());
            rc = 1;
        }
    }
    // Every task the table says needs a source refuses a run given none,
    // by the table's refusal -- not by a failure further in, which a
    // sourceless task also reaches -- with nothing handed out: the table's
    // needs_source column, row by row (#1057).
    for (const char * task : { "cover", "cover-nofsq", "repaint", "lego", "extract", "complete" }) {
        AceSynthEncodes none;
        const int       refused = ace_test_source_refusals();
        if (!run(task, NULL, NULL, none, false, 0) || ace_test_source_refusals() != refused + 1 ||
            !none.src_latents.empty() || !none.ref_latents.empty()) {
            fprintf(stderr, "[Test] FAIL: a %s given no source was not refused by the task table\n", task);
            rc = 1;
        }
    }
    // A region reaching past the source's end by less than a frame has that
    // frame, the pad rounded up to it -- latents and audio both -- and runs
    // to its DiT, not refused as covering none (#1138). The source is 2 s,
    // 50 whole frames.
    {
        AceSynthEncodes past_latents, past_audio;
        if (cover.src_latents.empty() ||
            !run("repaint", NULL, NULL, past_latents, true, 1, &cover.src_latents, 2.0f, 2.01f)) {
            fprintf(stderr, "[Test] FAIL: a latent repaint 2.00..2.01s, past a 2 s source, did not reach its DiT\n");
            rc = 1;
        }
        if (!run("repaint", source.data(), NULL, past_audio, true, 1, nullptr, 2.0f, 2.03f)) {
            fprintf(stderr, "[Test] FAIL: an audio repaint 2.00..2.03s, past a 2 s source, did not reach its DiT\n");
            rc = 1;
        }
        // And before it: a 0.01 s outpaint of latents, a pad the nearest frame
        // would make none, is padded a frame, not refused as covering none.
        AceSynthEncodes before;
        if (!run("repaint", NULL, NULL, before, true, 1, &cover.src_latents, -0.01f, -1.0f)) {
            fprintf(stderr, "[Test] FAIL: a latent repaint -0.01s before a 2 s source did not reach its DiT\n");
            rc = 1;
        }
        // A source whose end falls inside a frame -- 1.53 s, 38.25 frames, the
        // tone cut short; 38 whole frames, already patch-aligned -- repainted
        // from 1.52 s to its end has its last frame, the pad run on to it,
        // not a region of none refused.
        AceSynthEncodes to_end;
        if (!run("repaint", source.data(), NULL, to_end, true, 1, nullptr, 1.52f, -1.0f, 73440)) {
            fprintf(stderr,
                    "[Test] FAIL: an audio repaint 1.52s to the end of a 1.53 s source did not reach its DiT\n");
            rc = 1;
        }
    }
    // A repaint whose region covers no latent frame -- an outpaint of latents
    // within the frame slack of none, so nothing padded -- is refused before
    // its DiT, not run to give back its source unsaid (cadenza-audio #1129).
    {
        AceSynthEncodes none;
        if (cover.src_latents.empty() ||
            !run("repaint", NULL, NULL, none, true, 0, &cover.src_latents, -0.0001f, -1.0f) ||
            !none.src_latents.empty() || !none.ref_latents.empty()) {
            fprintf(stderr, "[Test] FAIL: a repaint whose region covers no frame returned a job, or reached its DiT\n");
            rc = 1;
        }
        // Nor a region whose end is far past what a run holds: refused before
        // its pad is allocated -- gigabytes of one, for 1e10 s (#1139) -- by
        // the bound, not by a failure further in (#1144).
        for (const bool as_latents : { true, false }) {
            AceSynthEncodes far;
            const int       before = ace_test_source_too_long();
            if ((as_latents && cover.src_latents.empty()) ||
                !run("repaint", as_latents ? NULL : source.data(), NULL, far, true, 0,
                     as_latents ? &cover.src_latents : nullptr, 0.5f, 1e10f) ||
                ace_test_source_too_long() != before + 1) {
                fprintf(stderr, "[Test] FAIL: a %s repaint to 1e10 s returned a job, or reached its DiT\n",
                        as_latents ? "latent" : "audio");
                rc = 1;
            }
        }
        // And at the most: a latent outpaint to 600.04 s on the 2 s source pads
        // to 15001 frames, one past a run's most, and is refused (#1139).
        AceSynthEncodes past_most;
        const int       before_most = ace_test_source_too_long();
        if (cover.src_latents.empty() ||
            !run("repaint", NULL, NULL, past_most, true, 0, &cover.src_latents, 1.0f, 600.04f) ||
            ace_test_source_too_long() != before_most + 1) {
            fprintf(stderr, "[Test] FAIL: a latent outpaint to 15001 frames returned a job, or reached its DiT\n");
            rc = 1;
        }
        // A source past a run's most frames is refused before its encode, by
        // every task that works from one -- not by ops_resolve_T after the
        // encode, minutes of one for an hour's audio (cadenza-audio #1144).
        // Latents of 15001 frames, one past the most, for each task; and the
        // audio of as many whole frames for a cover, whose encode it skips.
        {
            const std::vector<float> long_latents((size_t) (ACE_MAX_LATENT_FRAMES + 1) * 64, 0.0f);

            // A lego both whole and over a region: the region one is padded
            // (by none, here) before the bound sees it.
            struct Case {
                const char * task;
                bool         region;
            };

            for (const Case & c : {
                     Case{ "cover",       false },
                     Case{ "cover-nofsq", false },
                     Case{ "repaint",     true  },
                     Case{ "lego",        false },
                     Case{ "lego",        true  },
                     Case{ "extract",     false },
                     Case{ "complete",    false }
            }) {
                AceSynthEncodes too_long;
                const int       before = ace_test_source_too_long();
                if (!run(c.task, NULL, NULL, too_long, c.region, 0, &long_latents) ||
                    ace_test_source_too_long() != before + 1 || !too_long.src_latents.empty()) {
                    fprintf(stderr,
                            "[Test] FAIL: the %s%s from latents of %d frames was not refused before its encode\n",
                            c.task, c.region ? " region" : "", ACE_MAX_LATENT_FRAMES + 1);
                    rc = 1;
                }
            }
            const int                long_len = (ACE_MAX_LATENT_FRAMES + 1) * ACE_LATENT_HOP;
            // Zeros, by calloc: never touched when the bound holds (its 230 MB
            // are not faulted in), and silence, not indeterminate floats, for
            // an encode that runs when it does not.
            const std::unique_ptr<float, decltype(&free)> long_audio(
                static_cast<float *>(calloc((size_t) long_len * 2, sizeof(float))), &free);
            AceSynthEncodes          too_long;
            const int                before = ace_test_source_too_long();
            if (!long_audio) {
                fprintf(stderr, "[Test] FAIL: could not allocate %d frames of audio for the over-long cover\n",
                        ACE_MAX_LATENT_FRAMES + 1);
                rc = 1;
            } else if (!run("cover", long_audio.get(), NULL, too_long, false, 0, nullptr, 0.5f, 1.5f, long_len) ||
                       ace_test_source_too_long() != before + 1 || !too_long.src_latents.empty()) {
                fprintf(stderr, "[Test] FAIL: a cover from audio of %d frames was not refused before its encode\n",
                        ACE_MAX_LATENT_FRAMES + 1);
                rc = 1;
            }
        }
        // Nor a region whose bound is no number: refused, not read as 0 s --
        // before its pad and its source's encode, by a repaint and a lego
        // alike; a lego's would else read as no region and run whole (#1144).
        for (const char * task : { "repaint", "lego" }) {
            for (const bool nan_start : { true, false }) {
                AceSynthEncodes nan_bound;
                const int       before = ace_test_region_nonfinite();
                const float     rs     = nan_start ? std::nanf("") : 0.5f;
                const float     re     = nan_start ? 1.0f : std::nanf("");
                if (!run(task, source.data(), NULL, nan_bound, true, 0, nullptr, rs, re) ||
                    ace_test_region_nonfinite() != before + 1 || !nan_bound.src_latents.empty()) {
                    fprintf(stderr, "[Test] FAIL: the %s with a NaN %s was not refused before its encode\n", task,
                            nan_start ? "start" : "end");
                    rc = 1;
                }
            }
        }
    }
    // The cover knobs are a cover's alone (docs/ARCHITECTURE.md): a lego,
    // an extract, a complete and a repaint asked strength 0.5 and noise 0.5
    // run at 1.0 and 0.0 -- no switch to a non-cover encode, no step their
    // context turns to silence at -- where a whole-song lego asked 0.5 heard
    // silence for its last half (cadenza-audio #1135). A cover keeps both,
    // and switches: the record can tell a partial strength.
    {
        struct Case {
            const char * task;
            bool         region;
            float        noise;
            bool         cover;
        };

        // The cover with no noise first: its steps are the full schedule a
        // noise blend truncates, which no other task's may be.
        int full_steps = -1;
        for (const Case & c : {
                 Case{ "cover",    false, 0.0f, true  },
                 Case{ "cover",    false, 0.5f, true  },
                 Case{ "lego",     false, 0.5f, false },
                 Case{ "lego",     true,  0.5f, false },
                 Case{ "extract",  false, 0.5f, false },
                 Case{ "complete", false, 0.5f, false },
                 Case{ "repaint",  true,  0.5f, false }
        }) {
            AceSynthEncodes kept;
            const char *    what = c.region ? " region" : "";
            if (!run(c.task, source.data(), NULL, kept, c.region, 1, nullptr, 0.5f, 1.5f, 0, 0.5f, c.noise)) {
                fprintf(stderr, "[Test] FAIL: the %s%s asked strength 0.5 returned a job, or failed before its DiT\n",
                        c.task, what);
                rc = 1;
                continue;
            }
            const AceTestCoverRecord got = ace_test_last_cover();
            // The baseline is the plain cover's alone: if it failed, every
            // later case says so, not that its knobs were wrong.
            if (c.cover && c.noise == 0.0f) {
                full_steps = got.num_steps;
            }
            // A noise blend's effect is its truncated schedule: a cover's
            // with noise is shorter, every other task's the full one.
            const bool steps_ok = full_steps >= 0 && ((c.cover && c.noise > 0.0f) ? got.num_steps < full_steps :
                                                                                    got.num_steps == full_steps);
            const bool knobs_ok =
                c.cover ? got.strength == 0.5f && got.noise == c.noise && got.enc_switch && got.cover_steps >= 0 :
                          got.strength == 1.0f && got.noise == 0.0f && !got.enc_switch && got.cover_steps == -1;
            const bool ok = steps_ok && knobs_ok;
            if (!ok) {
                fprintf(stderr,
                        "[Test] FAIL: the %s%s asked strength 0.5, noise %.2f ran at %.2f, %.2f, %s its switch, "
                        "silence from step %d, %d of %d steps\n",
                        c.task, what, c.noise, got.strength, got.noise, got.enc_switch ? "with" : "without",
                        got.cover_steps, got.num_steps, full_steps);
                rc = 1;
            }
        }
    }
    // A cover's knobs out of [0, 1] -- an infinity too -- are clamped to it,
    // said; a NaN runs at its default (cadenza-audio #1152, owner decision).
    // Each run is the run of its clamped knobs: the same steps, the same
    // switch and silence step as a cover asked those knobs outright.
    {
        const float nan = std::nanf("");
        const float inf = std::numeric_limits<float>::infinity();
        struct Case {
            float strength, noise, want_strength, want_noise;
        };

        auto cover_run = [&](float strength, float noise, AceTestCoverRecord & got) {
            AceSynthEncodes kept;
            const bool ran = run("cover", source.data(), NULL, kept, false, 1, nullptr, 0.5f, 1.5f, 0, strength, noise);
            got            = ace_test_last_cover();
            return ran;
        };
        for (const Case & c : {
                 Case{ 1.5f,  0.0f,  1.0f, 0.0f },
                 Case{ -0.5f, 0.0f,  0.0f, 0.0f },
                 Case{ nan,   0.0f,  1.0f, 0.0f },
                 Case{ inf,   0.0f,  1.0f, 0.0f },
                 Case{ 0.5f,  1.5f,  0.5f, 1.0f },
                 Case{ 0.5f,  -0.5f, 0.5f, 0.0f },
                 Case{ 0.5f,  nan,   0.5f, 0.0f },
                 Case{ 0.5f,  inf,   0.5f, 1.0f }
        }) {
            AceTestCoverRecord got, want;
            if (!cover_run(c.strength, c.noise, got) || !cover_run(c.want_strength, c.want_noise, want)) {
                fprintf(stderr, "[Test] FAIL: a cover asked %g, %g returned a job, or failed before its DiT\n",
                        c.strength, c.noise);
                rc = 1;
                continue;
            }
            if (got.strength != c.want_strength || got.noise != c.want_noise || got.num_steps != want.num_steps ||
                got.enc_switch != want.enc_switch || got.cover_steps != want.cover_steps) {
                fprintf(stderr,
                        "[Test] FAIL: a cover asked %g, %g ran at %g, %g (%d steps, switch %d, silence at %d), want "
                        "%g, %g's run (%d, %d, %d)\n",
                        c.strength, c.noise, got.strength, got.noise, got.num_steps, got.enc_switch, got.cover_steps,
                        c.want_strength, c.want_noise, want.num_steps, want.enc_switch, want.cover_steps);
                rc = 1;
            }
        }
    }
    // A cover whose FSQ roundtrip cannot run fails before its DiT, and still
    // hands out its clean encode -- the one a cover-nofsq hands out (#1056).
    AceSynthEncodes failed;
    const int       fired = ace_test_fsq_require_failures();
    ace_test_fail_next_fsq_require(true);
    const bool failed_early = run("cover", source.data(), NULL, failed, false, 0);
    ace_test_fail_next_fsq_require(false);
    if (ace_test_fsq_require_failures() != fired + 1) {
        fprintf(stderr, "[Test] FAIL: the cover never reached the FSQ require the hook fails\n");
        rc = 1;
    } else if (!failed_early) {
        fprintf(stderr, "[Test] FAIL: a cover whose FSQ roundtrip failed returned a job, or reached its DiT\n");
        rc = 1;
    } else if (failed.src_latents != cover.src_latents) {
        fprintf(stderr,
                "[Test] FAIL: a cover whose FSQ roundtrip failed kept %zu source values, want its clean encode "
                "(%zu)\n",
                failed.src_latents.size(), cover.src_latents.size());
        rc = 1;
    }
    // A reference is no source: no run so far -- a text2music given only a
    // reference, every task given the source it works from, every one given
    // none, a cover whose FSQ roundtrip failed -- was warned of a source it
    // ignores (cadenza-audio #1095).
    if (ace_test_source_ignores() != ignores) {
        fprintf(stderr, "[Test] FAIL: %d runs given no source their task ignores were warned of one\n",
                ace_test_source_ignores() - ignores);
        rc = 1;
    }
    // A text2music given a source -- as audio, or as latents (the cover's
    // clean encode) -- runs as asked, without it: warned of the source it
    // ignores, not refused, and handing out no encode of it, only its
    // reference's (cadenza-audio #1095).
    for (const bool as_latents : { false, true }) {
        const char * form = as_latents ? "source latents" : "source audio";
        if (as_latents && cover.src_latents.empty()) {
            fprintf(stderr, "[Test] FAIL: no cover encode to give a text2music as source latents\n");
            rc = 1;
            continue;
        }
        AceSynthEncodes given;
        const int       refused = ace_test_source_refusals(), ignored = ace_test_source_ignores();
        const bool ran = as_latents ? run("text2music", NULL, reference.data(), given, false, 1, &cover.src_latents) :
                                      run("text2music", source.data(), reference.data(), given);
        // The counters first: a refusal stops the run before its DiT, which
        // is said as that, not as a failure further in.
        if (ace_test_source_refusals() != refused) {
            fprintf(stderr, "[Test] FAIL: a text2music given %s was refused\n", form);
            rc = 1;
        } else if (ace_test_source_ignores() != ignored + 1) {
            fprintf(stderr, "[Test] FAIL: a text2music given %s was not warned of it\n", form);
            rc = 1;
        } else if (!ran) {
            fprintf(stderr, "[Test] FAIL: a text2music given %s returned a job, or failed before its DiT\n", form);
            rc = 1;
        } else if (!given.src_latents.empty() || given.ref_latents != text.ref_latents) {
            fprintf(stderr,
                    "[Test] FAIL: a text2music given %s kept %zu source and %zu reference values, want none of the "
                    "source and the reference's (%zu)\n",
                    form, given.src_latents.size(), given.ref_latents.size(), text.ref_latents.size());
            rc = 1;
        }
    }
    ace_synth_free(synth);
    store_free(s);
    if (rc == 0) {
        fprintf(stderr,
                "[Test] scenario 11 PASS: cover %zu frames, reference %zu, repaint and lego region none, a "
                "whole-song lego, an extract, a complete and a failed FSQ roundtrip their clean encode, every "
                "run given no source refused, and a text2music's ignored source warned of\n",
                cover.src_latents.size() / 64, text.ref_latents.size() / 64);
    }
    return rc;
}

int main(int argc, char ** argv) {
    const char * models_dir = nullptr;
    const char * which      = "all";

    if (argc < 2) {
        usage(argv[0]);
        return 1;
    }

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--models") && i + 1 < argc) {
            models_dir = argv[++i];
        } else if (!strcmp(argv[i], "--scenarios") && i + 1 < argc) {
            which = argv[++i];
            if (strcmp(which, "all") && strcmp(which, "store") && strcmp(which, "encodes") && strcmp(which, "splice") &&
                strcmp(which, "tasks")) {
                fprintf(stderr, "[CLI] ERROR: --scenarios takes all, store, encodes, splice or tasks, not %s\n", which);
                return 1;
            }
        } else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) {
            usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            usage(argv[0]);
            return 1;
        }
    }

    // Scenarios 12 and 13 read no model: run alone, they need no --models.
    if (!strcmp(which, "splice") || !strcmp(which, "tasks")) {
        const int rc = !strcmp(which, "splice") ? scenario_splice_rule() : scenario_task_table();
        fprintf(stderr, rc == 0 ? "[Test] ALL PASS\n" : "[Test] FAIL\n");
        return rc;
    }

    if (!models_dir) {
        fprintf(stderr, "[CLI] ERROR: --models required\n");
        usage(argv[0]);
        return 1;
    }

    // Registry scan: one place to resolve all three GGUF paths, exactly like
    // the CLI binaries. The test is itself a user of the registry API, so any
    // drift there breaks scenario 7 just as it would break ace-server.
    ModelRegistry registry;
    if (!registry_scan(&registry, models_dir)) {
        fprintf(stderr, "[Test] FATAL: cannot scan --models %s\n", models_dir);
        return 1;
    }
    const bool store   = strcmp(which, "encodes") != 0;
    const bool encodes = strcmp(which, "store") != 0;
    // The planner LM only the store's scenarios load; the hand-out's runs
    // need the DiT, the VAE and (below) a text encoder.
    if ((store && registry.lm.empty()) || registry.dit.empty() || registry.vae.empty()) {
        fprintf(stderr, "[Test] FATAL: registry needs %sDiT and VAE models under %s\n", store ? "LM, " : "",
                models_dir);
        return 1;
    }

    const char * lm_path  = store ? registry.lm[0].path.c_str() : "";
    const char * dit_path = registry.dit[0].path.c_str();
    const char * vae_path = registry.vae[0].path.c_str();

    fprintf(stderr, "[Test] LM=%s DiT=%s VAE=%s\n", lm_path, dit_path, vae_path);

    int rc = 0;
    if (store) {
        rc |= scenario_strict(lm_path, dit_path, vae_path);
        rc |= scenario_never(lm_path, dit_path, vae_path);
        rc |= scenario_cpu(lm_path, dit_path);
        rc |= scenario_lm_sharing(lm_path);
        rc |= scenario_kind_split(vae_path);
        rc |= scenario_refcount(vae_path);
        rc |= scenario_integration(lm_path, dit_path, vae_path);
        rc |= scenario_release_unheld(vae_path);
        rc |= scenario_vram_except(vae_path);
        rc |= scenario_loaded_threads(vae_path);
    }
    // Scenarios 12 and 13 run with these only in a full run: ctest runs each
    // alone (--scenarios splice, tasks), so a failure of one is said once.
    if (strcmp(which, "all") == 0) {
        rc |= scenario_splice_rule();
        rc |= scenario_task_table();
    }
    if (encodes) {
        // Scenario 11 runs the synth, which needs a text encoder too: without one
        // it fails, after the others have run.
        if (registry.text_enc.empty()) {
            fprintf(stderr, "[Test] FAIL: scenario 11 needs a text-encoder model under %s\n", models_dir);
            rc |= 1;
        } else {
            rc |= scenario_failed_encodes(registry.text_enc[0].path.c_str(), dit_path, vae_path);
        }
    }

    if (rc == 0) {
        fprintf(stderr, "[Test] ALL PASS\n");
    } else {
        fprintf(stderr, "[Test] FAIL\n");
    }
    return rc;
}
