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
#include "model-registry.h"
#include "model-store.h"
#include "pipeline-lm.h"
#include "pipeline-understand.h"
#include "prompt.h"
#include "version.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

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
            "Usage: %s --models <dir>\n"
            "\n"
            "Required:\n"
            "  --models <dir>          Directory of GGUF model files\n",
            prog);
}

// Scenario 10: the CPU thread count a load ran with (cadenza-audio #889). A
// load shared from a live pair runs with that pair's count, not the one
// configured since; a pair made once none is alive runs with the new one.
static int scenario_loaded_threads(const char * vae_path) {
    fprintf(stderr, "[Test] scenario 10: loaded CPU threads\n");
    ModelKey k_enc = { MODEL_VAE_ENC, vae_path, 0, 0, "", 1.0f };
    ModelKey k_dec = { MODEL_VAE_DEC, vae_path, 0, 0, "", 1.0f };
    int      rc    = 0;

    ace_backend_set_threads(3);
    ModelStore * s   = store_create(EVICT_NEVER);
    auto *       enc = store_require_vae_enc(s, k_enc);
    if (!enc || store_backend_cpu_threads() != 3) {
        fprintf(stderr, "[Test] FAIL: a pair made at 3 threads reads %d\n", store_backend_cpu_threads());
        rc = 1;
    }
    // The encoder holds the pair, so the decoder shares it.
    ace_backend_set_threads(5);
    auto * dec = store_require_vae_dec(s, k_dec);
    if (!dec || store_backend_cpu_threads() != 3) {
        fprintf(stderr, "[Test] FAIL: a load sharing the 3-thread pair reads %d\n", store_backend_cpu_threads());
        rc = 1;
    }
    store_release(s, enc);
    store_release(s, dec);
    store_free(s);

    s   = store_create(EVICT_NEVER);
    dec = store_require_vae_dec(s, k_dec);
    if (!dec || store_backend_cpu_threads() != 5) {
        fprintf(stderr, "[Test] FAIL: a pair made at 5 threads reads %d\n", store_backend_cpu_threads());
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

int main(int argc, char ** argv) {
    const char * models_dir = nullptr;

    if (argc < 2) {
        usage(argv[0]);
        return 1;
    }

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--models") && i + 1 < argc) {
            models_dir = argv[++i];
        } else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) {
            usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            usage(argv[0]);
            return 1;
        }
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
    if (registry.lm.empty() || registry.dit.empty() || registry.vae.empty()) {
        fprintf(stderr, "[Test] FATAL: registry needs LM, DiT and VAE models under %s\n", models_dir);
        return 1;
    }

    const char * lm_path  = registry.lm[0].path.c_str();
    const char * dit_path = registry.dit[0].path.c_str();
    const char * vae_path = registry.vae[0].path.c_str();

    fprintf(stderr, "[Test] LM=%s DiT=%s VAE=%s\n", lm_path, dit_path, vae_path);

    int rc = 0;
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

    if (rc == 0) {
        fprintf(stderr, "[Test] ALL PASS\n");
    } else {
        fprintf(stderr, "[Test] FAIL\n");
    }
    return rc;
}
