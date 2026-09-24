#pragma once
// model-store.h: centralised ownership of GGML modules
//
// VRAM policy doctrine. READ THIS BEFORE CHANGING ANYTHING IN THIS FILE.
//
//   --keep-loaded (EVICT_NEVER)
//       Everything stays in VRAM. No reload, ever. The user is telling us
//       they have the budget for the full working set. Do not second-guess
//       them by adding smart eviction rules.
//
//   default (EVICT_STRICT)
//       Maximum VRAM optimisation. At most one GPU module resident at a
//       time. VAE tiles never coexist with DiT weights or LM weights by
//       construction, because only one module is ever loaded. No special
//       case needed.
//
//   invariant held under BOTH policies
//       Exactly ONE LM instance for the whole process. ace_lm (generate)
//       and ace_understand must share the same LM: duplicating it would
//       waste gigabytes for no gain. This is enforced by making the
//       ModelKey identical across both pipelines (same path, same
//       max_seq, same n_kv_sets).
//
// A ModelStore holds the GGML module instances that the pipelines need
// (Qwen3 LM, DiT, VAE encoder, VAE decoder, FSQ tokenizer, etc). Pipelines
// ask the store for a module by key and return it when done. The store
// decides what stays in VRAM and what gets evicted, following the policy
// above set at creation time.
//
// Keys
//   A module is uniquely identified by (kind, path, extras). Two requires
//   with the same key return the same instance. Two requires with different
//   extras (for instance two DiTs with different adapters) are two distinct
//   modules. The LM key deliberately fixes n_kv_sets at 2 * max_batch for
//   both ace_lm and ace_understand so they share one instance.
//
// Refcounting
//   Each module has a refcount. require increments it, release decrements.
//   In EVICT_STRICT, a module with refcount > 0 cannot be evicted: a
//   conflicting require is a programming error (asserts). This catches
//   accidental overlap between modules that must not coexist.
//
// Thread safety
//   All public entry points take a single mutex. Load / unload / hit
//   decisions are serialised. Compute itself runs outside the lock.

#include "bpe.h"
#include "cond-enc.h"
#include "dit.h"
#include "fsq-detok.h"
#include "fsq-tok.h"
#include "metadata-fsm.h"
#include "qwen3-enc.h"
#include "qwen3-lm.h"
#include "vae-enc.h"
#include "vae.h"

#include <cstddef>
#include <memory>
#include <string>

struct ModelStore;

enum ModelKind {
    MODEL_LM,         // Qwen3LM        from acestep-5Hz-lm-*.gguf
    MODEL_TEXT_ENC,   // Qwen3GGML      from Qwen3-Embedding-*.gguf
    MODEL_COND_ENC,   // CondGGML       from acestep-v15-*.gguf (cond_enc.*)
    MODEL_DIT,        // DiTGGML        from acestep-v15-*.gguf
    MODEL_VAE_ENC,    // VAEEncoder     from vae.gguf (encoder.*)
    MODEL_VAE_DEC,    // VAEGGML        from vae.gguf (decoder.*)
    MODEL_FSQ_TOK,    // TokGGML        from acestep-v15-*.gguf (tokenizer.*)
    MODEL_FSQ_DETOK,  // DetokGGML      from acestep-v15-*.gguf (detokenizer.*)
};

struct ModelKey {
    ModelKind   kind;
    std::string path;  // GGUF path the module is loaded from
    // LM-only extras (ignored for other kinds):
    int         max_seq;    // KV cache length
    int         n_kv_sets;  // number of KV sets (1 or 2*max_batch with CFG)
    // DiT-only extras (ignored for other kinds):
    std::string adapter_path;   // "" when no adapter
    float       adapter_scale;  // 1.0f default, significant when adapter_path is set
    // WHICH file at `path` (and at adapter_path) the module is read from:
    // store_key_identity's value when the caller stamped the key. A pipeline
    // stamps its keys once, at load, so every require it makes -- however
    // long its calls run -- asks for the same bytes its metadata came from.
    // A weight replaced at the same path (an installed update) is another
    // file: a key stamped after the replacement misses the cache and reads
    // it afresh, while one stamped before keeps getting its cached module,
    // or fails to load if that was freed -- never the new file's weights
    // (#309). Empty means "whatever file is there now": the store stamps it
    // at the require.
    std::string file_id;
};

// The identity of the file at `path` now (device, inode, size, change and
// modification times); empty if it cannot be stat'ed (gone, or unreadable).
std::string store_file_identity(const std::string & path);

// The identity of the files behind `k` now: its path's, plus its adapter's
// when it has one. What ModelKey::file_id records.
std::string store_key_identity(const ModelKey & k);

enum EvictPolicy {
    EVICT_STRICT,  // default: at most one GPU module resident at a time
    EVICT_NEVER,   // --keep-loaded: never evict, accumulate
};

// DiT metadata cached on the CPU: needed by text encoding and T resolution
// before the DiT itself is loaded on the GPU.
struct DiTMeta {
    DiTGGMLConfig      cfg;
    std::vector<float> silence_full;   // [15000, 64] f32, from silence_latent tensor
    std::vector<float> null_cond_cpu;  // [hidden_size] f32, empty when the model has none
    bool               is_turbo;
};

ModelStore * store_create(EvictPolicy policy);
void         store_free(ModelStore * s);

// Free what no longer matches its file: every cached module and CPU table
// read from a file that has since been replaced or deleted. A lookup never
// frees anything (a caller may hold a pointer from earlier in its call), so
// this is what releases a replaced model's memory. An idle GPU module is
// freed; one a caller holds is retired, counted as resident, and freed by its
// last store_release. Entries whose files are unchanged are untouched.
// Callers must not call this from inside a store call, and must not hold a
// CPU-table pointer across it (a pipeline's DiT metadata is shared, and
// survives).
void         store_release_stale(ModelStore * s);

// Typed GPU module accessors. Each returns a pointer owned by the store;
// never free it yourself. Returns NULL on load failure.
//
// After require, the module stays resident with a refcount > 0 until the
// matching release. In EVICT_STRICT, require evicts every other GPU module
// whose refcount is zero; if any conflicting module has refcount > 0 the
// store aborts (a programming error in the caller).
Qwen3LM *    store_require_lm(ModelStore * s, const ModelKey & k);
Qwen3GGML *  store_require_text_enc(ModelStore * s, const ModelKey & k);
CondGGML *   store_require_cond_enc(ModelStore * s, const ModelKey & k);
DiTGGML *    store_require_dit(ModelStore * s, const ModelKey & k);
VAEEncoder * store_require_vae_enc(ModelStore * s, const ModelKey & k);
VAEGGML *    store_require_vae_dec(ModelStore * s, const ModelKey & k);
TokGGML *    store_require_fsq_tok(ModelStore * s, const ModelKey & k);
DetokGGML *  store_require_fsq_detok(ModelStore * s, const ModelKey & k);

// Release decrements the refcount for the module behind this handle.
// Pass exactly the pointer returned by require. After release, the pointer
// must not be used: in EVICT_STRICT it may be unloaded immediately.
void store_release(ModelStore * s, void * handle);

// CPU-resident accessors. Loaded on first call and kept until
// store_release_stale finds their file replaced; never evicted. All small (a
// few MB total). Return NULL on load failure. `file_id` is the identity
// (store_file_identity) of the file the caller expects at the path, as a
// ModelKey's -- empty for whatever is there now. A file that is no longer
// that one is not read: NULL.
BPETokenizer *  store_bpe(ModelStore * s, const char * lm_path, const std::string & file_id = "");
const float *   store_silence(ModelStore * s, const char * dit_path, const std::string & file_id = "");
MetadataFSM *   store_fsm(ModelStore * s, const char * lm_path, int vocab_size, const std::string & file_id = "");
const DiTMeta * store_dit_meta(ModelStore * s, const char * dit_path, const std::string & file_id = "");
// The same metadata, shared: it outlives store_release_stale and the store's
// own entry for as long as the caller keeps it.
std::shared_ptr<const DiTMeta> store_dit_meta_shared(ModelStore *        s,
                                                     const char *        dit_path,
                                                     const std::string & file_id = "");

// Observability: sum of currently resident GPU module weight buffers, and
// the count of loaded GPU modules. Used by test-model-store to assert
// eviction policy invariants.
size_t store_vram_bytes(const ModelStore * s);
int    store_gpu_module_count(const ModelStore * s);

// RAII helper. Builds on top of store_release, nothing else.
struct ModelHandle {
    ModelStore * store;
    void *       ptr;

    ModelHandle(ModelStore * s, void * p) : store(s), ptr(p) {}

    ~ModelHandle() {
        if (store && ptr) {
            store_release(store, ptr);
        }
    }

    // non-copyable, movable
    ModelHandle(const ModelHandle &)             = delete;
    ModelHandle & operator=(const ModelHandle &) = delete;

    ModelHandle(ModelHandle && o) noexcept : store(o.store), ptr(o.ptr) {
        o.store = nullptr;
        o.ptr   = nullptr;
    }
};
