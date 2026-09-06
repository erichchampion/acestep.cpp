#pragma once
// weight-ctx.h: format-independent weight loading context for ggml backends
//
// Manages a ggml_context for weight tensors + their backend buffer.
// Used by gguf-weights.h for all model loaders.
//
// Usage:
//   WeightCtx wctx;
//   wctx_init(&wctx, n_tensors);
//   ggml_tensor * w = <loader>_load_tensor(&wctx, source, "name");
//   wctx_alloc(&wctx, backend);
//
// Loaders that copy straight out of a file mapping record where the bytes came
// from (wctx_push_file_copy in gguf-weights.h); wctx_alloc releases those staged
// pages as it copies (Apple: munmap, Linux: madvise), which is what keeps load-time
// page residency near one tensor instead of the whole file.

#include "ace-fatal.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

#ifdef __APPLE__
#    include <sys/mman.h>  // munmap
#    include <unistd.h>    // sysconf, _SC_PAGESIZE

#    include <cerrno>      // errno
#elif defined(__linux__)
#    include <sys/mman.h>  // madvise, MADV_DONTNEED
#    include <unistd.h>    // sysconf, _SC_PAGESIZE
#endif

struct WeightCtx {
    struct ggml_context * ctx;
    ggml_backend_buffer_t buffer;

    struct PendingCopy {
        struct ggml_tensor * tensor;
        const void *         src;
        size_t               nbytes;
        size_t               offset;  // byte offset into dst tensor (0 for regular loads)

        // Where the bytes came from, recorded at push time by the loaders that copy
        // straight out of a file mapping (wctx_push_file_copy in gguf-weights.h): the
        // original mapping address and extent to release once this copy is done. Null
        // for copies from heap staging or the stack, which have nothing to release.
        // Kept separate from src/nbytes so an adapter merge can repoint src at its
        // merged heap staging and the original mapping pages still get released by
        // wctx_alloc -- no second mapping parameter to mis-pair at the call site.
        const void * file_src    = nullptr;
        size_t       file_nbytes = 0;
        const void * file_base   = nullptr;
        size_t       file_len    = 0;
    };

    std::vector<PendingCopy> pending;

    // Staging buffers for type-converted data, kept alive until wctx_alloc.
    // unique_ptr keeps the data address stable even when the outer vector grows,
    // so src pointers stored in pending stay valid across staging.push_back().
    std::vector<std::unique_ptr<float[]>> staging;
};

static void wctx_init(WeightCtx * wctx, int n_tensors) {
    size_t                  ctx_size = (size_t) n_tensors * ggml_tensor_overhead() + 1024;
    struct ggml_init_params params   = {
        /*.mem_size   =*/ctx_size,
        /*.mem_buffer =*/NULL,
        /*.no_alloc   =*/true,
    };
    wctx->ctx    = ggml_init(params);
    wctx->buffer = NULL;
    wctx->pending.clear();
    wctx->pending.reserve(n_tensors);
}

// The ACE_NO_PAGE_UNMAP escape hatch for a destructive op in the load path: set it to
// any non-empty value other than "0" to keep the whole mapping resident until gf_close
// (the pre-2.7 behaviour), for A/B footprint measurement or to rule this out if a
// device ever misbehaves. "=0" or unset means release runs. This is a diagnostic
// switch, deliberately not part of the AceBackendConfig API: an application has no
// business disabling memory release, it exists for measurement and emergencies.
// Platform-neutral on purpose -- it gates the release on every platform, not just
// Apple, so it lives outside the ifdefs below.
static bool wctx_page_unmap_disabled(void) {
    static const bool disabled = [] {
        const char * v = std::getenv("ACE_NO_PAGE_UNMAP");
        return v != nullptr && v[0] != '\0' && strcmp(v, "0") != 0;
    }();
    return disabled;
}

#ifdef __APPLE__
// Release the staged GGUF mmap pages a just-copied tensor occupied, by unmapping the
// whole pages fully inside its byte range. The mmap is staging, not residency
// (gguf-weights.h copies each tensor into the backend buffer, then gf_close munmaps
// the rest), so once ggml_backend_tensor_set has consumed a tensor its file pages are
// dead weight. posix_madvise(POSIX_MADV_DONTNEED) -- and MADV_DONTNEED/MADV_FREE --
// are no-ops on macOS (measured with mincore: they free nothing), so a partial munmap
// is the only thing that actually drops the pages and lowers peak clean-page residency
// across a large load. gf_close later munmaps the whole original range, which tolerates
// these interior holes.
//
// Copy-before-unmap needs no guard call: ggml_backend_tensor_set is synchronous by
// construction -- ggml_backend_buffer_i has no async member, so a backend cannot
// defer a set_tensor -- and this is the synchronous variant, not
// ggml_backend_tensor_set_async. The src is fully read when the call returns, and
// nothing reads a tensor's src again afterwards. Unlike madvise, munmap is NOT safe
// on an unexpected re-read (it would fault, not refault), which is why the rules
// below are strict:
//
// Only whole pages fully inside BOTH [src, src+nbytes) and the file mapping are
// unmapped: start rounds up and end rounds down, so a page shared with an adjacent
// (possibly not-yet-copied) tensor survives, and a src not contained in the mapping --
// a heap staging buffer (adapter merge, the f32 and pre-permute loaders) or a stack
// scalar -- is skipped outright. So this only ever unmaps interior pages holding this
// one tensor's bytes, and only after its copy has read them.
static size_t wctx_unmap_file_pages(const void * src, size_t nbytes, const void * file_base, size_t file_len) {
    if (!file_base || file_len == 0 || nbytes == 0 || wctx_page_unmap_disabled()) {
        return 0;
    }
    static const long ps = sysconf(_SC_PAGESIZE);  // process-invariant; query once, not per tensor
    if (ps <= 0) {
        return 0;
    }
    const uintptr_t page = (uintptr_t) ps;
    const uintptr_t s    = (uintptr_t) src;
    const uintptr_t e    = s + nbytes;
    const uintptr_t fb   = (uintptr_t) file_base;
    const uintptr_t fe   = fb + file_len;
    if (s < fb || e > fe) {
        return 0;  // not (fully) inside the file mapping -- a staging/scalar src
    }
    const uintptr_t start = (s + page - 1) & ~(page - 1);  // first whole page at/after src
    const uintptr_t end   = e & ~(page - 1);               // last page boundary at/before src+nbytes
    if (end <= start) {
        return 0;                                          // tensor spans less than one whole interior page
    }
    if (munmap((void *) start, (size_t) (end - start)) != 0) {
        // A real failure must not look like the legitimate "nothing to unmap"
        // returns above: warn once so a systematically broken release is visible
        // instead of masquerading as "unmapped 0.0 MB".
        ace_warn_once("wctx-munmap-failed",
                      "[WeightCtx] WARNING: munmap of a staged tensor's pages failed "
                      "(%s); page release is not taking effect\n",
                      strerror(errno));
        return 0;
    }
    return (size_t) (end - start);
}
#elif defined(__linux__)
// Linux twin of the Apple body, with madvise(MADV_DONTNEED) in place of munmap:
// there the madvise family does work on file mappings, and it keeps the mapping
// intact -- the same interior-page rounding and return value, one call swapped.
// (Built by CI; behaviourally identical release semantics.)
static size_t wctx_unmap_file_pages(const void * src, size_t nbytes, const void * file_base, size_t file_len) {
    if (!file_base || file_len == 0 || nbytes == 0 || wctx_page_unmap_disabled()) {
        return 0;
    }
    static const long ps = sysconf(_SC_PAGESIZE);
    if (ps <= 0) {
        return 0;
    }
    const uintptr_t page = (uintptr_t) ps;
    const uintptr_t s    = (uintptr_t) src;
    const uintptr_t e    = s + nbytes;
    const uintptr_t fb   = (uintptr_t) file_base;
    const uintptr_t fe   = fb + file_len;
    if (s < fb || e > fe) {
        return 0;
    }
    const uintptr_t start = (s + page - 1) & ~(page - 1);
    const uintptr_t end   = e & ~(page - 1);
    if (end <= start) {
        return 0;
    }
    if (madvise((void *) start, (size_t) (end - start), MADV_DONTNEED) != 0) {
        ace_warn_once("wctx-madvise-failed",
                      "[WeightCtx] WARNING: madvise of a staged tensor's pages failed "
                      "(%s); page release is not taking effect\n",
                      strerror(errno));
        return 0;
    }
    return (size_t) (end - start);
}
#else
// No mechanism on this platform (Windows has neither munmap nor madvise for file
// mappings): the staged pages simply stay until gf_close. The log still reports
// what happened -- an honest "unmapped 0.0 MB" for a platform this engine is not
// shipped on.
static size_t wctx_unmap_file_pages(const void * src, size_t nbytes, const void * file_base, size_t file_len) {
    (void) src;
    (void) nbytes;
    (void) file_base;
    (void) file_len;
    return 0;
}
#endif

static bool wctx_alloc(WeightCtx * wctx, ggml_backend_t backend) {
    wctx->buffer = ggml_backend_alloc_ctx_tensors(wctx->ctx, backend);
    if (!wctx->buffer) {
        fprintf(stderr, "[WeightCtx] FATAL: failed to allocate backend buffer\n");
        return false;
    }
    // Mark as weight buffer so ggml_backend_sched assigns ops to the correct
    // backend based on weight location (avoids fallback through expansion).
    ggml_backend_buffer_set_usage(wctx->buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    size_t total    = 0;
    size_t unmapped = 0;
    for (auto & pc : wctx->pending) {
        ggml_backend_tensor_set(pc.tensor, pc.src, pc.offset, pc.nbytes);
        total += pc.nbytes;
        if (pc.file_src) {
            // Release the pages this copy came from -- recorded at push time, so it
            // still names the original mapping pages even after an adapter merge
            // repointed src at a heap staging buffer.
            unmapped += wctx_unmap_file_pages(pc.file_src, pc.file_nbytes, pc.file_base, pc.file_len);
        }
    }
    if (wctx_page_unmap_disabled()) {
        // Name the hatch explicitly: a deliberate off must not read like a
        // release that ran and freed nothing.
        fprintf(stderr,
                "[WeightCtx] Loaded %zu tensors, %.1f MB into backend (page release disabled by ACE_NO_PAGE_UNMAP)\n",
                wctx->pending.size(), (float) total / (1024 * 1024));
    } else {
        fprintf(stderr,
                "[WeightCtx] Loaded %zu tensors, %.1f MB into backend (unmapped %.1f MB of staged file pages)\n",
                wctx->pending.size(), (float) total / (1024 * 1024), (float) unmapped / (1024 * 1024));
    }
    wctx->pending.clear();
    wctx->staging.clear();
    return true;
}

static void wctx_free(WeightCtx * wctx) {
    if (wctx->buffer) {
        ggml_backend_buffer_free(wctx->buffer);
    }
    if (wctx->ctx) {
        ggml_free(wctx->ctx);
    }
    wctx->buffer = NULL;
    wctx->ctx    = NULL;
}
