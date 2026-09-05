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

#include "ggml-backend.h"
#include "ggml.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>

#ifdef __APPLE__
#    include <sys/mman.h>  // posix_madvise, POSIX_MADV_DONTNEED
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
    };

    std::vector<PendingCopy> pending;

    // Staging buffers for type-converted data, kept alive until wctx_alloc.
    // unique_ptr keeps the data address stable even when the outer vector grows,
    // so src pointers stored in pending stay valid across staging.push_back().
    std::vector<std::unique_ptr<float[]>> staging;

    // The source GGUF file mapping, if the loader recorded one (gf_note_mapping in
    // gguf-weights.h). wctx_alloc uses it to unmap each raw tensor's staged mmap pages
    // right after copying that tensor into the backend buffer -- keyed on this range
    // so only pages actually inside the file mapping are touched (heap staging buffers
    // and stack scalars, whose srcs fall outside it, are left alone). One range is
    // enough: every wctx_alloc caller stages from a single GGUF, and an adapter's
    // separate mapping is read and closed inside adapter_merge before wctx_alloc, so
    // it never reaches here. If a second distinct mapping were ever noted, the last
    // wins and the other file's pages simply stay mapped until its gf_close -- a
    // missed release, never an unsafe unmap (the range check still gates every call).
    const void * file_base = nullptr;
    size_t       file_len  = 0;
};

static void wctx_init(WeightCtx * wctx, int n_tensors) {
    size_t                  ctx_size = (size_t) n_tensors * ggml_tensor_overhead() + 1024;
    struct ggml_init_params params   = {
        /*.mem_size   =*/ctx_size,
        /*.mem_buffer =*/NULL,
        /*.no_alloc   =*/true,
    };
    wctx->ctx       = ggml_init(params);
    wctx->buffer    = NULL;
    wctx->file_base = nullptr;
    wctx->file_len  = 0;
    wctx->pending.clear();
    wctx->pending.reserve(n_tensors);
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
// Only whole pages fully inside BOTH [src, src+nbytes) and the file mapping are
// unmapped: start rounds up and end rounds down, so a page shared with an adjacent
// (possibly not-yet-copied) tensor survives, and a src not contained in the mapping --
// a heap staging buffer (adapter merge, the f32 and pre-permute loaders) or a stack
// scalar -- is skipped outright. So this only ever unmaps interior pages holding this
// one tensor's bytes, and only after its copy has read them. Unlike madvise, munmap is
// NOT safe on an unexpected re-read (it would fault, not refault), which is exactly why
// the containment and interior-only rules are strict -- and they hold because nothing
// reads a tensor's src again once wctx_alloc has copied it.
static size_t wctx_unmap_file_pages(const void * src, size_t nbytes, const void * file_base, size_t file_len) {
    if (!file_base || file_len == 0 || nbytes == 0) {
        return 0;
    }
    // Escape hatch for a destructive op in the load path: ACE_NO_PAGE_UNMAP=1 keeps the
    // whole mapping resident until gf_close (the pre-2.7 behaviour), for A/B footprint
    // measurement or to rule this out if a device ever misbehaves. Queried once.
    static const bool disabled = std::getenv("ACE_NO_PAGE_UNMAP") != nullptr;
    if (disabled) {
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
        return 0;
    }
    return (size_t) (end - start);
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
    size_t total = 0;
#ifdef __APPLE__
    size_t unmapped = 0;
#endif
    for (auto & pc : wctx->pending) {
        // ggml_backend_tensor_set (the synchronous copy, not the _async variant) has
        // fully read pc.src by the time it returns, so unmapping the source pages
        // right after is safe -- the data now lives in the backend buffer.
        ggml_backend_tensor_set(pc.tensor, pc.src, pc.offset, pc.nbytes);
        total += pc.nbytes;
#ifdef __APPLE__
        unmapped += wctx_unmap_file_pages(pc.src, pc.nbytes, wctx->file_base, wctx->file_len);
#endif
    }
#ifdef __APPLE__
    fprintf(stderr, "[WeightCtx] Loaded %zu tensors, %.1f MB into backend (unmapped %.1f MB of staged file pages)\n",
            wctx->pending.size(), (float) total / (1024 * 1024), (float) unmapped / (1024 * 1024));
#else
    fprintf(stderr, "[WeightCtx] Loaded %zu tensors, %.1f MB into backend\n", wctx->pending.size(),
            (float) total / (1024 * 1024));
#endif
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
