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
    // gguf-weights.h). wctx_alloc uses it to release each raw tensor's staged mmap
    // pages right after copying that tensor into the backend buffer -- keyed on this
    // range so only pages actually inside the file mapping are touched (heap staging
    // buffers and stack scalars, whose srcs fall outside it, are left alone).
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
// Release the staged mmap pages a just-copied tensor occupied. The GGUF mmap is
// staging, not residency (gguf-weights.h copies each tensor into the backend buffer,
// then gf_close munmaps), so once ggml_backend_tensor_set has consumed a tensor its
// file pages are dead weight until the munmap at the end of the load. Dropping them
// per tensor keeps peak clean-page residency down across a large load instead of
// holding the whole file mapped until then.
//
// Only whole pages fully inside BOTH [src, src+nbytes) and the file mapping are
// dropped: the start rounds up and the end rounds down, so a page shared with an
// adjacent tensor survives, and a src not contained in the mapping (a heap staging
// buffer, a stack scalar) is skipped outright. So this can only ever discard clean
// file-backed pages, which refault from the file on the (unexpected) next read --
// never dirty or anonymous memory, so it cannot corrupt or zero live data.
static size_t wctx_release_file_pages(const void * src, size_t nbytes, const void * file_base, size_t file_len) {
    if (!file_base || file_len == 0 || nbytes == 0) {
        return 0;
    }
    const long ps = sysconf(_SC_PAGESIZE);
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
    if (posix_madvise((void *) start, (size_t) (end - start), POSIX_MADV_DONTNEED) != 0) {
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
    size_t released = 0;
#endif
    for (auto & pc : wctx->pending) {
        // ggml_backend_tensor_set (the synchronous copy, not the _async variant) has
        // fully read pc.src by the time it returns, so releasing the source pages
        // right after is safe -- the data now lives in the backend buffer.
        ggml_backend_tensor_set(pc.tensor, pc.src, pc.offset, pc.nbytes);
        total += pc.nbytes;
#ifdef __APPLE__
        released += wctx_release_file_pages(pc.src, pc.nbytes, wctx->file_base, wctx->file_len);
#endif
    }
#ifdef __APPLE__
    fprintf(stderr, "[WeightCtx] Loaded %zu tensors, %.1f MB into backend (released %.1f MB of staged file pages)\n",
            wctx->pending.size(), (float) total / (1024 * 1024), (float) released / (1024 * 1024));
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
