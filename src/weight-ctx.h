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
#elif defined(__linux__)
#    include <sys/mman.h>  // madvise, MADV_DONTNEED
#    include <unistd.h>    // sysconf, _SC_PAGESIZE
#endif

#include <cerrno>  // errno, for the release-failure warning

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
        //
        // Invariant: two PendingCopys must never name overlapping file ranges. The
        // first release unmaps the pages, and the second copy's read then faults on
        // Apple (on Linux madvise would silently refault, hiding the same bug). Tied
        // weights -- one GGUF byte range serving two tensors -- are the natural way
        // to violate this; push one copy and reuse the tensor instead. Loaders that
        // convert out of the mapping (norms, biases, the pre-permutes) push plain
        // copies with these fields null; their (small) source pages release at
        // gf_close instead, which is part of the gap the completion line reports.
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
// Deliberately NOT cached: the A/B use case flips it between loads in one process, so
// a latched value would make the second leg measure the first. getenv is a linear
// scan of the environment -- nothing against a multi-GB weight copy.
static bool wctx_page_unmap_disabled(void) {
    const char * v = std::getenv("ACE_NO_PAGE_UNMAP");
    return v != nullptr && v[0] != '\0' && strcmp(v, "0") != 0;
}

#if defined(__APPLE__) || defined(__linux__)
// The per-platform release primitive over a page-aligned interior range. Same
// contract on both: drop the pages, return false + set errno on failure.
#    if defined(__APPLE__)
// munmap is the only call that works on macOS: the madvise family (POSIX_MADV_DONTNEED,
// MADV_DONTNEED, MADV_FREE) is a no-op there -- measured with mincore, they free
// nothing. munmap is destructive (an unexpected re-read faults instead of refaulting),
// which is why the containment and interior-only rules in the caller matter.
static bool wctx_release_pages(void * start, size_t len) {
    return munmap(start, len) == 0;
}

static const char * wctx_release_op = "munmap";
#    else
// Linux: madvise(MADV_DONTNEED) does work on file mappings there and keeps the
// mapping intact -- same drop-and-refault semantics for these read-only staged pages.
static bool wctx_release_pages(void * start, size_t len) {
    return madvise(start, len, MADV_DONTNEED) == 0;
}

static const char * wctx_release_op = "madvise";
#    endif

// Release the staged GGUF mmap pages a just-copied tensor occupied: the whole pages
// fully inside its byte range. The mmap is staging, not residency (gguf-weights.h
// copies each tensor into the backend buffer, then gf_close releases the rest), so
// once ggml_backend_tensor_set has consumed a tensor its file pages are dead weight.
// Releasing them per tensor keeps peak clean-page residency across a large load near
// one tensor instead of the whole file; gf_close later releases the whole original
// range, which tolerates these interior holes.
//
// Copy-before-release: wctx_alloc calls ggml_backend_tensor_set -- the synchronous
// variant, never ggml_backend_tensor_set_async. The buffer-level iface it dispatches
// to (ggml_backend_buffer_i::set_tensor) has no deferred form; the async operations
// that exist live on the backend-level ggml_backend_i and are separate API calls
// this code does not use. Verified per vendored backend that the synchronous call
// consumes src before returning: CPU memcpy, Metal blit + semaphore wait (shared
// buffers are a memcpy), CUDA memcpyAsync + cudaStreamSynchronize, Vulkan write with
// waitForFences. The src is fully read when the call returns, and nothing reads a
// tensor's src again afterwards.
//
// Only whole pages fully inside BOTH [src, src+nbytes) and the file mapping are
// released: start rounds up and end rounds down, so a page shared with an adjacent
// (possibly not-yet-copied) tensor survives, and a src not contained in the mapping --
// a heap staging buffer (adapter merge, the f32 and pre-permute loaders) or a stack
// scalar -- is skipped outright. The containment test is unreachable for correctly
// recorded copies (wctx_push_file_copy records in-mapping sources, and wctx_alloc
// gates on file_src), and that is the point: it keeps a future mis-recording a
// harmless no-op instead of a destructive call on the wrong memory.
//
// failed_bytes accumulates the ranges the kernel refused to release, so wctx_alloc
// can report the shortfall for THIS load -- a persistent failure must not be
// invisible just because its first occurrence already warned.
static size_t wctx_unmap_file_pages(const void * src,
                                    size_t       nbytes,
                                    const void * file_base,
                                    size_t       file_len,
                                    size_t *     failed_bytes) {
    const long ps = sysconf(_SC_PAGESIZE);  // nanoseconds against a multi-GB copy; no caching
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
    if (!wctx_release_pages((void *) start, (size_t) (end - start))) {
        // The strerror detail once per process; the per-load summary below makes
        // every later failure visible regardless.
        ace_warn_once(
            "[WeightCtx] WARNING: %s of a staged tensor's pages failed (%s); page release is not "
            "taking effect\n",
            wctx_release_op, strerror(errno));
        *failed_bytes += (size_t) (end - start);
        return 0;
    }
    return (size_t) (end - start);
}
#else
// No release mechanism on this platform (Windows has neither munmap nor madvise for
// file mappings): the staged pages simply stay until gf_close.
static size_t wctx_unmap_file_pages(const void *, size_t, const void *, size_t, size_t *) {
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
#ifndef NDEBUG
    // Enforce the PendingCopy overlap invariant (see the struct comment) in debug
    // builds: a duplicated file range would fault the second copy's read once the
    // first release has run. O(n^2) over a few hundred entries is nothing next to
    // the copies it guards.
    for (size_t i = 0; i < wctx->pending.size(); i++) {
        if (!wctx->pending[i].file_src) {
            continue;
        }
        for (size_t j = i + 1; j < wctx->pending.size(); j++) {
            const WeightCtx::PendingCopy & a = wctx->pending[i];
            const WeightCtx::PendingCopy & b = wctx->pending[j];
            if (!b.file_src) {
                continue;
            }
            const uintptr_t as = (uintptr_t) a.file_src;
            const uintptr_t ae = as + a.file_nbytes;
            const uintptr_t bs = (uintptr_t) b.file_src;
            const uintptr_t be = bs + b.file_nbytes;
            if (as < be && bs < ae) {
                fprintf(stderr,
                        "[WeightCtx] FATAL: two pending copies share file bytes -- tied weights "
                        "must push one copy\n");
                return false;
            }
        }
    }
#endif
    // The hatch is decided once per load, here -- not inside the release helper -- so
    // the off switch is one decision a second caller cannot forget to re-apply.
    const bool disabled = wctx_page_unmap_disabled();
    size_t     total    = 0;
    size_t     recorded = 0;  // staged bytes with a recorded file range
    size_t     unmapped = 0;
    size_t     failed   = 0;
    for (auto & pc : wctx->pending) {
        ggml_backend_tensor_set(pc.tensor, pc.src, pc.offset, pc.nbytes);
        total += pc.nbytes;
        if (pc.file_src) {
            // Release the pages this copy came from -- recorded at push time, so it
            // still names the original mapping pages even after an adapter merge
            // repointed src at a heap staging buffer.
            recorded += pc.file_nbytes;
            if (!disabled) {
                unmapped += wctx_unmap_file_pages(pc.file_src, pc.file_nbytes, pc.file_base, pc.file_len, &failed);
            }
        }
    }
#if defined(__APPLE__) || defined(__linux__)
    if (disabled) {
        // Name the hatch explicitly: a deliberate off must not read like a
        // release that ran and freed nothing.
        fprintf(stderr,
                "[WeightCtx] Loaded %zu tensors, %.1f MB into backend (page release disabled by ACE_NO_PAGE_UNMAP)\n",
                wctx->pending.size(), (float) total / (1024 * 1024));
    } else {
        // "X of Y": the gap is the boundary pages shared between adjacent tensors,
        // plus every copy made from heap staging -- both release at gf_close. The
        // reader sees the residual and its scale instead of guessing what 0.0 MB
        // means for a module of sub-page tensors.
        fprintf(stderr,
                "[WeightCtx] Loaded %zu tensors, %.1f MB into backend (unmapped %.1f of %.1f MB of staged file "
                "pages)\n",
                wctx->pending.size(), (float) total / (1024 * 1024), (float) unmapped / (1024 * 1024),
                (float) recorded / (1024 * 1024));
        if (failed > 0) {
            fprintf(stderr,
                    "[WeightCtx] WARNING: %.1f MB of staged pages could not be released this load; they "
                    "stay resident until gf_close\n",
                    (float) failed / (1024 * 1024));
        }
    }
#else
    (void) recorded;  // no mechanism on this platform: the counters are still
    (void) unmapped;  // accumulated for symmetry but there is nothing to report
    (void) failed;
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
