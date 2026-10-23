// test-page-release: releasing a copied tensor's staged pages keeps their
// addresses reserved (cadenza-audio #819).
//
// wctx_alloc drops each copied tensor's whole interior pages from the GGUF
// mapping, and gf_close later unmaps the whole original range. Released by
// munmap, those pages were holes the kernel gave to the next allocation from
// any thread -- and gf_close's munmap then unmapped it under its owner. A
// released range must still be taken -- nothing can be allocated at its
// addresses -- hold none of the file's pages, and leave the pages around it
// readable; once the whole range is unmapped, its addresses are free.
//
//   ./test-page-release

#include "weight-ctx.h"

#include <fcntl.h>
#include <sys/mman.h>
#ifdef __APPLE__
#    include <mach/mach.h>
#endif
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "test-util.h"

#ifdef ACE_HAS_PAGE_RELEASE
#    ifdef __APPLE__
typedef char mincore_vec_t;
#    else
typedef unsigned char mincore_vec_t;
#    endif

// Whether an allocation could be placed at exactly [addr, addr+len): the hole a
// released range must not become. (mincore cannot tell: on macOS it does not
// fail over an unmapped range.)
static bool range_is_free(void * addr, size_t len) {
#    ifdef __APPLE__
    vm_address_t a = (vm_address_t) addr;
    if (vm_allocate(mach_task_self(), &a, len, VM_FLAGS_FIXED) != KERN_SUCCESS) {
        return false;
    }
    vm_deallocate(mach_task_self(), a, len);
    return true;
#    else
    void * p = mmap(addr, len, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0);
    if (p == MAP_FAILED) {
        return false;
    }
    munmap(p, len);
    return p == addr;
#    endif
}

int main(void) {
    const size_t page  = (size_t) sysconf(_SC_PAGESIZE);
    const size_t pages = 16;
    const size_t len   = pages * page;

    char path[] = "/tmp/test-page-release-XXXXXX";
    int  fd     = mkstemp(path);
    if (fd < 0) {
        fprintf(stderr, "test-page-release: mkstemp failed\n");
        return 1;
    }
    unlink(path);
    std::vector<unsigned char> bytes(len);
    for (size_t i = 0; i < len; i++) {
        bytes[i] = (unsigned char) (i / page + 1);
    }
    CHECK(write(fd, bytes.data(), len) == (ssize_t) len);

    // As gf_load maps a GGUF.
    auto * base = (unsigned char *) mmap(NULL, len, PROT_READ, MAP_PRIVATE, fd, 0);
    CHECK(base != MAP_FAILED);
    if (base == MAP_FAILED) {
        return test_summary("test-page-release");
    }
    unsigned sum = 0;
    for (size_t p = 0; p < pages; p++) {
        sum += base[p * page];  // fault every page in
    }
    CHECK(sum == pages * (pages + 1) / 2);

    // A tensor from 100 bytes into page 0 to 100 bytes into page 10: its whole
    // interior pages are 1 through 9.
    size_t failed   = 0;
    size_t released = wctx_unmap_file_pages(base + 100, 10 * page, base, len, &failed);
    CHECK(released == 9 * page);
    CHECK(failed == 0);

    // Still taken, so nothing else can be placed inside the mapping.
    CHECK(!range_is_free(base + page, 9 * page));
    std::vector<mincore_vec_t> vec(pages);
    CHECK(mincore(base + page, 9 * page, vec.data()) == 0);
    for (size_t p = 0; p < 9; p++) {
        CHECK((vec[p] & 1) == 0);  // none of the file's pages resident there
    }

    // The pages it shares with neighbours keep the file's bytes.
    CHECK(base[0] == 1);
    CHECK(base[10 * page] == 11);
    CHECK(base[len - 1] == pages);

    // As gf_close: the whole original range unmaps, and is gone.
    CHECK(munmap(base, len) == 0);
    CHECK(range_is_free(base, len));

    close(fd);
    return test_summary("test-page-release");
}
#else
int main(void) {
    printf("test-page-release: SKIP: no staged-page release on this platform\n");
    return TEST_SKIPPED;
}
#endif
