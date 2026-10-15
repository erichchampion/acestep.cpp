# test-ace-fatal: the fatal-throws path (ace_fatal / ace_fatal_error, #17).
# Header-only, no engine link: it defines ACESTEP_FATAL_THROWS for itself.
add_executable(test-ace-fatal test-ace-fatal.cpp)
target_include_directories(test-ace-fatal PRIVATE ${PROJECT_SOURCE_DIR}/src)

# test-progress: the AceProgress report/cancel contract (#18). Header-only.
add_executable(test-progress test-progress.cpp)
target_include_directories(test-progress PRIVATE ${PROJECT_SOURCE_DIR}/src)

# test-backend-config: explicit thread count + device selection (#19). Includes
# backend.h, so it links the ggml backends.
add_executable(test-backend-config test-backend-config.cpp)
target_link_libraries(test-backend-config PRIVATE acestep-core)
link_ggml_backends(test-backend-config)

# test-gpu-fail: the ggml Metal error paths the engine leans on
# (cadenza-audio #102, #402, #405): a failed Metal allocation returns NULL,
# has_error is exported through get_proc_address, and exit with a live Metal
# buffer is clean -- the device and buffer-type statics are leaked on purpose.
# Needs a Metal device; skips without one.
add_executable(test-gpu-fail test-gpu-fail.cpp)
target_link_libraries(test-gpu-fail PRIVATE acestep-core)
link_ggml_backends(test-gpu-fail)

# test-audio-layout: the channel-layout contract at the pipeline boundary
# (#28). Needs only the VAE GGUF: run with --vae <gguf>.
add_executable(test-audio-layout test-audio-layout.cpp)
target_link_libraries(test-audio-layout PRIVATE acestep-core)
link_ggml_backends(test-audio-layout)

# test-vae-tiling: the VAE's tiles join bit-exactly at the default overlap,
# encoder and decoder alike (cadenza-audio #91, #684). Needs only the VAE GGUF.
add_executable(test-vae-tiling test-vae-tiling.cpp)
target_link_libraries(test-vae-tiling PRIVATE acestep-core)
link_ggml_backends(test-vae-tiling)

# test-understand-cot: what an understand pass keeps of a metadata block its
# decode stopped inside, and where it closes one that runs on (cadenza-audio
# #648). No model is loaded.
add_executable(test-understand-cot test-understand-cot.cpp)
target_link_libraries(test-understand-cot PRIVATE acestep-core)

# test-philox: Philox RNG reference dumper.
add_executable(test-philox test-philox.cpp)
target_include_directories(test-philox PRIVATE ${PROJECT_SOURCE_DIR}/src)
if(NOT MSVC)
    target_link_libraries(test-philox PRIVATE m)
endif()

# test-model-store: exercise ModelStore load/evict policies against real GGUFs.
add_executable(test-model-store test-model-store.cpp)
target_link_libraries(test-model-store PRIVATE acestep-core)
link_ggml_backends(test-model-store)

# test-lm-prompt: byte for byte check of Phase 2 LM prompt builders
# against the Qwen3 chat template reference.
add_executable(test-lm-prompt test-lm-prompt.cpp)
target_link_libraries(test-lm-prompt PRIVATE acestep-core)
link_ggml_backends(test-lm-prompt)

# ctest (cadenza-audio #134): the checks above are run, not only built. The
# self-contained ones always; the ones that read GGUFs only when
# ACESTEP_TEST_MODELS names a directory holding them (-DACESTEP_TEST_MODELS=
# <dir>). A test that cannot run here exits 77 (tests/test-util.h), which
# ctest reports as skipped -- a runner that must run it treats that as a
# failure (cadenza-audio scripts/test-engine.sh does).
add_test(NAME test-ace-fatal COMMAND test-ace-fatal)
add_test(NAME test-progress COMMAND test-progress)
add_test(NAME test-backend-config COMMAND test-backend-config)
add_test(NAME test-understand-cot COMMAND test-understand-cot)
add_test(NAME test-gpu-fail COMMAND test-gpu-fail)
set_tests_properties(test-gpu-fail PROPERTIES SKIP_RETURN_CODE 77)
set(ACESTEP_TEST_MODELS "" CACHE PATH "Directory of GGUFs for the tests that read them (empty: not run)")
if(ACESTEP_TEST_MODELS)
    add_test(NAME test-audio-layout COMMAND test-audio-layout --vae ${ACESTEP_TEST_MODELS}/vae-BF16.gguf)
    add_test(NAME test-vae-tiling COMMAND test-vae-tiling --vae ${ACESTEP_TEST_MODELS}/vae-BF16.gguf)
    add_test(NAME test-lm-prompt COMMAND test-lm-prompt --models ${ACESTEP_TEST_MODELS})
    add_test(NAME test-model-store COMMAND test-model-store --models ${ACESTEP_TEST_MODELS})
endif()
