# test-ace-fatal: the fatal-throws path (ace_fatal / ace_fatal_error, #17).
# Header-only, no engine link: it defines ACESTEP_FATAL_THROWS for itself.
add_executable(test-ace-fatal test-ace-fatal.cpp)
target_include_directories(test-ace-fatal PRIVATE ${CMAKE_SOURCE_DIR}/src)

# test-progress: the AceProgress report/cancel contract (#18). Header-only.
add_executable(test-progress test-progress.cpp)
target_include_directories(test-progress PRIVATE ${CMAKE_SOURCE_DIR}/src)

# test-backend-config: explicit thread count + device selection (#19). Includes
# backend.h, so it links the ggml backends.
add_executable(test-backend-config test-backend-config.cpp)
target_link_libraries(test-backend-config PRIVATE acestep-core)
link_ggml_backends(test-backend-config)

# test-audio-layout: the channel-layout contract at the pipeline boundary
# (#28). Needs only the VAE GGUF: run with --vae <gguf>.
add_executable(test-audio-layout test-audio-layout.cpp)
target_link_libraries(test-audio-layout PRIVATE acestep-core)
link_ggml_backends(test-audio-layout)

# test-philox: Philox RNG reference dumper.
add_executable(test-philox test-philox.cpp)
target_include_directories(test-philox PRIVATE ${CMAKE_SOURCE_DIR}/src)
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
