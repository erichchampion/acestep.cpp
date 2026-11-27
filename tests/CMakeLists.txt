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

# test-page-release: a released staged-page range stays reserved until gf_close
# unmaps the whole mapping (cadenza-audio #819). No model, no GPU.
add_executable(test-page-release test-page-release.cpp)
target_link_libraries(test-page-release PRIVATE acestep-core)

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

# test-synth-batch: the tools' batch expansion and its clamp log, by the
# helpers ace-synth and ace-server share (cadenza-audio #1269). No model.
add_executable(test-synth-batch test-synth-batch.cpp)
target_link_libraries(test-synth-batch PRIVATE acestep-core)
target_include_directories(test-synth-batch PRIVATE ${PROJECT_SOURCE_DIR}/tools)

# test-resample: a long source's tail resampled as finely as its head, not
# nearest-sample past 2^23 samples (cadenza-audio #1310). Header-only, no model.
add_executable(test-resample test-resample.cpp)
target_include_directories(test-resample PRIVATE ${PROJECT_SOURCE_DIR}/src)

# test-tiling-load: the engine's loads refuse a VAE tiling it can't run
# (cadenza-audio #1311). Its understand half needs no model; its synth half
# reads the DiT's metadata, so runs with --models only.
add_executable(test-tiling-load test-tiling-load.cpp)
target_link_libraries(test-tiling-load PRIVATE acestep-core)
link_ggml_backends(test-tiling-load)

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
add_test(NAME test-page-release COMMAND test-page-release)
add_test(NAME test-synth-batch COMMAND test-synth-batch)
add_test(NAME test-resample COMMAND test-resample)
# And the tools held to them: no expansion loop of their own (#1269).
add_test(NAME lint-batch-expansion
         COMMAND ${CMAKE_COMMAND} -DTOOLS=${PROJECT_SOURCE_DIR}/tools -P ${CMAKE_CURRENT_SOURCE_DIR}/lint-batch-expansion.cmake)
set_tests_properties(test-page-release PROPERTIES SKIP_RETURN_CODE 77)
# A timeout: the underlying-error walk's cycle case hangs, not fails, if the
# walk stops ending (cadenza-audio #857). Generous: it covers the whole binary,
# Metal's init and a dozen backends, on a runner others load (#357).
set_tests_properties(test-gpu-fail PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 600)
# Scenario 12, the splice rule a failed run's hand-out reads, needs no model
# (cadenza-audio #1056): run in every build, the weight-free ones too.
add_test(NAME test-splice-rule COMMAND test-model-store --scenarios splice)
# Scenario 13, the task table as the tools read it, needs none either
# (cadenza-audio #1094).
add_test(NAME test-task-table COMMAND test-model-store --scenarios tasks)
# Scenario 14, the timbre reference's crop, needs none either (cadenza-audio
# #1147).
add_test(NAME test-timbre-crop COMMAND test-model-store --scenarios timbre)
# The webui's batch rule, run by Node's own TypeScript support with no
# packages (cadenza-audio #1164): the form's synth sections are every
# request's, a field cleared after Compose included. Only a node that strips
# the types itself, unflagged (22.18 to 22.x, or 23.6 and later), runs it; a
# build without one does not.
find_program(ACESTEP_NODE node)
if(ACESTEP_NODE)
    execute_process(COMMAND ${ACESTEP_NODE} --version OUTPUT_VARIABLE ACESTEP_NODE_VERSION
                    OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    string(REGEX REPLACE "^v" "" ACESTEP_NODE_VERSION "${ACESTEP_NODE_VERSION}")
    if((ACESTEP_NODE_VERSION VERSION_GREATER_EQUAL 22.18 AND ACESTEP_NODE_VERSION VERSION_LESS 23)
       OR ACESTEP_NODE_VERSION VERSION_GREATER_EQUAL 23.6)
        add_test(NAME test-webui-fields
                 COMMAND ${ACESTEP_NODE} ${PROJECT_SOURCE_DIR}/tools/webui/test/fields.test.ts)
    else()
        message(STATUS "test-webui-fields: node ${ACESTEP_NODE_VERSION} does not strip TypeScript, not run")
    endif()
endif()
# ace-synth says it leaves a source the task ignores unread, before any
# model loads (cadenza-audio #1157): a text2music given a --src-audio that
# does not exist warns, a cover does not. Both then stop at the missing
# --models, before the source would be read, so no model is needed -- and
# the read itself is not reached: these hold the decision, not the decode.
add_test(NAME test-synth-ignored-source
         COMMAND ace-synth --request ${CMAKE_CURRENT_SOURCE_DIR}/request-text2music.json
                 --src-audio /nonexistent/source.wav --models /nonexistent)
set_tests_properties(test-synth-ignored-source PROPERTIES
    PASS_REGULAR_EXPRESSION "task 'text2music' ignores the given source: --src-audio not read")
add_test(NAME test-synth-cover-keeps-source
         COMMAND ace-synth --request ${CMAKE_CURRENT_SOURCE_DIR}/request-cover.json
                 --src-audio /nonexistent/source.wav --models /nonexistent)
set_tests_properties(test-synth-cover-keeps-source PROPERTIES
    PASS_REGULAR_EXPRESSION "cannot scan --models" FAIL_REGULAR_EXPRESSION "ignores the given source")
# And refuses a task that works from a source given none, before any model
# loads, as the run would (cadenza-audio #1160): a cover with no --src-audio
# stops there, not at the missing --models.
add_test(NAME test-synth-needs-source
         COMMAND ace-synth --request ${CMAKE_CURRENT_SOURCE_DIR}/request-cover.json --models /nonexistent)
set_tests_properties(test-synth-needs-source PROPERTIES
    PASS_REGULAR_EXPRESSION "task 'cover': the task requires source audio" FAIL_REGULAR_EXPRESSION "cannot scan --models")
# The tools' refusals (cadenza-audio #1266), each a failure that says why:
# tool-refuses.cmake requires a non-zero exit and the reason, which
# PASS_REGULAR_EXPRESSION alone would not. make-tool-inputs writes the files
# they read first, as a fixture. None loads a model: each is refused before.
add_executable(make-tool-inputs make-tool-inputs.cpp)
target_include_directories(make-tool-inputs PRIVATE ${PROJECT_SOURCE_DIR}/src)
set(ACESTEP_TOOL_INPUTS ${CMAKE_CURRENT_BINARY_DIR}/tool-inputs)
add_test(NAME tool-inputs COMMAND ${CMAKE_COMMAND} -E make_directory ${ACESTEP_TOOL_INPUTS})
add_test(NAME tool-inputs-write COMMAND make-tool-inputs ${ACESTEP_TOOL_INPUTS})
set_tests_properties(tool-inputs PROPERTIES FIXTURES_SETUP tool_inputs_dir)
set_tests_properties(tool-inputs-write PROPERTIES FIXTURES_REQUIRED tool_inputs_dir FIXTURES_SETUP tool_inputs)
# A tool refused, saying `expect`: add_tool_refusal(<name> <expect> <tool> <args...>).
function(add_tool_refusal name expect)
    add_test(NAME ${name}
             COMMAND ${CMAKE_COMMAND} "-DEXPECT=${expect}" -P ${CMAKE_CURRENT_SOURCE_DIR}/tool-refuses.cmake -- ${ARGN})
    set_tests_properties(${name} PROPERTIES FIXTURES_REQUIRED tool_inputs)
endfunction()
# Tiles the VAE can't run, refused where each tool parses them: a chunk of 0
# (a stride of 0, SIGFPE), below 0 (no tile, no error), past a run's most
# frames (chunk * ACE_LATENT_HOP overflows past ~1.1M), an overlap below 0.
add_tool_refusal(test-codec-chunk-zero "VAE chunk must be 1 to 15000 latent frames \\(--vae-chunk"
                 $<TARGET_FILE:neural-codec> --vae /nonexistent --decode -i /nonexistent --vae-chunk 0)
add_tool_refusal(test-codec-chunk-negative "VAE chunk must be 1 to 15000 latent frames \\(--vae-chunk"
                 $<TARGET_FILE:neural-codec> --vae /nonexistent --decode -i /nonexistent --vae-chunk -1)
add_tool_refusal(test-codec-chunk-past-most "VAE chunk must be 1 to 15000 latent frames \\(--vae-chunk"
                 $<TARGET_FILE:neural-codec> --vae /nonexistent --decode -i /nonexistent --vae-chunk 15001)
add_tool_refusal(test-codec-overlap-negative "VAE overlap must be 0 to under half the chunk \\(--vae-chunk"
                 $<TARGET_FILE:neural-codec> --vae /nonexistent --decode -i /nonexistent --vae-overlap -1)
# An overlap of half the chunk or more: a tile with no core, once halved to
# fit (4 and 3 dropped most frames before #1309).
add_tool_refusal(test-codec-overlap-past-half "VAE overlap must be 0 to under half the chunk \\(--vae-chunk"
                 $<TARGET_FILE:neural-codec> --vae /nonexistent --encode -i /nonexistent --vae-chunk 4
                 --vae-overlap 3)
add_tool_refusal(test-synth-chunk-zero "VAE chunk must be 1 to 15000 latent frames \\(--vae-chunk"
                 $<TARGET_FILE:ace-synth> --request ${CMAKE_CURRENT_SOURCE_DIR}/request-text2music.json
                 --models /nonexistent --vae-chunk 0)
# ace-server's numeric flags, each argument read whole and held to its range
# (cadenza-audio #1194): refused as it is parsed, before any model is scanned.
add_tool_refusal(test-server-port-trailing "--port takes a whole number from 0 to 65535"
                 $<TARGET_FILE:ace-server> --models /nonexistent --port 80x)
add_tool_refusal(test-server-port-past-most "--port takes a whole number from 0 to 65535"
                 $<TARGET_FILE:ace-server> --models /nonexistent --port 65536)
add_tool_refusal(test-server-max-seq-none "--max-seq takes a whole number from 1"
                 $<TARGET_FILE:ace-server> --models /nonexistent --max-seq 0)
add_tool_refusal(test-understand-chunk-zero "VAE chunk must be 1 to 15000 latent frames \\(--vae-chunk"
                 $<TARGET_FILE:ace-understand> --models /nonexistent --src-audio /nonexistent --vae-chunk 0)
# neural-codec's latent files, bounded as ace-server's payloads are: a header
# claiming 1.2M frames, a file of no frame (Q4 and f32), a part frame. The
# bound is a decode's (what its int math holds), not a run's: a header of
# 20000 frames passes it, and is refused only for the frames it lacks.
add_tool_refusal(test-codec-latent-past-most "more latent frames than a decode holds"
                 $<TARGET_FILE:neural-codec> --vae /nonexistent --decode -i ${ACESTEP_TOOL_INPUTS}/past-most.nac8)
add_tool_refusal(test-codec-latent-past-run "Q8 size mismatch: expected 1320008"
                 $<TARGET_FILE:neural-codec> --vae /nonexistent --decode -i ${ACESTEP_TOOL_INPUTS}/past-run.nac8)
add_tool_refusal(test-codec-latent-empty-q4 "no latent frame"
                 $<TARGET_FILE:neural-codec> --vae /nonexistent --decode -i ${ACESTEP_TOOL_INPUTS}/empty.nac4)
add_tool_refusal(test-codec-latent-empty-f32 "no latent frame"
                 $<TARGET_FILE:neural-codec> --vae /nonexistent --decode -i ${ACESTEP_TOOL_INPUTS}/empty.vae)
add_tool_refusal(test-codec-latent-ragged "not a whole number of 64-channel f32 frames"
                 $<TARGET_FILE:neural-codec> --vae /nonexistent --decode -i ${ACESTEP_TOOL_INPUTS}/ragged.vae)
# A header's rate outside a source's, refused before any resample: 1 Hz. A
# low rate a recording has still resamples: 4 kHz, read and resampled before
# the missing VAE stops the encode.
add_tool_refusal(test-codec-rate-1hz "a sample rate of 1 Hz"
                 $<TARGET_FILE:neural-codec> --vae /nonexistent --encode -i ${ACESTEP_TOOL_INPUTS}/rate-1hz.wav)
add_test(NAME test-codec-rate-4khz
         COMMAND neural-codec --vae /nonexistent --encode -i ${ACESTEP_TOOL_INPUTS}/rate-4khz.wav)
set_tests_properties(test-codec-rate-4khz PROPERTIES FIXTURES_REQUIRED tool_inputs
                     PASS_REGULAR_EXPRESSION "Done: 400 -> 4800 samples" FAIL_REGULAR_EXPRESSION "Refused")
set(ACESTEP_TEST_MODELS "" CACHE PATH "Directory of GGUFs for the tests that read them (empty: not run)")
# The loads' tiling refusal (#1311): with the models when there are some, so
# ace_synth_load's half runs too; its understand half needs none.
if(ACESTEP_TEST_MODELS)
    add_test(NAME test-tiling-load COMMAND test-tiling-load --models ${ACESTEP_TEST_MODELS})
else()
    add_test(NAME test-tiling-load COMMAND test-tiling-load)
endif()
if(ACESTEP_TEST_MODELS)
    add_test(NAME test-audio-layout COMMAND test-audio-layout --vae ${ACESTEP_TEST_MODELS}/vae-BF16.gguf)
    add_test(NAME test-vae-tiling COMMAND test-vae-tiling --vae ${ACESTEP_TEST_MODELS}/vae-BF16.gguf)
    add_test(NAME test-lm-prompt COMMAND test-lm-prompt --models ${ACESTEP_TEST_MODELS})
    # The store's policies, and apart from them a failed run's hand-out, which
    # runs the synth (cadenza-audio #1056): each fails, and is timed, alone.
    add_test(NAME test-model-store COMMAND test-model-store --models ${ACESTEP_TEST_MODELS} --scenarios store)
    add_test(NAME test-failed-encodes COMMAND test-model-store --models ${ACESTEP_TEST_MODELS} --scenarios encodes)
    # neural-codec and quantize with the VAE (cadenza-audio #1266): a write
    # that fails -- to a folder that is not there -- fails the tool, encode,
    # decode and quantize alike, rather than exiting 0. (An encode of no frame is refused too,
    # but no input reaches one: the VAE rounds a source up, 100 samples to a
    # frame.)
    add_tool_refusal(test-codec-encode-write-fails "failed to write"
                     $<TARGET_FILE:neural-codec> --vae ${ACESTEP_TEST_MODELS}/vae-BF16.gguf --encode
                     -i ${ACESTEP_TOOL_INPUTS}/one-second.wav -o /nonexistent/one-second.vae)
    # ace-synth reads its source to what a run takes whole, before any model
    # loads, and before the resample: 601 s at 1 kHz is refused unresampled.
    add_tool_refusal(test-synth-source-past-run "past the 28801919 this read takes"
                     $<TARGET_FILE:ace-synth> --request ${CMAKE_CURRENT_SOURCE_DIR}/request-cover.json
                     --src-audio ${ACESTEP_TOOL_INPUTS}/long-1khz.wav --models ${ACESTEP_TEST_MODELS})
    add_tool_refusal(test-quantize-write-fails "Failed to write metadata"
                     $<TARGET_FILE:quantize> ${ACESTEP_TEST_MODELS}/vae-BF16.gguf /nonexistent/vae-Q8_0.gguf Q8_0)
    add_tool_refusal(test-codec-decode-write-fails "failed to write"
                     $<TARGET_FILE:neural-codec> --vae ${ACESTEP_TEST_MODELS}/vae-BF16.gguf --decode
                     -i ${ACESTEP_TOOL_INPUTS}/one-frame.vae -o /nonexistent/one-frame.wav)
    # ace-server's /synth refusals against a running server on a free port
    # of 127.0.0.1 (cadenza-audio #1194): each a 400 saying why, the server
    # answering after -- a cover sent a reference and no source among them,
    # refused before the reference is read. The script stops the server
    # however it ends; the timeout bounds a server that never answers.
    add_test(NAME test-server-synth
             COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/test-server-synth.sh $<TARGET_FILE:ace-server>
                     ${ACESTEP_TEST_MODELS} ${ACESTEP_TOOL_INPUTS}/one-second.wav)
    set_tests_properties(test-server-synth PROPERTIES FIXTURES_REQUIRED tool_inputs TIMEOUT 180)
    # A bind that fails is the server's failure, exit 1 (cadenza-audio #1194):
    # TEST-NET-1 (192.0.2.1, RFC 5737), no address of this host's, and no
    # name to look up.
    add_tool_refusal(test-server-bind-fails "cannot bind"
                     $<TARGET_FILE:ace-server> --models ${ACESTEP_TEST_MODELS} --host 192.0.2.1 --port 0)
endif()
