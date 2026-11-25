# lint-batch-expansion.cmake: the tools expand a batch by the engine's one
# rule, once per job (cadenza-audio #1243, #1269). test-synth-batch pins the
# helpers; this holds ace-synth and ace-server to them, so a per-tool
# expansion loop put back fails here: each tool's code -- its comments left
# out -- calls ace_synth_batch_takes once and synth_batch_expand once, and
# reads neither synth_batch_size nor ace_synth_request_takes nor
# ace_synth_requests_run (the batch asked again) itself.
#
#   cmake -DTOOLS=<engine>/tools -P lint-batch-expansion.cmake

if(NOT DEFINED TOOLS)
    message(FATAL_ERROR "usage: cmake -DTOOLS=<engine>/tools -P lint-batch-expansion.cmake")
endif()

set(failed FALSE)
foreach(tool ace-synth.cpp ace-server.cpp)
    file(READ "${TOOLS}/${tool}" text)
    # Comments out: a line's // to its end, and /* ... */ blocks.
    string(REGEX REPLACE "/\\*([^*]|\\*+[^*/])*\\*+/" "" text "${text}")
    string(REGEX REPLACE "//[^\n]*" "" text "${text}")
    foreach(call ace_synth_batch_takes synth_batch_expand)
        string(REGEX MATCHALL "${call}\\(" hits "${text}")
        list(LENGTH hits n)
        if(NOT n EQUAL 1)
            message(SEND_ERROR "${tool}: ${call}( is called ${n} time(s), want once per job")
            set(failed TRUE)
        endif()
    endforeach()
    foreach(banned synth_batch_size ace_synth_request_takes ace_synth_requests_run)
        if(text MATCHES "${banned}")
            message(SEND_ERROR "${tool}: reads ${banned} -- expand by ace_synth_batch_takes and synth_batch_expand")
            set(failed TRUE)
        endif()
    endforeach()
endforeach()
if(failed)
    message(FATAL_ERROR "a tool expands a batch by its own rule")
endif()
message(STATUS "the tools expand a batch by the engine's rule, once")
