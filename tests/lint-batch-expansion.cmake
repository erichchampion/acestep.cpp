# lint-batch-expansion.cmake: the tools expand a batch by the engine's one
# rule, once per job (cadenza-audio #1243, #1269). test-synth-batch pins the
# helpers; this holds ace-synth and ace-server to them, so a per-tool
# expansion loop put back fails here: each tool's code -- its string and
# character literals and its comments left out -- calls
# ace_synth_batch_takes once and synth_batch_expand once, and reads neither
# synth_batch_size nor ace_synth_request_takes nor ace_synth_requests_run
# (the batch asked again) itself. Literals go first, so a "http://" or an
# "audio/*" can't start a comment that hides a call; a self-check on a line
# of both runs before the tools are read.
#
#   cmake -DTOOLS=<engine>/tools -P lint-batch-expansion.cmake

if(NOT DEFINED TOOLS)
    message(FATAL_ERROR "usage: cmake -DTOOLS=<engine>/tools -P lint-batch-expansion.cmake")
endif()

# `var`'s C++ with its literals emptied and its comments out.
function(code_only var)
    set(text "${${var}}")
    string(REGEX REPLACE "'([^'\\\\\n]|\\\\.)*'" "''" text "${text}")
    string(REGEX REPLACE "\"([^\"\\\\\n]|\\\\.)*\"" "\"\"" text "${text}")
    string(REGEX REPLACE "/\\*([^*]|\\*+[^*/])*\\*+/" "" text "${text}")
    string(REGEX REPLACE "//[^\n]*" "" text "${text}")
    set(${var} "${text}" PARENT_SCOPE)
endfunction()

# How many times `call(` is in `text`.
function(count_calls text call out)
    string(REGEX MATCHALL "${call}\\(" hits "${text}")
    list(LENGTH hits n)
    set(${out} ${n} PARENT_SCOPE)
endfunction()

# The self-check: a URL and a MIME glob in literals hide nothing after them,
# an escaped quote ends no literal early, a commented call is not counted,
# and a call inside a literal is not one.
set(sample "a(\"http://x\", '\"'); synth_batch_expand(); b(\"audio/*\"); synth_batch_expand();\n"
           "c(\"say \\\"synth_batch_expand()\\\"\"); /* synth_batch_expand(); */ // synth_batch_expand();\n")
code_only(sample)
count_calls("${sample}" synth_batch_expand n)
if(NOT n EQUAL 2)
    message(FATAL_ERROR "lint self-check: ${n} call(s) seen in the sample, want 2:\n${sample}")
endif()

set(failed FALSE)
foreach(tool ace-synth.cpp ace-server.cpp)
    file(READ "${TOOLS}/${tool}" text)
    code_only(text)
    foreach(call ace_synth_batch_takes synth_batch_expand)
        count_calls("${text}" ${call} n)
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
