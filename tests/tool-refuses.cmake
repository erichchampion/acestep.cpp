# tool-refuses.cmake: run a tool and require that it fails, saying why
# (cadenza-audio #1266) -- a non-zero exit AND a line matching EXPECT. ctest's
# PASS_REGULAR_EXPRESSION alone ignores the exit status, and a tool that says
# FATAL and exits 0 is the bug these tests hold. A crash -- a signal, not an
# exit -- fails too, whatever it printed first.
#
#   cmake -DEXPECT=<regex> -P tool-refuses.cmake -- <tool> <args...>

set(cmd "")
set(seen_dashes FALSE)
foreach(i RANGE ${CMAKE_ARGC})
    if(seen_dashes)
        list(APPEND cmd "${CMAKE_ARGV${i}}")
    elseif("${CMAKE_ARGV${i}}" STREQUAL "--")
        set(seen_dashes TRUE)
    endif()
endforeach()
list(FILTER cmd EXCLUDE REGEX "^$")
if(NOT cmd OR NOT DEFINED EXPECT)
    message(FATAL_ERROR "usage: cmake -DEXPECT=<regex> -P tool-refuses.cmake -- <tool> <args...>")
endif()

execute_process(COMMAND ${cmd} RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err)
set(all "${out}${err}")
if(NOT rc MATCHES "^[0-9]+$")
    message(FATAL_ERROR "${cmd} did not exit (${rc}), want a failure:\n${all}")
endif()
if(rc EQUAL 0)
    message(FATAL_ERROR "${cmd} exited 0, want a failure:\n${all}")
endif()
if(NOT all MATCHES "${EXPECT}")
    message(FATAL_ERROR "${cmd} failed (${rc}) without saying '${EXPECT}':\n${all}")
endif()
message(STATUS "refused (${rc}): ${EXPECT}")
