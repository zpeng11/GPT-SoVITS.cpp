# tests/cli/check_cli.cmake
#
# Script-mode helper for the no-model CLI smoke tests. Invoked as:
#
#   cmake -P check_cli.cmake <expect-exit> <must-not-exist> <cli-binary> [cli args...]
#
# Runs the CLI, asserts the process exit code equals <expect-exit>, and asserts
# that the file at <must-not-exist> ("-" to skip the check) was not created.
# Reports failures via message(FATAL_ERROR) so ctest shows the reason.

if(CMAKE_ARGC LESS 6)
    message(FATAL_ERROR
        "usage: cmake -P check_cli.cmake <expect-exit> <must-not-exist> <cli-binary> [args...]")
endif()

set(expect_exit "${CMAKE_ARGV3}")
set(must_not_exist "${CMAKE_ARGV4}")
set(cmd "")
set(i 5)
while(i LESS CMAKE_ARGC)
    list(APPEND cmd "${CMAKE_ARGV${i}}")
    math(EXPR i "${i} + 1")
endwhile()

execute_process(COMMAND ${cmd}
    RESULT_VARIABLE exit_code
    OUTPUT_QUIET ERROR_QUIET)

if(NOT exit_code EQUAL expect_exit)
    message(FATAL_ERROR "CLI exited with ${exit_code}, expected ${expect_exit}: ${cmd}")
endif()

if(NOT must_not_exist STREQUAL "-" AND EXISTS "${must_not_exist}")
    message(FATAL_ERROR "CLI failure must not create '${must_not_exist}'")
endif()
