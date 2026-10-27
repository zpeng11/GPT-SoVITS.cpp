cmake_minimum_required(VERSION 3.14)

foreach(name IN ITEMS CLI_BINARY WAV_CHECKER REFERENCE_CASE TEST_SOURCE_DIR TEST_OUTPUT_DIR)
    if(NOT DEFINED ${name} OR "${${name}}" STREQUAL "")
        message(FATAL_ERROR "${name} must be set for CLI synthesis validation")
    endif()
endforeach()

if(NOT IS_DIRECTORY "${REFERENCE_CASE}")
    message(FATAL_ERROR "reference case directory is missing: ${REFERENCE_CASE}")
endif()

set(hubert "${TEST_SOURCE_DIR}/hubert/models/chinese-hubert-base-f32.gguf")
set(roberta "${TEST_SOURCE_DIR}/roberta/chinese-roberta-wwm-ext-large-f16.gguf")
set(t2s "${TEST_SOURCE_DIR}/t2s/models/s1v3-s2Gv2Pro-f16.gguf")
set(sovits "${TEST_SOURCE_DIR}/sovits/models/v2pro-sovits-f16.gguf")
set(sv "${TEST_SOURCE_DIR}/sovits/models/sv-eres2netv2-f32.gguf")
set(ref_audio "${REFERENCE_CASE}/ref.wav")
set(ref_json "${REFERENCE_CASE}/ref_frontend.json")
set(target_json "${REFERENCE_CASE}/target_frontend.json")

foreach(input IN ITEMS "${CLI_BINARY}" "${WAV_CHECKER}"
        "${hubert}" "${roberta}" "${t2s}" "${sovits}" "${sv}"
        "${ref_audio}" "${ref_json}" "${target_json}")
    if(NOT EXISTS "${input}" OR IS_DIRECTORY "${input}")
        message(FATAL_ERROR "required CLI synthesis input is missing or not a file: ${input}")
    endif()
endforeach()

string(RANDOM LENGTH 16 ALPHABET 0123456789abcdef run_id)
set(work_dir "${TEST_OUTPUT_DIR}/run-${run_id}")
if(EXISTS "${work_dir}")
    message(FATAL_ERROR "test directory already exists: ${work_dir}")
endif()
file(MAKE_DIRECTORY "${work_dir}")
set(wav "${work_dir}/output audio.wav")
set(max_new_tokens 100)
set(cli_args
    --hubert "${hubert}"
    --roberta "${roberta}"
    --t2s "${t2s}"
    --sovits "${sovits}"
    --sv "${sv}"
    --ref-audio "${ref_audio}"
    --ref-frontend-json "${ref_json}"
    --target-frontend-json "${target_json}"
    --output "${wav}"
    --top-k 1 --max-new-tokens ${max_new_tokens}
    --seed 42 --noise-seed 7 --noise-scale 0.5)

message(STATUS "CLI synthesis logs and output: ${work_dir}")
execute_process(COMMAND "${CLI_BINARY}" ${cli_args}
    RESULT_VARIABLE result TIMEOUT 600
    OUTPUT_FILE "${work_dir}/synthesis.stdout.log"
    ERROR_FILE "${work_dir}/synthesis.stderr.log")
if(NOT "${result}" STREQUAL "0")
    message(FATAL_ERROR
        "CLI synthesis exited with '${result}', expected 0; see ${work_dir}/synthesis.stderr.log")
endif()

execute_process(COMMAND "${WAV_CHECKER}" "${wav}"
        "${work_dir}/synthesis.stderr.log" ${max_new_tokens}
    RESULT_VARIABLE result TIMEOUT 30
    OUTPUT_FILE "${work_dir}/validation.stdout.log"
    ERROR_FILE "${work_dir}/validation.stderr.log")
if(NOT "${result}" STREQUAL "0")
    file(READ "${work_dir}/validation.stderr.log" detail)
    message(FATAL_ERROR
        "CLI output validation failed ('${result}'): ${detail}\nArtifacts: ${work_dir}")
endif()

set(snapshot "${work_dir}/original.wav")
configure_file("${wav}" "${snapshot}" COPYONLY)
execute_process(COMMAND "${CLI_BINARY}" ${cli_args}
    RESULT_VARIABLE result TIMEOUT 30
    OUTPUT_FILE "${work_dir}/overwrite.stdout.log"
    ERROR_FILE "${work_dir}/overwrite.stderr.log")
if(NOT "${result}" STREQUAL "1")
    message(FATAL_ERROR
        "CLI overwrite protection exited with '${result}', expected 1; see ${work_dir}/overwrite.stderr.log")
endif()
file(READ "${work_dir}/overwrite.stderr.log" refusal)
string(FIND "${refusal}" "exists (use --overwrite to replace it)" refusal_pos)
if(refusal_pos EQUAL -1)
    message(FATAL_ERROR
        "CLI did not report the existing-output error; see ${work_dir}/overwrite.stderr.log")
endif()
execute_process(COMMAND "${CMAKE_COMMAND}" -E compare_files "${wav}" "${snapshot}"
    RESULT_VARIABLE result TIMEOUT 30)
if(NOT "${result}" STREQUAL "0")
    message(FATAL_ERROR "CLI changed the existing output without --overwrite; artifacts: ${work_dir}")
endif()

file(READ "${work_dir}/validation.stdout.log" validation)
message(STATUS "${validation}")
file(REMOVE_RECURSE "${work_dir}")
