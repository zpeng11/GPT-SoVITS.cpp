include(FetchContent)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.15.2
)
# Prevent GoogleTest from overriding our compiler/linker options
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

include(GoogleTest)

# Common test utilities (cnpy for .npy loading)
set(COMMON_TEST_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/common/cnpy.cpp
)

# Common test include path (for cnpy, npy_loader.h, test_backend.h)
set(COMMON_TEST_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/common)

set(HUBERT_TEST_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/hubert/test_hubert_load.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/hubert/test_hubert_parity.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/hubert/test_hubert_flac.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/hubert/test_hubert_extract.cpp
)

foreach(test_src ${HUBERT_TEST_SOURCES})
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} ${test_src})
    target_include_directories(${test_name} PRIVATE
        ${COMMON_TEST_INCLUDE_DIR}
    )
    target_link_libraries(${test_name} PRIVATE
        gpt_sovits_cpp_static
        GTest::gtest_main
    )
    target_compile_definitions(${test_name} PRIVATE
        HUBERT_TEST_DIR="${CMAKE_CURRENT_SOURCE_DIR}/hubert/"
    )
    gtest_discover_tests(${test_name})
endforeach()

# RoBERTa tests
set(ROBERTA_TEST_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/roberta/test_roberta_load.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/roberta/test_roberta_parity.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/roberta/test_roberta_extract.cpp
)

foreach(test_src ${ROBERTA_TEST_SOURCES})
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} ${test_src})
    target_include_directories(${test_name} PRIVATE
        ${COMMON_TEST_INCLUDE_DIR}
    )
    target_link_libraries(${test_name} PRIVATE
        gpt_sovits_cpp_static
        GTest::gtest_main
    )
    target_compile_definitions(${test_name} PRIVATE
        ROBERTA_MODEL_DIR="${CMAKE_CURRENT_SOURCE_DIR}/roberta/"
    )
    gtest_discover_tests(${test_name})
endforeach()

# SoVITS ref_enc tests
set(SOVITS_TEST_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/sovits/test_sovits_ref_enc.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sovits/test_sovits_quantizer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sovits/test_sovits_text_encoder.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sovits/test_sovits_flow.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sovits/test_sovits_generator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sovits/test_sovits_helpers.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sovits/test_sovits_session.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sovits/test_sovits_v2pro_parity.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sovits/test_sovits_loader.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sovits/test_sv_parity.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sovits/test_sv_loader.cpp
)

foreach(test_src ${SOVITS_TEST_SOURCES})
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} ${test_src} ${COMMON_TEST_SOURCES})
    target_include_directories(${test_name} PRIVATE
        ${COMMON_TEST_INCLUDE_DIR}
    )
    target_link_libraries(${test_name} PRIVATE
        gpt_sovits_cpp_static
        GTest::gtest_main
    )
    target_compile_definitions(${test_name} PRIVATE
        SOVITS_TEST_DIR="${CMAKE_CURRENT_SOURCE_DIR}/sovits/"
    )
    gtest_discover_tests(${test_name})
endforeach()

# Audio DSP/IO tests (use cnpy for .npy loading)
set(AUDIO_TEST_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/audio/test_audio_stft.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/audio/test_audio_io.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/audio/test_audio_resample.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/audio/test_audio_reference.cpp
)

foreach(test_src ${AUDIO_TEST_SOURCES})
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} ${test_src} ${COMMON_TEST_SOURCES})
    target_include_directories(${test_name} PRIVATE
        ${COMMON_TEST_INCLUDE_DIR}
    )
    target_link_libraries(${test_name} PRIVATE
        gpt_sovits_cpp_static
        GTest::gtest_main
    )
    target_compile_definitions(${test_name} PRIVATE
        AUDIO_TEST_DIR="${CMAKE_CURRENT_SOURCE_DIR}/audio/"
    )
    gtest_discover_tests(${test_name})
endforeach()

# T2S tests (use cnpy for .npy loading)
set(T2S_TEST_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/t2s/test_t2s_loader.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/t2s/test_t2s_session.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/t2s/test_t2s_infer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/t2s/test_t2s_generate.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/t2s/test_t2s_generate_integration.cpp
)

foreach(test_src ${T2S_TEST_SOURCES})
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} ${test_src} ${COMMON_TEST_SOURCES})
    target_include_directories(${test_name} PRIVATE
        ${COMMON_TEST_INCLUDE_DIR}
    )
    target_link_libraries(${test_name} PRIVATE
        gpt_sovits_cpp_static
        GTest::gtest_main
    )
    target_compile_definitions(${test_name} PRIVATE
        T2S_TEST_DIR="${CMAKE_CURRENT_SOURCE_DIR}/t2s/"
    )
    gtest_discover_tests(${test_name})
endforeach()

set(GPTSOVITS_T2S_DECODE_REFERENCE "" CACHE PATH "Local teacher-forced T2S decode reference export")
set(GPTSOVITS_T2S_DECODE_MODEL "" CACHE FILEPATH "Matching F32 T2S GGUF for decode reference tests")

add_executable(test_t2s_decode_reference_contract t2s/test_t2s_decode_reference_contract.cpp)
target_include_directories(test_t2s_decode_reference_contract PRIVATE
    ${CMAKE_SOURCE_DIR}/third_party/nlohmann/include
)
target_link_libraries(test_t2s_decode_reference_contract PRIVATE gpt_sovits_cpp_static GTest::gtest_main)
gtest_discover_tests(test_t2s_decode_reference_contract)

add_executable(test_t2s_decode_reference t2s/test_t2s_decode_reference.cpp)
target_include_directories(test_t2s_decode_reference PRIVATE
    ${CMAKE_SOURCE_DIR}/third_party/nlohmann/include
)
target_link_libraries(test_t2s_decode_reference PRIVATE gpt_sovits_cpp_static GTest::gtest_main)
target_compile_definitions(test_t2s_decode_reference PRIVATE
    T2S_DECODE_REFERENCE_DIR="${GPTSOVITS_T2S_DECODE_REFERENCE}"
    T2S_DECODE_MODEL_PATH="${GPTSOVITS_T2S_DECODE_MODEL}"
)
gtest_discover_tests(test_t2s_decode_reference
    PROPERTIES LABELS "cpu\\;integration" TIMEOUT 300
)

set(GPTSOVITS_T2S_GENERATION_REFERENCE "" CACHE PATH "Matched-policy T2S generation references with rp1/rp135 cases")

add_executable(test_t2s_generation_reference_contract t2s/test_t2s_generation_reference_contract.cpp)
target_include_directories(test_t2s_generation_reference_contract PRIVATE
    ${CMAKE_SOURCE_DIR}/third_party/nlohmann/include
)
target_link_libraries(test_t2s_generation_reference_contract PRIVATE gpt_sovits_cpp_static GTest::gtest_main)
gtest_discover_tests(test_t2s_generation_reference_contract)

add_executable(test_t2s_generation_reference t2s/test_t2s_generation_reference.cpp)
target_include_directories(test_t2s_generation_reference PRIVATE
    ${CMAKE_SOURCE_DIR}/third_party/nlohmann/include
)
target_link_libraries(test_t2s_generation_reference PRIVATE gpt_sovits_cpp_static GTest::gtest_main)
target_compile_definitions(test_t2s_generation_reference PRIVATE
    T2S_GENERATION_REFERENCE_DIR="${GPTSOVITS_T2S_GENERATION_REFERENCE}"
    T2S_DECODE_MODEL_PATH="${GPTSOVITS_T2S_DECODE_MODEL}"
)
gtest_discover_tests(test_t2s_generation_reference
    PROPERTIES LABELS "cpu\\;integration" TIMEOUT 300
)

add_executable(test_text_frontend_json frontend/test_text_frontend_json.cpp)
target_include_directories(test_text_frontend_json PRIVATE
    ${CMAKE_SOURCE_DIR}/third_party/nlohmann/include
)
target_link_libraries(test_text_frontend_json PRIVATE
    gpt_sovits_cpp_static
    GTest::gtest_main
)
target_compile_definitions(test_text_frontend_json PRIVATE
    FRONTEND_TEST_DIR="${CMAKE_CURRENT_SOURCE_DIR}/frontend/"
)
gtest_discover_tests(test_text_frontend_json)

add_executable(test_text_frontend_bert frontend/test_text_frontend_bert.cpp)
target_include_directories(test_text_frontend_bert PRIVATE
    ${CMAKE_SOURCE_DIR}/src/frontend
)
target_link_libraries(test_text_frontend_bert PRIVATE
    gpt_sovits_cpp_static
    GTest::gtest_main
)
target_compile_definitions(test_text_frontend_bert PRIVATE
    FRONTEND_TEST_DIR="${CMAKE_CURRENT_SOURCE_DIR}/frontend/"
)
gtest_discover_tests(test_text_frontend_bert)

add_executable(test_text_frontend_bert_integration frontend/test_text_frontend_bert_integration.cpp)
target_link_libraries(test_text_frontend_bert_integration PRIVATE
    gpt_sovits_cpp_static
    GTest::gtest_main
)
target_compile_definitions(test_text_frontend_bert_integration PRIVATE
    FRONTEND_TEST_DIR="${CMAKE_CURRENT_SOURCE_DIR}/frontend/"
    ROBERTA_MODEL_DIR="${CMAKE_CURRENT_SOURCE_DIR}/roberta/"
)
gtest_discover_tests(test_text_frontend_bert_integration
    PROPERTIES LABELS "cpu\\;integration" TIMEOUT 300
)

set(GPTSOVITS_REFERENCE_CASE "" CACHE PATH "Local matched inference dump for CPU reference integration tests")

add_executable(test_reference pipeline/test_reference.cpp)
target_include_directories(test_reference PRIVATE ${CMAKE_SOURCE_DIR}/src/pipeline)
target_link_libraries(test_reference PRIVATE gpt_sovits_cpp_static GTest::gtest_main)
gtest_discover_tests(test_reference)

add_executable(test_reference_integration
    pipeline/test_reference_integration.cpp
    ${COMMON_TEST_SOURCES}
)
target_include_directories(test_reference_integration PRIVATE
    ${COMMON_TEST_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/src/pipeline
    ${CMAKE_SOURCE_DIR}/third_party/nlohmann/include
)
target_link_libraries(test_reference_integration PRIVATE gpt_sovits_cpp_static GTest::gtest_main)
target_compile_definitions(test_reference_integration PRIVATE
    PIPELINE_REFERENCE_CASE="${GPTSOVITS_REFERENCE_CASE}"
    HUBERT_TEST_DIR="${CMAKE_CURRENT_SOURCE_DIR}/hubert/"
    ROBERTA_MODEL_DIR="${CMAKE_CURRENT_SOURCE_DIR}/roberta/"
    T2S_TEST_DIR="${CMAKE_CURRENT_SOURCE_DIR}/t2s/"
    SOVITS_TEST_DIR="${CMAKE_CURRENT_SOURCE_DIR}/sovits/"
)
gtest_discover_tests(test_reference_integration
    PROPERTIES LABELS "cpu\\;integration" TIMEOUT 300
)

add_executable(test_synthesis pipeline/test_synthesis.cpp)
target_include_directories(test_synthesis PRIVATE
    ${CMAKE_SOURCE_DIR}/src/pipeline
    ${CMAKE_SOURCE_DIR}/third_party/ggml/src
)
target_link_libraries(test_synthesis PRIVATE gpt_sovits_cpp_static GTest::gtest_main)
gtest_discover_tests(test_synthesis)

add_executable(test_synthesis_integration
    pipeline/test_synthesis_integration.cpp
    ${COMMON_TEST_SOURCES}
)
target_include_directories(test_synthesis_integration PRIVATE
    ${COMMON_TEST_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/src/pipeline
    ${CMAKE_SOURCE_DIR}/third_party/ggml/src
    ${CMAKE_SOURCE_DIR}/third_party/nlohmann/include
)
target_link_libraries(test_synthesis_integration PRIVATE gpt_sovits_cpp_static GTest::gtest_main)
target_compile_definitions(test_synthesis_integration PRIVATE
    PIPELINE_REFERENCE_CASE="${GPTSOVITS_REFERENCE_CASE}"
    HUBERT_TEST_DIR="${CMAKE_CURRENT_SOURCE_DIR}/hubert/"
    ROBERTA_MODEL_DIR="${CMAKE_CURRENT_SOURCE_DIR}/roberta/"
    T2S_TEST_DIR="${CMAKE_CURRENT_SOURCE_DIR}/t2s/"
    SOVITS_TEST_DIR="${CMAKE_CURRENT_SOURCE_DIR}/sovits/"
)
gtest_discover_tests(test_synthesis_integration
    PROPERTIES LABELS "cpu\\;integration" TIMEOUT 600
)

add_executable(test_t2s_sovits_bridge
    ${CMAKE_CURRENT_SOURCE_DIR}/pipeline/test_t2s_sovits_bridge.cpp
    ${COMMON_TEST_SOURCES}
)
target_include_directories(test_t2s_sovits_bridge PRIVATE
    ${COMMON_TEST_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/src/pipeline
)
target_link_libraries(test_t2s_sovits_bridge PRIVATE
    gpt_sovits_cpp_static
    GTest::gtest_main
)
target_compile_definitions(test_t2s_sovits_bridge PRIVATE
    T2S_TEST_DIR="${CMAKE_CURRENT_SOURCE_DIR}/t2s/"
    SOVITS_TEST_DIR="${CMAKE_CURRENT_SOURCE_DIR}/sovits/"
)
gtest_discover_tests(test_t2s_sovits_bridge
    PROPERTIES LABELS "cpu\\;integration" TIMEOUT 300
)

# CLI smoke tests and optional local CPU synthesis integration.
if(TARGET gpt_sovits_cli)
    add_test(NAME cli_help COMMAND gpt_sovits_cli --help)

    add_test(NAME cli_missing_required_args COMMAND gpt_sovits_cli)
    set_tests_properties(cli_missing_required_args PROPERTIES WILL_FAIL TRUE)

    add_test(NAME cli_bad_input_path COMMAND gpt_sovits_cli
        --hubert /nonexistent/hubert.gguf
        --t2s /nonexistent/t2s.gguf
        --sovits /nonexistent/sovits.gguf
        --sv /nonexistent/sv.gguf
        --ref-audio /nonexistent/ref.wav
        --ref-frontend-json /nonexistent/ref.json
        --target-frontend-json /nonexistent/target.json
        --output ${CMAKE_CURRENT_BINARY_DIR}/cli_bad_input_path.wav)
    set_tests_properties(cli_bad_input_path PROPERTIES WILL_FAIL TRUE)

    # A non-GGUF --hubert passes the early path checks and fails inside
    # hubert_model_load after the CPU backend is up: the CLI must exit 1
    # without creating the output file.
    add_test(NAME cli_loader_failure_keeps_output_absent
        COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_SOURCE_DIR}/cli/check_cli.cmake
        1 ${CMAKE_CURRENT_BINARY_DIR}/cli_loader_failure.wav
        $<TARGET_FILE:gpt_sovits_cli>
        --hubert ${CMAKE_CURRENT_SOURCE_DIR}/frontend/fixtures/valid_mixed.json
        --t2s ${CMAKE_CURRENT_SOURCE_DIR}/frontend/fixtures/valid_mixed.json
        --sovits ${CMAKE_CURRENT_SOURCE_DIR}/frontend/fixtures/valid_mixed.json
        --sv ${CMAKE_CURRENT_SOURCE_DIR}/frontend/fixtures/valid_mixed.json
        --ref-audio ${CMAKE_CURRENT_SOURCE_DIR}/hubert/Narsil.asr_dummy.4.flac
        --ref-frontend-json ${CMAKE_CURRENT_SOURCE_DIR}/frontend/fixtures/valid_mixed.json
        --target-frontend-json ${CMAKE_CURRENT_SOURCE_DIR}/frontend/fixtures/valid_mixed.json
        --output ${CMAKE_CURRENT_BINARY_DIR}/cli_loader_failure.wav)

    add_executable(cli_synthesis_check cli/check_synthesis.cpp)
    target_include_directories(cli_synthesis_check PRIVATE ${CMAKE_SOURCE_DIR}/third_party/dr_libs)
    target_link_libraries(cli_synthesis_check PRIVATE gpt_sovits_cpp_static)
    add_dependencies(cli_synthesis_check gpt_sovits_cli)

    if("${GPTSOVITS_REFERENCE_CASE}" STREQUAL "")
        add_test(NAME cli_synthesis_e2e COMMAND cli_synthesis_check --skip)
        set_tests_properties(cli_synthesis_e2e PROPERTIES SKIP_RETURN_CODE 77)
    else()
        add_test(NAME cli_synthesis_e2e COMMAND ${CMAKE_COMMAND}
            "-DCLI_BINARY=$<TARGET_FILE:gpt_sovits_cli>"
            "-DWAV_CHECKER=$<TARGET_FILE:cli_synthesis_check>"
            "-DREFERENCE_CASE=${GPTSOVITS_REFERENCE_CASE}"
            "-DTEST_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}"
            "-DTEST_OUTPUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/cli-synthesis"
            -P ${CMAKE_CURRENT_SOURCE_DIR}/cli/check_synthesis.cmake)
    endif()
    set_tests_properties(cli_synthesis_e2e PROPERTIES LABELS "cpu;integration" TIMEOUT 900)
endif()
