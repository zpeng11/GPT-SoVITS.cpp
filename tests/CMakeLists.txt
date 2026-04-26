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
    ${CMAKE_CURRENT_SOURCE_DIR}/sovits/test_sovits_text_encoder_ssl.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sovits/test_sovits_text_encoder_text.cpp
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

# T2S tests (use cnpy for .npy loading)
set(T2S_TEST_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/t2s/test_t2s_loader.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/t2s/test_t2s_session.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/t2s/test_t2s_infer.cpp
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
