set(GPTSOVITS_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/audio/stft.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/audio/io.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/audio/resample.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/audio/reference.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/frontend/json.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/frontend/bert_features.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/pipeline/reference.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/pipeline/synthesize.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/pipeline/synthesis_validation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/hubert/block.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/hubert/loader.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/hubert/session.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sovits/block.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sovits/loader.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sovits/session.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sv/sv.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/t2s/block.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/t2s/generate.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/t2s/loader.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/t2s/session.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/roberta/block.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/roberta/loader.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/roberta/session.cpp
)

set(GPTSOVITS_PUBLIC_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)
set(GPTSOVITS_JSON_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/third_party/nlohmann/include)

# Static library
if(GPTSOVITS_BUILD_STATIC)
    add_library(gpt_sovits_cpp_static STATIC ${GPTSOVITS_SOURCES})
    target_include_directories(gpt_sovits_cpp_static PUBLIC
        $<BUILD_INTERFACE:${GPTSOVITS_PUBLIC_INCLUDE_DIR}>
        PRIVATE ${GPTSOVITS_JSON_INCLUDE_DIR}
    )
    target_link_libraries(gpt_sovits_cpp_static PUBLIC ggml::ggml)
    set_target_properties(gpt_sovits_cpp_static PROPERTIES OUTPUT_NAME gpt_sovits_cpp)
endif()

# Shared library
if(GPTSOVITS_BUILD_SHARED)
    add_library(gpt_sovits_cpp_shared SHARED ${GPTSOVITS_SOURCES})
    target_include_directories(gpt_sovits_cpp_shared PUBLIC
        $<BUILD_INTERFACE:${GPTSOVITS_PUBLIC_INCLUDE_DIR}>
        PRIVATE ${GPTSOVITS_JSON_INCLUDE_DIR}
    )
    target_link_libraries(gpt_sovits_cpp_shared PUBLIC ggml::ggml)
    set_target_properties(gpt_sovits_cpp_shared PROPERTIES OUTPUT_NAME gpt_sovits_cpp)
endif()
