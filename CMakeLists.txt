cmake_minimum_required(VERSION 3.14)
project(gpt_sovits_cpp VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Keep build artifacts out of per-subdirectory build trees.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

foreach(cfg Debug Release RelWithDebInfo MinSizeRel)
    string(TOUPPER ${cfg} cfg_upper)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${cfg_upper} ${CMAKE_BINARY_DIR}/bin)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${cfg_upper} ${CMAKE_BINARY_DIR}/lib)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${cfg_upper} ${CMAKE_BINARY_DIR}/lib)
endforeach()

# --- Project options ---
option(GPTSOVITS_BUILD_SHARED "Build shared library" ON)
option(GPTSOVITS_BUILD_STATIC "Build static library" ON)
option(GPTSOVITS_BUILD_TESTS  "Build tests"          OFF)

# --- ggml submodule ---
# ggml options are passed through naturally via add_subdirectory.
# Common options (pass via -D on cmake command line):
#   GGML_CUDA=ON       - enable CUDA backend
#   GGML_METAL=ON      - enable Metal backend
#   GGML_VULKAN=ON     - enable Vulkan backend
#   GGML_BLAS=ON       - enable BLAS
#   GGML_OPENMP=OFF    - disable OpenMP (ON by default)
# See third_party/ggml/CMakeLists.txt for all available options.
add_subdirectory(third_party/ggml)

# --- Project sources ---
add_subdirectory(src)

# --- Tests ---
if(GPTSOVITS_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
