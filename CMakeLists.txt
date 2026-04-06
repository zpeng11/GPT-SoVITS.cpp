cmake_minimum_required(VERSION 3.14)
project(gpt_sovits_cpp VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --- Project options ---
option(GPTSOVITS_BUILD_SHARED "Build shared library" ON)
option(GPTSOVITS_BUILD_STATIC "Build static library" ON)

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
