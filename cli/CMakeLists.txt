# CLI executable: single-request offline CPU synthesis over the public API.
add_executable(gpt_sovits_cli main.cpp)
target_link_libraries(gpt_sovits_cli PRIVATE gpt_sovits_cpp_static)
