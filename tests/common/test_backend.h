// tests/common/test_backend.h
//
// Shared helper for creating a ggml backend in tests.
// On Apple platforms with Metal compiled in (GGML_USE_METAL), prefers the
// Metal GPU backend and falls back to CPU.  On all other platforms the
// preprocessor guard compiles out the Metal path entirely.

#pragma once

#include "ggml-backend.h"

#if defined(GGML_USE_METAL)
#include "ggml-metal.h"
#endif

#include "ggml-cpu.h"

static inline ggml_backend_t create_test_backend() {
#if defined(GGML_USE_METAL)
    ggml_backend_t backend = ggml_backend_metal_init();
    if (backend) return backend;
#endif
    return ggml_backend_cpu_init();
}
