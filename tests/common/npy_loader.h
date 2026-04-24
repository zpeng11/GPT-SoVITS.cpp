// tests/common/npy_loader.h
//
// Helper to load .npy files as std::vector<float> with automatic fp16->fp32
// conversion. Eliminates the need for a Python npy->bin conversion step.

#ifndef NPY_LOADER_H_
#define NPY_LOADER_H_

#include "cnpy.h"

#include "ggml.h"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

// Load a .npy file and return its data as a flat std::vector<float>.
// Supports:
//   - float32 (.npy descr '<f4'): returned as-is.
//   - float16 (.npy descr '<f2'): converted to float32 via ggml_fp16_to_fp32.
//   - int32   (.npy descr '<i4'): reinterpreted as float (bit-cast).
//
// The shape information is discarded — the caller knows the expected shape.
inline std::vector<float> load_npy_as_f32(const std::string & path) {
    cnpy::NpyArray arr = cnpy::npy_load(path);

    const size_t n = arr.num_vals;

    if (arr.word_size == 4) {
        // Already float32 or int32 — just copy.
        const float * src = arr.data<float>();
        return std::vector<float>(src, src + n);
    }

    if (arr.word_size == 2) {
        // float16 -> float32 conversion.
        std::vector<float> result(n);
        const uint16_t * src = arr.data<uint16_t>();
        for (size_t i = 0; i < n; i++) {
            ggml_fp16_t h;
            memcpy(&h, &src[i], sizeof(ggml_fp16_t));
            result[i] = ggml_fp16_to_fp32(h);
        }
        return result;
    }

    fprintf(stderr, "load_npy_as_f32: unsupported word_size=%zu in '%s'\n",
            arr.word_size, path.c_str());
    return {};
}

#endif // NPY_LOADER_H_
