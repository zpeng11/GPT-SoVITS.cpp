#pragma once

#include "gpt_sovits/roberta.h"

namespace gpt_sovits {
namespace detail {

struct ggml_tensor * roberta_model_block_forward_n_layers(
    struct ggml_context                  * ctx,
    struct ggml_tensor                   * input_ids,
    struct ggml_tensor                   * token_type_ids,
    const roberta_model_block_weights    & weights,
    int                                    n_layers);

} // namespace detail
} // namespace gpt_sovits
