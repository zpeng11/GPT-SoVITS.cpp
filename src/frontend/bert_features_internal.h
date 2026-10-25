#pragma once

#include "gpt_sovits/text_frontend.h"

namespace gpt_sovits::detail {

using frontend_token_feature_provider = bool (*)(
    const void * context,
    const text_frontend_tokenizer & tokenizer,
    float * out,
    int64_t out_cap,
    std::string & error);

bool text_frontend_build_bert_features_with_provider(
    const text_frontend_result & text,
    frontend_token_feature_provider provider,
    const void * context,
    std::vector<float> & out,
    std::string * error = nullptr);

} // namespace gpt_sovits::detail
