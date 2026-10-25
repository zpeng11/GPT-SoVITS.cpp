#pragma once

#include <cstdint>
#include <cstdio>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>

namespace gpt_sovits::frontend_detail {

class contract_error : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

[[noreturn]] inline void invalid(const std::string & path, const std::string & reason) {
    throw contract_error(path + ": " + reason);
}

inline std::string indexed(const std::string & path, size_t i) {
    return path + "[" + std::to_string(i) + "]";
}

inline int64_t codepoint_count(const std::string & text, const std::string & path) {
    int64_t count = 0;
    for (size_t i = 0; i < text.size();) {
        const auto lead = static_cast<unsigned char>(text[i]);
        size_t width = 0;
        uint32_t cp = 0;
        if (lead < 0x80) { width = 1; cp = lead; }
        else if (lead >= 0xc2 && lead <= 0xdf) { width = 2; cp = lead & 0x1f; }
        else if (lead >= 0xe0 && lead <= 0xef) { width = 3; cp = lead & 0x0f; }
        else if (lead >= 0xf0 && lead <= 0xf4) { width = 4; cp = lead & 0x07; }
        else invalid(path, "invalid UTF-8");
        if (width > text.size() - i) invalid(path, "truncated UTF-8");
        for (size_t j = 1; j < width; ++j) {
            const auto c = static_cast<unsigned char>(text[i + j]);
            if ((c & 0xc0) != 0x80) invalid(path, "invalid UTF-8 continuation");
            cp = (cp << 6) | (c & 0x3f);
        }
        if ((width == 2 && cp < 0x80) || (width == 3 && cp < 0x800) ||
            (width == 4 && cp < 0x10000) || cp > 0x10ffff ||
            (cp >= 0xd800 && cp <= 0xdfff)) invalid(path, "invalid UTF-8 code point");
        if (count == std::numeric_limits<int64_t>::max()) invalid(path, "text is too long");
        ++count;
        i += width;
    }
    return count;
}

inline bool report_error(std::string * error, const char * message) noexcept {
    if (error) {
        try {
            *error = message;
            return false;
        } catch (const std::bad_alloc &) {
        } catch (const std::length_error &) {
        }
    }
    fprintf(stderr, "text_frontend: %s\n", message);
    return false;
}

} // namespace gpt_sovits::frontend_detail
