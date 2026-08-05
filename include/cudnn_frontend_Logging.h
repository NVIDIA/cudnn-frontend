

/*
 * SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <iostream>
#include <fstream>
#include <cstring>
#include <cstdlib>

#include "cudnn_frontend_shim.h"  // cudnn_frontend::get_environment

namespace cudnn_frontend {

inline int
getLogLevel() {
#ifdef NV_CUDNN_FRONTEND_DISABLE_LOGGING
    static int log_level = 0;
#else
    static int log_level = []() {
        const char *env_val = get_environment("CUDNN_FRONTEND_LOG_INFO");
        return env_val ? std::atoi(env_val) : 0;
    }();
#endif
    return log_level;
}

inline bool &
isLoggingEnabled() {
#ifdef NV_CUDNN_FRONTEND_DISABLE_LOGGING
    static bool log_enabled = false;
#else
    static bool log_enabled = (getLogLevel() > 0);
#endif
    return log_enabled;
}

inline bool &
isLoggingTensorDumpEnabled() {
#ifdef NV_CUDNN_FRONTEND_DISABLE_LOGGING
    static bool tensor_dump_enabled = false;
#else
    static bool tensor_dump_enabled = []() {
        int level = getLogLevel();
        return level >= 1 && level < 10;
    }();
#endif
    return tensor_dump_enabled;
}

inline std::ostream &
getStream() {
    static std::ofstream outFile;
    static std::ostream &stream =
        get_environment("CUDNN_FRONTEND_LOG_FILE")
            ? (std::strncmp(get_environment("CUDNN_FRONTEND_LOG_FILE"), "stdout", 6) == 0
                   ? std::cout
                   : (std::strncmp(get_environment("CUDNN_FRONTEND_LOG_FILE"), "stderr", 6) == 0
                          ? std::cerr
                          : (outFile.open(get_environment("CUDNN_FRONTEND_LOG_FILE"), std::ios::out), outFile)))
            : (isLoggingEnabled() = false, std::cout);
    return stream;
}

class ConditionalStreamer {
   private:
    std::ostream &stream;

   public:
    ConditionalStreamer(std::ostream &stream_) : stream(stream_) {}

    template <typename T>
    const ConditionalStreamer &
    operator<<(const T &t) const {
        if (isLoggingEnabled()) {
            stream << t;
        }
        return *this;
    }

    const ConditionalStreamer &
    operator<<(std::ostream &(*spl)(std::ostream &)) const {
        if (isLoggingEnabled()) {
            stream << spl;
        }
        return *this;
    }
};

inline ConditionalStreamer &
getLogger() {
    static ConditionalStreamer opt(getStream());
    return opt;
}

#define CUDNN_FE_LOG(X)           \
    do {                          \
        if (isLoggingEnabled()) { \
            getLogger() << X;     \
        }                         \
    } while (0);

#define CUDNN_FE_LOG_LABEL(X)                        \
    do {                                             \
        if (isLoggingEnabled()) {                    \
            getLogger() << "[cudnn_frontend] " << X; \
        }                                            \
    } while (0);

#define CUDNN_FE_LOG_LABEL_ENDL(X)                                \
    do {                                                          \
        if (isLoggingEnabled()) {                                 \
            getLogger() << "[cudnn_frontend] " << X << std::endl; \
        }                                                         \
    } while (0);

#define CUDNN_FE_LOG_BANNER(X)                                                         \
    do {                                                                               \
        if (isLoggingEnabled()) {                                                      \
            {                                                                          \
                constexpr int total_width = 128;                                       \
                std::ostringstream oss;                                                \
                oss << "[cudnn_frontend] ||| === " << X << " === |||";                 \
                std::string banner_line = oss.str();                                   \
                int banner_len          = static_cast<int>(banner_line.size());        \
                int pad                 = total_width - banner_len;                    \
                if (pad > 0) {                                                         \
                    banner_line.insert(banner_line.size() - 5, std::string(pad, ' ')); \
                }                                                                      \
                getLogger() << std::string(total_width, '=') << std::endl;             \
                getLogger() << banner_line << std::endl;                               \
                getLogger() << std::string(total_width, '=') << std::endl;             \
            }                                                                          \
        }                                                                              \
    } while (0);

static std::ostream &
operator<<(std::ostream &os, const BackendDescriptor &desc) {
    if (isLoggingEnabled()) {
        os << desc.describe();
    }
    return os;
}

}  // namespace cudnn_frontend
