/*
 * SPDX-FileCopyrightText: Copyright (c) 2020 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#if !defined(_ERROR_UTIL_H_)
#define _ERROR_UTIL_H_

#include <functional>
#include <sstream>
#include <stdlib.h>
#include <iostream>

#include <cudnn_frontend.h>

#define FatalError(s)                                                     \
    {                                                                     \
        std::stringstream _where, _message;                               \
        _where << __FILE__ << ':' << __LINE__;                            \
        _message << std::string(s) + "\n" << __FILE__ << ':' << __LINE__; \
        std::cerr << _message.str() << "\nAborting...\n";                 \
        cudaDeviceReset();                                                \
        exit(EXIT_FAILURE);                                               \
    }

#define checkCudaErrors(status)                                              \
    {                                                                        \
        std::stringstream _error;                                            \
        if (status != 0) {                                                   \
            _error << "Cuda failure\nError: " << cudaGetErrorString(status); \
            FatalError(_error.str());                                        \
        }                                                                    \
    }

namespace cudnn_frontend {
static inline void
throw_if(std::function<bool()> expr, [[maybe_unused]] const char *message, [[maybe_unused]] cudnnStatus_t status) {
    if (expr()) {
#ifndef NV_CUDNN_DISABLE_EXCEPTION
        throw cudnn_frontend::cudnnException(message, status);
#endif
    }
}
static inline void
throw_if(bool expr, [[maybe_unused]] const char *message, [[maybe_unused]] cudnnStatus_t status) {
    if (expr) {
#ifndef NV_CUDNN_DISABLE_EXCEPTION
        throw cudnn_frontend::cudnnException(message, status);
#endif
    }
}
}  // namespace cudnn_frontend

#endif  // _ERROR_UTIL_H_
