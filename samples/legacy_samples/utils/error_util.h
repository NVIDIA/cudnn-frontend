/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
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
