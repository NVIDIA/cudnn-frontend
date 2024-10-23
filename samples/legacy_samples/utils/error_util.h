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
#include <stdio.h>
#include <iostream>
#include <vector>

#include <cudnn_frontend.h>

#define TOSTR_(s) #s
#define TOSTR(s) TOSTR_(s)
#if defined(__GNUC__)
#define COMPILER_NAME "GCC"
#define COMPILER_VER TOSTR(__GNUC__) "." TOSTR(__GNUC_MINOR__) "." TOSTR(__GNUC_PATCHLEVEL__)
#elif defined(_MSC_VER)
#if _MSC_VER < 1500
#define COMPILER_NAME "MSVC_2005"
#elif _MSC_VER < 1600
#define COMPILER_NAME "MSVC_2008"
#elif _MSC_VER < 1700
#define COMPILER_NAME "MSVC_2010"
#elif _MSC_VER < 1800
#define COMPILER_NAME "MSVC_2012"
#elif _MSC_VER < 1900
#define COMPILER_NAME "MSVC_2013"
#elif _MSC_VER < 2000
#define COMPILER_NAME "MSVC_2014"
#else
#define COMPILER_NAME "MSVC"
#endif
#define COMPILER_VER TOSTR(_MSC_FULL_VER) "." TOSTR(_MSC_BUILD)
#elif defined(__clang_major__)
#define COMPILER_NAME "CLANG"
#define COMPILER_VER TOSTR(__clang_major__) "." TOSTR(__clang_minor__) "." TOSTR(__clang_patchlevel__)
#elif defined(__INTEL_COMPILER)
#define COMPILER_NAME "ICC"
#define COMPILER_VER TOSTR(__INTEL_COMPILER) "." TOSTR(__INTEL_COMPILER_BUILD_DATE)
#else
#define COMPILER_NAME "unknown"
#define COMPILER_VER "???"
#endif

#define CUDNN_VERSION_STR TOSTR(CUDNN_MAJOR) "." TOSTR(CUDNN_MINOR) "." TOSTR(CUDNN_PATCHLEVEL)

#define FatalError(s)                                                     \
    {                                                                     \
        std::stringstream _where, _message;                               \
        _where << __FILE__ << ':' << __LINE__;                            \
        _message << std::string(s) + "\n" << __FILE__ << ':' << __LINE__; \
        std::cerr << _message.str() << "\nAborting...\n";                 \
        cudaDeviceReset();                                                \
        exit(EXIT_FAILURE);                                               \
    }

#define checkCUDNN(status)                                                     \
    {                                                                          \
        std::stringstream _error;                                              \
        if (status != CUDNN_STATUS_SUCCESS) {                                  \
            _error << "CUDNN failure\nError: " << cudnnGetErrorString(status); \
            FatalError(_error.str());                                          \
        }                                                                      \
    }

#define checkCudaErrors(status)                                              \
    {                                                                        \
        std::stringstream _error;                                            \
        if (status != 0) {                                                   \
            _error << "Cuda failure\nError: " << cudaGetErrorString(status); \
            FatalError(_error.str());                                        \
        }                                                                    \
    }

#define checkCublasErrors(status)                              \
    {                                                          \
        std::stringstream _error;                              \
        if (status != 0) {                                     \
            _error << "Cublas failure\nError code " << status; \
            FatalError(_error.str());                          \
        }                                                      \
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
// CUDA Utility Helper Functions

static void
showDevices(void) {
    int totalDevices;
    checkCudaErrors(cudaGetDeviceCount(&totalDevices));
    printf("\nThere are %d CUDA capable devices on your machine :\n", totalDevices);
    for (int i = 0; i < totalDevices; i++) {
        struct cudaDeviceProp prop;
        checkCudaErrors(cudaGetDeviceProperties(&prop, i));
        printf(
            "device %d : sms %2d  Capabilities %d.%d, SmClock %.1f Mhz, MemSize (Mb) %d, MemClock %.1f Mhz, Ecc=%d, "
            "boardGroupID=%d\n",
            i,
            prop.multiProcessorCount,
            prop.major,
            prop.minor,
            (float)prop.clockRate * 1e-3,
            (int)(prop.totalGlobalMem / (1024 * 1024)),
            (float)prop.memoryClockRate * 1e-3,
            prop.ECCEnabled,
            prop.multiGpuBoardGroupID);
    }
}

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#ifndef _CRT_SECURE_NO_DEPRECATE
#define _CRT_SECURE_NO_DEPRECATE
#endif
#ifndef STRNCASECMP
#define STRNCASECMP _strnicmp
#endif
#else  // Linux Includes
#include <string.h>
#include <strings.h>
#ifndef STRNCASECMP
#define STRNCASECMP strncasecmp
#endif
#endif
inline int
stringRemoveDelimiter(char delimiter, const char *string) {
    int string_start = 0;

    while (string[string_start] == delimiter) {
        string_start++;
    }

    if (string_start >= (int)strlen(string) - 1) {
        return 0;
    }

    return string_start;
}

inline bool
checkCmdLineFlag(const int argc, const char **argv, const char *string_ref) {
    bool bFound = false;

    if (argc >= 1) {
        for (int i = 1; i < argc; i++) {
            int string_start        = stringRemoveDelimiter('-', argv[i]);
            const char *string_argv = &argv[i][string_start];

            const char *equal_pos = strchr(string_argv, '=');
            int argv_length       = (int)(equal_pos == 0 ? strlen(string_argv) : equal_pos - string_argv);

            int length = (int)strlen(string_ref);

            if (length == argv_length && !STRNCASECMP(string_argv, string_ref, length)) {
                bFound = true;
                continue;
            }
        }
    }

    return bFound;
}

inline int
getCmdLineArgumentInt(const int argc, const char **argv, const char *string_ref) {
    bool bFound = false;
    int value   = -1;

    if (argc >= 1) {
        for (int i = 1; i < argc; i++) {
            int string_start        = stringRemoveDelimiter('-', argv[i]);
            const char *string_argv = &argv[i][string_start];
            int length              = (int)strlen(string_ref);

            if (!STRNCASECMP(string_argv, string_ref, length)) {
                if (length + 1 <= (int)strlen(string_argv)) {
                    int auto_inc = (string_argv[length] == '=') ? 1 : 0;
                    value        = atoi(&string_argv[length + auto_inc]);
                } else {
                    value = 0;
                }

                bFound = true;
                continue;
            }
        }
    }

    if (bFound) {
        return value;
    } else {
        printf("Not found int\n");
        return 0;
    }
}

inline bool
getCmdLineArgumentString(const int argc, const char **argv, const char *string_ref, char **string_retval) {
    bool bFound = false;

    if (argc >= 1) {
        for (int i = 1; i < argc; i++) {
            int string_start  = stringRemoveDelimiter('-', argv[i]);
            char *string_argv = (char *)&argv[i][string_start];
            int length        = (int)strlen(string_ref);

            if (!STRNCASECMP(string_argv, string_ref, length)) {
                *string_retval = &string_argv[length + 1];
                bFound         = true;
                continue;
            }
        }
    }

    if (!bFound) {
        *string_retval = NULL;
    }

    return bFound;
}

#endif  // _ERROR_UTIL_H_
