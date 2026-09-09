/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

// cudaLibrary_t, cudaKernel_t, cudaJitOption, cudaLibraryOption etc.
// were introduced in CUDA 12.8. On older toolkits (or when using dynamic
// loading where cuda_runtime.h may lack these types), provide typedefs
// from the driver API equivalents (available via cuda.h) so that
// engine headers can declare members without #ifdef clutter.
// The actual runtime API functions are guarded by CUDART_VERSION checks below.
#if !defined(CUDART_VERSION) || CUDART_VERSION < 12080
using cudaLibrary_t     = CUlibrary;
using cudaKernel_t      = CUkernel;
using cudaJitOption     = CUjit_option;
using cudaLibraryOption = CUlibraryOption;
#endif

namespace cudnn_frontend::experimental {

inline std::vector<std::string>
parse_flags_string(const char* data, size_t len) {
    std::vector<std::string> flags;
    std::string content(data, len);
    std::istringstream stream(content);
    std::string line;
    while (std::getline(stream, line)) {
        size_t start = line.find_first_not_of(" \t\r\n");
        if (start == std::string::npos) continue;
        size_t end = line.find_last_not_of(" \t\r\n");
        line       = line.substr(start, end - start + 1);
        if (!line.empty()) {
            flags.push_back(line);
        }
    }
    return flags;
}

// ============================================================
// CUDA runtime API wrappers (using NV_FE_CALL_TO_CUDA)
// ============================================================

namespace detail {

// NV_FE_CALL_TO_CUDA macros reference symbols in cudnn_frontend::detail
// via unqualified lookup. Import them into this namespace.
#if defined NV_CUDNN_FRONTEND_USE_DYNAMIC_LOADING
using cudnn_frontend::detail::CudaLibrary;
using cudnn_frontend::detail::get_cuda_symbol;
#endif

// Re-export CUDA runtime wrappers from the main shim so the engine
// never calls cudaMalloc/cudaFree/etc. directly (required for dynamic loading).
using cudnn_frontend::detail::cuda_get_device;
using cudnn_frontend::detail::cuda_get_device_properties;
using cudnn_frontend::detail::cuda_get_error_string;
using cudnn_frontend::detail::cuda_mem_cpy_async;
using cudnn_frontend::detail::cuda_mem_set_async;

// Write a 32-bit pattern to device memory (async, on stream).
// Uses thread_local storage so the source buffer persists through the async copy.
// Only used with N=1 in practice (writing float 1.0f for FP8 default scale).
// Cannot use NV_FE_CALL_TO_CUDA because cudaMemcpyAsync has more args than this wrapper.
inline cudaError_t
cuda_mem_set_d32_async(void* dstDevice, unsigned int ui, size_t N, cudaStream_t stream) {
    static thread_local unsigned int val;
    val = ui;
#if defined NV_CUDNN_FRONTEND_USE_DYNAMIC_LOADING
    using fn_t = cudaError_t (*)(void*, const void*, size_t, cudaMemcpyKind, cudaStream_t);
    auto _fn = reinterpret_cast<fn_t>(cudnn_frontend::detail::get_cuda_symbol(CudaLibrary::CUDART, "cudaMemcpyAsync"));
    return _fn(dstDevice, &val, N * sizeof(unsigned int), cudaMemcpyHostToDevice, stream);
#else
    return cudaMemcpyAsync(dstDevice, &val, N * sizeof(unsigned int), cudaMemcpyHostToDevice, stream);
#endif
}

// Convert cudaError_t to a descriptive string (e.g., "invalid argument")
inline std::string
cuda_error_to_string(cudaError_t err) {
    const char* str = cuda_get_error_string(err);
    return str ? std::string(str) : ("cudaError=" + std::to_string(static_cast<int>(err)));
}

// cudaGetLastError takes zero arguments — can't use NV_FE_CALL_TO_CUDA
// (variadic macro requires at least one arg). Handle both paths manually.
inline cudaError_t
cuda_get_last_error() {
#if defined NV_CUDNN_FRONTEND_USE_DYNAMIC_LOADING
    using fn_t = cudaError_t (*)();
    auto _fn = reinterpret_cast<fn_t>(cudnn_frontend::detail::get_cuda_symbol(CudaLibrary::CUDART, "cudaGetLastError"));
    return _fn();
#else
    return cudaGetLastError();
#endif
}

// ============================================================
// CUDA 12.8+ runtime API wrappers for library/kernel management.
// These APIs (cudaLibraryLoadData, cudaLibraryGetKernel, cudaLibraryUnload,
// cudaKernelSetAttributeForDevice, cudaGetDriverEntryPointByVersion) require
// CUDART_VERSION >= 12080. On older toolkits, the OSS engine check_support()
// will reject the configuration before these are called.
// ============================================================

#if !defined(NV_CUDNN_FRONTEND_USE_DYNAMIC_LOADING) && defined(CUDART_VERSION) && CUDART_VERSION < 12080

// Stubs that return errors — OSS engines require CUDA 12.8+.
inline cudaError_t
cuda_library_load_data(void*, const void*, void*, void**, unsigned int, void*, void**, unsigned int) {
    return cudaErrorNotSupported;
}
inline cudaError_t
cuda_library_get_kernel(void*, void*, const char*) {
    return cudaErrorNotSupported;
}
inline cudaError_t
cuda_library_unload(void*) {
    return cudaErrorNotSupported;
}
inline cudaError_t
cuda_kernel_set_attribute_for_device(void*, int, int, int) {
    return cudaErrorNotSupported;
}

#else  // CUDART_VERSION >= 12080 or dynamic loading

inline cudaError_t
cuda_library_load_data(cudaLibrary_t* library,
                       const void* code,
                       cudaJitOption* jitOptions,
                       void** jitOptionsValues,
                       unsigned int numJitOptions,
                       cudaLibraryOption* libraryOptions,
                       void** libraryOptionsValues,
                       unsigned int numLibraryOptions) {
    NV_FE_CALL_TO_CUDA(cuda_library_load_data,
                       cudaLibraryLoadData,
                       library,
                       code,
                       jitOptions,
                       jitOptionsValues,
                       numJitOptions,
                       libraryOptions,
                       libraryOptionsValues,
                       numLibraryOptions);
}

inline cudaError_t
cuda_library_get_kernel(cudaKernel_t* pKernel, cudaLibrary_t library, const char* name) {
    NV_FE_CALL_TO_CUDA(cuda_library_get_kernel, cudaLibraryGetKernel, pKernel, library, name);
}

inline cudaError_t
cuda_library_unload(cudaLibrary_t library) {
    NV_FE_CALL_TO_CUDA(cuda_library_unload, cudaLibraryUnload, library);
}

inline cudaError_t
cuda_kernel_set_attribute_for_device(cudaKernel_t kernel, cudaFuncAttribute attrib, int val, int dev) {
    NV_FE_CALL_TO_CUDA(cuda_kernel_set_attribute_for_device, cudaKernelSetAttributeForDevice, kernel, attrib, val, dev);
}

#endif  // CUDART_VERSION check

inline cudaError_t
cuda_launch_kernel(const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, cudaStream_t stream) {
    NV_FE_CALL_TO_CUDA(cuda_launch_kernel, cudaLaunchKernel, func, gridDim, blockDim, args, sharedMem, stream);
}

}  // namespace detail

}  // namespace cudnn_frontend::experimental