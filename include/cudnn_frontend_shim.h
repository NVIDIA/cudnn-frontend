/*
 * Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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

#pragma once

#include <ostream>
#include <iostream>
#if defined NV_CUDNN_FRONTEND_USE_DYNAMIC_LOADING
#include <dlfcn.h>
#include <mutex>
#endif

namespace cudnn_frontend {

// cudnn package initialization set this global handle
extern void *cudnn_dlhandle;

namespace detail {

#if defined NV_CUDNN_FRONTEND_USE_DYNAMIC_LOADING

inline void *
get_symbol(const char *function_name) {
    void *ret = dlsym(cudnn_dlhandle, function_name);
    return ret;
}

inline void *
get_cuda_symbol(const char *function_name) {
    static std::mutex cuda_fe_lib_mutex;
    std::lock_guard<std::mutex> lock(cuda_fe_lib_mutex);
    char *c                = NULL;
    c                      = dlerror();
    static void *dl_handle = dlopen("libcudart.so", RTLD_NOW);
    c                      = dlerror();
    (void)c;
    if (dl_handle == nullptr) {
        std::string error_msg = std::string("Unable to dlopen libcudart.so") + std::string(c);
        throw std::runtime_error(error_msg.c_str());
    }

    void *ret = dlsym(dl_handle, function_name);
    return ret;
}
#endif

#if defined NV_CUDNN_FRONTEND_USE_DYNAMIC_LOADING
#define NV_CUDNN_FE_DYNAMIC_CHECK_BACKEND_DESCRIPTOR(MINIMUM_VERSION, DESCRIPTOR, MESSAGE) \
    if (MINIMUM_VERSION > detail::get_backend_version()) {                                 \
        set_error_and_throw_exception(&DESCRIPTOR, CUDNN_STATUS_INVALID_VALUE, MESSAGE);   \
        return std::move(DESCRIPTOR);                                                      \
    }
#else
#define NV_CUDNN_FE_DYNAMIC_CHECK_BACKEND_DESCRIPTOR(MINIMUM_VERSION, DESCRIPTOR, MESSAGE)
#endif

#if defined NV_CUDNN_FRONTEND_USE_DYNAMIC_LOADING
#define NV_CUDNN_FE_DYNAMIC_CHECK_CUDNN_BACKEND_VERSION(MINIMUM_VERSION, STATUS) \
    if (MINIMUM_VERSION > detail::get_backend_version()) {                       \
        return STATUS;                                                           \
    }
#else
#define NV_CUDNN_FE_DYNAMIC_CHECK_CUDNN_BACKEND_VERSION(MINIMUM_VERSION, STATUS)
#endif

#if defined NV_CUDNN_FRONTEND_USE_DYNAMIC_LOADING
#define NV_FE_CALL_TO_BACKEND(function_name, backend_symbol, ...)           \
    static void *fptr = get_symbol(#backend_symbol);                        \
    if (fptr == nullptr) {                                                  \
        throw std::runtime_error("Unable to find symbol " #backend_symbol); \
    }                                                                       \
    return reinterpret_cast<decltype(function_name) *>(fptr)(__VA_ARGS__);
#else
#define NV_FE_CALL_TO_BACKEND(function_name, backend_symbol, ...) return backend_symbol(__VA_ARGS__);
#endif

#if defined NV_CUDNN_FRONTEND_USE_DYNAMIC_LOADING
#define NV_FE_CALL_TO_CUDA(function_name, cuda_symbol, ...)              \
    static void *fptr = get_cuda_symbol(#cuda_symbol);                   \
    if (fptr == nullptr) {                                               \
        throw std::runtime_error("Unable to find symbol " #cuda_symbol); \
    }                                                                    \
    return reinterpret_cast<decltype(function_name) *>(fptr)(__VA_ARGS__);
#else
#define NV_FE_CALL_TO_CUDA(function_name, cuda_symbol, ...) return cuda_symbol(__VA_ARGS__);
#endif

inline cudaError_t
cuda_event_create(cudaEvent_t *event) {
    NV_FE_CALL_TO_CUDA(cuda_event_create, cudaEventCreate, event);
}

inline cudaError_t
cuda_event_destroy(cudaEvent_t event) {
    NV_FE_CALL_TO_CUDA(cuda_event_destroy, cudaEventDestroy, event);
}

inline cudaError_t
cuda_event_record(cudaEvent_t event, cudaStream_t stream) {
    NV_FE_CALL_TO_CUDA(cuda_event_record, cudaEventRecord, event, stream);
}

inline cudaError_t
cuda_event_synchronize(cudaEvent_t event) {
    NV_FE_CALL_TO_CUDA(cuda_event_synchronize, cudaEventSynchronize, event);
}

inline cudaError_t
cuda_event_elapsed_time(float *ms, cudaEvent_t start, cudaEvent_t end) {
    NV_FE_CALL_TO_CUDA(cuda_event_elapsed_time, cudaEventElapsedTime, ms, start, end);
}

inline cudaError_t
cuda_mem_cpy_async(void *dst, const void *src, size_t count, cudaMemcpyKind kind, cudaStream_t stream) {
    NV_FE_CALL_TO_CUDA(cuda_mem_cpy_async, cudaMemcpyAsync, dst, src, count, kind, stream);
}

inline cudaError_t
cuda_mem_set_async(void *devPtr, int value, size_t count, cudaStream_t stream) {
    NV_FE_CALL_TO_CUDA(cuda_mem_set_async, cudaMemsetAsync, devPtr, value, count, stream);
}

inline cudaError_t
cuda_get_device_properties(cudaDeviceProp *prop, int device) {
    NV_FE_CALL_TO_CUDA(cuda_get_device_properties, cudaGetDeviceProperties, prop, device);
}

inline cudaError_t
cuda_get_device(int *device) {
    NV_FE_CALL_TO_CUDA(cuda_get_device, cudaGetDevice, device);
}

inline const char *
cuda_get_error_string(cudaError_t error) {
    NV_FE_CALL_TO_CUDA(cuda_get_error_string, cudaGetErrorString, error);
}

inline cudaError_t
cuda_device_synchronize() {
#if defined NV_CUDNN_FRONTEND_USE_DYNAMIC_LOADING
    static void *fptr = get_cuda_symbol("cudaDeviceSynchronize");
    if (fptr == nullptr) {
        throw std::runtime_error("Unable to find symbol cudaDeviceSynchronize");
    }
    return reinterpret_cast<decltype(cuda_device_synchronize) *>(fptr)();
#else
    return cudaDeviceSynchronize();
#endif
}

inline cudnnStatus_t
create_handle(cudnnHandle_t *handle) {
    NV_FE_CALL_TO_BACKEND(create_handle, cudnnCreate, handle);
}

inline cudnnStatus_t
destroy_handle(cudnnHandle_t handle) {
    NV_FE_CALL_TO_BACKEND(destroy_handle, cudnnDestroy, handle);
}

inline size_t
get_backend_version(void) {
#if defined NV_CUDNN_FRONTEND_USE_DYNAMIC_LOADING
    static void *fptr = get_symbol("cudnnGetVersion");
    if (fptr == nullptr) {
        throw std::runtime_error("Unable to find symbol cudnnGetVersion");
    }
    return reinterpret_cast<decltype(get_backend_version) *>(fptr)();
#else
    return cudnnGetVersion();
#endif
}

inline constexpr size_t
get_compiled_version(void) {
    return CUDNN_VERSION;
}

inline std::string
convert_version_to_str(size_t const version) {
    // The multiplier for major version pre-v9 and post-v9 are different.
    size_t major = version / 10000;
    size_t minor = (version / 100) % 100;
    if (major == 0) {
        major = version / 1000;
        minor = (version / 100) % 10;
    }
    auto patch = version % 100;

    return std::to_string(major) + "." + std::to_string(minor) + "." + std::to_string(patch);
}

inline std::string
get_backend_version_string() {
    return convert_version_to_str(get_backend_version());
}

inline cudnnStatus_t
create_descriptor(cudnnBackendDescriptorType_t descriptorType, cudnnBackendDescriptor_t *descriptor) {
    NV_FE_CALL_TO_BACKEND(create_descriptor, cudnnBackendCreateDescriptor, descriptorType, descriptor);
}

inline cudnnStatus_t
destroy_descriptor(cudnnBackendDescriptor_t descriptor) {
    NV_FE_CALL_TO_BACKEND(destroy_descriptor, cudnnBackendDestroyDescriptor, descriptor);
}

inline cudnnStatus_t
set_attribute(cudnnBackendDescriptor_t descriptor,
              cudnnBackendAttributeName_t attributeName,
              cudnnBackendAttributeType_t attributeType,
              int64_t elementCount,
              const void *arrayOfElements) {
    NV_FE_CALL_TO_BACKEND(set_attribute,
                          cudnnBackendSetAttribute,
                          descriptor,
                          attributeName,
                          attributeType,
                          elementCount,
                          arrayOfElements);
}

inline cudnnStatus_t
get_attribute(cudnnBackendDescriptor_t const descriptor,
              cudnnBackendAttributeName_t attributeName,
              cudnnBackendAttributeType_t attributeType,
              int64_t requestedElementCount,
              int64_t *elementCount,
              void *arrayOfElements) {
    NV_FE_CALL_TO_BACKEND(get_attribute,
                          cudnnBackendGetAttribute,
                          descriptor,
                          attributeName,
                          attributeType,
                          requestedElementCount,
                          elementCount,
                          arrayOfElements)
}

inline cudnnStatus_t
finalize(cudnnBackendDescriptor_t descriptor) {
    NV_FE_CALL_TO_BACKEND(finalize, cudnnBackendFinalize, descriptor);
}

inline cudnnStatus_t
execute(cudnnHandle_t handle, cudnnBackendDescriptor_t executionPlan, cudnnBackendDescriptor_t variantPack) {
    NV_FE_CALL_TO_BACKEND(execute, cudnnBackendExecute, handle, executionPlan, variantPack);
}

inline const char *
get_error_string(cudnnStatus_t status) {
    NV_FE_CALL_TO_BACKEND(get_error_string, cudnnGetErrorString, status);
}

inline void
get_last_error_string(char *message, size_t size) {
    if (detail::get_backend_version() > 90000 && detail::get_compiled_version() < 90000) {
#if CUDNN_VERSION >= 90000
        NV_FE_CALL_TO_BACKEND(get_last_error_string, cudnnGetLastErrorString, message, size);
#endif
    } else {
        std::string default_message = "Can't retrieve backend error messages for CUDNN version < 9.0";
        strncpy(message, default_message.c_str(), size - 1);
        message[size - 1] = '\0';  // Ensure null terminator at the end of the string
    }
}

inline std::string
get_last_error_string_() {
    const size_t size = 65535;

    std::string message;

    message.reserve(size);

    get_last_error_string(message.data(), size);

    return message;
}

inline cudnnStatus_t
set_stream(cudnnHandle_t handle, cudaStream_t stream) {
    NV_FE_CALL_TO_BACKEND(set_stream, cudnnSetStream, handle, stream);
}

inline cudnnStatus_t
get_stream(cudnnHandle_t handle, cudaStream_t *stream) {
    NV_FE_CALL_TO_BACKEND(get_stream, cudnnGetStream, handle, stream);
}

inline cudnnStatus_t
create_filter_desc_v7(cudnnFilterDescriptor_t *filter) {
    NV_FE_CALL_TO_BACKEND(create_filter_desc_v7, cudnnCreateFilterDescriptor, filter);
}

inline cudnnStatus_t
set_ndfilter_desc_v7(cudnnFilterDescriptor_t filter,
                     cudnnDataType_t type,
                     cudnnTensorFormat_t format,
                     int x,
                     const int filterDimA[]) {
    NV_FE_CALL_TO_BACKEND(set_ndfilter_desc_v7, cudnnSetFilterNdDescriptor, filter, type, format, x, filterDimA);
}

inline cudnnStatus_t
reorder_filter_bias(cudnnHandle_t handle,
                    const cudnnFilterDescriptor_t filterDesc,
                    cudnnReorderType_t reorderType,
                    const void *filterData,
                    void *reorderedFilterData,
                    int reorderBias,
                    const void *biasData,
                    void *reorderedBiasData) {
    NV_FE_CALL_TO_BACKEND(reorder_filter_bias,
                          cudnnReorderFilterAndBias,
                          handle,
                          filterDesc,
                          reorderType,
                          filterData,
                          reorderedFilterData,
                          reorderBias,
                          biasData,
                          reorderedBiasData);
}

inline cudnnStatus_t
destroy_filter(cudnnFilterDescriptor_t filter) {
    NV_FE_CALL_TO_BACKEND(destroy_filter, cudnnDestroyFilterDescriptor, filter);
}

}  // namespace detail
}  // namespace cudnn_frontend
