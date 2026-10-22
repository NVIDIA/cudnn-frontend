// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// The host-side CUDA entry points a CuTeDSL export_to_c object imports, so the library does not link the CuTeDSL
// runtime (libcute_dsl_runtime.so). Each one reproduces the CuTeDSL 4.8 runtime's wrapper of the same name, read from
// its disassembly (nvidia_cutlass_dsl/cu13/lib): the cuda* ones forward to cudart; the cu* ones resolve the driver
// function through cudaGetDriverEntryPointByVersion and return CUDA_ERROR_UNKNOWN when it cannot be resolved.
// aot/sm120/build.py --runtime cute-dsl links the original instead; test_sdpa_sm120 --compare-lib checks both agree.

#include <cuda.h>
#include <cuda_runtime.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace {

template <class Fn>
Fn
driver_entry(std::atomic<void*>& slot, const char* name, unsigned int version) {
    void* fn = slot.load(std::memory_order_acquire);
    if (!fn) {
        cudaDriverEntryPointQueryResult status = cudaDriverEntryPointSymbolNotFound;
        void* found                            = nullptr;
        if (cudaGetDriverEntryPointByVersion(name, &found, version, cudaEnableDefault, &status) != cudaSuccess ||
            status != cudaDriverEntryPointSuccess || !found)
            return nullptr;
        void* expected = nullptr;
        slot.compare_exchange_strong(expected, found, std::memory_order_acq_rel);
        fn = slot.load(std::memory_order_acquire);
    }
    return reinterpret_cast<Fn>(fn);
}

std::atomic<void*> g_cuDeviceGet{nullptr};
std::atomic<void*> g_cuKernelGetAttribute{nullptr};
std::atomic<void*> g_cuTensorMapEncodeTiled{nullptr};

// Bits per element of each CUtensorMapDataType (0 .. 15), the CuTeDSL runtime's table.
constexpr uint64_t kTensorMapElementBits[16] = {8, 16, 32, 32, 64, 64, 16, 32, 64, 16, 32, 32, 32, 4, 4, 6};

}  // namespace

extern "C" {

__attribute__((visibility("hidden"))) cudaError_t
_cudaGetDevice(int* device) {
    return cudaGetDevice(device);
}

__attribute__((visibility("hidden"))) cudaError_t
_cudaDeviceGetAttribute(int* value, cudaDeviceAttr attr, int device) {
    return cudaDeviceGetAttribute(value, attr, device);
}

__attribute__((visibility("hidden"))) cudaError_t
_cudaFuncSetAttribute(const void* func, cudaFuncAttribute attr, int value) {
    return cudaFuncSetAttribute(func, attr, value);
}

__attribute__((visibility("hidden"))) cudaError_t
_cudaKernelSetAttributeForDevice(cudaKernel_t kernel, cudaFuncAttribute attr, int value, int device) {
    return cudaKernelSetAttributeForDevice(kernel, attr, value, device);
}

__attribute__((visibility("hidden"))) cudaError_t
_cudaLaunchKernelEx(const cudaLaunchConfig_t* config, const void* func, void** args) {
    return cudaLaunchKernelExC(config, func, args);
}

__attribute__((visibility("hidden"))) cudaError_t
_cudaLibraryLoadData(cudaLibrary_t* library,
                     const void* code,
                     cudaJitOption* jit_options,
                     void** jit_option_values,
                     unsigned int num_jit_options,
                     cudaLibraryOption* library_options,
                     void** library_option_values,
                     unsigned int num_library_options) {
    return cudaLibraryLoadData(library,
                               code,
                               jit_options,
                               jit_option_values,
                               num_jit_options,
                               library_options,
                               library_option_values,
                               num_library_options);
}

__attribute__((visibility("hidden"))) cudaError_t
_cudaLibraryGetKernel(cudaKernel_t* kernel, cudaLibrary_t library, const char* name) {
    return cudaLibraryGetKernel(kernel, library, name);
}

// The last argument is a device ORDINAL, mapped to a CUdevice with cuDeviceGet first.
__attribute__((visibility("hidden"))) CUresult
_cuKernelGetAttribute(int* value, CUfunction_attribute attr, CUkernel kernel, int device_ordinal) {
    using DeviceGet    = CUresult (*)(CUdevice*, int);
    using GetAttribute = CUresult (*)(int*, CUfunction_attribute, CUkernel, CUdevice);
    auto device_get    = driver_entry<DeviceGet>(g_cuDeviceGet, "cuDeviceGet", 12080);
    if (!device_get) return CUDA_ERROR_UNKNOWN;
    CUdevice device = 0;
    if (device_get(&device, device_ordinal) != CUDA_SUCCESS) return CUDA_ERROR_UNKNOWN;
    auto get_attribute = driver_entry<GetAttribute>(g_cuKernelGetAttribute, "cuKernelGetAttribute", 12080);
    if (!get_attribute) return CUDA_ERROR_UNKNOWN;
    return get_attribute(value, attr, kernel, device);
}

// After a successful encode on a driver of CUDA 13.1 or older, a tensor whose addressed span is at most 0x1ffff bytes
// gets bit 21 of the descriptor's second 64-bit word cleared, as the CuTeDSL runtime does.
__attribute__((visibility("hidden"))) CUresult
_cuTensorMapEncodeTiled(CUtensorMap* map,
                        CUtensorMapDataType dtype,
                        cuuint32_t rank,
                        void* address,
                        const cuuint64_t* dims,
                        const cuuint64_t* strides,
                        const cuuint32_t* box,
                        const cuuint32_t* element_strides,
                        CUtensorMapInterleave interleave,
                        CUtensorMapSwizzle swizzle,
                        CUtensorMapL2promotion l2_promotion,
                        CUtensorMapFloatOOBfill oob_fill) {
    using Encode = CUresult (*)(CUtensorMap*,
                                CUtensorMapDataType,
                                cuuint32_t,
                                void*,
                                const cuuint64_t*,
                                const cuuint64_t*,
                                const cuuint32_t*,
                                const cuuint32_t*,
                                CUtensorMapInterleave,
                                CUtensorMapSwizzle,
                                CUtensorMapL2promotion,
                                CUtensorMapFloatOOBfill);
    auto encode  = driver_entry<Encode>(g_cuTensorMapEncodeTiled, "cuTensorMapEncodeTiled", 12000);
    if (!encode) return CUDA_ERROR_UNKNOWN;
    const CUresult res = encode(
        map, dtype, rank, address, dims, strides, box, element_strides, interleave, swizzle, l2_promotion, oob_fill);
    if (res != CUDA_SUCCESS) return res;
    int driver = 0;
    if (const cudaError_t err = cudaDriverGetVersion(&driver)) return static_cast<CUresult>(err);
    if (driver > 13010 || static_cast<unsigned>(dtype) > 15u) return CUDA_SUCCESS;
    const uint64_t bits = kTensorMapElementBits[static_cast<unsigned>(dtype)];
    if (rank != 0) {
        uint64_t span = (bits * dims[0] + 7) >> 3;
        for (cuuint32_t i = 1; i < rank; ++i) span += (dims[i] - 1) * strides[i - 1];
        if (span > 0x1ffff) return CUDA_SUCCESS;
    }
    reinterpret_cast<uint64_t*>(map)[1] &= ~(uint64_t{1} << 21);
    return CUDA_SUCCESS;
}

__attribute__((visibility("hidden"))) void
cuda_dialect_assert_success(int err) {
    if (err == 0) return;
    const cudaError_t e = static_cast<cudaError_t>(err);
    std::fprintf(
        stderr, "CUDA Dialect Assertion failed on CUDA error: %s - %s\n", cudaGetErrorName(e), cudaGetErrorString(e));
    std::abort();
}

}  // extern "C"
