// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cuda_runtime.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>

namespace {
__global__ void
validate_prefix(const int* cu, int sequences, int tokens) {
    if (threadIdx.x == 0 && (cu[0] != 0 || cu[sequences] != tokens)) {
        asm volatile("trap;");
    }
    for (int64_t i = threadIdx.x; i < sequences; i += blockDim.x) {
        if (cu[i] < 0 || cu[i + 1] <= cu[i] || cu[i + 1] > tokens) {
            asm volatile("trap;");
        }
    }
}

void
validate(tvm::ffi::TensorView cu, int64_t tokens) {
    TVM_FFI_CHECK(cu.device().device_type == kDLCUDA && cu.ndim() == 1 && cu.size(0) >= 2 && cu.IsContiguous(),
                  ValueError)
        << "cu_seqlens must be a contiguous CUDA vector of at least two elements";
    TVM_FFI_CHECK(cu.dtype().code == kDLInt && cu.dtype().bits == 32 && cu.dtype().lanes == 1, ValueError)
        << "cu_seqlens must be int32";
    TVM_FFI_CHECK(tokens > 0 && tokens <= INT32_MAX && cu.size(0) - 1 <= INT32_MAX, ValueError)
        << "tokens and sequence count must fit signed int32";
    auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, cu.device().device_id));
    auto data   = reinterpret_cast<const int*>(static_cast<const char*>(cu.data_ptr()) + cu.byte_offset());
    validate_prefix<<<1, 256, 0, stream>>>(data, static_cast<int>(cu.size(0) - 1), static_cast<int>(tokens));
    auto status = cudaGetLastError();
    TVM_FFI_CHECK(status == cudaSuccess, RuntimeError) << cudaGetErrorString(status);
}

int64_t
build_cuda_version() {
    return CUDART_VERSION;
}
}  // namespace

TVM_FFI_DLL_EXPORT_TYPED_FUNC(validate, validate);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(build_cuda_version, build_cuda_version);
