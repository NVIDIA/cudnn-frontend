/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Host surface for the generated Cake launchers (``*_binding.cu``) on TVM-FFI: a read-only tensor
// view, argument checks that raise Python exceptions, the current CUDA stream and the module export.
// The launchers use only the names below. tools/vendor_cake.py writes this file into every vendored
// package of generated miles sources in place of their torch-extension host surface, so those
// launchers build with nvcc and load with tvm-ffi like every other vendored package.
#pragma once

#include <cuda_runtime_api.h>
#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/dtype.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/extra/cuda/device_guard.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/optional.h>

#include <cstdint>

namespace cake_host {

using TensorView = tvm::ffi::TensorView;
using CUDADeviceGuard = tvm::ffi::CUDADeviceGuard;
template <class T>
using Optional = tvm::ffi::Optional<T>;

// The stream the caller made current on the device; the launchers enqueue on it.
inline void* CurrentStream(int device_type, int device_id) { return TVMFFIEnvGetStream(device_type, device_id); }

// The CUDA runtime this library was compiled against (``CUDART_VERSION``), exported as
// ``build_cuda_version`` so the loader can compare it with the toolkit before the first launch.
inline int64_t BuildCudaVersion() { return CUDART_VERSION; }

}  // namespace cake_host

#define CAKE_HOST_CHECK(cond, Kind) \
  if (cond) {                       \
  } else                            \
    TVM_FFI_THROW(Kind)

#define CAKE_HOST_CHECK_CUDA(expr)                                                         \
  if (const cudaError_t cake_host_cuda_err = (expr); cake_host_cuda_err == cudaSuccess) { \
  } else                                                                                   \
    TVM_FFI_THROW(RuntimeError) << cudaGetErrorString(cake_host_cuda_err)

#define CAKE_HOST_EXPORT(name, fn)         \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(name, fn); \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(build_cuda_version, ::cake_host::BuildCudaVersion)

// Named tensor-argument checks, the same functions as the torch-extension host surface.
inline void check_cuda_tensor(const cake_host::TensorView& t, const char* name) {
  CAKE_HOST_CHECK(t.device().device_type == kDLCUDA, ValueError)
      << name << " must be a CUDA tensor, got device_type=" << static_cast<int>(t.device().device_type);
}

inline void check_dtype(const cake_host::TensorView& t, DLDataType expected, const char* name) {
  DLDataType d = t.dtype();
  CAKE_HOST_CHECK(d.code == expected.code && d.bits == expected.bits && d.lanes == expected.lanes, TypeError)
      << name << " dtype mismatch: expected DLDataType(code=" << static_cast<int>(expected.code)
      << ", bits=" << static_cast<int>(expected.bits) << ", lanes=" << static_cast<int>(expected.lanes)
      << "), got (code=" << static_cast<int>(d.code) << ", bits=" << static_cast<int>(d.bits)
      << ", lanes=" << static_cast<int>(d.lanes) << ")";
}

inline void check_contiguous(const cake_host::TensorView& t, const char* name) {
  CAKE_HOST_CHECK(t.IsContiguous(), ValueError) << name << " must be contiguous";
}

inline void check_same_device(const cake_host::TensorView& t, const cake_host::TensorView& reference,
                              const char* name, const char* reference_name) {
  CAKE_HOST_CHECK(t.device().device_type == reference.device().device_type &&
                      t.device().device_id == reference.device().device_id,
                  ValueError)
      << name << " must be on the same device as " << reference_name << ": got device_type="
      << static_cast<int>(t.device().device_type) << " id=" << t.device().device_id
      << " versus device_type=" << static_cast<int>(reference.device().device_type)
      << " id=" << reference.device().device_id;
}

inline void check_dense_leading_dims(const cake_host::TensorView& t, int trailing, const char* name) {
  CAKE_HOST_CHECK(trailing > 0 && t.ndim() >= trailing, ValueError)
      << name << " cannot fold leading dimensions above " << trailing << " trailing dims from ndim=" << t.ndim();
  int outer_last = t.ndim() - trailing - 1;
  if (outer_last <= 0) {
    return;
  }
  int64_t step = t.stride(outer_last);
  CAKE_HOST_CHECK(step > 0, ValueError) << name << " physical strides must be positive";
  int64_t expected = step;
  for (int axis = outer_last - 1; axis >= 0; --axis) {
    expected *= t.size(axis + 1);
    if (t.size(axis) > 1) {
      CAKE_HOST_CHECK(t.stride(axis) == expected, ValueError)
          << name << " leading dims are not physically foldable above " << trailing
          << " trailing dims: stride(" << axis << ")=" << t.stride(axis) << ", expected " << expected;
    }
  }
}
