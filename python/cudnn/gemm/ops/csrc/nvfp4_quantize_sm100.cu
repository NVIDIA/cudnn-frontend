/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cuda_runtime.h>
#include <pybind11/pybind11.h>

#include <cstdint>
#include <stdexcept>
#include <string>

#include "nvfp4_smooth_quantize_sm100.cuh"

namespace {

void
launch(std::uintptr_t x,
       std::uintptr_t pre_quant_scale,
       std::uintptr_t global_scale,
       std::uintptr_t output,
       std::uintptr_t scale_factors,
       int m,
       int k,
       int multiprocessor_count,
       std::uintptr_t stream,
       bool enable_pdl) {
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    flashinfer::gemm::nvfp4_smooth_quantize(reinterpret_cast<void*>(output),
                                            reinterpret_cast<void*>(scale_factors),
                                            reinterpret_cast<void const*>(x),
                                            reinterpret_cast<void const*>(pre_quant_scale),
                                            reinterpret_cast<float const*>(global_scale),
                                            m,
                                            k,
                                            multiprocessor_count,
                                            cuda_stream,
                                            enable_pdl);

    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("nvfp4 quantize kernel launch failed: ") + cudaGetErrorString(status));
    }
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("launch",
               &launch,
               pybind11::arg("x"),
               pybind11::arg("pre_quant_scale"),
               pybind11::arg("global_scale"),
               pybind11::arg("output"),
               pybind11::arg("scale_factors"),
               pybind11::arg("m"),
               pybind11::arg("k"),
               pybind11::arg("multiprocessor_count"),
               pybind11::arg("stream"),
               pybind11::arg("enable_pdl"));
}
