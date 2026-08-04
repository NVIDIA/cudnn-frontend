/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdexcept>
#include <utility>

#include "pybind11/pybind11.h"
#include "pybind11/cast.h"
#include "pybind11/stl.h"

#include "cudnn_frontend.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace cudnn_frontend {

#ifdef _WIN32
HMODULE cudnn_dlhandle = nullptr;
#else
void *cudnn_dlhandle = nullptr;
#endif

namespace python_bindings {

namespace {

constexpr int CAUSAL_CONV1D_MIN_KERNEL_SIZE           = 2;
constexpr int CAUSAL_CONV1D_MAX_KERNEL_SIZE           = 256;
constexpr int CAUSAL_CONV1D_NWH_MAX_KERNEL_SIZE       = 128;
constexpr int B2B_CAUSAL_CONV1D_PROJ_MAX_KERNEL_SIZE  = 32;
constexpr int B2B_CAUSAL_CONV1D_MIXER_MAX_KERNEL_SIZE = 256;

cudnnStatus_t
precheck_causal_conv1d_kernel_size(int const kernel_size, int const max_kernel_size) {
    return kernel_size > max_kernel_size ? CUDNN_STATUS_NOT_SUPPORTED : CUDNN_STATUS_SUCCESS;
}

bool
is_causal_conv1d_kernel_size_out_of_range(int const kernel_size, int const max_kernel_size) {
    return kernel_size < CAUSAL_CONV1D_MIN_KERNEL_SIZE || kernel_size > max_kernel_size;
}

[[noreturn]] void
throw_causal_conv1d_kernel_size_error(char const *api_name,
                                      char const *argument_name,
                                      int const kernel_size,
                                      int const max_kernel_size) {
    throw std::invalid_argument(std::string(api_name) + " " + argument_name + " must be between " +
                                std::to_string(CAUSAL_CONV1D_MIN_KERNEL_SIZE) + " and " +
                                std::to_string(max_kernel_size) + ", inclusive; got " + std::to_string(kernel_size));
}

void
throw_if_causal_conv1d_failed(cudnnStatus_t const status,
                              char const *operation,
                              char const *api_name,
                              int const kernel_size,
                              int const max_kernel_size) {
    if (status == CUDNN_STATUS_SUCCESS) return;

    if (is_causal_conv1d_kernel_size_out_of_range(kernel_size, max_kernel_size)) {
        throw_causal_conv1d_kernel_size_error(api_name, "kernel_size", kernel_size, max_kernel_size);
    }

    throw std::runtime_error(std::string(operation) + " failed with status " + std::to_string(status));
}

void
throw_if_b2b_causal_conv1d_failed(cudnnStatus_t const status,
                                  char const *operation,
                                  int const kernel_size_proj,
                                  int const kernel_size_mixer) {
    if (status == CUDNN_STATUS_SUCCESS) return;

    if (is_causal_conv1d_kernel_size_out_of_range(kernel_size_proj, B2B_CAUSAL_CONV1D_PROJ_MAX_KERNEL_SIZE)) {
        throw_causal_conv1d_kernel_size_error(
            "b2b_causal_conv1d", "kernel_size_proj", kernel_size_proj, B2B_CAUSAL_CONV1D_PROJ_MAX_KERNEL_SIZE);
    }
    if (is_causal_conv1d_kernel_size_out_of_range(kernel_size_mixer, B2B_CAUSAL_CONV1D_MIXER_MAX_KERNEL_SIZE)) {
        throw_causal_conv1d_kernel_size_error(
            "b2b_causal_conv1d", "kernel_size_mixer", kernel_size_mixer, B2B_CAUSAL_CONV1D_MIXER_MAX_KERNEL_SIZE);
    }

    throw std::runtime_error(std::string(operation) + " failed with status " + std::to_string(status));
}

}  // namespace

// Raise C++ exceptions corresponding to C++ FE error codes.
// Pybinds will automatically convert C++ exceptions to python exceptions.
void
throw_if(bool const cond, cudnn_frontend::error_code_t const error_code, std::string const &error_msg) {
    if (cond == false) return;

    switch (error_code) {
        case cudnn_frontend::error_code_t::OK:
            return;
        case cudnn_frontend::error_code_t::ATTRIBUTE_NOT_SET:
            throw std::invalid_argument(error_msg);
        case cudnn_frontend::error_code_t::SHAPE_DEDUCTION_FAILED:
            throw std::invalid_argument(error_msg);
        case cudnn_frontend::error_code_t::INVALID_TENSOR_NAME:
            throw std::invalid_argument(error_msg);
        case cudnn_frontend::error_code_t::INVALID_VARIANT_PACK:
            throw std::invalid_argument(error_msg);
        case cudnn_frontend::error_code_t::GRAPH_EXECUTION_PLAN_CREATION_FAILED:
            throw cudnn_frontend::cudnnGraphNotSupportedException(error_msg.c_str());
        case cudnn_frontend::error_code_t::GRAPH_EXECUTION_FAILED:
            throw std::runtime_error(error_msg);
        case cudnn_frontend::error_code_t::HEURISTIC_QUERY_FAILED:
            throw cudnn_frontend::cudnnGraphNotSupportedException(error_msg.c_str());
        case cudnn_frontend::error_code_t::CUDNN_BACKEND_API_FAILED:
            throw std::runtime_error(error_msg);
        case cudnn_frontend::error_code_t::CUDA_API_FAILED:
            throw std::runtime_error(error_msg);
        case cudnn_frontend::error_code_t::INVALID_CUDA_DEVICE:
            throw std::runtime_error(error_msg);
        case cudnn_frontend::error_code_t::UNSUPPORTED_GRAPH_FORMAT:
            throw cudnn_frontend::cudnnGraphNotSupportedException(error_msg.c_str());
        case cudnn_frontend::error_code_t::GRAPH_NOT_SUPPORTED:
            throw cudnn_frontend::cudnnGraphNotSupportedException(error_msg.c_str());
        case cudnn_frontend::error_code_t::HANDLE_ERROR:
            throw std::runtime_error(error_msg);
        case cudnn_frontend::error_code_t::INVALID_VALUE:
            throw std::runtime_error(error_msg);
        case cudnn_frontend::error_code_t::NVRTC_COMPILATION_FAILED:
            throw std::runtime_error(error_msg);
    }
}

// pybinds for pygraph class
void
init_pygraph_submodule(py::module_ &);

// pybinds for kernel_cache class
void
create_kernel_cache_submodule(py::module_ &);

// pybinds for all properties and helpers
void
init_properties(py::module_ &);

void
set_dlhandle_cudnn(std::intptr_t dlhandle) {
#ifdef _WIN32
    cudnn_dlhandle = reinterpret_cast<HMODULE>(dlhandle);
#else
    cudnn_dlhandle = reinterpret_cast<void *>(dlhandle);
#endif
}

PYBIND11_MODULE(_compiled_module, m) {
    m.def("backend_version", &detail::get_backend_version);
    m.def("backend_version_string", &detail::get_backend_version_string);

    init_properties(m);
    init_pygraph_submodule(m);

    m.def("_set_dlhandle_cudnn", &set_dlhandle_cudnn);

    py::register_exception<cudnnGraphNotSupportedException>(m, "cudnnGraphNotSupportedError");

#if CUDNN_VERSION >= 92200
    m.def("causal_conv1d_forward",
          [](std::intptr_t stream,
             std::intptr_t x_ptr,
             std::intptr_t weight_ptr,
             std::intptr_t bias_ptr,
             std::intptr_t out_ptr,
             int batch,
             int dim,
             int seq_len,
             int kernel_size,
             int data_type,
             int activation) {
              auto status = precheck_causal_conv1d_kernel_size(kernel_size, CAUSAL_CONV1D_MAX_KERNEL_SIZE);
              if (status == CUDNN_STATUS_SUCCESS) {
                  status = detail::causal_conv1d_forward(reinterpret_cast<cudaStream_t>(stream),
                                                         reinterpret_cast<const void *>(x_ptr),
                                                         reinterpret_cast<const void *>(weight_ptr),
                                                         reinterpret_cast<const void *>(bias_ptr),
                                                         reinterpret_cast<void *>(out_ptr),
                                                         batch,
                                                         dim,
                                                         seq_len,
                                                         kernel_size,
                                                         static_cast<cudnnDataType_t>(data_type),
                                                         static_cast<cudnnCausalConv1dActivation_t>(activation));
              }
              throw_if_causal_conv1d_failed(
                  status, "cudnnCausalConv1dForward", "causal_conv1d", kernel_size, CAUSAL_CONV1D_MAX_KERNEL_SIZE);
          });

    m.def("causal_conv1d_backward",
          [](std::intptr_t stream,
             std::intptr_t x_ptr,
             std::intptr_t weight_ptr,
             std::intptr_t bias_ptr,
             std::intptr_t dy_ptr,
             std::intptr_t dx_ptr,
             std::intptr_t dweight_ptr,
             std::intptr_t dbias_ptr,
             int batch,
             int dim,
             int seq_len,
             int kernel_size,
             int data_type,
             int dw_data_type,
             int activation) {
              auto status = precheck_causal_conv1d_kernel_size(kernel_size, CAUSAL_CONV1D_MAX_KERNEL_SIZE);
              if (status == CUDNN_STATUS_SUCCESS) {
                  status = detail::causal_conv1d_backward(reinterpret_cast<cudaStream_t>(stream),
                                                          reinterpret_cast<const void *>(x_ptr),
                                                          reinterpret_cast<const void *>(weight_ptr),
                                                          reinterpret_cast<const void *>(bias_ptr),
                                                          reinterpret_cast<const void *>(dy_ptr),
                                                          reinterpret_cast<void *>(dx_ptr),
                                                          reinterpret_cast<void *>(dweight_ptr),
                                                          reinterpret_cast<void *>(dbias_ptr),
                                                          batch,
                                                          dim,
                                                          seq_len,
                                                          kernel_size,
                                                          static_cast<cudnnDataType_t>(data_type),
                                                          static_cast<cudnnDataType_t>(dw_data_type),
                                                          static_cast<cudnnCausalConv1dActivation_t>(activation));
              }
              throw_if_causal_conv1d_failed(
                  status, "cudnnCausalConv1dBackward", "causal_conv1d", kernel_size, CAUSAL_CONV1D_MAX_KERNEL_SIZE);
          });

    m.def("causal_conv1d_nwh_forward",
          [](std::intptr_t stream,
             std::intptr_t x_ptr,
             std::intptr_t weight_ptr,
             std::intptr_t bias_ptr,
             std::intptr_t out_ptr,
             int batch,
             int dim,
             int seq_len,
             int kernel_size,
             int data_type,
             int activation) {
              auto status = precheck_causal_conv1d_kernel_size(kernel_size, CAUSAL_CONV1D_NWH_MAX_KERNEL_SIZE);
              if (status == CUDNN_STATUS_SUCCESS) {
                  status = detail::causal_conv1d_nwh_forward(reinterpret_cast<cudaStream_t>(stream),
                                                             reinterpret_cast<const void *>(x_ptr),
                                                             reinterpret_cast<const void *>(weight_ptr),
                                                             reinterpret_cast<const void *>(bias_ptr),
                                                             reinterpret_cast<void *>(out_ptr),
                                                             batch,
                                                             dim,
                                                             seq_len,
                                                             kernel_size,
                                                             static_cast<cudnnDataType_t>(data_type),
                                                             static_cast<cudnnCausalConv1dActivation_t>(activation));
              }
              throw_if_causal_conv1d_failed(status,
                                            "cudnnCausalConv1dNwhForward",
                                            "causal_conv1d_nwh",
                                            kernel_size,
                                            CAUSAL_CONV1D_NWH_MAX_KERNEL_SIZE);
          });

    m.def("causal_conv1d_nwh_backward",
          [](std::intptr_t stream,
             std::intptr_t x_ptr,
             std::intptr_t weight_ptr,
             std::intptr_t bias_ptr,
             std::intptr_t dy_ptr,
             std::intptr_t dx_ptr,
             std::intptr_t dweight_ptr,
             std::intptr_t dbias_ptr,
             int batch,
             int dim,
             int seq_len,
             int kernel_size,
             int data_type,
             int dw_data_type,
             int activation) {
              auto status = precheck_causal_conv1d_kernel_size(kernel_size, CAUSAL_CONV1D_NWH_MAX_KERNEL_SIZE);
              if (status == CUDNN_STATUS_SUCCESS) {
                  status = detail::causal_conv1d_nwh_backward(reinterpret_cast<cudaStream_t>(stream),
                                                              reinterpret_cast<const void *>(x_ptr),
                                                              reinterpret_cast<const void *>(weight_ptr),
                                                              reinterpret_cast<const void *>(bias_ptr),
                                                              reinterpret_cast<const void *>(dy_ptr),
                                                              reinterpret_cast<void *>(dx_ptr),
                                                              reinterpret_cast<void *>(dweight_ptr),
                                                              reinterpret_cast<void *>(dbias_ptr),
                                                              batch,
                                                              dim,
                                                              seq_len,
                                                              kernel_size,
                                                              static_cast<cudnnDataType_t>(data_type),
                                                              static_cast<cudnnDataType_t>(dw_data_type),
                                                              static_cast<cudnnCausalConv1dActivation_t>(activation));
              }
              throw_if_causal_conv1d_failed(status,
                                            "cudnnCausalConv1dNwhBackward",
                                            "causal_conv1d_nwh",
                                            kernel_size,
                                            CAUSAL_CONV1D_NWH_MAX_KERNEL_SIZE);
          });

    m.def(
        "b2b_causal_conv1d_forward",
        [](std::intptr_t stream,
           std::intptr_t x_ptr,
           std::intptr_t weights_proj_ptr,
           std::intptr_t weights_mixer_ptr,
           std::intptr_t skip_bias_ptr,
           std::intptr_t y_ptr,
           std::intptr_t y_gated_ptr,
           int batch,
           int dim,
           int seq_len,
           int kernel_size_proj,
           int kernel_size_mixer,
           int data_type) {
            auto status = precheck_causal_conv1d_kernel_size(kernel_size_proj, B2B_CAUSAL_CONV1D_PROJ_MAX_KERNEL_SIZE);
            if (status == CUDNN_STATUS_SUCCESS) {
                status = precheck_causal_conv1d_kernel_size(kernel_size_mixer, B2B_CAUSAL_CONV1D_MIXER_MAX_KERNEL_SIZE);
            }
            if (status == CUDNN_STATUS_SUCCESS) {
                status = detail::b2b_causal_conv1d_forward(reinterpret_cast<cudaStream_t>(stream),
                                                           reinterpret_cast<const void *>(x_ptr),
                                                           reinterpret_cast<const void *>(weights_proj_ptr),
                                                           reinterpret_cast<const void *>(weights_mixer_ptr),
                                                           reinterpret_cast<const void *>(skip_bias_ptr),
                                                           reinterpret_cast<void *>(y_ptr),
                                                           reinterpret_cast<void *>(y_gated_ptr),
                                                           batch,
                                                           dim,
                                                           seq_len,
                                                           kernel_size_proj,
                                                           kernel_size_mixer,
                                                           static_cast<cudnnDataType_t>(data_type));
            }
            throw_if_b2b_causal_conv1d_failed(
                status, "cudnnB2BCausalConv1dForward", kernel_size_proj, kernel_size_mixer);
        });

    m.def(
        "b2b_causal_conv1d_backward",
        [](std::intptr_t stream,
           std::intptr_t x_ptr,
           std::intptr_t weights_proj_ptr,
           std::intptr_t weights_mixer_ptr,
           std::intptr_t skip_bias_ptr,
           std::intptr_t y_ptr,
           std::intptr_t dy_ptr,
           std::intptr_t dx_ptr,
           std::intptr_t dweights_proj_ptr,
           std::intptr_t dweights_mixer_ptr,
           std::intptr_t dskip_bias_ptr,
           int batch,
           int dim,
           int seq_len,
           int kernel_size_proj,
           int kernel_size_mixer,
           int data_type,
           int dw_data_type) {
            auto status = precheck_causal_conv1d_kernel_size(kernel_size_proj, B2B_CAUSAL_CONV1D_PROJ_MAX_KERNEL_SIZE);
            if (status == CUDNN_STATUS_SUCCESS) {
                status = precheck_causal_conv1d_kernel_size(kernel_size_mixer, B2B_CAUSAL_CONV1D_MIXER_MAX_KERNEL_SIZE);
            }
            if (status == CUDNN_STATUS_SUCCESS) {
                status = detail::b2b_causal_conv1d_backward(reinterpret_cast<cudaStream_t>(stream),
                                                            reinterpret_cast<const void *>(x_ptr),
                                                            reinterpret_cast<const void *>(weights_proj_ptr),
                                                            reinterpret_cast<const void *>(weights_mixer_ptr),
                                                            reinterpret_cast<const void *>(skip_bias_ptr),
                                                            reinterpret_cast<const void *>(y_ptr),
                                                            reinterpret_cast<const void *>(dy_ptr),
                                                            reinterpret_cast<void *>(dx_ptr),
                                                            reinterpret_cast<void *>(dweights_proj_ptr),
                                                            reinterpret_cast<void *>(dweights_mixer_ptr),
                                                            reinterpret_cast<void *>(dskip_bias_ptr),
                                                            batch,
                                                            dim,
                                                            seq_len,
                                                            kernel_size_proj,
                                                            kernel_size_mixer,
                                                            static_cast<cudnnDataType_t>(data_type),
                                                            static_cast<cudnnDataType_t>(dw_data_type));
            }
            throw_if_b2b_causal_conv1d_failed(
                status, "cudnnB2BCausalConv1dBackward", kernel_size_proj, kernel_size_mixer);
        });
#endif
}

}  // namespace python_bindings
}  // namespace cudnn_frontend
