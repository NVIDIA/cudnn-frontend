/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <optional>
#include <utility>
#include <unordered_map>
#include <vector>

#include "dlpack/dlpack.h"

// Part of the Array API specification.
#define CUDNN_FRONTEND_DLPACK_CAPSULE_NAME "dltensor"
#define CUDNN_FRONTEND_DLPACK_USED_CAPSULE_NAME "used_dltensor"

#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/vector.h>

#include "cudnn_frontend.h"
#include "pygraph.h"
#include "variant_pack.h"

namespace py = nanobind;

namespace cudnn_frontend::python_bindings {

void
throw_if(bool const cond, cudnn_frontend::error_code_t const error_code, std::string const& error_msg);

void
init_pygraph_norm_submodule(PyGraphClass&);

void
init_pygraph_sdpa_submodule(PyGraphClass&);

void
init_pygraph_pointwise_submodule(PyGraphClass&);

void
init_pygraph_custom_reduction_submodule(PyGraphClass&);

namespace {

int
pygraph_traverse(PyObject* self, visitproc visit, void* arg) {
    Py_VISIT(Py_TYPE(self));
    if (!py::inst_ready(self)) {
        return 0;
    }
    auto* graph = py::inst_ptr<PyGraph>(self);
    Py_VISIT(graph->callback_gc_owner.ptr());
    for (auto const& weak : graph->custom_callback_states) {
        if (auto state = weak.lock()) {
            Py_VISIT(state->reducer.ptr());
            Py_VISIT(state->owner_weakref.ptr());
        }
    }
    for (auto const* callback : {&graph->callback_fn, &graph->callback_fn_bprop}) {
        if (callback->has_value()) {
            py::handle function = py::find(callback->value());
            Py_VISIT(function.ptr());
        }
    }
    return 0;
}

int
pygraph_clear(PyObject* self) {
    if (!py::inst_ready(self)) {
        return 0;
    }
    auto* graph = py::inst_ptr<PyGraph>(self);
    // Detach every member before decref can run a finalizer that reenters the graph.
    decltype(graph->custom_callback_states) states;
    states.swap(graph->custom_callback_states);
    std::optional<PyGraph::PyCallback> callback;
    callback.swap(graph->callback_fn);
    std::optional<PyGraph::PyCallback> callback_bprop;
    callback_bprop.swap(graph->callback_fn_bprop);
    auto owner = std::move(graph->callback_gc_owner);
    for (auto const& weak : states) {
        if (auto state = weak.lock()) {
            auto reducer       = std::move(state->reducer);
            auto owner_weakref = std::move(state->owner_weakref);
        }
    }
    return 0;
}

PyType_Slot pygraph_gc_slots[] = {{Py_tp_traverse, reinterpret_cast<void*>(pygraph_traverse)},
                                  {Py_tp_clear, reinterpret_cast<void*>(pygraph_clear)},
                                  {0, nullptr}};

}  // namespace

cudnn_frontend::DataType_t
convert_to_cudnn_data_type(const DLDataType& dtype) {
    switch (dtype.code) {
        case DLDataTypeCode::kDLUInt:
            switch (dtype.bits) {
                case 8:
                    return cudnn_frontend::DataType_t::UINT8;
            }
            break;
        case DLDataTypeCode::kDLInt:
            switch (dtype.bits) {
                case 8:
                    return cudnn_frontend::DataType_t::INT8;
                case 32:
                    return cudnn_frontend::DataType_t::INT32;
                case 64:
                    return cudnn_frontend::DataType_t::INT64;
            }
            break;
        case DLDataTypeCode::kDLFloat:
            switch (dtype.bits) {
                case 16:
                    return cudnn_frontend::DataType_t::HALF;
                case 32:
                    return cudnn_frontend::DataType_t::FLOAT;
                case 64:
                    return cudnn_frontend::DataType_t::DOUBLE;
            }
            break;
        case DLDataTypeCode::kDLBfloat:
            switch (dtype.bits) {
                case 16:
                    return cudnn_frontend::DataType_t::BFLOAT16;
            }
            break;
        case DLDataTypeCode::kDLBool:
            switch (dtype.bits) {
                case 8:
                    return cudnn_frontend::DataType_t::BOOLEAN;
            }
            break;
    }
    return cudnn_frontend::DataType_t::NOT_SET;
}

std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>
PyGraph::tensor(std::vector<int64_t> const& dim,
                std::vector<int64_t> const& stride,
                cudnn_frontend::DataType_t const& data_type,
                bool const& is_virtual,
                bool const& is_pass_by_value,
                std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> const& ragged_offset,
                cudnn_frontend::TensorReordering_t const reordering_type,
                std::string const& name,
                int64_t const& uid,
                int64_t const& ragged_offset_multiplier) {
    auto props = cudnn_frontend::graph::Tensor_attributes()
                     .set_data_type(data_type)
                     .set_is_virtual(is_virtual)
                     .set_is_pass_by_value(is_pass_by_value)
                     .set_dim(dim)
                     .set_stride(stride)
                     .set_ragged_offset(ragged_offset)
                     .set_reordering_type(reordering_type)
                     .set_name(name)
                     .set_ragged_offset_multiplier(ragged_offset_multiplier);

    if (uid != -1) {
        props.set_uid(uid);
    }

    return graph->tensor(props);
}

std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>
PyGraph::tensor_like(std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> const& tensor, std::string const& name) {
    return graph->tensor_like(tensor, name);
}

static std::intptr_t
extract_data_pointer(py::object const& obj) {
    throw_if(!py::hasattr(obj, "__dlpack__"),
             cudnn_frontend::error_code_t::INVALID_VARIANT_PACK,
             "Object does not have the __dlpack__() method");

    py::capsule capsule = py::cast<py::capsule>(obj.attr("__dlpack__")());
    throw_if(capsule.is_none(),
             cudnn_frontend::error_code_t::INVALID_VARIANT_PACK,
             "Failed to retrieve the DLPack capsule.");

    DLManagedTensor* managed =
        static_cast<DLManagedTensor*>(PyCapsule_GetPointer(capsule.ptr(), CUDNN_FRONTEND_DLPACK_CAPSULE_NAME));
    throw_if(managed == nullptr, cudnn_frontend::error_code_t::INVALID_VARIANT_PACK, "Invalid DLPack capsule.");

    DLDeviceType device_type = managed->dl_tensor.device.device_type;
    throw_if(
        device_type != kDLCPU && device_type != kDLCUDAHost && device_type != kDLCUDA && device_type != kDLCUDAManaged,
        cudnn_frontend::error_code_t::INVALID_VARIANT_PACK,
        "Invalid device type.");

    void* p     = (char*)managed->dl_tensor.data + managed->dl_tensor.byte_offset;
    auto result = reinterpret_cast<std::intptr_t>(p);
    return result;
}

std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>
PyGraph::tensor_like(py::object const& pyobj) {
    throw_if(!py::hasattr(pyobj, "__dlpack__"),
             cudnn_frontend::error_code_t::INVALID_VARIANT_PACK,
             "Object does not have the __dlpack__() method");

    py::capsule capsule = py::cast<py::capsule>(pyobj.attr("__dlpack__")());
    throw_if(capsule.is_none(),
             cudnn_frontend::error_code_t::INVALID_VARIANT_PACK,
             "Failed to retrieve the DLPack capsule.");

    DLManagedTensor* managed =
        static_cast<DLManagedTensor*>(PyCapsule_GetPointer(capsule.ptr(), CUDNN_FRONTEND_DLPACK_CAPSULE_NAME));
    throw_if(managed == nullptr, cudnn_frontend::error_code_t::INVALID_VARIANT_PACK, "Invalid DLPack capsule.");

    DLDeviceType device_type = managed->dl_tensor.device.device_type;
    throw_if(
        device_type != kDLCPU && device_type != kDLCUDAHost && device_type != kDLCUDA && device_type != kDLCUDAManaged,
        cudnn_frontend::error_code_t::INVALID_VARIANT_PACK,
        "Invalid device type.");

    auto ndim = managed->dl_tensor.ndim;
    std::vector<int64_t> dim(managed->dl_tensor.shape, managed->dl_tensor.shape + ndim);

    auto props = cudnn_frontend::graph::Tensor_attributes()
                     .set_data_type(convert_to_cudnn_data_type(managed->dl_tensor.dtype))
                     .set_is_virtual(false)
                     .set_is_pass_by_value(managed->dl_tensor.device.device_type == kDLCPU)
                     .set_dim(dim);

    if (managed->dl_tensor.strides == nullptr) {
        // dlpack says "can be NULL, indicating tensor is compact and row-majored"
        auto stride_order = detail::generate_row_major_stride_order(ndim);
        props.set_stride(detail::generate_stride(dim, stride_order));
    } else {
        std::vector<int64_t> stride(managed->dl_tensor.strides, managed->dl_tensor.strides + ndim);
        props.set_stride(stride);
    }

    return graph->tensor(props);
}

std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>
PyGraph::slice(std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& input,
               std::vector<py::slice> const& slices,
               cudnn_frontend::DataType_t const& compute_data_type,
               std::string const& name) {
    auto input_dim = input->get_dim();

    std::vector<std::pair<int64_t, int64_t>> start_end_indices;
    std::vector<int64_t> steps;
    steps.reserve(slices.size());
    for (size_t i = 0; i < slices.size(); ++i) {
        auto [start, stop, step, length] = slices[i].compute(input_dim[i]);
        CUDNN_FRONTEND_UNUSED(length);
        start_end_indices.push_back({start, stop});
        steps.push_back(step);
    }

    auto attributes = cudnn_frontend::graph::Slice_attributes()
                          .set_slices(start_end_indices)
                          .set_strides(steps)
                          .set_compute_data_type(compute_data_type)
                          .set_name(name);

    auto output = graph->slice(input, attributes);
    return output;
}

std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>
PyGraph::conv_fprop(std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& image,
                    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& weight,
                    std::vector<int64_t> const& pre_padding,
                    std::vector<int64_t> const& post_padding,
                    std::vector<int64_t> const& stride,
                    std::vector<int64_t> const& dilation,
                    cudnn_frontend::ConvolutionMode_t const& conv_mode,
                    cudnn_frontend::DataType_t const& compute_data_type,
                    std::string const& name) {
    auto attributes = cudnn_frontend::graph::Conv_fprop_attributes()
                          .set_pre_padding(pre_padding)
                          .set_post_padding(post_padding)
                          .set_stride(stride)
                          .set_dilation(dilation)
                          .set_convolution_mode(conv_mode)
                          .set_compute_data_type(compute_data_type)
                          .set_name(name);

    auto Y = graph->conv_fprop(image, weight, attributes);
    return Y;
}

std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>
PyGraph::conv_dgrad(std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& loss,
                    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& filter,
                    std::vector<int64_t> const& pre_padding,
                    std::vector<int64_t> const& post_padding,
                    std::vector<int64_t> const& stride,
                    std::vector<int64_t> const& dilation,
                    cudnn_frontend::ConvolutionMode_t const& conv_mode,
                    cudnn_frontend::DataType_t const& compute_data_type,
                    std::string const& name) {
    auto attributes = cudnn_frontend::graph::Conv_dgrad_attributes()
                          .set_pre_padding(pre_padding)
                          .set_post_padding(post_padding)
                          .set_stride(stride)
                          .set_dilation(dilation)
                          .set_convolution_mode(conv_mode)
                          .set_compute_data_type(compute_data_type)
                          .set_name(name);
    auto DX = graph->conv_dgrad(loss, filter, attributes);
    return DX;
}

std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>
PyGraph::conv_wgrad(std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& image,
                    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& loss,
                    std::vector<int64_t> const& pre_padding,
                    std::vector<int64_t> const& post_padding,
                    std::vector<int64_t> const& stride,
                    std::vector<int64_t> const& dilation,
                    cudnn_frontend::ConvolutionMode_t const& conv_mode,
                    cudnn_frontend::DataType_t const& compute_data_type,
                    std::string const& name) {
    auto attributes = cudnn_frontend::graph::Conv_wgrad_attributes()
                          .set_pre_padding(pre_padding)
                          .set_post_padding(post_padding)
                          .set_stride(stride)
                          .set_dilation(dilation)
                          .set_convolution_mode(conv_mode)
                          .set_compute_data_type(compute_data_type)
                          .set_name(name);
    auto DW = graph->conv_wgrad(loss, image, attributes);
    return DW;
}

std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>
PyGraph::block_scale_dequantize(std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& input,
                                std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& descale,
                                std::vector<int32_t> const& block_size,
                                bool const is_negative_scale,
                                cudnn_frontend::DataType_t const& compute_data_type,
                                std::string const& name) {
    auto attributes = cudnn_frontend::graph::Block_scale_dequantize_attributes()
                          .set_block_size(block_size)
                          .set_is_negative_scale(is_negative_scale)
                          .set_compute_data_type(compute_data_type)
                          .set_name(name);
    if (compute_data_type != cudnn_frontend::DataType_t::NOT_SET) {
        attributes.set_compute_data_type(compute_data_type);
    }
    auto output = graph->block_scale_dequantize(input, descale, attributes);
    return output;
}

std::array<std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>, 2>
PyGraph::block_scale_quantize(std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& input,
                              int32_t block_size,
                              std::optional<int64_t> axis,
                              bool transpose,
                              cudnn_frontend::DataType_t const& compute_data_type,
                              std::string const& name) {
    auto attributes = cudnn_frontend::graph::Block_scale_quantize_attributes()
                          .set_block_size(block_size)
                          .set_transpose(transpose)
                          .set_name(name);
    if (axis.has_value()) {
        attributes.set_axis(axis.value());
    }
    if (compute_data_type != cudnn_frontend::DataType_t::NOT_SET) {
        attributes.set_compute_data_type(compute_data_type);
    }
    auto outputs = graph->block_scale_quantize(input, attributes);
    return outputs;
}

std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>
PyGraph::matmul(std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& A,
                std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& B,
                cudnn_frontend::DataType_t const& compute_data_type,
                double const padding,
                std::string const& name) {
    auto attributes = cudnn_frontend::graph::Matmul_attributes()
                          .set_compute_data_type(compute_data_type)
                          .set_name(name)
                          .set_padding(padding);

    auto C = graph->matmul(A, B, attributes);
    return C;
}

std::array<std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>, 2UL>
PyGraph::genstats(std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& input,
                  cudnn_frontend::DataType_t const& compute_data_type,
                  std::string const& name) {
    auto attributes =
        cudnn_frontend::graph::Genstats_attributes().set_compute_data_type(compute_data_type).set_name(name);

    auto [SUM, SQ_SUM] = graph->genstats(input, attributes);
    return {SUM, SQ_SUM};
}

std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>
PyGraph::reduction(std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& input,
                   cudnn_frontend::ReductionMode_t const mode,
                   cudnn_frontend::DataType_t const& compute_data_type,
                   std::string const& name,
                   std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> group_offset) {
    auto attributes = cudnn_frontend::graph::Reduction_attributes()
                          .set_mode(mode)
                          .set_compute_data_type(compute_data_type)
                          .set_name(name);
    if (group_offset != nullptr) {
        attributes.set_group_offset(std::move(group_offset));
    }

    auto OUT_0 = graph->reduction(input, attributes);
    return OUT_0;
}

std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>
PyGraph::reshape(std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& input,
                 std::string const& name,
                 cudnn_frontend::ReshapeMode_t reshape_mode) {
    auto attributes = cudnn_frontend::graph::Reshape_attributes().set_name(name).set_reshape_mode(reshape_mode);

    auto OUT_0 = graph->reshape(input, attributes);
    return OUT_0;
}

std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>
PyGraph::transpose(std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& input,
                   std::vector<int64_t> const& permutation,
                   cudnn_frontend::DataType_t const& compute_data_type,
                   std::string const& name) {
    auto attributes = cudnn_frontend::graph::Transpose_attributes()
                          .set_name(name)
                          .set_permutation(permutation)
                          .set_compute_data_type(compute_data_type);

    return graph->transpose(input, attributes);
}

std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>
PyGraph::concatenate(std::vector<std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>> inputs,
                     int64_t axis,
                     std::optional<int64_t> in_place_index,
                     std::string const& name) {
    auto attributes = cudnn_frontend::graph::Concatenate_attributes().set_axis(axis).set_name(name);
    if (in_place_index.has_value()) {
        attributes.set_in_place_index(in_place_index.value());
    }
    return graph->concatenate(std::move(inputs), attributes);
}

std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>
PyGraph::tensor_scalar(float const& value, cudnn_frontend::graph::ScalarType scalar_type) {
    return graph->tensor(value, scalar_type);
}

std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>
PyGraph::tensor_scalar(double const& value, cudnn_frontend::graph::ScalarType scalar_type) {
    return graph->tensor(value, scalar_type);
}

std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>
PyGraph::tensor_scalar(int32_t const& value, cudnn_frontend::graph::ScalarType scalar_type) {
    return graph->tensor(value, scalar_type);
}

std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>
PyGraph::tensor_scalar(int64_t const& value, cudnn_frontend::graph::ScalarType scalar_type) {
    return graph->tensor(value, scalar_type);
}

std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>
PyGraph::moe_grouped_matmul(std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& token,
                            std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& weight,
                            std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& first_token_offset,
                            std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& token_index,
                            std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& token_ks,
                            cudnn_frontend::MoeGroupedMatmulMode_t const& mode,
                            cudnn_frontend::DataType_t const& compute_data_type,
                            int32_t const& top_k,
                            std::string const& name,
                            std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> top_k_scores) {
    auto attributes = cudnn_frontend::graph::Moe_grouped_matmul_attributes()
                          .set_name(name)
                          .set_mode(mode)
                          .set_compute_data_type(compute_data_type)
                          .set_top_k(top_k)
                          .set_top_k_scores(std::move(top_k_scores));

    auto output = graph->moe_grouped_matmul(token, weight, first_token_offset, token_index, token_ks, attributes);
    return output;
}

std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>
PyGraph::moe_grouped_matmul_bwd(std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& doutput,
                                std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& token,
                                std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& first_token_offset,
                                cudnn_frontend::DataType_t const& compute_data_type,
                                std::string const& name) {
    auto attributes = cudnn_frontend::graph::Moe_grouped_matmul_bwd_attributes().set_name(name).set_compute_data_type(
        compute_data_type);

    auto dweight = graph->moe_grouped_matmul_bwd(doutput, token, first_token_offset, attributes);
    return dweight;
}

void
PyGraph::validate() {
    auto status = graph->validate();
    throw_if(status.is_bad(), status.get_code(), status.get_message());
}

size_t
PyGraph::key() {
    return graph->key();
}

void
PyGraph::build_operation_graph() {
    auto status = graph->build_operation_graph(handle);
    throw_if(status.is_bad(), status.get_code(), status.get_message());
}

std::vector<BehaviorNote_t>
PyGraph::get_behavior_notes() {
    std::vector<BehaviorNote_t> notes;
    auto status = graph->get_behavior_notes(notes);
    throw_if(status.is_bad(), status.get_code(), status.get_message());
    return notes;
}

std::vector<BehaviorNote_t>
PyGraph::get_behavior_notes_for_plan_at_index(int64_t const index) {
    std::vector<BehaviorNote_t> notes;
    auto status = graph->get_behavior_notes_for_plan_at_index(index, notes);
    throw_if(status.is_bad(), status.get_code(), status.get_message());
    return notes;
}

void
PyGraph::create_execution_plans(std::vector<cudnn_frontend::HeurMode_t> const& modes) {
    auto status = graph->create_execution_plans(modes);
    throw_if(status.is_bad(), status.get_code(), status.get_message());
}

void
PyGraph::create_execution_plan(int64_t const engine_id, std::unordered_map<KnobType_t, int64_t> const& knobs) {
    auto status = graph->create_execution_plan(engine_id, knobs);
    throw_if(status.is_bad(), status.get_code(), status.get_message());
}

int64_t
PyGraph::get_engine_count() {
    int64_t engine_count = 0;
    auto status          = graph->get_engine_count(engine_count);
    throw_if(status.is_bad(), status.get_code(), status.get_message());
    return engine_count;
}

std::vector<Knob>
PyGraph::get_knobs_for_engine(int64_t const engine_id) {
    std::vector<Knob> knobs;
    auto status = graph->get_knobs_for_engine(engine_id, knobs);
    throw_if(status.is_bad(), status.get_code(), status.get_message());
    return knobs;
}

void
PyGraph::build_plans(BuildPlanPolicy_t const policy) {
    // TODO: Add multithreaded support in python
    auto status = graph->build_plans(policy, false);
    throw_if(status.is_bad(), status.get_code(), status.get_message());
}

void
PyGraph::build_plan_at_index(int64_t const index) {
    auto status = graph->build_plan_at_index(index);
    throw_if(status.is_bad(), status.get_code(), status.get_message());
}

void
PyGraph::build(std::vector<cudnn_frontend::HeurMode_t> const& modes) {
    validate();
    build_operation_graph();
    create_execution_plans(modes);
    check_support();
    build_plans(cudnn_frontend::BuildPlanPolicy_t::HEURISTICS_CHOICE);
}

void
PyGraph::build() {
    validate();
    build_operation_graph();
}

void
PyGraph::check_support() {
    auto status = graph->check_support();
    throw_if(status.is_bad(), status.get_code(), status.get_message());
}

int64_t
PyGraph::get_workspace_size() {
    int64_t workspace = 0;

    auto status = graph->get_workspace_size(workspace);
    throw_if(status.is_bad(), status.get_code(), status.get_message());

    return workspace;
}

int64_t
PyGraph::get_workspace_size(std::optional<std::intptr_t> exec_handle,
                            py::object override_uids,
                            py::object override_shapes,
                            py::object override_strides) {
    std::vector<int64_t> override_uids_vec =
        override_uids.is_none() ? std::vector<int64_t>() : py::cast<std::vector<int64_t>>(override_uids);
    std::vector<std::vector<int64_t>> override_shapes_vec =
        override_shapes.is_none() ? std::vector<std::vector<int64_t>>()
                                  : py::cast<std::vector<std::vector<int64_t>>>(override_shapes);
    std::vector<std::vector<int64_t>> override_strides_vec =
        override_strides.is_none() ? std::vector<std::vector<int64_t>>()
                                   : py::cast<std::vector<std::vector<int64_t>>>(override_strides);

    int64_t workspace     = 0;
    cudnnHandle_t handle_ = exec_handle.has_value() ? static_cast<cudnnHandle_t>((void*)(exec_handle.value())) : handle;
    auto status =
        graph->get_workspace_size(handle_, workspace, override_uids_vec, override_shapes_vec, override_strides_vec);
    throw_if(status.is_bad(), status.get_code(), status.get_message());

    return workspace;
}

int64_t
PyGraph::get_workspace_size_plan_at_index(int64_t index) {
    int64_t workspace;

    auto status = graph->get_workspace_size_plan_at_index(index, workspace);
    throw_if(status.is_bad(), status.get_code(), status.get_message());

    return workspace;
}

int64_t
PyGraph::get_workspace_size_plan_at_index(int64_t index,
                                          std::optional<std::intptr_t> exec_handle,
                                          py::object override_uids,
                                          py::object override_shapes,
                                          py::object override_strides) {
    std::vector<int64_t> override_uids_vec =
        override_uids.is_none() ? std::vector<int64_t>() : py::cast<std::vector<int64_t>>(override_uids);
    std::vector<std::vector<int64_t>> override_shapes_vec =
        override_shapes.is_none() ? std::vector<std::vector<int64_t>>()
                                  : py::cast<std::vector<std::vector<int64_t>>>(override_shapes);
    std::vector<std::vector<int64_t>> override_strides_vec =
        override_strides.is_none() ? std::vector<std::vector<int64_t>>()
                                   : py::cast<std::vector<std::vector<int64_t>>>(override_strides);

    int64_t workspace     = 0;
    cudnnHandle_t handle_ = exec_handle.has_value() ? static_cast<cudnnHandle_t>((void*)(exec_handle.value())) : handle;
    auto status           = graph->get_workspace_size_plan_at_index(
        handle_, index, workspace, override_uids_vec, override_shapes_vec, override_strides_vec);
    throw_if(status.is_bad(), status.get_code(), status.get_message());

    return workspace;
}

namespace {

PyGraph::MetadataBatchBounds
metadata_batch_bounds(json const& structure) {
    PyGraph::MetadataBatchBounds bounds;
    auto nodes = structure.find("nodes");
    if (nodes == structure.end() || nodes->empty()) {
        auto saved = structure.find("frontend_metadata_batch_bounds");
        return saved == structure.end() ? bounds : saved->get<PyGraph::MetadataBatchBounds>();
    }
    // Older structural encodings may embed tensor objects in ports. Keep the
    // conservative bound when the flat UID schema cannot establish ownership.
    if (!structure.contains("tensors") || !structure.at("tensors").is_array()) return {};
    std::unordered_map<int64_t, json const*> tensors;
    for (auto const& tensor : structure.at("tensors")) {
        if (!tensor.contains("uid") || !tensor.at("uid").is_number_integer()) return {};
        tensors.emplace(tensor.at("uid").get<int64_t>(), &tensor);
    }
    std::unordered_set<int64_t> other_uses;
    for (auto const& node : *nodes) {
        auto const& inputs  = node.at("inputs");
        auto const& outputs = node.at("outputs");
        if (!inputs.is_object() || !outputs.is_object()) return {};
        for (auto const* ports : {&inputs, &outputs})
            for (auto const& ref : *ports)
                if (!ref.is_number_integer()) return {};
        auto tag      = node.at("tag").get<std::string>();
        auto q_ref    = inputs.find("Q");
        int64_t q_uid = q_ref == inputs.end() ? -1 : q_ref->get<int64_t>();
        auto q        = tensors.find(q_uid);
        // Dense SDPA may retain the declared batch despite enabled overrides
        // (#1570). The smaller-prefix contract is qualified for unified half
        // forward only. Quantized/backward plans retain the declared floor
        // until their runtime-batch storage behavior is independently checked.
        bool half =
            q != tensors.end() && (q->second->at("data_type") == "HALF" || q->second->at("data_type") == "BFLOAT16");
        bool packed = tag == "SDPA" && node.value("implementation", "AUTO") == "UNIFIED" && half &&
                      q->second->contains("ragged_offset_uid") && !q->second->at("ragged_offset_uid").is_null();
        std::unordered_set<int64_t> metadata;
        auto add = [&](int64_t uid, int64_t extra) {
            auto tensor = tensors.find(uid);
            if (tensor == tensors.end()) return;
            auto dtype = tensor->second->at("data_type").get<std::string>();
            if (dtype != "INT32" && dtype != "INT64") return;
            auto const& shape = tensor->second->at("dim");
            if (shape.empty()) return;
            auto batch = q->second->at("dim").at(0).get<int64_t>();
            bounds[uid].push_back({q_uid, batch, extra});
            metadata.insert(uid);
        };
        if (packed) {
            for (auto const& port : inputs.items()) {
                if (port.key() == "CU_SEQ_LEN_Q" || port.key() == "CU_SEQ_LEN_KV") {
                    add(port.value().get<int64_t>(), 1);
                } else if (port.key() == "SEQ_LEN_Q" || port.key() == "SEQ_LEN_KV" || port.key() == "Page_table_K" ||
                           port.key() == "Page_table_V") {
                    add(port.value().get<int64_t>(), 0);
                }
            }
            for (auto const* ports : {&inputs, &outputs}) {
                for (auto const& ref : *ports) {
                    auto tensor = tensors.find(ref.get<int64_t>());
                    if (tensor == tensors.end()) continue;
                    auto offset = tensor->second->find("ragged_offset_uid");
                    if (offset != tensor->second->end() && !offset->is_null()) add(offset->get<int64_t>(), 1);
                }
            }
        }
        for (auto const* ports : {&inputs, &outputs})
            for (auto const& ref : *ports) {
                auto uid = ref.get<int64_t>();
                if (!metadata.count(uid)) other_uses.insert(uid);
                auto tensor = tensors.find(uid);
                if (tensor != tensors.end()) {
                    auto offset = tensor->second->find("ragged_offset_uid");
                    if (offset != tensor->second->end() && !offset->is_null()) {
                        auto offset_uid = offset->get<int64_t>();
                        if (!metadata.count(offset_uid)) other_uses.insert(offset_uid);
                    }
                }
            }
    }
    // A tensor can also be an ordinary operand elsewhere in a fused graph.
    // Its metadata use must not erase that other operation's storage bound.
    for (auto uid : other_uses) bounds.erase(uid);
    for (auto& [uid, rules] : bounds) {
        CUDNN_FRONTEND_UNUSED(uid);
        std::sort(rules.begin(), rules.end());
        rules.erase(std::unique(rules.begin(), rules.end()), rules.end());
    }
    return bounds;
}

}  // namespace

PyGraph::MetadataBatchBounds
PyGraph::get_metadata_batch_bounds() const {
    if (!graph->context.get_override_shape_enabled()) return {};
    json structure;
    graph->serialize(structure);
    return structure.at("nodes").empty() ? restored_metadata_batch_bounds : metadata_batch_bounds(structure);
}

std::vector<uint8_t>
PyGraph::serialize() const {
    std::vector<uint8_t> data;
    auto status = graph->serialize(data);
    throw_if(status.is_bad(), status.get_code(), status.get_message());
    auto bounds = get_metadata_batch_bounds();
    if (!bounds.empty()) {
        auto saved                              = json::from_ubjson(data);
        saved["frontend_metadata_batch_bounds"] = std::move(bounds);
        data                                    = json::to_ubjson(saved);
    }
    return data;
}

void
PyGraph::deserialize(std::optional<std::intptr_t> handle_, py::object const& pyobj, bool const enforce_precompiled) {
    if (py::isinstance<py::str>(pyobj)) {
        json j      = json::parse(py::cast<std::string>(pyobj));
        auto bounds = j.value("context", json::object()).value("is_override_shape_enabled", false)
                          ? metadata_batch_bounds(j)
                          : MetadataBatchBounds{};

        auto status = graph->deserialize(j, enforce_precompiled);

        throw_if(status.is_bad(), status.get_code(), status.get_message());
        restored_metadata_batch_bounds = std::move(bounds);
    } else {
        std::vector<uint8_t> data;
        if (py::isinstance<py::bytes>(pyobj)) {
            auto bytes = py::borrow<py::bytes>(pyobj);
            auto first = static_cast<const uint8_t*>(bytes.data());
            data.assign(first, first + bytes.size());
        } else {
            data = py::cast<std::vector<uint8_t>>(pyobj);
        }

        // Validate the additional binding facts before publishing a new plan.
        // A malformed blob still goes through Graph's existing diagnostic path.
        auto saved = json::from_ubjson(data, true, false);
        bool overriding =
            saved.is_object() && saved.value("context", json::object()).value("is_override_shape_enabled", false);
        auto bounds = overriding ? metadata_batch_bounds(saved) : MetadataBatchBounds{};
        if (!handle_.has_value() && this->handle == nullptr && this->device_properties != nullptr) {
            // Handle-less path: use the device properties set at construction.
            auto status = graph->deserialize(data, enforce_precompiled);
            throw_if(status.is_bad(), status.get_code(), status.get_message());
        } else {
            // Uses explicit handle if provided, otherwise falls back to the construction-time handle.
            cudnnHandle_t handle =
                handle_.has_value() ? static_cast<cudnnHandle_t>((void*)(handle_.value())) : this->handle;
            auto status = graph->deserialize(handle, data, enforce_precompiled);
            throw_if(status.is_bad(), status.get_code(), status.get_message());
        }
        // A loaded backend plan retains its override contract, even though
        // Graph's plan-only restore does not reconstruct the node context.
        // Read the source flag rather than inheriting the destination graph's
        // constructor; fixed graphs must keep their declared capacity floor.
        graph->set_override_shape_enabled(overriding);
        restored_metadata_batch_bounds = std::move(bounds);
    }
}

void
PyGraph::deserialize(py::object const& pyobj, bool const enforce_precompiled) {
    // Call the overloaded version with default handle (nullopt)
    deserialize(std::nullopt, pyobj, enforce_precompiled);
}

void
PyGraph::update_cuda_graph(std::intptr_t handle,
                           std::unordered_map<cudnn_frontend::graph::Tensor_attributes::uid_t, std::intptr_t> var_pack,
                           std::intptr_t workspace,
                           std::intptr_t cuda_graph) {
    std::unordered_map<int64_t, void*> var_pack_;
    var_pack_.reserve(var_pack.size());
    for (auto const& [uid, device_pointer] : var_pack) {
        var_pack_.emplace(uid, (void*)device_pointer);
    }

    auto status = graph->update_cuda_graph(reinterpret_cast<cudnnHandle_t>(handle),
                                           var_pack_,
                                           reinterpret_cast<void*>(workspace),
                                           reinterpret_cast<cudaGraph_t>(cuda_graph));
    throw_if(status.is_bad(), status.get_code(), status.get_message());

    return;
}

void
PyGraph::populate_cuda_graph(
    std::intptr_t handle,
    std::unordered_map<cudnn_frontend::graph::Tensor_attributes::uid_t, std::intptr_t> var_pack,
    std::intptr_t workspace,
    std::intptr_t cuda_graph) {
    std::unordered_map<int64_t, void*> var_pack_;
    var_pack_.reserve(var_pack.size());
    for (auto const& [uid, device_pointer] : var_pack) {
        var_pack_.emplace(uid, (void*)device_pointer);
    }

    auto status = graph->populate_cuda_graph(reinterpret_cast<cudnnHandle_t>(handle),
                                             var_pack_,
                                             reinterpret_cast<void*>(workspace),
                                             reinterpret_cast<cudaGraph_t>(cuda_graph));
    throw_if(status.is_bad(), status.get_code(), status.get_message());

    return;
}

void
PyGraph::execute(std::unordered_map<int64_t, std::intptr_t> var_pack,
                 std::intptr_t workspace,
                 std::optional<std::intptr_t> exec_handle,
                 py::object override_uids,
                 py::object override_shapes,
                 py::object override_strides,
                 std::optional<int64_t> workspace_bytes) {
    std::unordered_map<int64_t, void*> var_pack_;
    var_pack_.reserve(var_pack.size());
    for (auto const& [uid, device_pointer] : var_pack) {
        var_pack_.emplace(uid, (void*)device_pointer);
    }

    // Convert override_uids to a vector of int64_t (one-liner)
    std::vector<int64_t> override_uids_vec =
        override_uids.is_none() ? std::vector<int64_t>() : py::cast<std::vector<int64_t>>(override_uids);
    std::vector<std::vector<int64_t>> override_shapes_vec =
        override_shapes.is_none() ? std::vector<std::vector<int64_t>>()
                                  : py::cast<std::vector<std::vector<int64_t>>>(override_shapes);
    std::vector<std::vector<int64_t>> override_strides_vec =
        override_strides.is_none() ? std::vector<std::vector<int64_t>>()
                                   : py::cast<std::vector<std::vector<int64_t>>>(override_strides);

    auto workspace_ptr = (void*)workspace;

    cudnnHandle_t handle_ = exec_handle.has_value() ? static_cast<cudnnHandle_t>((void*)(exec_handle.value())) : handle;

    check_workspace_capacity(
        workspace, workspace_bytes, handle_, -1, override_uids_vec, override_shapes_vec, override_strides_vec);

    cudnn_frontend::error_t status = {error_code_t::OK, ""};
    if (override_uids_vec.empty()) {
        status = graph->execute(handle_, var_pack_, workspace_ptr);
    } else {
        status = graph->execute(
            handle_, var_pack_, workspace_ptr, override_uids_vec, override_shapes_vec, override_strides_vec);
    }
    throw_if(status.is_bad(), status.get_code(), status.get_message());
    return;
}

void
PyGraph::prepare_variant_pack_template() {
    auto status = graph->prepare_variant_pack_template();
    throw_if(status.is_bad(), status.get_code(), status.get_message());
}

void
PyGraph::check_workspace_capacity(std::intptr_t workspace,
                                  std::optional<int64_t> workspace_bytes,
                                  cudnnHandle_t exec_handle,
                                  int64_t plan_index,
                                  const std::vector<int64_t>& override_uids,
                                  const std::vector<std::vector<int64_t>>& override_shapes,
                                  const std::vector<std::vector<int64_t>>& override_strides) {
    // Preserve the existing null-workspace diagnostic in the common C++ path.
    // Integer pointers expose no allocation bound and remain caller-sized.
    if (!workspace || !workspace_bytes) return;
    if (*workspace_bytes < 0) throw py::value_error("workspace byte capacity must be nonnegative");
    int64_t required = 0;
    error_t status;
#if (CUDNN_VERSION >= 92300)
    if (!override_uids.empty() && cudnn_frontend::detail::get_backend_version() >= 92300) {
        status =
            plan_index < 0
                ? graph->get_workspace_size(exec_handle, required, override_uids, override_shapes, override_strides)
                : graph->get_workspace_size_plan_at_index(
                      exec_handle, plan_index, required, override_uids, override_shapes, override_strides);
    } else
#endif
    {
        status = plan_index < 0 ? graph->get_workspace_size(required)
                                : graph->get_workspace_size_plan_at_index(plan_index, required);
    }
    throw_if(status.is_bad(), status.get_code(), status.get_message());
    if (*workspace_bytes < required)
        throw py::value_error(("Plan requires a " + std::to_string(required) + "-byte workspace but received a " +
                               std::to_string(*workspace_bytes) + "-byte buffer.")
                                  .c_str());
}

void
PyGraph::execute_with_raw_ptrs(std::intptr_t user_ptrs_array,
                               int64_t n_user,
                               std::intptr_t workspace,
                               std::intptr_t exec_handle,
                               int64_t plan_index,
                               std::optional<int64_t> workspace_bytes) {
    static_assert(sizeof(std::intptr_t) == sizeof(void*), "intptr_t and void* must be the same size");
    throw_if(n_user < 0, error_code_t::INVALID_VALUE, "n_user must be non-negative");
    throw_if(n_user > 0 && user_ptrs_array == 0, error_code_t::INVALID_VALUE, "user_ptrs_array is null");
    // user_ptrs_array points to a contiguous intptr_t[] of device pointers — zero copy
    void** ptrs     = reinterpret_cast<void**>(user_ptrs_array);
    cudnnHandle_t h = exec_handle ? static_cast<cudnnHandle_t>((void*)exec_handle) : handle;
    check_workspace_capacity(workspace, workspace_bytes, h, plan_index);
    // Address the plan the python walk built, which stops being the graph's own
    // candidate once the walk has skipped an entry. -1 defers to the candidate.
    auto status = plan_index < 0 ? graph->execute(h, ptrs, (int)n_user, (void*)workspace)
                                 : graph->execute_plan_at_index(h, ptrs, (int)n_user, (void*)workspace, plan_index);
    throw_if(status.is_bad(), status.get_code(), status.get_message());
}

void
PyGraph::execute_ordered_pack(py::handle pack,
                              std::intptr_t workspace,
                              std::intptr_t exec_handle,
                              int64_t plan_index,
                              std::optional<int64_t> workspace_bytes) {
    const auto frame      = read_native_execution_bindings(pack);
    const auto& overrides = frame.overrides;
    auto h                = exec_handle ? reinterpret_cast<cudnnHandle_t>(exec_handle) : handle;
    auto ws               = reinterpret_cast<void*>(workspace);
    check_workspace_capacity(
        workspace, workspace_bytes, h, plan_index, overrides.uids, overrides.shapes, overrides.strides);
    error_t status;
    if (plan_index >= 0) {
        status = graph->execute_plan_at_index(h,
                                              frame.pointers,
                                              static_cast<int>(frame.size),
                                              ws,
                                              plan_index,
                                              overrides.uids,
                                              overrides.shapes,
                                              overrides.strides);
    } else if (overrides.uids.empty()) {
        status = graph->execute(h, frame.pointers, static_cast<int>(frame.size), ws);
    } else {
        // The delegating C++ candidate has no sorted-pointer override overload.
        // Preserve that rare path's semantics; concrete native plans stay sorted.
        auto uids = graph->get_variant_pack_uids_sorted();
        if (uids.size() != frame.size) throw py::value_error("Wrong number of ordered buffers");
        std::unordered_map<int64_t, void*> pointers;
        for (size_t i = 0; i < uids.size(); ++i) pointers.emplace(uids[i], frame.pointers[i]);
        status = graph->execute(h, pointers, ws, overrides.uids, overrides.shapes, overrides.strides);
    }
    throw_if(status.is_bad(), status.get_code(), status.get_message());
}

py::object
PyGraph::execute_ordered(py::handle schema,
                         py::handle buffers,
                         py::handle tensor_uids,
                         const py::dict& auto_bindings,
                         py::handle workspace,
                         py::handle override_uids,
                         py::handle override_shapes,
                         py::handle override_strides,
                         std::intptr_t exec_handle,
                         int64_t plan_index) {
    auto read = read_ordered_binding(
        schema, buffers, tensor_uids, auto_bindings, workspace, override_uids, override_shapes, override_strides);
    if (!py::cast<py::list>(read[1]).empty() || read[2].is_none()) return read;
    const auto extent = py::cast<py::tuple>(read[2]);
    execute_ordered_pack(read[0],
                         py::cast<std::intptr_t>(extent[0]),
                         exec_handle,
                         plan_index,
                         py::cast<std::optional<int64_t>>(extent[1]));
    return py::none();
}

void
PyGraph::execute_plan_at_index(std::unordered_map<int64_t, std::intptr_t> var_pack,
                               std::intptr_t workspace,
                               int64_t index,
                               std::optional<std::intptr_t> exec_handle,
                               py::object override_uids,
                               py::object override_shapes,
                               py::object override_strides,
                               std::optional<int64_t> workspace_bytes) {
    std::unordered_map<int64_t, void*> var_pack_;
    for (auto const& [uid, device_pointer] : var_pack) {
        var_pack_.emplace(uid, (void*)device_pointer);
    }

    // Convert override_uids to a vector of int64_t (one-liner)
    std::vector<int64_t> override_uids_vec =
        override_uids.is_none() ? std::vector<int64_t>() : py::cast<std::vector<int64_t>>(override_uids);
    std::vector<std::vector<int64_t>> override_shapes_vec =
        override_shapes.is_none() ? std::vector<std::vector<int64_t>>()
                                  : py::cast<std::vector<std::vector<int64_t>>>(override_shapes);
    std::vector<std::vector<int64_t>> override_strides_vec =
        override_strides.is_none() ? std::vector<std::vector<int64_t>>()
                                   : py::cast<std::vector<std::vector<int64_t>>>(override_strides);

    auto workspace_ptr = (void*)workspace;

    cudnnHandle_t handle_ = exec_handle.has_value() ? static_cast<cudnnHandle_t>((void*)(exec_handle.value())) : handle;

    check_workspace_capacity(
        workspace, workspace_bytes, handle_, index, override_uids_vec, override_shapes_vec, override_strides_vec);

    cudnn_frontend::error_t status = {error_code_t::OK, ""};
    if (override_uids_vec.empty()) {
        status = graph->execute_plan_at_index(handle_, var_pack_, workspace_ptr, index);
    } else {
        status = graph->execute_plan_at_index(
            handle_, var_pack_, workspace_ptr, index, override_uids_vec, override_shapes_vec, override_strides_vec);
    }
    throw_if(status.is_bad(), status.get_code(), status.get_message());
    return;
}

std::shared_ptr<graph::Tensor_attributes>
PyGraph::query_tensor_attributes_of_uid(int64_t const uid) const {
    graph::Tensor_attributes tensor;
    auto status = graph->query_tensor_attributes_of_uid(uid, tensor);
    throw_if(status.is_bad(), status.get_code(), status.get_message());
    return std::make_shared<graph::Tensor_attributes>(tensor);
}

std::string
PyGraph::get_plan_name_at_index(int64_t index) {
    std::string plan_name;
    auto status = graph->get_plan_name_at_index(index, plan_name);
    throw_if(status.is_bad(), status.get_code(), status.get_message());
    return plan_name;
}

std::pair<int64_t, std::unordered_map<KnobType_t, int64_t>>
PyGraph::get_engine_and_knobs_at_index(int64_t index) {
    int64_t engine_id = -1;
    std::unordered_map<KnobType_t, int64_t> knobs;
    auto status = graph->get_engine_and_knobs_at_index(index, engine_id, knobs);
    throw_if(status.is_bad(), status.get_code(), status.get_message());
    return {engine_id, knobs};
}

std::vector<int64_t>
default_vector(void) {
    return {};
}

void
init_pygraph_submodule(py::module_& m) {
    PyGraphClass pygraph_(m, "backend_graph", py::is_weak_referenceable(), py::type_slots(pygraph_gc_slots));
    pygraph_
        .def(py::init<std::string const&,
                      cudnn_frontend::DataType_t,
                      cudnn_frontend::DataType_t,
                      cudnn_frontend::DataType_t,
                      std::optional<std::intptr_t>,
                      py::object,
                      py::object,
                      std::shared_ptr<KernelCache>,
                      std::shared_ptr<cudnn_frontend::DeviceProperties>,
                      bool,
                      bool,
                      std::optional<cudnn_frontend::CudaGraphRetention_t>>(),
             py::arg("name")                      = "test_graph",
             py::arg("io_data_type")              = cudnn_frontend::DataType_t::NOT_SET,
             py::arg("intermediate_data_type")    = cudnn_frontend::DataType_t::NOT_SET,
             py::arg("compute_data_type")         = cudnn_frontend::DataType_t::NOT_SET,
             py::arg("handle")                    = std::nullopt,
             py::arg("sm_count")                  = py::none(),
             py::arg("sm_version")                = py::none(),
             py::arg("kernel_cache")              = py::none(),
             py::arg("device_property")           = py::none(),
             py::arg("is_dynamic_shape_enabled")  = false,
             py::arg("is_override_shape_enabled") = false,
             py::arg("cuda_graph_retention")      = py::none())
        .def(
            "get_cuda_graph_retention",
            [](PyGraph const& self) { return self.graph->get_cuda_graph_retention(); },
            R"pbdoc(
                The CUDA graph retention mode in effect for this graph: the one passed as
                cuda_graph_retention, else the CUDNN_CUDA_GRAPH_RETENTION environment variable,
                else cuda_graph_retention.PER_GRAPH.
            )pbdoc")
        .def("tensor_like",
             py::overload_cast<std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> const&, std::string const&>(
                 &PyGraph::tensor_like),
             py::arg("input"),
             py::arg("name") = "")
        .def("tensor_like", py::overload_cast<py::object const&>(&PyGraph::tensor_like))
        .def("_make_tensor",
             &PyGraph::tensor,
             py::arg{"dim"},
             py::arg{"stride"},
             py::arg("data_type")                = cudnn_frontend::DataType_t::NOT_SET,
             py::arg("is_virtual")               = false,
             py::arg("is_pass_by_value")         = false,
             py::arg("ragged_offset")            = py::none(),
             py::arg("reordering_type")          = cudnn_frontend::TensorReordering_t::NONE,
             py::arg("name")                     = "",
             py::arg("uid")                      = -1,
             py::arg("ragged_offset_multiplier") = int64_t{1})
        .def("genstats",
             &PyGraph::genstats,
             py::arg("input"),
             py::arg("compute_data_type") = cudnn_frontend::DataType_t::NOT_SET,
             py::arg("name")              = "")
        .def("slice",
             &PyGraph::slice,
             py::arg("input"),
             py::arg("slices")            = default_vector(),
             py::arg("compute_data_type") = cudnn_frontend::DataType_t::NOT_SET,
             py::arg("name")              = "",
             R"pbdoc(
                Perform slice operation on the given input tensor.

                Args:
                    input (cudnn_tensor): The input tensor to be sliced.
                    slices (List[slice]): A list of Python slice objects, one for each dimension.
                        Per-axis step comes from each slice's ``step`` (default 1), after normalization
                        for the tensor shape (same semantics as indexing a sequence of that length).
                    compute_data_type (Optional[cudnn.data_type]): The data type for computation.
                        Default is NOT_SET.
                    name (Optional[str]): A name for the slice operation.

                Returns:
                    cudnn_tensor: The resulting sliced tensor.

                Example:
                    >>> input_tensor = graph.tensor([4, 8, 16])
                    >>> sliced_tensor = graph.slice(input_tensor, [slice(0, 2), slice(1, 8, 2), slice(0, 16)])
            )pbdoc")
        .def(
            "conv_fprop",
            [](PyGraph& self,
               std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& image,
               std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& weight,
               std::vector<int64_t> const& padding,
               std::vector<int64_t> const& stride,
               std::vector<int64_t> const& dilation,
               cudnn_frontend::ConvolutionMode_t const convolution_mode,
               cudnn_frontend::DataType_t const& compute_data_type,
               std::string const& name) {
                return self.conv_fprop(
                    image, weight, padding, padding, stride, dilation, convolution_mode, compute_data_type, name);
            },
            py::arg("image"),
            py::arg("weight"),
            py::arg("padding")           = default_vector(),
            py::arg("stride")            = default_vector(),
            py::arg("dilation")          = default_vector(),
            py::arg("convolution_mode")  = cudnn_frontend::ConvolutionMode_t::CROSS_CORRELATION,
            py::arg("compute_data_type") = cudnn_frontend::DataType_t::NOT_SET,
            py::arg("name")              = "")
        .def("conv_fprop",
             &PyGraph::conv_fprop,
             py::arg("image"),
             py::arg("weight"),
             py::arg("pre_padding")       = default_vector(),
             py::arg("post_padding")      = default_vector(),
             py::arg("stride")            = default_vector(),
             py::arg("dilation")          = default_vector(),
             py::arg("convolution_mode")  = cudnn_frontend::ConvolutionMode_t::CROSS_CORRELATION,
             py::arg("compute_data_type") = cudnn_frontend::DataType_t::NOT_SET,
             py::arg("name")              = "",
             R"pbdoc(
                Perform convolution operation with the given inputs.

                Args:
                    image (cudnn_tensor): The image tensor.
                    weight (cudnn_tensor): The weight tensor.
                    pre_padding (Optional[List[int]]): The pre padding values for the operation. Default is an empty list.
                    post_padding (Optional[List[int]]): The post padding values for the operation. Default is an empty list.
                    stride (Optional[List[int]]): The stride values for the operation. Default is an empty list.
                    dilation (Optional[List[int]]): The dilation values for the operation. Default is an empty list.
                    compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
                    name (Optional[str]): A name for the operation to be performed.

                Returns:
                    cudnn_tensor: The created tensor.
            )pbdoc")
        .def(
            "conv_wgrad",
            [](PyGraph& self,
               std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& image,
               std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& loss,
               std::vector<int64_t> const& padding,
               std::vector<int64_t> const& stride,
               std::vector<int64_t> const& dilation,
               cudnn_frontend::ConvolutionMode_t const convolution_mode,
               cudnn_frontend::DataType_t const& compute_data_type,
               std::string const& name) {
                return self.conv_wgrad(
                    image, loss, padding, padding, stride, dilation, convolution_mode, compute_data_type, name);
            },
            py::arg("image"),
            py::arg("loss"),
            py::arg("padding")           = default_vector(),
            py::arg("stride")            = default_vector(),
            py::arg("dilation")          = default_vector(),
            py::arg("convolution_mode")  = cudnn_frontend::ConvolutionMode_t::CROSS_CORRELATION,
            py::arg("compute_data_type") = cudnn_frontend::DataType_t::NOT_SET,
            py::arg("name")              = "")
        .def("conv_wgrad",
             &PyGraph::conv_wgrad,
             py::arg("image"),
             py::arg("loss"),
             py::arg("pre_padding")       = default_vector(),
             py::arg("post_padding")      = default_vector(),
             py::arg("stride")            = default_vector(),
             py::arg("dilation")          = default_vector(),
             py::arg("convolution_mode")  = cudnn_frontend::ConvolutionMode_t::CROSS_CORRELATION,
             py::arg("compute_data_type") = cudnn_frontend::DataType_t::NOT_SET,
             py::arg("name")              = "",
             R"pbdoc(
                Compute weight gradients using the given inputs and loss.

                Args:
                    image (cudnn_tensor): The image tensor.
                    loss (cudnn_tensor): The loss tensor.
                    pre_padding (Optional[List[int]]): The pre padding values for the operation. Default is an empty list.
                    post_padding (Optional[List[int]]): The post padding values for the operation. Default is an empty list.                    stride (Optional[List[int]]): The stride values for the operation. Default is an empty list.
                    dilation (Optional[List[int]]): The dilation values for the operation. Default is an empty list.
                    compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
                    name (Optional[str]): A name for the operation to be performed.

                Returns:
                    cudnn_tensor: The created tensor.
            )pbdoc")
        .def(
            "conv_dgrad",
            [](PyGraph& self,
               std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& loss,
               std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& filter,
               std::vector<int64_t> const& padding,
               std::vector<int64_t> const& stride,
               std::vector<int64_t> const& dilation,
               cudnn_frontend::ConvolutionMode_t const convolution_mode,
               cudnn_frontend::DataType_t const& compute_data_type,
               std::string const& name) {
                return self.conv_dgrad(
                    loss, filter, padding, padding, stride, dilation, convolution_mode, compute_data_type, name);
            },
            py::arg("loss"),
            py::arg("filter"),
            py::arg("padding")           = default_vector(),
            py::arg("stride")            = default_vector(),
            py::arg("dilation")          = default_vector(),
            py::arg("convolution_mode")  = cudnn_frontend::ConvolutionMode_t::CROSS_CORRELATION,
            py::arg("compute_data_type") = cudnn_frontend::DataType_t::NOT_SET,
            py::arg("name")              = "")
        .def("conv_dgrad",
             &PyGraph::conv_dgrad,
             py::arg("loss"),
             py::arg("filter"),
             py::arg("pre_padding")       = default_vector(),
             py::arg("post_padding")      = default_vector(),
             py::arg("stride")            = default_vector(),
             py::arg("dilation")          = default_vector(),
             py::arg("convolution_mode")  = cudnn_frontend::ConvolutionMode_t::CROSS_CORRELATION,
             py::arg("compute_data_type") = cudnn_frontend::DataType_t::NOT_SET,
             py::arg("name")              = "",
             R"pbdoc(
                Compute filter gradients using the given inputs and loss.

                Args:
                    loss (cudnn_tensor): The loss tensor.
                    filter (cudnn_tensor): The filter tensor.
                    pre_padding (Optional[List[int]]): The pre padding values for the operation. Default is an empty list.
                    post_padding (Optional[List[int]]): The post padding values for the operation. Default is an empty list.
                    stride (Optional[List[int]]): The stride values for the operation. Default is an empty list.
                    dilation (Optional[List[int]]): The dilation values for the operation. Default is an empty list.
                    compute_data_type (Optional[pycudnn.data_type]): The data type for computation. Default is NOT_SET.
                    name (Optional[str]): A name for the operation to be performed.

                Returns:
                    cudnn_tensor: The created tensor.
            )pbdoc")
        .def("matmul",
             &PyGraph::matmul,
             py::arg("A"),
             py::arg("B"),
             py::arg("compute_data_type") = cudnn_frontend::DataType_t::NOT_SET,
             py::arg("padding")           = 0.0,
             py::arg("name")              = "",
             R"pbdoc(
                Perform matrix multiplication of two tensors A and B.

                Args:
                    A (cudnn_tensor): The first tensor.
                    B (cudnn_tensor): The second matrix tensor.
                    compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
                    name (Optional[str]): A name for the operation to be performed.

                Returns:
                    cudnn_tensor: The result of the matrix multiplication.
            )pbdoc")
        .def("reduction",
             &PyGraph::reduction,
             py::arg("input"),
             py::arg("mode"),
             py::arg("compute_data_type") = cudnn_frontend::DataType_t::NOT_SET,
             py::arg("name")              = "",
             py::arg("group_offset")      = py::none(),
             R"pbdoc(
                Reduce an input tensor along certain dimensions. These dimensions to reduce on are inferred from output tensor shape.

                Args:
                    input (cudnn_tensor): The input tensor.
                    mode (cudnn.reduction_mode): The mode to use to reduce along a dimension.
                    compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
                    name (Optional[str]): A name for the operation to be performed.
                    group_offset (Optional[cudnn_tensor]): The group offset tensor for grouped reduction.

                Returns:
                    cudnn_tensor: The result of reduction operation.
            )pbdoc")
        .def("block_scale_dequantize",
             &PyGraph::block_scale_dequantize,
             py::arg("input"),
             py::arg("descale"),
             py::arg("block_size"),
             py::arg("is_negative_scale") = false,
             py::arg("compute_data_type") = cudnn_frontend::DataType_t::NOT_SET,
             py::arg("name")              = "",
             R"pbdoc(
                Dequantize an input tensor to other dimensions without changing the actual memory layout.
                
                Args:
                    input (cudnn_tensor): The input tensor to dequantize.
                    descale (cudnn_tensor): The scale tensor for dequantization.
                    block_size (List[int]): The block size for dequantization.
                    is_negative_scale (Optional[bool]): Whether the scale values can be negative. Default is False.
                    compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
                    name (Optional[str]): A name for the operation.
                
                Returns:
                    cudnn_tensor: The dequantized output tensor.
            )pbdoc")
        .def("block_scale_quantize",
             &PyGraph::block_scale_quantize,
             py::arg("input"),
             py::arg("block_size"),
             py::arg("axis")              = std::nullopt,
             py::arg("transpose")         = false,
             py::arg("compute_data_type") = cudnn_frontend::DataType_t::NOT_SET,
             py::arg("name")              = "",
             R"pbdoc(
                Quantize an input tensor with block scaling to produce quantized output and scale tensors.
                
                Args:
                    input (cudnn_tensor): The input tensor to quantize.
                    block_size (int): The block size for quantization.
                    axis (Optional[int]): The axis along which to quantize. Default is None.
                    transpose (Optional[bool]): Whether to transpose during quantization. Default is False.
                    compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
                    name (Optional[str]): A name for the operation.
                
                Returns:
                    Tuple[cudnn_tensor, cudnn_tensor]: A tuple of (quantized_output, scale) tensors.
            )pbdoc")
        .def("reshape",
             &PyGraph::reshape,
             py::arg("input"),
             py::arg("name")         = "",
             py::arg("reshape_mode") = cudnn_frontend::ReshapeMode_t::VIEW_ONLY,
             R"pbdoc(
                Reshape an input tensor to other dimensions without changing the actual memory layout.
                These dimensions to reshape to are inferred from output tensor shape.

                Args:
                    input (cudnn_tensor): The input tensor.
                    name (Optional[str]): A name for the operation to be performed.
                    reshape_mode (cudnn.reshape_mode): VIEW_ONLY (default) or LOGICAL for lexicographic logical reshapes.

                Returns:
                    cudnn_tensor: The result of reshape operation. Please set the dims for the output tensor.
            )pbdoc")
        .def("transpose",
             &PyGraph::transpose,
             py::arg("input"),
             py::arg("permutation"),
             py::arg("compute_data_type") = cudnn_frontend::DataType_t::NOT_SET,
             py::arg("name")              = "",
             R"pbdoc(
                Permute tensor dimensions using a permutation vector (output axis i reads input axis permutation[i]).

                Args:
                    input (cudnn_tensor): The input tensor.
                    permutation (List[int]): Permutation of axis indices.
                    compute_data_type (Optional[cudnn.data_type]): Optional compute type; default NOT_SET.
                    name (Optional[str]): Operation name.

                Returns:
                    cudnn_tensor: Transposed tensor (dims/strides inferred when not set).
            )pbdoc")
        .def("concatenate",
             &PyGraph::concatenate,
             py::arg("inputs"),
             py::arg("axis"),
             py::arg("in_place_index") = std::optional<int64_t>{},
             py::arg("name")           = "",
             R"pbdoc(
                Concatenate tensors along an axis.

                Args:
                    inputs (List[cudnn_tensor]): Tensors to concatenate.
                    axis (int): Concatenation axis.
                    in_place_index (Optional[int]): When set, optional in-place concat per backend semantics.
                    name (Optional[str]): Operation name.

                Returns:
                    cudnn_tensor: Concatenated output tensor.
            )pbdoc")
        .def("tensor_scalar",
             py::overload_cast<float const&, cudnn_frontend::graph::ScalarType>(&PyGraph::tensor_scalar),
             py::arg("value"),
             py::arg("scalar_type"),
             R"pbdoc(
                Create a rank-1 scalar tensor from a Python float, marked runtime or compile-time.

                Args:
                    value (float): Scalar value.
                    scalar_type (cudnn.scalar_type): RUNTIME_PARAM or COMPILE_TIME_CONST.

                Returns:
                    cudnn_tensor: Scalar tensor (set dim/stride/name as needed for your graph).
            )pbdoc")
        .def("tensor_scalar",
             py::overload_cast<double const&, cudnn_frontend::graph::ScalarType>(&PyGraph::tensor_scalar),
             py::arg("value"),
             py::arg("scalar_type"))
        .def("tensor_scalar",
             py::overload_cast<int32_t const&, cudnn_frontend::graph::ScalarType>(&PyGraph::tensor_scalar),
             py::arg("value"),
             py::arg("scalar_type"))
        .def("tensor_scalar",
             py::overload_cast<int64_t const&, cudnn_frontend::graph::ScalarType>(&PyGraph::tensor_scalar),
             py::arg("value"),
             py::arg("scalar_type"))
        .def("moe_grouped_matmul",
             &PyGraph::moe_grouped_matmul,
             py::arg("token"),
             py::arg("weight"),
             py::arg("first_token_offset"),
             py::arg("token_index")       = py::none(),
             py::arg("token_ks")          = py::none(),
             py::arg("mode")              = cudnn_frontend::MoeGroupedMatmulMode_t::NONE,
             py::arg("compute_data_type") = cudnn_frontend::DataType_t::FLOAT,
             py::arg("top_k")             = 0,
             py::arg("name")              = "",
             py::arg("top_k_scores")      = py::none(),
             R"pbdoc(
                Perform MoE Grouped Matmul operation.

                Args:
                    token (cudnn_tensor): The token tensor.
                    weight (cudnn_tensor): The weight tensor.
                    first_token_offset (cudnn_tensor): The first token offset tensor.
                    token_index (cudnn_tensor): The token index tensor or nullptr.
                    token_ks (cudnn_tensor): The token ks tensor or nullptr.
                    mode (cudnn.moe_grouped_matmul_mode): The mode of the operation.
                    compute_data_type (cudnn.data_type): The data type for computation.
                    top_k (int): The top k value.
                    name (str): The name of the operation.
                    top_k_scores (cudnn_tensor): COMBINE scores [1, tokens, top_k]. Frontend-only.
            )pbdoc")
        .def("moe_grouped_matmul_bwd",
             &PyGraph::moe_grouped_matmul_bwd,
             py::arg("doutput"),
             py::arg("token"),
             py::arg("first_token_offset"),
             py::arg("compute_data_type") = cudnn_frontend::DataType_t::FLOAT,
             py::arg("name")              = "",
             R"pbdoc(
                Perform MoE Grouped Matmul Bwd operation.

                Args:
                    doutput (cudnn_tensor): The doutput tensor.
                    token (cudnn_tensor): The token tensor.
                    first_token_offset (cudnn_tensor): The first token offset tensor.
                    compute_data_type (cudnn.data_type): The data type for computation.
                    name (str): The name of the operation.
            )pbdoc")
        .def("get_behavior_notes", &PyGraph::get_behavior_notes)
        .def("get_behavior_notes_for_plan_at_index", &PyGraph::get_behavior_notes_for_plan_at_index)
        .def("deselect_engines", &PyGraph::deselect_engines)
        .def("deselect_numeric_notes", &PyGraph::deselect_numeric_notes)
        .def("deselect_behavior_notes", &PyGraph::deselect_behavior_notes)
        .def("select_numeric_notes", &PyGraph::select_numeric_notes)
        .def("select_behavior_notes", &PyGraph::select_behavior_notes)
        .def("deselect_workspace_greater_than", &PyGraph::deselect_workspace_greater_than)
        .def("validate", &PyGraph::validate)
        .def("key", &PyGraph::key)
        .def("build_operation_graph", &PyGraph::build_operation_graph)
        .def("create_execution_plans", &PyGraph::create_execution_plans)
        .def("create_execution_plan",
             &PyGraph::create_execution_plan,
             R"pbdoc(
                Gets the knob configurations available for the given engine.
                Args:
                    engine_id (int): The ID of the engine to create the execution plan on.
                    knobs (dict[Knob, int]): The map of knobs to knob values.
            )pbdoc")
        .def("get_engine_count", &PyGraph::get_engine_count)
        .def("get_knobs_for_engine",
             &PyGraph::get_knobs_for_engine,
             R"pbdoc(
                Gets the knob configurations available for the given engine.
                Args:
                    engine_id (int): The ID of the engine to query knob configurations for.
            )pbdoc")
        .def("check_support", &PyGraph::check_support)
        .def("build_plans",
             &PyGraph::build_plans,
             py::arg("policy") = cudnn_frontend::BuildPlanPolicy_t::HEURISTICS_CHOICE)
        .def("build_plan_at_index",
             &PyGraph::build_plan_at_index,
             py::arg("index"),
             R"pbdoc(
                Build a plan at the given index.
                Args:
                    index (int): The index of the plan to build.
            )pbdoc")
        .def("build", (void (PyGraph::*)(std::vector<cudnn_frontend::HeurMode_t> const&))&PyGraph::build)
        .def("build", (void (PyGraph::*)())&PyGraph::build)
        .def("get_execution_plan_count",
             &PyGraph::get_execution_plan_count,
             R"pbdoc(
                Get the number of execution plan candidates.
            )pbdoc")
        .def("get_workspace_size", (int64_t (PyGraph::*)())&PyGraph::get_workspace_size)
        .def("get_workspace_size",
             (int64_t (PyGraph::*)(
                 std::optional<std::intptr_t>, py::object, py::object, py::object))&PyGraph::get_workspace_size,
             py::arg("handle")           = std::nullopt,
             py::arg("override_uids")    = py::none(),
             py::arg("override_shapes")  = py::none(),
             py::arg("override_strides") = py::none())
        .def("get_workspace_size_plan_at_index",
             (int64_t (PyGraph::*)(int64_t))&PyGraph::get_workspace_size_plan_at_index,
             py::arg("index"),
             R"pbdoc(
                Get workspace for a plan at the given index.
                Args:
                    index (int): The index of the plan to get workspace from.
                    If the graph is not built at the index, this will return 0.
            )pbdoc")
        .def("get_workspace_size_plan_at_index",
             (int64_t (PyGraph::*)(int64_t,
                                   std::optional<std::intptr_t>,
                                   py::object,
                                   py::object,
                                   py::object))&PyGraph::get_workspace_size_plan_at_index,
             py::arg("index"),
             py::arg("handle")           = std::nullopt,
             py::arg("override_uids")    = py::none(),
             py::arg("override_shapes")  = py::none(),
             py::arg("override_strides") = py::none())
        .def("query_tensor_attributes_of_uid",
             &PyGraph::query_tensor_attributes_of_uid,
             py::arg("uid"),
             R"pbdoc(
                    Get tensor_attributes for a given UID
                    Args:
                    uid (int): The uid of tensor to be queried
                    If the graph does not have the UID, this will raise an error
                )pbdoc")
        .def("get_plan_name_at_index",
             &PyGraph::get_plan_name_at_index,
             py::arg("index"),
             R"pbdoc(
                    Get the name for a plan at the given index.
                    Args:
                    index (int): The index of the plan to get workspace from.
                )pbdoc")
        .def("get_engine_and_knobs_at_index",
             &PyGraph::get_engine_and_knobs_at_index,
             py::arg("index"),
             R"pbdoc(
                    Get the engine id and knob choices for a plan at the given index.
                    Structured counterpart of get_plan_name_at_index; the result can be
                    passed to create_execution_plan() to rebuild the same kernel.
                    Args:
                    index (int): The index of the plan to query.
                )pbdoc")
        .def("_execute",
             &PyGraph::execute,
             py::arg("var_pack"),
             py::arg("workspace"),
             py::arg("handle"),
             py::arg("override_uids")    = py::none(),
             py::arg("override_shapes")  = py::none(),
             py::arg("override_strides") = py::none(),
             py::arg("workspace_bytes")  = std::nullopt)
        .def("_prepare_variant_pack_template", &PyGraph::prepare_variant_pack_template)
        .def("_get_variant_pack_uids_sorted", &PyGraph::get_variant_pack_uids_sorted)
        .def("_get_buffer_bounds",
             [](PyGraph& self) {
                 py::list bounds;
                 auto metadata = self.get_metadata_batch_bounds();
                 for (auto uid : self.graph->get_variant_pack_uids_sorted()) {
                     graph::Tensor_attributes tensor;
                     auto status = self.graph->query_tensor_attributes_of_uid(uid, tensor);
                     throw_if(status.is_bad(), status.get_code(), status.get_message());
                     // Dense affine ranges are descriptor facts. Ragged offsets and
                     // reordered/vectorized layouts require their operation's storage
                     // contract; their logical envelope is not an allocation size.
                     const bool affine = !tensor.get_ragged_offset() &&
                                         tensor.get_reordering_type() == TensorReordering_t::NONE &&
                                         tensor.get_vector_count() == 1;
                     bounds.append(py::make_tuple(
                         uid,
                         tensor.get_dim(),
                         tensor.get_stride(),
                         tensor.get_data_type(),
                         affine ? cudnn_frontend::detail::get_element_size_in_bits(tensor.get_data_type()) : 0,
                         self.graph->context.get_override_shape_enabled(),
                         metadata[uid]));
                 }
                 return bounds;
             })
        .def("_execute_with_raw_ptrs",
             &PyGraph::execute_with_raw_ptrs,
             py::arg("user_ptrs_array"),
             py::arg("n_user"),
             py::arg("workspace"),
             py::arg("handle"),
             py::arg("plan_index")      = -1,
             py::arg("workspace_bytes") = std::nullopt)
        .def("_execute_ordered_pack",
             &PyGraph::execute_ordered_pack,
             py::arg("pack"),
             py::arg("workspace"),
             py::arg("handle"),
             py::arg("plan_index")      = -1,
             py::arg("workspace_bytes") = std::nullopt)
        .def("_execute_ordered",
             &PyGraph::execute_ordered,
             py::arg("schema"),
             py::arg("buffers"),
             py::arg("tensor_uids"),
             py::arg("auto_bindings"),
             py::arg("workspace").none(),
             py::arg("override_uids").none(),
             py::arg("override_shapes").none(),
             py::arg("override_strides").none(),
             py::arg("handle"),
             py::arg("plan_index"))
        .def("populate_cuda_graph", &PyGraph::populate_cuda_graph)
        .def("update_cuda_graph", &PyGraph::update_cuda_graph)
        .def(
            "serialize",
            [](PyGraph const& self, bool as_bytes) -> py::object {
                auto data = self.serialize();
                if (as_bytes) return py::bytes(data.data(), data.size());
                return py::cast(data);
            },
            py::kw_only(),
            py::arg("as_bytes") = false)
        .def("deserialize",
             (void (PyGraph::*)(std::optional<std::intptr_t>, py::object const&, bool const))&PyGraph::deserialize,
             py::arg("handle_"),
             py::arg("pyobj"),
             py::arg("enforce_precompiled") = false)
        .def("deserialize",
             (void (PyGraph::*)(py::object const&, bool const))&PyGraph::deserialize,
             py::arg("pyobj"),
             py::arg("enforce_precompiled") = false)
        .def("_execute_plan_at_index",
             &PyGraph::execute_plan_at_index,
             py::arg("var_pack"),
             py::arg("workspace"),
             py::arg("index"),
             py::arg("handle"),
             py::arg("override_uids")    = py::none(),
             py::arg("override_shapes")  = py::none(),
             py::arg("override_strides") = py::none(),
             py::arg("workspace_bytes")  = std::nullopt)
        .def("__repr__", [](PyGraph const& pygraph) {
            std::stringstream ss;
            json j = pygraph.graph;
            ss << j.dump(4);
            return ss.str();
        });

    m.def("_get_data_ptr", &extract_data_pointer);

    init_pygraph_norm_submodule(pygraph_);
    init_pygraph_sdpa_submodule(pygraph_);
    init_pygraph_pointwise_submodule(pygraph_);
    init_pygraph_custom_reduction_submodule(pygraph_);
}

}  // namespace cudnn_frontend::python_bindings
