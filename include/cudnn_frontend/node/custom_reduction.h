/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../graph_helpers.h"
#include "../node_interface.h"

namespace cudnn_frontend::graph {

class Graph;

class CustomReductionNode : public NodeCRTP<CustomReductionNode> {
   public:
    // The backend accepts 1 to 8 streams (CUDNN_ATTR_OPERATION_REDUCTION_XDESC / YDESC).
    static constexpr size_t max_stream_count = 8;

    Custom_reduction_attributes attributes;

    CustomReductionNode(Custom_reduction_attributes &&attributes_, detail::Context const &context)
        : NodeCRTP(context), attributes(std::move(attributes_)) {}

    Type
    getType() override final {
        return Type::CUSTOM_REDUCTION;
    }

    error_t
    pre_validate_node() const override final {
        auto const stream_count = attributes.inputs.size();
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            stream_count == 0, error_code_t::ATTRIBUTE_NOT_SET, "Custom reduction requires at least one input stream.");
        RETURN_CUDNN_FRONTEND_ERROR_IF(stream_count > max_stream_count,
                                       error_code_t::INVALID_VALUE,
                                       "Custom reduction supports at most " + std::to_string(max_stream_count) +
                                           " input streams; got " + std::to_string(stream_count) + ".");
        RETURN_CUDNN_FRONTEND_ERROR_IF(attributes.reducer == nullptr,
                                       error_code_t::ATTRIBUTE_NOT_SET,
                                       "Custom reduction reducer callback is not set.");
        RETURN_CUDNN_FRONTEND_ERROR_IF(attributes.reducer_semantic_key.empty(),
                                       error_code_t::ATTRIBUTE_NOT_SET,
                                       "Custom reduction reducer semantic key is not set.");
        RETURN_CUDNN_FRONTEND_ERROR_IF(attributes.reduction_dims.empty(),
                                       error_code_t::ATTRIBUTE_NOT_SET,
                                       "Custom reduction requires at least one reduction dimension.");
        RETURN_CUDNN_FRONTEND_ERROR_IF(attributes.identity_values.size() != stream_count,
                                       error_code_t::INVALID_VALUE,
                                       "Custom reduction identity count must equal its input stream count.");
        RETURN_CUDNN_FRONTEND_ERROR_IF(attributes.outputs.size() != stream_count,
                                       error_code_t::INVALID_VALUE,
                                       "Custom reduction output count must equal its input stream count.");
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            !attributes.compute_data_types.empty() && attributes.compute_data_types.size() != stream_count,
            error_code_t::INVALID_VALUE,
            "Custom reduction compute-type count must "
            "equal its input stream count.");

        std::vector<int64_t> common_shape;
        for (size_t i = 0; i < stream_count; ++i) {
            auto const &input = attributes.inputs[i];
            RETURN_CUDNN_FRONTEND_ERROR_IF(input == nullptr,
                                           error_code_t::ATTRIBUTE_NOT_SET,
                                           "Custom reduction input stream " + std::to_string(i) + " is not set.");
            RETURN_CUDNN_FRONTEND_ERROR_IF(input->get_dim().empty(),
                                           error_code_t::ATTRIBUTE_NOT_SET,
                                           "Custom reduction input stream " + std::to_string(i) + " has no shape.");
            if (common_shape.empty()) {
                common_shape = input->get_dim();
            } else {
                RETURN_CUDNN_FRONTEND_ERROR_IF(input->get_dim() != common_shape,
                                               error_code_t::INVALID_VALUE,
                                               "Custom reduction input streams must have the same shape.");
            }
        }

        auto const rank = static_cast<int64_t>(common_shape.size());
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            rank > CUDNN_DIM_MAX, error_code_t::INVALID_VALUE, "Custom reduction input rank exceeds CUDNN_DIM_MAX.");
        for (size_t i = 0; i < attributes.reduction_dims.size(); ++i) {
            auto const dim = attributes.reduction_dims[i];
            RETURN_CUDNN_FRONTEND_ERROR_IF(
                dim < 0 || dim >= rank,
                error_code_t::INVALID_VALUE,
                "Custom reduction dimension " + std::to_string(dim) + " is outside [0, " + std::to_string(rank) + ").");
            RETURN_CUDNN_FRONTEND_ERROR_IF(i > 0 && dim <= attributes.reduction_dims[i - 1],
                                           error_code_t::INVALID_VALUE,
                                           "Custom reduction dimensions must be unique and in strictly "
                                           "ascending order.");
        }

        for (auto const &[index, output] : attributes.outputs) {
            CUDNN_FRONTEND_UNUSED(index);
            RETURN_CUDNN_FRONTEND_ERROR_IF(
                output == nullptr, error_code_t::ATTRIBUTE_NOT_SET, "Custom reduction output is not set.");
        }

        return {error_code_t::OK, ""};
    }

    error_t
    infer_properties_node() override final {
        if (attributes.compute_data_types.empty() && attributes.compute_data_type != DataType_t::NOT_SET) {
            attributes.compute_data_types.assign(attributes.inputs.size(), attributes.compute_data_type);
        }

        for (size_t i = 0; i < attributes.inputs.size(); ++i) {
            auto const &input  = attributes.inputs[i];
            auto const &output = attributes.outputs.at(static_cast<int64_t>(i));

            input->fill_from_context(context);
            if (output->get_data_type() == DataType_t::NOT_SET) {
                output->set_data_type(input->get_data_type());
            }
            if (output->get_dim().empty()) {
                auto output_dim = input->get_dim();
                for (auto const dim : attributes.reduction_dims) {
                    output_dim[static_cast<size_t>(dim)] = 1;
                }
                output->set_dim(std::move(output_dim));
            }
            if (output->get_stride().empty()) {
                auto const &output_dim = output->get_dim();
                // generate_NHWC_stride_order indexes axis 1 and therefore
                // requires rank >= 2. A rank-1 keep-dims reduction has the
                // canonical contiguous vector stride instead.
                auto const stride_order = output_dim.size() == 1
                                              ? detail::generate_row_major_stride_order(output_dim.size())
                                              : detail::generate_NHWC_stride_order(output_dim.size());
                std::vector<int64_t> output_stride;
                CHECK_CUDNN_FRONTEND_ERROR(detail::generate_stride_checked(output_dim, stride_order, output_stride));
                output->set_stride(std::move(output_stride));
            }
            output->fill_from_context(context);
        }

        return {error_code_t::OK, ""};
    }

    error_t
    expand_node() override final;
    error_t
    post_validate_node() const override final;

    error_t
    create_cudnn_operations(
        std::unordered_set<Tensor_attributes::uid_t> &uids_involved_in_operations,
        std::vector<std::shared_ptr<cudnn_frontend::Operation>> &operations,
        managed_backend_descriptor_t &raw_operations,
        std::unordered_map<int64_t, std::shared_ptr<cudnn_frontend::Tensor>> &tensors) const override final;

#ifndef CUDNN_FRONTEND_SKIP_JSON_LIB
    void
    serialize(json &j) const override final {
        auto compute_data_types = attributes.compute_data_types;
        if (compute_data_types.empty() && attributes.compute_data_type != DataType_t::NOT_SET) {
            compute_data_types.assign(attributes.inputs.size(), attributes.compute_data_type);
        }

        j["tag"]                  = "CUSTOM_REDUCTION";
        j["name"]                 = attributes.name;
        j["reducer_semantic_key"] = attributes.reducer_semantic_key;
        j["reduction_dims"]       = attributes.reduction_dims;
        j["compute_data_types"]   = std::move(compute_data_types);
        j["identity_types"]       = json::array();
        j["identity_raw"]         = json::array();
        for (auto const &identity : attributes.identity_values) {
            j["identity_types"].push_back(identity.get_data_type());
            j["identity_raw"].push_back(identity.get_raw_hex());
        }

        // This is inspection-only structure for execution-plan blobs. The reducer
        // callback is deliberately neither serialized nor reconstructible.
        j["inputs"] = json::array();
        for (size_t i = 0; i < attributes.inputs.size(); ++i) {
            j["inputs"].push_back({std::to_string(i), attributes.inputs[i]});
        }
        j["outputs"] = json::array();
        for (auto const &[index, output] : attributes.outputs) {
            j["outputs"].push_back({std::to_string(index), output});
        }
    }
#endif

   private:
    // The backend's integer class: every integer and boolean type. A compute type must be in the same class as its
    // stream type.
    static bool
    is_integer_class(DataType_t type) {
        switch (type) {
            case DataType_t::INT4:
            case DataType_t::INT8:
            case DataType_t::UINT8:
            case DataType_t::INT32:
            case DataType_t::UINT32:
            case DataType_t::INT64:
            case DataType_t::INT8x4:
            case DataType_t::UINT8x4:
            case DataType_t::INT8x32:
            case DataType_t::BOOLEAN:
            case DataType_t::BYTE_BOOLEAN:
                return true;
            default:
                return false;
        }
    }

    static error_t
    validate_reducer_body(Graph const &body,
                          std::vector<std::shared_ptr<Tensor_attributes>> const &lhs,
                          std::vector<std::shared_ptr<Tensor_attributes>> const &rhs,
                          std::vector<std::shared_ptr<Tensor_attributes>> const &outputs);

    // A reducer that builds the graph of this node again must not run this reducer again.
    bool reducer_running = false;

    std::shared_ptr<Graph> subgraph;
    std::vector<std::shared_ptr<Tensor_attributes>> lhs_tensors;
    std::vector<std::shared_ptr<Tensor_attributes>> rhs_tensors;
    std::vector<std::shared_ptr<Tensor_attributes>> reducer_output_tensors;
};

inline std::vector<std::shared_ptr<Tensor_attributes>>
INode::custom_reduction(std::vector<std::shared_ptr<Tensor_attributes>> inputs,
                        Custom_reduction_attributes attributes) {
    attributes.inputs = std::move(inputs);

    std::vector<std::shared_ptr<Tensor_attributes>> outputs;
    outputs.reserve(attributes.inputs.size());
    for (size_t i = 0; i < attributes.inputs.size(); ++i) {
        auto output                                 = output_tensor(attributes.name + "::Y" + std::to_string(i));
        attributes.outputs[static_cast<int64_t>(i)] = output;
        outputs.emplace_back(std::move(output));
    }

    sub_nodes.emplace_back(std::make_unique<CustomReductionNode>(std::move(attributes), context));
    return outputs;
}

}  // namespace cudnn_frontend::graph
