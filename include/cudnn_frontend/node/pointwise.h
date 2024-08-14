#pragma once

#include "../../cudnn_frontend_PointWiseDesc.h"
#include "../../cudnn_frontend_Heuristics.h"
#include "../../cudnn_frontend_Logging.h"

#include "../graph_helpers.h"
#include "../node_interface.h"

namespace cudnn_frontend::graph {

class PointwiseNode : public NodeCRTP<PointwiseNode> {
   public:
    Pointwise_attributes attributes;

    PointwiseNode(Pointwise_attributes&& attributes_, detail::Context const& context)
        : NodeCRTP(context), attributes(std::move(attributes_)) {}

    Type
    getType() override final {
        return Type::POINTWISE;
    }

    error_t
    infer_properties_node() override final {
        CUDNN_FE_LOG_LABEL_ENDL("INFO: Inferrencing properties for pointwise node " << attributes.name << "...");

        attributes.fill_from_context(context);

        auto out_0_tensor = attributes.outputs.at(Pointwise_attributes::output_names::OUT_0);

        auto output_dim = out_0_tensor->get_dim();
        // Only infer dims and strides if user did not set them
        if (output_dim.empty()) {
            std::vector<std::vector<int64_t>> input_shapes;
            for (const auto& [input_name, input_tensor] : attributes.inputs) {
                if (!input_tensor) {
                    continue;
                }
                input_shapes.push_back(input_tensor->get_dim());
            }

            CHECK_CUDNN_FRONTEND_ERROR(detail::compute_broadcast_shape(input_shapes, output_dim));
            out_0_tensor->set_dim(output_dim);
        }

        if (out_0_tensor->get_stride().empty()) {
            auto input_stride = attributes.inputs.at(Pointwise_attributes::input_names::IN_0)->get_stride();
            std::vector<int64_t> stride_order;
            CHECK_CUDNN_FRONTEND_ERROR(
                detail::generate_stride_order_preserving_format(input_stride, output_dim.size(), stride_order));
            out_0_tensor->set_stride(detail::generate_stride(output_dim, stride_order));
        }

        return {error_code_t::OK, ""};
    }

    error_t
    create_cudnn_operations(
        std::unordered_set<uid_t>& uids_involved_in_operations,
        std::vector<std::shared_ptr<cudnn_frontend::Operation>>& operations,
        managed_backend_descriptor_t& raw_operations,
        std::unordered_map<int64_t, std::shared_ptr<cudnn_frontend::Tensor>>& tensors) const override final {
        CUDNN_FRONTEND_UNUSED(raw_operations);
        CUDNN_FE_LOG_LABEL_ENDL("INFO: " << "Building PointwiseNode operations " << attributes.name << "...");

        auto&& pointwise_descriptor_builder = cudnn_frontend::PointwiseDescBuilder();

        if (attributes.get_axis().has_value()) {
            pointwise_descriptor_builder.setAxis(attributes.get_axis().value());
        }

        if (attributes.relu_lower_clip_slope.has_value()) {
            pointwise_descriptor_builder.setReluLowerClipSlope(attributes.relu_lower_clip_slope.value());
        }

        if (attributes.relu_lower_clip.has_value()) {
            pointwise_descriptor_builder.setReluLowerClip(attributes.relu_lower_clip.value());
        }

        if (attributes.relu_upper_clip.has_value()) {
            pointwise_descriptor_builder.setReluUpperClip(attributes.relu_upper_clip.value());
        }

        pointwise_descriptor_builder.setComputeType(attributes.compute_data_type);
        pointwise_descriptor_builder.setMode(attributes.mode);
        auto pointwise_descriptor = pointwise_descriptor_builder.build();

        auto const port_count = get_pointwise_mode_port_count(attributes.mode);

        auto&& pointwise_operation_builder =
            cudnn_frontend::OperationBuilder(DescriptorType_t::OPERATION_POINTWISE_DESCRIPTOR);
        pointwise_operation_builder.setpwDesc(pointwise_descriptor);

        if (detail::is_activation_backward_mode(attributes.mode)) {
            CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(IN_0, Pointwise_attributes::input_names::IN_0);
            pointwise_operation_builder.setdyDesc(*(tensors.at(IN_0->second->get_uid())));

            CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(IN_1, Pointwise_attributes::input_names::IN_1);
            pointwise_operation_builder.setxDesc(*(tensors.at(IN_1->second->get_uid())));

            CUDNN_FE_VALIDATE_AND_ASSIGN_OUTPUT_TENSOR(OUT_0, Pointwise_attributes::output_names::OUT_0);
            pointwise_operation_builder.setdxDesc(*(tensors.at(OUT_0->second->get_uid())));
        } else {
            CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(IN_0, Pointwise_attributes::input_names::IN_0);
            pointwise_operation_builder.setxDesc(*(tensors.at(IN_0->second->get_uid())));

            if (port_count >= 3) {
                CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(IN_1, Pointwise_attributes::input_names::IN_1);
                pointwise_operation_builder.setbDesc(*(tensors.at(IN_1->second->get_uid())));
            }

            if (port_count >= 4) {
                CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(IN_2, Pointwise_attributes::input_names::IN_2);
                pointwise_operation_builder.settDesc(*(tensors.at(IN_2->second->get_uid())));
            }

            CUDNN_FE_VALIDATE_AND_ASSIGN_OUTPUT_TENSOR(OUT_0, Pointwise_attributes::output_names::OUT_0);
            pointwise_operation_builder.setyDesc(*(tensors.at(OUT_0->second->get_uid())));
        }

#ifdef NV_CUDNN_DISABLE_EXCEPTION
        // disable exception macro is defined. Calling build will not throw.
        // Check status of desc and return error.
        auto operation = pointwise_operation_builder.build();
        RETURN_CUDNN_FRONTEND_ERROR_IF(operation.get_status() != CUDNN_STATUS_SUCCESS,
                                       error_code_t::CUDNN_BACKEND_API_FAILED,
                                       operation.get_error());
        operations.push_back(std::make_shared<Operation_v8>(std::move(operation)));
#else
        // build() can throw
        // wrap in try catch
        try {
            auto operation = pointwise_operation_builder.build();
            operations.push_back(std::make_shared<Operation_v8>(std::move(operation)));
        } catch (cudnn_frontend::cudnnException& e) {
            RETURN_CUDNN_FRONTEND_ERROR_IF(
                e.getCudnnStatus() != CUDNN_STATUS_SUCCESS, error_code_t::CUDNN_BACKEND_API_FAILED, e.what());
        }
#endif

        auto const& non_virtual_uids = attributes.get_non_virtual_uids();
        uids_involved_in_operations.insert(non_virtual_uids.begin(), non_virtual_uids.end());
        return {error_code_t::OK, ""};
    }

#ifndef CUDNN_FRONTEND_SKIP_JSON_LIB
    virtual void
    serialize(json& j) const override final {
        j = attributes;
        j.update(R"({"tag": "POINTWISE"})"_json);
    }
#endif
};

inline void
INode::pointwise(std::shared_ptr<Tensor_attributes> a,
                 Pointwise_attributes attributes,
                 std::shared_ptr<Tensor_attributes> c) {
    attributes.inputs[Pointwise_attributes::input_names::IN_0]    = a;
    attributes.outputs[Pointwise_attributes::output_names::OUT_0] = c;
    sub_nodes.emplace_back(std::make_unique<PointwiseNode>(std::move(attributes), context));
}

inline void
INode::pointwise(std::shared_ptr<Tensor_attributes> a,
                 std::shared_ptr<Tensor_attributes> b,
                 Pointwise_attributes attributes,
                 std::shared_ptr<Tensor_attributes> c) {
    attributes.inputs[Pointwise_attributes::input_names::IN_0]    = a;
    attributes.inputs[Pointwise_attributes::input_names::IN_1]    = b;
    attributes.outputs[Pointwise_attributes::output_names::OUT_0] = c;
    sub_nodes.emplace_back(std::make_unique<PointwiseNode>(std::move(attributes), context));
}

inline std::shared_ptr<Tensor_attributes>
INode::pointwise(std::shared_ptr<Tensor_attributes> a, Pointwise_attributes attributes) {
    if (attributes.name.empty()) {
        attributes.name += std::to_string(sub_nodes.size());
    }
    attributes.inputs[Pointwise_attributes::input_names::IN_0] = a;
    if (a->get_name().empty()) {
        a->set_name(attributes.name + "::IN_0");
    };
    auto OUT_0 = attributes.outputs[Pointwise_attributes::output_names::OUT_0] =
        output_tensor(attributes.name + "::OUT_0");

    sub_nodes.emplace_back(std::make_unique<PointwiseNode>(std::move(attributes), context));
    return OUT_0;
}

inline std::shared_ptr<Tensor_attributes>
INode::pointwise(std::shared_ptr<Tensor_attributes> a,
                 std::shared_ptr<Tensor_attributes> b,
                 Pointwise_attributes attributes) {
    if (attributes.name.empty()) {
        attributes.name += std::to_string(sub_nodes.size());
    }
    attributes.inputs[Pointwise_attributes::input_names::IN_0] = a;
    attributes.inputs[Pointwise_attributes::input_names::IN_1] = b;
    if (a->get_name().empty()) {
        a->set_name(attributes.name + "::IN_0");
    };
    if (b->get_name().empty()) {
        b->set_name(attributes.name + "::IN_1");
    };
    auto OUT_0 = attributes.outputs[Pointwise_attributes::output_names::OUT_0] =
        output_tensor(attributes.name + "::OUT_0");

    sub_nodes.emplace_back(std::make_unique<PointwiseNode>(std::move(attributes), context));
    return OUT_0;
}

inline std::shared_ptr<Tensor_attributes>
INode::pointwise(std::shared_ptr<Tensor_attributes> a,
                 std::shared_ptr<Tensor_attributes> b,
                 std::shared_ptr<Tensor_attributes> c,
                 Pointwise_attributes attributes) {
    if (attributes.name.empty()) {
        attributes.name += std::to_string(sub_nodes.size());
    }
    attributes.inputs[Pointwise_attributes::input_names::IN_0] = a;
    attributes.inputs[Pointwise_attributes::input_names::IN_1] = b;
    attributes.inputs[Pointwise_attributes::input_names::IN_2] = c;
    if (a->get_name().empty()) {
        a->set_name(attributes.name + "::IN_0");
    };
    if (b->get_name().empty()) {
        b->set_name(attributes.name + "::IN_1");
    };
    if (c->get_name().empty()) {
        c->set_name(attributes.name + "::IN_2");
    };
    auto OUT_0 = attributes.outputs[Pointwise_attributes::output_names::OUT_0] =
        output_tensor(attributes.name + "::OUT_0");

    sub_nodes.emplace_back(std::make_unique<PointwiseNode>(std::move(attributes), context));
    return OUT_0;
}
}  // namespace cudnn_frontend::graph