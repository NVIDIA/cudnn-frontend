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
    pre_validate_node() const override final {
        getLogger() << "[cudnn_frontend] INFO: "
                    << "Validating pointwise node " << attributes.name << "..." << std::endl;

        RETURN_CUDNN_FRONTEND_ERROR_IF(
            attributes.mode == PointwiseMode_t::NOT_SET, error_code_t::ATTRIBUTE_NOT_SET, "pointwise mode not set.");

        CUDNN_FE_VALIDATE_INPUT_TENSOR(Pointwise_attributes::input_names::IN_0);

        auto const port_count = get_pointwise_mode_port_count(attributes.mode);
        if (port_count >= 3) {
            CUDNN_FE_VALIDATE_INPUT_TENSOR(Pointwise_attributes::input_names::IN_1);
        }

        if (port_count >= 4) {
            CUDNN_FE_VALIDATE_INPUT_TENSOR(Pointwise_attributes::input_names::IN_2);
        }

        CUDNN_FE_VALIDATE_OUTPUT_TENSOR(Pointwise_attributes::output_names::OUT_0);

        CHECK_CUDNN_FRONTEND_ERROR(attributes.validate_inputs());

        return {error_code_t::OK, ""};
    }

    error_t
    expand_and_infer_properties_node() override final {
        getLogger() << "[cudnn_frontend] INFO: Inferrencing properties for pointwise node " << attributes.name << "..."
                    << std::endl;

        attributes.fill_from_context(context);

        // Only inferrencing from IN_0 to OUT_0 works today.
        auto in_0_tensor  = attributes.inputs[Pointwise_attributes::input_names::IN_0];
        auto out_0_tensor = attributes.outputs[Pointwise_attributes::output_names::OUT_0];

        auto out_0_tensor_dim = out_0_tensor->get_dim();
        // Only infer dims and strides if user did not set them
        if (out_0_tensor_dim.empty()) {
            out_0_tensor->set_dim(in_0_tensor->get_dim());
        }
        // Special case here where input strides are being copied over
        if (out_0_tensor->get_stride().empty()) {
            out_0_tensor->set_stride(in_0_tensor->get_stride());
        }

        return {error_code_t::OK, ""};
    }

    error_t
    post_validate_node() const override final {
        // Validate outputs
        // All properties of output tensors should have been set now.
        CHECK_CUDNN_FRONTEND_ERROR(attributes.validate_outputs());

        return {error_code_t::OK, ""};
    }

    error_t
    create_cudnn_operations(
        std::unordered_set<uid_t>& uids_involved_in_operations,
        std::vector<std::shared_ptr<cudnn_frontend::Operation>>& operations,
        std::unordered_map<int64_t, std::shared_ptr<cudnn_frontend::Tensor>>& tensors) const override final {
        getLogger() << "[cudnn_frontend] INFO: "
                    << "Building PointwiseNode operations " << attributes.name << "..." << std::endl;

        auto pointwise_descriptor = cudnn_frontend::PointwiseDescBuilder()
                                        .setAxis(attributes.get_axis().value_or(-1))
                                        .setReluLowerClipSlope(attributes.relu_lower_clip_slope.value_or(0.0))
                                        .setComputeType(attributes.compute_data_type)
                                        .setMode(attributes.mode)
                                        .build();

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

    virtual void
    serialize(json& j) const override final {
        j = attributes;
        j.update(R"({"tag": "POINTWISE"})"_json);
    }
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
    attributes.inputs[Pointwise_attributes::input_names::IN_0] = a;
    auto OUT_0 = attributes.outputs[Pointwise_attributes::output_names::OUT_0] =
        output_tensor(attributes.name + "::OUT_0");

    sub_nodes.emplace_back(std::make_unique<PointwiseNode>(std::move(attributes), context));
    return OUT_0;
}

inline std::shared_ptr<Tensor_attributes>
INode::pointwise(std::shared_ptr<Tensor_attributes> a,
                 std::shared_ptr<Tensor_attributes> b,
                 Pointwise_attributes attributes) {
    attributes.inputs[Pointwise_attributes::input_names::IN_0] = a;
    attributes.inputs[Pointwise_attributes::input_names::IN_1] = b;
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
    attributes.inputs[Pointwise_attributes::input_names::IN_0] = a;
    attributes.inputs[Pointwise_attributes::input_names::IN_1] = b;
    attributes.inputs[Pointwise_attributes::input_names::IN_2] = c;
    auto OUT_0 = attributes.outputs[Pointwise_attributes::output_names::OUT_0] =
        output_tensor(attributes.name + "::OUT_0");

    sub_nodes.emplace_back(std::make_unique<PointwiseNode>(std::move(attributes), context));
    return OUT_0;
}
}  // namespace cudnn_frontend::graph