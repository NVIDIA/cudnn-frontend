#pragma once

#include "../../cudnn_frontend_ReductionDesc.h"
#include "../../cudnn_frontend_Logging.h"

#include "../graph_helpers.h"
#include "../node_interface.h"

namespace cudnn_frontend::graph {

class ReductionNode : public NodeCRTP<ReductionNode> {
   public:
    Reduction_attributes attributes;

    ReductionNode(Reduction_attributes&& attributes_, detail::Context const& context)
        : NodeCRTP(context), attributes(std::move(attributes_)) {}

    Type
    getType() override final {
        return Type::REDUCTION;
    }

    error_t
    infer_properties_node() override final {
        CUDNN_FE_LOG_LABEL_ENDL("INFO: Inferrencing properties for reduction node " << attributes.name << "...");

        attributes.fill_from_context(context);

        // Only inferrencing from IN_0 to OUT_0 works today.
        auto x_tensor = attributes.inputs[Reduction_attributes::input_names::X];
        auto y_tensor = attributes.outputs[Reduction_attributes::output_names::Y];

        auto const& x_tensor_dim = x_tensor->get_dim();
        auto y_tensor_dim        = y_tensor->get_dim();
        // Only infer dims and strides if user did not set them
        if (y_tensor_dim.empty()) {
            y_tensor->set_dim(x_tensor_dim);
        }
        if (y_tensor->get_stride().empty()) {
            auto const& y_dim = y_tensor->get_dim();
            // Default to NHWC
            auto const& stride_order = detail::generate_NHWC_stride_order(y_dim.size());
            y_tensor->set_stride(detail::generate_stride(y_dim, stride_order));
        }

        return {error_code_t::OK, ""};
    }

    error_t
    create_cudnn_operations(
        std::unordered_set<Tensor_attributes::uid_t>& uids_involved_in_operations,
        std::vector<std::shared_ptr<cudnn_frontend::Operation>>& operations,
        managed_backend_descriptor_t& raw_operations,
        std::unordered_map<int64_t, std::shared_ptr<cudnn_frontend::Tensor>>& tensors) const override final {
        CUDNN_FRONTEND_UNUSED(raw_operations);
        CUDNN_FE_LOG_LABEL_ENDL("INFO: " << "Building ReductionNode operations " << attributes.name << "...");

        auto reduction_descriptor = cudnn_frontend::ReductionDescBuilder()
                                        .setComputeType(attributes.compute_data_type)
                                        .setReductionOp(attributes.get_mode().value())
                                        .build();

        auto&& reduction_operation_builder =
            cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR);

        CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(X, Reduction_attributes::input_names::X);
        reduction_operation_builder.setxDesc(*(tensors.at(X->second->get_uid())));

        CUDNN_FE_VALIDATE_AND_ASSIGN_OUTPUT_TENSOR(Y, Reduction_attributes::output_names::Y);
        reduction_operation_builder.setyDesc(*(tensors.at(Y->second->get_uid())));

        reduction_operation_builder.setreductionDesc(reduction_descriptor);

#ifdef NV_CUDNN_DISABLE_EXCEPTION
        // disable exception macro is defined. Calling build will not throw.
        // Check status of desc and return error.
        auto operation = reduction_operation_builder.build();
        RETURN_CUDNN_FRONTEND_ERROR_IF(operation.get_status() != CUDNN_STATUS_SUCCESS,
                                       error_code_t::CUDNN_BACKEND_API_FAILED,
                                       operation.get_error());
        operations.push_back(std::make_shared<Operation_v8>(std::move(operation)));
#else
        // build() can throw
        // wrap in try catch
        try {
            auto operation = reduction_operation_builder.build();
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
        j.update(R"({"tag": "REDUCTION"})"_json);
    }
#endif
};

inline void
INode::reduction(std::shared_ptr<Tensor_attributes> a,
                 Reduction_attributes attributes,
                 std::shared_ptr<Tensor_attributes> c) {
    attributes.inputs[Reduction_attributes::input_names::X]   = a;
    attributes.outputs[Reduction_attributes::output_names::Y] = c;
    sub_nodes.emplace_back(std::make_unique<ReductionNode>(std::move(attributes), context));
}

inline std::shared_ptr<Tensor_attributes>
INode::reduction(std::shared_ptr<Tensor_attributes> input, Reduction_attributes attributes) {
    attributes.inputs[Reduction_attributes::input_names::X] = input;
    auto Y = attributes.outputs[Reduction_attributes::output_names::Y] = output_tensor(attributes.name + "::Y");

    sub_nodes.emplace_back(std::make_unique<ReductionNode>(std::move(attributes), context));
    return Y;
}
}  // namespace cudnn_frontend::graph