#pragma once

#include "../../cudnn_frontend_ConvDesc.h"
#include "../../cudnn_frontend_Heuristics.h"
#include "../../cudnn_frontend_Logging.h"

#include "../graph_helpers.h"
#include "../node_interface.h"

namespace cudnn_frontend::graph {

class WgradNode : public INode {
    Conv_wgrad_attributes attributes;

   public:
    WgradNode(Conv_wgrad_attributes&& attributes_, detail::Context const& context)
        : INode(context), attributes(std::move(attributes_)) {}

    Type
    getType() override final {
        return Type::WGRAD;
    }

    error_t
    pre_validate_node() const override final {
        getLogger() << "[cudnn_frontend] INFO: "
                    << "Validating Node Type::WGRAD " << attributes.name << "..." << std::endl;

        CUDNN_FE_VALIDATE_INPUT_TENSOR(Conv_wgrad_attributes::input_names::X);
        CUDNN_FE_VALIDATE_INPUT_TENSOR(Conv_wgrad_attributes::input_names::DY);

        CUDNN_FE_VALIDATE_OUTPUT_TENSOR(Conv_wgrad_attributes::output_names::DW);

        RETURN_CUDNN_FRONTEND_ERROR_IF(
            attributes.get_pre_padding().empty(), error_code_t::ATTRIBUTE_NOT_SET, "Pre padding not set.");
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            attributes.get_post_padding().empty(), error_code_t::ATTRIBUTE_NOT_SET, "Post padding not set.");
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            attributes.get_stride().empty(), error_code_t::ATTRIBUTE_NOT_SET, "Conv strides not set.");
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            attributes.get_dilation().empty(), error_code_t::ATTRIBUTE_NOT_SET, "Conv dilation not set.");

        CHECK_CUDNN_FRONTEND_ERROR(attributes.validate_inputs());
        return {error_code_t::OK, ""};
    }

    error_t
    expand_and_infer_properties() override final {
        getLogger() << "[cudnn_frontend] INFO: Inferrencing properties for conv node " << attributes.name << "."
                    << std::endl;

        attributes.fill_from_context(context);

        // TODO: Only inferrencing from (X, DY) -> DW works today.
        auto X  = attributes.inputs[Conv_wgrad_attributes::input_names::X];
        auto DW = attributes.outputs[Conv_wgrad_attributes::output_names::DW];
        auto DY = attributes.inputs[Conv_wgrad_attributes::input_names::DY];

        auto const x_tensor_dim  = X->get_dim();
        auto const dy_tensor_dim = DY->get_dim();
        auto dw_tensor_dim       = DW->get_dim();

        // No dim inferencing as inverse mapping from DY, X to DX is not unique.
        // Only infer strides if user did not set them
        if (DW->get_stride().empty()) {
            auto const& DW_dim = DW->get_dim();
            // Default to NHWC
            auto const& stride_order = detail::generate_NHWC_stride_order(DW_dim.size());
            DW->set_stride(detail::generate_stride(DW_dim, stride_order));
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
    collect_pre_assigned_uids(std::unordered_set<int64_t>& pre_assigned_uids) const override final {
        return attributes.get_prefilled_uids(pre_assigned_uids);
    }

    error_t
    create_cudnn_tensors(int64_t& uid,
                         std::unordered_map<int64_t, std::shared_ptr<cudnn_frontend::Tensor>>& tensors,
                         std::unordered_set<int64_t> const& invalid_uids) const override final {
        getLogger() << "[cudnn_frontend] INFO: "
                    << "Building WgradNode tensors " << attributes.name << "..." << std::endl;

        for (auto const& [name, tensor] : attributes.inputs) {
            (void)name;
            if (tensor) {
                CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(tensor, uid, tensors, invalid_uids));
            }
        }
        for (auto const& [name, tensor] : attributes.outputs) {
            (void)name;
            if (tensor) {
                CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(tensor, uid, tensors, invalid_uids));
            }
        }
        return {error_code_t::OK, ""};
    }

    error_t
    create_cudnn_operations(
        std::unordered_set<uid_t>& uids_involved_in_operations,
        std::vector<std::shared_ptr<cudnn_frontend::Operation>>& operations,
        std::unordered_map<int64_t, std::shared_ptr<cudnn_frontend::Tensor>>& tensors) const override final {
        getLogger() << "[cudnn_frontend] INFO: "
                    << "Building WgradNode operations " << attributes.name << "..." << std::endl;

        // wgrad descriptor
        int64_t const spatial_dim_count = attributes.get_pre_padding().size();
        auto wgrad_descriptor           = cudnn_frontend::ConvDescBuilder()
                                    .setComputeType(attributes.compute_data_type)
                                    .setMathMode(CUDNN_CROSS_CORRELATION)
                                    .setSpatialDimCount(spatial_dim_count)
                                    .setSpatialStride(spatial_dim_count, attributes.get_stride().data())
                                    .setPrePadding(spatial_dim_count, attributes.get_pre_padding().data())
                                    .setPostPadding(spatial_dim_count, attributes.get_post_padding().data())
                                    .setDilation(spatial_dim_count, attributes.get_dilation().data())
                                    .build();

        // Create the wgrad operation.
        auto&& wgrad_operation_builder =
            cudnn_frontend::OperationBuilder(DescriptorType_t::OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR);

        CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(X, Conv_wgrad_attributes::input_names::X);
        wgrad_operation_builder.setxDesc(*(tensors.at(X->second->get_uid())));

        CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(DY, Conv_wgrad_attributes::input_names::DY);
        wgrad_operation_builder.setdyDesc(*(tensors.at(DY->second->get_uid())));

        CUDNN_FE_VALIDATE_AND_ASSIGN_OUTPUT_TENSOR(DW, Conv_wgrad_attributes::output_names::DW);
        wgrad_operation_builder.setdwDesc(*(tensors.at(DW->second->get_uid())));

        wgrad_operation_builder.setcDesc(wgrad_descriptor).setAlpha(1.f).setBeta(0.f);

#ifdef NV_CUDNN_DISABLE_EXCEPTION
        // disable exception macro is defined. Calling build will not throw.
        // Check status of desc and return error.
        auto operation = wgrad_operation_builder.build();
        RETURN_CUDNN_FRONTEND_ERROR_IF(operation.get_status() != CUDNN_STATUS_SUCCESS,
                                       error_code_t::CUDNN_BACKEND_API_FAILED,
                                       operation.get_error());
        operations.push_back(std::make_shared<Operation_v8>(std::move(operation)));
#else
        // build() can throw
        // wrap in try catch
        try {
            auto operation = wgrad_operation_builder.build();
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
        j.update(R"( {"tag": "CONV_WGRAD"})"_json);
    }
};

}  // namespace cudnn_frontend::graph