#pragma once

#include "../../cudnn_frontend_ConvDesc.h"
#include "../../cudnn_frontend_Heuristics.h"
#include "../../cudnn_frontend_Logging.h"

#include "../graph_helpers.h"
#include "../node_interface.h"

namespace cudnn_frontend::graph {

class ConvolutionNode : public INode {
   public:
    Conv_fprop_attributes attributes;

    ConvolutionNode(Conv_fprop_attributes&& attributes_, detail::Context const& context)
        : INode(context), attributes(std::move(attributes_)) {}

    Type
    getType() override final {
        return Type::CONVOLUTION;
    }

    error_t
    collect_pre_assigned_uids(std::unordered_set<int64_t>& pre_assigned_uids) const override final {
        return attributes.get_prefilled_uids(pre_assigned_uids);
    }

    error_t
    pre_validate_node() const override final {
        getLogger() << "[cudnn_frontend] INFO: "
                    << "Validating Node Type::CONVOLUTION " << attributes.name << "..." << std::endl;

        CUDNN_FE_VALIDATE_INPUT_TENSOR(Conv_fprop_attributes::input_names::X);
        CUDNN_FE_VALIDATE_INPUT_TENSOR(Conv_fprop_attributes::input_names::W);

        CUDNN_FE_VALIDATE_OUTPUT_TENSOR(Conv_fprop_attributes::output_names::Y);

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
        getLogger() << "[cudnn_frontend] INFO: Inferrencing properties for conv node " << attributes.name << "..."
                    << std::endl;

        attributes.fill_from_context(context);

        // TODO: Only inferrencing from (X, W) -> Y works today.
        auto& X = attributes.inputs.find(Conv_fprop_attributes::input_names::X)->second;
        auto& W = attributes.inputs.find(Conv_fprop_attributes::input_names::W)->second;
        auto& Y = attributes.outputs.find(Conv_fprop_attributes::output_names::Y)->second;

        auto const x_tensor_dim = X->get_dim();
        auto const w_tensor_dim = W->get_dim();
        auto y_tensor_dim       = Y->get_dim();

        // Only infer dims and strides if user did not set them
        if (y_tensor_dim.empty()) {
            y_tensor_dim.resize(x_tensor_dim.size());
            auto const& pre_padding  = attributes.get_pre_padding();
            auto const& post_padding = attributes.get_post_padding();
            auto const& stride       = attributes.get_stride();
            auto const& dilation     = attributes.get_dilation();
            // N
            y_tensor_dim[0] = x_tensor_dim[0];
            // PQ
            for (size_t dim = 2; dim < x_tensor_dim.size(); ++dim) {
                y_tensor_dim[dim] = 1 + (x_tensor_dim[dim] - dilation[dim - 2] * (w_tensor_dim[dim] - 1) - 1 +
                                         pre_padding[dim - 2] + post_padding[dim - 2]) /
                                            stride[dim - 2];
            }
            // K
            y_tensor_dim[1] = w_tensor_dim[0];
            Y->set_dim(y_tensor_dim);
        }
        if (Y->get_stride().empty()) {
            auto const& Y_dim = Y->get_dim();
            // Default to NHWC
            auto const& stride_order = detail::generate_NHWC_stride_order(Y_dim.size());
            Y->set_stride(detail::generate_stride(Y_dim, stride_order));
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
    create_cudnn_tensors(int64_t& uid,
                         std::unordered_map<int64_t, std::shared_ptr<cudnn_frontend::Tensor>>& tensors,
                         std::unordered_set<int64_t> const& invalid_uids) const override final {
        getLogger() << "[cudnn_frontend] INFO: "
                    << "Building ConvolutionNode tensors " << attributes.name << "..." << std::endl;

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
                    << "Building ConvolutionNode operations " << attributes.name << "..." << std::endl;

        // convolution descriptor
        int64_t const spatial_dim_count = attributes.get_pre_padding().size();
        auto convolution_descriptor     = cudnn_frontend::ConvDescBuilder()
                                          .setComputeType(attributes.compute_data_type)
                                          .setMathMode(CUDNN_CROSS_CORRELATION)
                                          .setSpatialDimCount(spatial_dim_count)
                                          .setSpatialStride(spatial_dim_count, attributes.get_stride().data())
                                          .setPrePadding(spatial_dim_count, attributes.get_pre_padding().data())
                                          .setPostPadding(spatial_dim_count, attributes.get_post_padding().data())
                                          .setDilation(spatial_dim_count, attributes.get_dilation().data())
                                          .build();

        // Create the convolution operation.
        auto&& convolution_operation_builder =
            cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR);

        CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(X, Conv_fprop_attributes::input_names::X);
        convolution_operation_builder.setxDesc(*(tensors[X->second->get_uid()]));

        CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(W, Conv_fprop_attributes::input_names::W);
        convolution_operation_builder.setwDesc(*(tensors[W->second->get_uid()]));

        CUDNN_FE_VALIDATE_AND_ASSIGN_OUTPUT_TENSOR(Y, Conv_fprop_attributes::output_names::Y);
        convolution_operation_builder.setyDesc(*(tensors[Y->second->get_uid()]));

        convolution_operation_builder.setcDesc(convolution_descriptor).setAlpha(1.f).setBeta(0.f);

#ifdef NV_CUDNN_DISABLE_EXCEPTION
        // disable exception macro is defined. Calling build will not throw.
        // Check status of desc and return error.
        auto operation = convolution_operation_builder.build();
        RETURN_CUDNN_FRONTEND_ERROR_IF(operation.get_status() != CUDNN_STATUS_SUCCESS,
                                       error_code_t::CUDNN_BACKEND_API_FAILED,
                                       operation.get_error());
        operations.push_back(std::make_shared<Operation_v8>(std::move(operation)));
#else
        // build() can throw
        // wrap in try catch
        try {
            auto operation = convolution_operation_builder.build();
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
        j.update(R"({"tag": "CONV_FPROP"})"_json);
    }
};

}  // namespace cudnn_frontend::graph