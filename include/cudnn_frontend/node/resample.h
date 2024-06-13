#pragma once

#include "../../cudnn_frontend_Resample.h"
#include "../../cudnn_frontend_Logging.h"

#include "../graph_helpers.h"
#include "../node_interface.h"

namespace cudnn_frontend::graph {

class ResampleNode : public NodeCRTP<ResampleNode> {
   public:
    Resample_attributes attributes;

    ResampleNode(Resample_attributes&& attributes_, detail::Context const& context)
        : NodeCRTP(context), attributes(std::move(attributes_)) {}

    Type
    getType() override final {
        return Type::RESAMPLE;
    }

    error_t
    pre_validate_node() const override final {
        getLogger() << "[cudnn_frontend] INFO: " << "Validating ResampleNode " << attributes.name << "..." << std::endl;

        CUDNN_FE_VALIDATE_INPUT_TENSOR(Resample_attributes::input_names::X);
        CUDNN_FE_VALIDATE_OUTPUT_TENSOR(Resample_attributes::output_names::Y);

        RETURN_CUDNN_FRONTEND_ERROR_IF(attributes.is_inference.has_value() == false,
                                       error_code_t::ATTRIBUTE_NOT_SET,
                                       "is_inference attribute not set");

        if (attributes.is_inference.value() == false && attributes.resample_mode == ResampleMode_t::MAXPOOL) {
            CUDNN_FE_VALIDATE_OUTPUT_TENSOR(Resample_attributes::output_names::Index);
        }

        // Make sure that the mode can be lowered to BE
        cudnnResampleMode_t dummy;
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            detail::convert_to_cudnn_type(attributes.resample_mode, dummy) != CUDNN_STATUS_SUCCESS,
            error_code_t::ATTRIBUTE_NOT_SET,
            "Invalid resample mode.");

        CHECK_CUDNN_FRONTEND_ERROR(attributes.validate_inputs());

        return {error_code_t::OK, ""};
    }

    error_t
    expand_and_infer_properties_node() override final {
        getLogger() << "[cudnn_frontend] INFO: Inferrencing properties for resample node " << attributes.name << "..."
                    << std::endl;

        auto y_tensor = attributes.outputs[Resample_attributes::output_names::Y];
        auto x_tensor = attributes.inputs[Resample_attributes::input_names::X];

        attributes.fill_from_context(context);

        // If user does not set shape and layout of the output tensor,
        // Get it from node attributes
        if (y_tensor->get_dim().empty()) {
            auto const x_dim = x_tensor->get_dim();
            auto y_dim       = y_tensor->get_dim();
            y_dim            = x_dim;

            // 2 cause first two dimensions are batch and channels
            for (auto dim = 2u; dim < x_dim.size(); ++dim) {
                auto spatial_dim = dim - 2u;
                y_dim[dim] =
                    1 + (x_dim[dim] + attributes.pre_padding[spatial_dim].numerator +
                         attributes.post_padding[spatial_dim].numerator - attributes.window[spatial_dim].numerator) /
                            attributes.stride[spatial_dim].numerator;
            }

            y_tensor->set_dim(y_dim);
        }

        // If layout is not set, generate the strides from layout
        if (y_tensor->get_stride().empty()) {
            auto const& y_dim = y_tensor->get_dim();
            // Default to NHWC
            auto const& stride_order = detail::generate_NHWC_stride_order(y_dim.size());
            y_tensor->set_stride(detail::generate_stride(y_dim, stride_order));
        }

        if (attributes.outputs[Resample_attributes::output_names::Index]) {
            auto index_tensor = attributes.outputs[Resample_attributes::output_names::Index];
            index_tensor->set_dim(y_tensor->get_dim());

            // If layout is not set, generate the strides from layout
            if (index_tensor->get_stride().empty()) {
                auto const& index_dim = index_tensor->get_dim();
                // Default to NHWC
                auto const& stride_order = detail::generate_NHWC_stride_order(index_dim.size());
                index_tensor->set_stride(detail::generate_stride(index_dim, stride_order));
            }
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
        getLogger() << "[cudnn_frontend] INFO: " << "Building ResampleNode operations " << attributes.name << "..."
                    << std::endl;

        auto number_of_spatial_dim = static_cast<int64_t>(attributes.window.size());

        // Define the resample descriptor
        auto resample_descriptor = cudnn_frontend::ResampleDescBuilder_v8()
                                       .setComputeType(attributes.compute_data_type)
                                       .setNanPropagation(CUDNN_PROPAGATE_NAN)
                                       .setResampleMode(attributes.resample_mode)
                                       .setPaddingMode(attributes.padding_mode)
                                       .setSpatialDim(number_of_spatial_dim, attributes.window.data())
                                       .setSpatialStride(number_of_spatial_dim, attributes.stride.data())
                                       .setPrePadding(number_of_spatial_dim, attributes.pre_padding.data())
                                       .setPostPadding(number_of_spatial_dim, attributes.post_padding.data())
                                       .build();

        auto&& resample_op_builder =
            cudnn_frontend::OperationBuilder(DescriptorType_t::OPERATION_RESAMPLE_FWD_DESCRIPTOR);

        CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(X, Resample_attributes::input_names::X);
        resample_op_builder.setxDesc(*(tensors.at(X->second->get_uid())));

        CUDNN_FE_VALIDATE_AND_ASSIGN_OUTPUT_TENSOR(Y, Resample_attributes::output_names::Y);
        resample_op_builder.setyDesc(*(tensors.at(Y->second->get_uid())));

        resample_op_builder.setResampleDesc(resample_descriptor);

        auto index = attributes.outputs.find(Resample_attributes::output_names::Index);
        if ((index != attributes.outputs.end()) && (index->second != nullptr)) {
            resample_op_builder.setidxDesc(*tensors.at(index->second->get_uid()));
        }

#ifdef NV_CUDNN_DISABLE_EXCEPTION
        // disable exception macro is defined. Calling build will not throw.
        // Check status of desc and return error.
        auto operation = resample_op_builder.build();
        RETURN_CUDNN_FRONTEND_ERROR_IF(operation.get_status() != CUDNN_STATUS_SUCCESS,
                                       error_code_t::CUDNN_BACKEND_API_FAILED,
                                       operation.get_error());
        operations.push_back(std::make_shared<Operation_v8>(std::move(operation)));
#else
        // build() can throw
        // wrap in try catch
        try {
            auto operation = resample_op_builder.build();
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
        j.update(R"( {"tag": "RESAMPLE"})"_json);
    }
#endif
};

inline std::array<std::shared_ptr<Tensor_attributes>, 2>
INode::resample(std::shared_ptr<Tensor_attributes> input, Resample_attributes attributes) {
    attributes.inputs[Resample_attributes::input_names::X] = input;
    auto Y = attributes.outputs[Resample_attributes::output_names::Y] = output_tensor(attributes.name + "::Y");
    std::shared_ptr<Tensor_attributes> Index                          = nullptr;
    if (attributes.is_inference.has_value() && attributes.is_inference.value() == false &&
        attributes.resample_mode == ResampleMode_t::MAXPOOL) {
        Index = attributes.outputs[Resample_attributes::output_names::Index] =
            output_tensor(attributes.name + "::Index");
    }

    sub_nodes.emplace_back(std::make_unique<ResampleNode>(std::move(attributes), context));
    return {Y, Index};
}

}  // namespace cudnn_frontend::graph