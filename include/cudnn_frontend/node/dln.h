#pragma once

#include "../../cudnn_frontend_Heuristics.h"
#include "../../cudnn_frontend_Logging.h"

#include "../graph_helpers.h"
#include "../node_interface.h"

namespace cudnn_frontend {

namespace graph {

class DLNNode : public NodeCRTP<DLNNode> {
   public:
    Layernorm_backward_attributes attributes;

    DLNNode(Layernorm_backward_attributes&& attributes_, detail::Context const& context)
        : NodeCRTP(context), attributes(std::move(attributes_)) {}

    Type
    getType() override final {
        return Type::DLN;
    }
    error_t
    infer_properties_node() override final {
        CUDNN_FE_LOG_LABEL_ENDL("INFO: Inferencing properties for DLN node " << attributes.name << "...");

        // WAR as epsilon was required in previous versions
        if (detail::get_backend_version() < 8906) {
            attributes.inputs[Layernorm_backward_attributes::input_names::EPSILON] =
                std::make_shared<Tensor_attributes>(0.0f);
        }

        attributes.fill_from_context(context);

        // TODO: Only inferencing from X works today.
        auto X                  = attributes.inputs[Layernorm_backward_attributes::input_names::X];
        auto const x_tensor_dim = X->get_dim();

        auto DY            = attributes.inputs[Layernorm_backward_attributes::input_names::DY];
        auto dy_tensor_dim = DY->get_dim();

        // Only infer dims and strides if user did not set them
        if (dy_tensor_dim.empty()) {
            dy_tensor_dim.resize(x_tensor_dim.size());
            DY->set_dim(x_tensor_dim);
        }
        if (DY->get_stride().empty()) {
            auto const& DY_dim = DY->get_dim();
            // Default to NHWC
            auto const& stride_order = detail::generate_NHWC_stride_order(DY_dim.size());
            DY->set_stride(detail::generate_stride(DY_dim, stride_order));
        }

        auto DX            = attributes.outputs[Layernorm_backward_attributes::output_names::DX];
        auto dx_tensor_dim = DX->get_dim();
        // Only infer dims and strides if user did not set them
        if (dx_tensor_dim.empty()) {
            dx_tensor_dim.resize(x_tensor_dim.size());
            DX->set_dim(x_tensor_dim);
        }
        if (DX->get_stride().empty()) {
            auto const& DX_dim = DX->get_dim();
            // Default to NHWC
            auto const& stride_order = detail::generate_NHWC_stride_order(DX_dim.size());
            DX->set_stride(detail::generate_stride(DX_dim, stride_order));
        }

        auto scale_bias_dim = X->get_dim();
        scale_bias_dim[0]   = 1;

        // Set channel length tensors
        auto infer_scale_bias_tensors = [&scale_bias_dim](std::shared_ptr<Tensor_attributes>& T) {
            auto tensor_dim = T->get_dim();
            // Only infer dims and strides if user did not set them
            if (tensor_dim.empty()) {
                T->set_dim(scale_bias_dim);
            }
            if (T->get_stride().empty()) {
                auto const& T_dim = T->get_dim();
                // Default to NHWC
                auto const& stride_order = detail::generate_NHWC_stride_order(T_dim.size());
                T->set_stride(detail::generate_stride(T_dim, stride_order));
            }
        };

        infer_scale_bias_tensors(attributes.outputs[Layernorm_backward_attributes::output_names::DSCALE]);
        infer_scale_bias_tensors(attributes.outputs[Layernorm_backward_attributes::output_names::DBIAS]);

        return {error_code_t::OK, ""};
    }

    error_t
    create_cudnn_operations(
        std::unordered_set<Tensor_attributes::uid_t>& uids_involved_in_operations,
        std::vector<std::shared_ptr<cudnn_frontend::Operation>>& operations,
        managed_backend_descriptor_t& raw_operations,
        std::unordered_map<int64_t, std::shared_ptr<cudnn_frontend::Tensor>>& tensors) const override final {
        CUDNN_FRONTEND_UNUSED(raw_operations);
        CUDNN_FE_LOG_LABEL_ENDL("INFO: " << "Building DLNNode operations " << attributes.name << "...");

        // Create the DLN operation.
        auto&& DLN_op_builder = cudnn_frontend::OperationBuilder(DescriptorType_t::OPERATION_NORM_BACKWARD_DESCRIPTOR);

        DLN_op_builder.setNormalizationMode(NormMode_t::LAYER_NORM);

        CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(X, Layernorm_backward_attributes::input_names::X);
        DLN_op_builder.setxDesc(*(tensors.at(X->second->get_uid())));

        CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(DY, Layernorm_backward_attributes::input_names::DY);
        DLN_op_builder.setdyDesc(*(tensors.at(DY->second->get_uid())));

        CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(SCALE, Layernorm_backward_attributes::input_names::SCALE);
        DLN_op_builder.setScale(*(tensors.at(SCALE->second->get_uid())));

        CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(MEAN, Layernorm_backward_attributes::input_names::MEAN);
        CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(INV_VARIANCE,
                                                  Layernorm_backward_attributes::input_names::INV_VARIANCE);
        DLN_op_builder.setSavedMeanAndInvVar(*(tensors.at(MEAN->second->get_uid())),
                                             *(tensors.at(INV_VARIANCE->second->get_uid())));

        CUDNN_FE_VALIDATE_AND_ASSIGN_OUTPUT_TENSOR(DSCALE, Layernorm_backward_attributes::output_names::DSCALE);
        CUDNN_FE_VALIDATE_AND_ASSIGN_OUTPUT_TENSOR(DBIAS, Layernorm_backward_attributes::output_names::DBIAS);
        DLN_op_builder.setDScaleAndDBias(*(tensors.at(DSCALE->second->get_uid())),
                                         *(tensors.at(DBIAS->second->get_uid())));

        CUDNN_FE_VALIDATE_AND_ASSIGN_OUTPUT_TENSOR(DX, Layernorm_backward_attributes::output_names::DX);
        DLN_op_builder.setdxDesc(*(tensors.at(DX->second->get_uid())));

        if (detail::get_backend_version() < 8906) {
            CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(EPSILON, Layernorm_backward_attributes::input_names::EPSILON);
            DLN_op_builder.setEpsilonTensor(*(tensors.at(EPSILON->second->get_uid())));
        }

#ifdef NV_CUDNN_DISABLE_EXCEPTION
        // disable exception macro is defined. Calling build will not throw.
        // Check status of desc and return error.
        auto operation = DLN_op_builder.build();
        RETURN_CUDNN_FRONTEND_ERROR_IF(operation.get_status() != CUDNN_STATUS_SUCCESS,
                                       error_code_t::CUDNN_BACKEND_API_FAILED,
                                       operation.get_error());
        operations.push_back(std::make_shared<Operation_v8>(std::move(operation)));
#else
        // build() can throw
        // wrap in try catch
        try {
            auto operation = DLN_op_builder.build();
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
        j.update(R"( {"tag": "LAYER_NORM_BPROP"})"_json);
    }
#endif
};

}  // namespace graph

}  // namespace cudnn_frontend