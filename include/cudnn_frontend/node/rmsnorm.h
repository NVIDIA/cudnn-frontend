#pragma once

#include "../../cudnn_frontend_Heuristics.h"
#include "../../cudnn_frontend_Logging.h"

#include "../graph_helpers.h"
#include "../node_interface.h"

namespace cudnn_frontend {

namespace graph {
class RMSNormNode : public NodeCRTP<RMSNormNode> {
   public:
    Rmsnorm_attributes attributes;

    RMSNormNode(Rmsnorm_attributes&& attributes_, detail::Context const& context)
        : NodeCRTP(context), attributes(std::move(attributes_)) {}

    Type
    getType() override final {
        return Type::RMSNORM;
    }

    error_t
    infer_properties_node() override final {
        CUDNN_FE_LOG_LABEL_ENDL("INFO: Inferencing properties for rmsnorm node " << attributes.name << "...");

        attributes.fill_from_context(context);

        auto X                  = attributes.inputs[Rmsnorm_attributes::input_names::X];
        auto const x_tensor_dim = X->get_dim();

        auto Y            = attributes.outputs[Rmsnorm_attributes::output_names::Y];
        auto y_tensor_dim = Y->get_dim();

        // Only infer dims and strides if user did not set them
        if (y_tensor_dim.empty()) {
            y_tensor_dim.resize(x_tensor_dim.size());
            Y->set_dim(x_tensor_dim);
        }
        if (Y->get_stride().empty()) {
            auto const& Y_dim = Y->get_dim();
            // Default to NHWC
            auto const& stride_order = detail::generate_NHWC_stride_order(Y_dim.size());
            Y->set_stride(detail::generate_stride(Y_dim, stride_order));
        }

        if (attributes.forward_phase == NormFwdPhase_t::TRAINING) {
            auto inv_var = attributes.outputs[Rmsnorm_attributes::output_names::INV_VARIANCE];
            if (auto inv_var_dim = inv_var->get_dim(); inv_var_dim.empty()) {
                inv_var_dim.resize(x_tensor_dim.size(), 1);
                inv_var_dim[0] = x_tensor_dim[0];
                inv_var->set_dim(inv_var_dim);
            }
            if (inv_var->get_stride().empty()) {
                auto const& inv_var_dim = inv_var->get_dim();
                // Default to NHWC
                auto const& stride_order = detail::generate_NHWC_stride_order(inv_var_dim.size());
                inv_var->set_stride(detail::generate_stride(inv_var_dim, stride_order));
            }
        }
        return {error_code_t::OK, ""};
    }

    error_t
    pre_validate_node() const override final {
        CUDNN_FE_LOG_LABEL_ENDL("INFO: Validating RMSNormNode " << attributes.name << "...");

        // Norm forward phase should be set
        RETURN_CUDNN_FRONTEND_ERROR_IF(attributes.forward_phase == NormFwdPhase_t::NOT_SET,
                                       error_code_t::ATTRIBUTE_NOT_SET,
                                       "Forward phase not set of rmsnorm node.");

        return {error_code_t::OK, ""};
    }

    error_t
    create_cudnn_operations(
        std::unordered_set<Tensor_attributes::uid_t>& uids_involved_in_operations,
        std::vector<std::shared_ptr<cudnn_frontend::Operation>>& operations,
        managed_backend_descriptor_t& raw_operations,
        std::unordered_map<int64_t, std::shared_ptr<cudnn_frontend::Tensor>>& tensors) const override final {
        CUDNN_FRONTEND_UNUSED(raw_operations);
        CUDNN_FE_LOG_LABEL_ENDL("INFO: Building RMSNormNode operations " << attributes.name << "...");

        auto&& rmsnorm_operation_builder =
            cudnn_frontend::OperationBuilder(DescriptorType_t::OPERATION_NORM_FORWARD_DESCRIPTOR);

        rmsnorm_operation_builder.setNormalizationMode(NormMode_t::RMS_NORM).setNormFwdPhase(attributes.forward_phase);

        CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(X, Rmsnorm_attributes::input_names::X);
        rmsnorm_operation_builder.setxDesc(*(tensors.at(X->second->get_uid())));

        CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(SCALE, Rmsnorm_attributes::input_names::SCALE);
        rmsnorm_operation_builder.setScale(*(tensors.at(SCALE->second->get_uid())));

        CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(EPSILON, Rmsnorm_attributes::input_names::EPSILON);
        rmsnorm_operation_builder.setEpsilonTensor(*(tensors.at(EPSILON->second->get_uid())));

        CUDNN_FE_VALIDATE_AND_ASSIGN_OUTPUT_TENSOR(Y, Rmsnorm_attributes::output_names::Y);
        rmsnorm_operation_builder.setyDesc(*(tensors.at(Y->second->get_uid())));

        if (attributes.forward_phase == NormFwdPhase_t::TRAINING) {
            CUDNN_FE_VALIDATE_AND_ASSIGN_OUTPUT_TENSOR(INV_VARIANCE, Rmsnorm_attributes::output_names::INV_VARIANCE);
            rmsnorm_operation_builder.setSavedInvVar(*(tensors.at(INV_VARIANCE->second->get_uid())));
        }

        auto BIAS = attributes.inputs.find(Rmsnorm_attributes::input_names::BIAS);
        if ((BIAS != attributes.inputs.end()) && (BIAS->second != nullptr)) {
            rmsnorm_operation_builder.setBias(*(tensors.at(BIAS->second->get_uid())));
        }
#ifdef NV_CUDNN_DISABLE_EXCEPTION
        // disable exception macro is defined. Calling build will not throw.
        // Check status of desc and return error.
        auto operation = rmsnorm_operation_builder.build();
        RETURN_CUDNN_FRONTEND_ERROR_IF(operation.get_status() != CUDNN_STATUS_SUCCESS,
                                       error_code_t::CUDNN_BACKEND_API_FAILED,
                                       operation.get_error());
        operations.push_back(std::make_shared<Operation_v8>(std::move(operation)));
#else
        // build() can throw
        // wrap in try catch
        try {
            auto operation = rmsnorm_operation_builder.build();
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
        j.update(R"( {"tag": "RMS_NORM"})"_json);
    }
#endif
};

class DRMSNormNode : public NodeCRTP<DRMSNormNode> {
   public:
    Rmsnorm_backward_attributes attributes;

    DRMSNormNode(Rmsnorm_backward_attributes&& attributes_, detail::Context const& context)
        : NodeCRTP(context), attributes(std::move(attributes_)) {}

    Type
    getType() override final {
        return Type::DRMSNorm;
    }

    error_t
    pre_validate_node() const override final {
        CUDNN_FE_LOG_LABEL_ENDL("INFO: Validating DRMSNormNode node " << attributes.name << "...");

        RETURN_CUDNN_FRONTEND_ERROR_IF(attributes.use_dbias.has_value() == false,
                                       error_code_t::ATTRIBUTE_NOT_SET,
                                       "DRMSNormNode node needs has_bias(bool) to be called.");

        return {error_code_t::OK, ""};
    }

    error_t
    infer_properties_node() override final {
        CUDNN_FE_LOG_LABEL_ENDL("INFO: Inferencing properties for DRMSNorm node " << attributes.name << "...");

        attributes.fill_from_context(context);

        // TODO: Only inferencing from X works today.
        auto X                  = attributes.inputs[Rmsnorm_backward_attributes::input_names::X];
        auto const x_tensor_dim = X->get_dim();

        auto DY            = attributes.inputs[Rmsnorm_backward_attributes::input_names::DY];
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

        auto DX            = attributes.outputs[Rmsnorm_backward_attributes::output_names::DX];
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

        infer_scale_bias_tensors(attributes.outputs[Rmsnorm_backward_attributes::output_names::DSCALE]);
        if (attributes.use_dbias.value()) {
            infer_scale_bias_tensors(attributes.outputs[Rmsnorm_backward_attributes::output_names::DBIAS]);
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
        CUDNN_FE_LOG_LABEL_ENDL("INFO: Building DRMSNormNode operations " << attributes.name << "...");

        auto&& DRMSNorm_operation_builder =
            cudnn_frontend::OperationBuilder(DescriptorType_t::OPERATION_NORM_BACKWARD_DESCRIPTOR);

        DRMSNorm_operation_builder.setNormalizationMode(NormMode_t::RMS_NORM);

        CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(X, Rmsnorm_backward_attributes::input_names::X);
        DRMSNorm_operation_builder.setxDesc(*(tensors.at(X->second->get_uid())));

        CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(DY, Rmsnorm_backward_attributes::input_names::DY);
        DRMSNorm_operation_builder.setdyDesc(*(tensors.at(DY->second->get_uid())));

        CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(SCALE, Rmsnorm_backward_attributes::input_names::SCALE);
        DRMSNorm_operation_builder.setScale(*(tensors.at(SCALE->second->get_uid())));

        CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(INV_VARIANCE, Rmsnorm_backward_attributes::input_names::INV_VARIANCE);
        DRMSNorm_operation_builder.setSavedInvVar(*(tensors.at(INV_VARIANCE->second->get_uid())));

        CUDNN_FE_VALIDATE_AND_ASSIGN_OUTPUT_TENSOR(DSCALE, Rmsnorm_backward_attributes::output_names::DSCALE);
        DRMSNorm_operation_builder.setDScale(*(tensors.at(DSCALE->second->get_uid())));

        if (attributes.use_dbias.value()) {
            CUDNN_FE_VALIDATE_AND_ASSIGN_OUTPUT_TENSOR(DBIAS, Rmsnorm_backward_attributes::output_names::DBIAS);
            DRMSNorm_operation_builder.setDBias(*(tensors.at(DBIAS->second->get_uid())));
        }

        CUDNN_FE_VALIDATE_AND_ASSIGN_OUTPUT_TENSOR(DX, Rmsnorm_backward_attributes::output_names::DX);
        DRMSNorm_operation_builder.setdxDesc(*(tensors.at(DX->second->get_uid())));
#ifdef NV_CUDNN_DISABLE_EXCEPTION
        // disable exception macro is defined. Calling build will not throw.
        // Check status of desc and return error.
        auto operation = DRMSNorm_operation_builder.build();
        RETURN_CUDNN_FRONTEND_ERROR_IF(operation.get_status() != CUDNN_STATUS_SUCCESS,
                                       error_code_t::CUDNN_BACKEND_API_FAILED,
                                       operation.get_error());
        operations.push_back(std::make_shared<Operation_v8>(std::move(operation)));
#else
        // build() can throw
        // wrap in try catch
        try {
            auto operation = DRMSNorm_operation_builder.build();
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
        j.update(R"( {"tag": "RMS_NORM_BPROP"})"_json);
    }
#endif
};

}  // namespace graph

}  // namespace cudnn_frontend