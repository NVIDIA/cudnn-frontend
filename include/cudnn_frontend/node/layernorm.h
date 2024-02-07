#pragma once

#include "../../cudnn_frontend_Heuristics.h"
#include "../../cudnn_frontend_Logging.h"

#include "../graph_helpers.h"
#include "../node_interface.h"

namespace cudnn_frontend {

namespace graph {
class LayerNormNode : public INode {
   public:
    Layernorm_attributes attributes;

    LayerNormNode(Layernorm_attributes&& attributes_, detail::Context const& context)
        : INode(context), attributes(std::move(attributes_)) {}

    Type
    getType() override final {
        return Type::LAYERNORM;
    }

    error_t
    collect_pre_assigned_uids(std::unordered_set<int64_t>& pre_assigned_uids) const override final {
        return attributes.get_prefilled_uids(pre_assigned_uids);
    }

    error_t
    expand_and_infer_properties() override final {
        getLogger() << "[cudnn_frontend] INFO: Inferencing properties for layernorm node " << attributes.name << "..."
                    << std::endl;

        attributes.fill_from_context(context);

        auto X                  = attributes.inputs[Layernorm_attributes::input_names::X];
        auto const x_tensor_dim = X->get_dim();

        auto Y            = attributes.outputs[Layernorm_attributes::output_names::Y];
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

        // scale_bias   dim is 1,c,h,w
        // mean inv_var dim is n,1,1,1
        auto scale_bias_dim = X->get_dim();
        scale_bias_dim[0]   = 1;

        auto stats_dim = X->get_dim();
        for (size_t i = 1; i < stats_dim.size(); i++) {
            stats_dim[i] = 1;
        }

        auto scale = attributes.inputs[Layernorm_attributes::input_names::SCALE];
        if (scale->get_dim().empty()) {
            scale->set_dim(scale_bias_dim);
        }
        if (scale->get_stride().empty()) {
            auto const& scale_dim = scale->get_dim();
            // Default to NHWC
            auto const& stride_order = detail::generate_NHWC_stride_order(scale_dim.size());
            scale->set_stride(detail::generate_stride(scale_dim, stride_order));
        }

        auto bias = attributes.inputs[Layernorm_attributes::input_names::BIAS];
        if (bias->get_dim().empty()) {
            bias->set_dim(scale_bias_dim);
        }
        if (bias->get_stride().empty()) {
            auto const& bias_dim = bias->get_dim();
            // Default to NHWC
            auto const& stride_order = detail::generate_NHWC_stride_order(bias_dim.size());
            bias->set_stride(detail::generate_stride(bias_dim, stride_order));
        }

        if (attributes.forward_phase == NormFwdPhase_t::TRAINING) {
            auto mean = attributes.outputs[Layernorm_attributes::output_names::MEAN];
            if (mean->get_dim().empty()) {
                mean->set_dim(stats_dim);
            }
            if (mean->get_stride().empty()) {
                auto const& mean_dim = mean->get_dim();
                // Default to NHWC
                auto const& stride_order = detail::generate_NHWC_stride_order(mean_dim.size());
                mean->set_stride(detail::generate_stride(mean_dim, stride_order));
            }

            auto inv_var = attributes.outputs[Layernorm_attributes::output_names::INV_VARIANCE];
            if (inv_var->get_dim().empty()) {
                inv_var->set_dim(stats_dim);
            }
            if (inv_var->get_stride().empty()) {
                auto const& inv_var_dim = inv_var->get_dim();
                // Default to NHWC
                auto const& stride_order = detail::generate_NHWC_stride_order(inv_var_dim.size());
                inv_var->set_stride(detail::generate_stride(inv_var_dim, stride_order));
            }
        }

        // Set scalar tensors
        auto infer_scalar_tensors = [&x_tensor_dim](std::shared_ptr<Tensor_attributes>& T) {
            auto tensor_dim = T->get_dim();
            // Only infer dims and strides if user did not set them
            if (tensor_dim.empty()) {
                tensor_dim.resize(x_tensor_dim.size(), 1);
                T->set_dim(tensor_dim);
            }
            if (T->get_stride().empty()) {
                auto const& T_dim = T->get_dim();
                // Default to NHWC
                auto const& stride_order = detail::generate_NHWC_stride_order(T_dim.size());
                T->set_stride(detail::generate_stride(T_dim, stride_order));
            }
        };
        infer_scalar_tensors(attributes.inputs[Layernorm_attributes::input_names::EPSILON]);

        return {error_code_t::OK, ""};
    }

    error_t
    pre_validate_node() const override final {
        getLogger() << "[cudnn_frontend] INFO: "
                    << "Validating LayerNormNode " << attributes.name << "..." << std::endl;

        // Norm forward phase should be set
        RETURN_CUDNN_FRONTEND_ERROR_IF(attributes.forward_phase == NormFwdPhase_t::NOT_SET,
                                       error_code_t::ATTRIBUTE_NOT_SET,
                                       "Forward phase not set of layernorm node.");

        CHECK_CUDNN_FRONTEND_ERROR(attributes.validate_inputs());

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
                    << "Building LayerNormNode tensors " << attributes.name << "..." << std::endl;

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
                    << "Building LayerNormNode operations " << attributes.name << "..." << std::endl;

        auto&& layernorm_operation_builder =
            cudnn_frontend::OperationBuilder(DescriptorType_t::OPERATION_NORM_FORWARD_DESCRIPTOR);
        layernorm_operation_builder.setNormalizationMode(NormMode_t::LAYER_NORM)
            .setNormFwdPhase(attributes.forward_phase);

        CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(X, Layernorm_attributes::input_names::X);
        layernorm_operation_builder.setxDesc(*(tensors.at(X->second->get_uid())));

        CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(SCALE, Layernorm_attributes::input_names::SCALE);
        CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(BIAS, Layernorm_attributes::input_names::BIAS);
        layernorm_operation_builder.setScaleAndBias(*(tensors.at(SCALE->second->get_uid())),
                                                    *(tensors.at(BIAS->second->get_uid())));

        CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(EPSILON, Layernorm_attributes::input_names::EPSILON);
        layernorm_operation_builder.setEpsilonTensor(*(tensors.at(EPSILON->second->get_uid())));

        CUDNN_FE_VALIDATE_AND_ASSIGN_OUTPUT_TENSOR(Y, Layernorm_attributes::output_names::Y);
        layernorm_operation_builder.setyDesc(*(tensors.at(Y->second->get_uid())));

        if (attributes.forward_phase == NormFwdPhase_t::TRAINING) {
            CUDNN_FE_VALIDATE_AND_ASSIGN_OUTPUT_TENSOR(MEAN, Layernorm_attributes::output_names::MEAN);
            CUDNN_FE_VALIDATE_AND_ASSIGN_OUTPUT_TENSOR(INV_VARIANCE, Layernorm_attributes::output_names::INV_VARIANCE);
            layernorm_operation_builder.setSavedMeanAndInvVar(*(tensors.at(MEAN->second->get_uid())),
                                                              *(tensors.at(INV_VARIANCE->second->get_uid())));
        }
#ifdef NV_CUDNN_DISABLE_EXCEPTION
        // disable exception macro is defined. Calling build will not throw.
        // Check status of desc and return error.
        auto operation = layernorm_operation_builder.build();
        RETURN_CUDNN_FRONTEND_ERROR_IF(operation.get_status() != CUDNN_STATUS_SUCCESS,
                                       error_code_t::CUDNN_BACKEND_API_FAILED,
                                       operation.get_error());
        operations.push_back(std::make_shared<Operation_v8>(std::move(operation)));
#else
        // build() can throw
        // wrap in try catch
        try {
            auto operation = layernorm_operation_builder.build();
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
        j.update(R"( {"tag": "LAYER_NORM"})"_json);
    }
};

}  // namespace graph

}  // namespace cudnn_frontend