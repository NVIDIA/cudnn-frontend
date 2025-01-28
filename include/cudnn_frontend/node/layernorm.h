#pragma once

#include "../../cudnn_frontend_Heuristics.h"
#include "../../cudnn_frontend_Logging.h"

#include "../graph_helpers.h"
#include "../node_interface.h"

namespace cudnn_frontend {

namespace graph {
class LayerNormNode : public NodeCRTP<LayerNormNode> {
   public:
    Layernorm_attributes attributes;

    LayerNormNode(Layernorm_attributes&& attributes_, detail::Context const& context)
        : NodeCRTP(context), attributes(std::move(attributes_)) {}

    Type
    getType() override final {
        return Type::LAYERNORM;
    }

    error_t
    infer_properties_node() override final {
        CUDNN_FE_LOG_LABEL_ENDL("INFO: Inferencing properties for layernorm node " << attributes.name << "...");

        attributes.fill_from_context(context);

        auto X = attributes.inputs[Layernorm_attributes::input_names::X];
        auto Y = attributes.outputs[Layernorm_attributes::output_names::Y];

        // Only infer dims and strides if user did not set them
        if (Y->get_dim().empty()) {
            Y->set_dim(X->get_dim());
        }
        if (Y->get_stride().empty()) {
            Y->set_stride(X->get_stride());
        }

        // scale_bias dim is 1,c,h,w
        auto scale_bias_dim = X->get_dim();
        scale_bias_dim[0]   = 1;

        auto scale = attributes.inputs[Layernorm_attributes::input_names::SCALE];
        // Only infer dims and strides if user did not set them
        if (scale->get_dim().empty()) {
            scale->set_dim(scale_bias_dim);
        }
        if (scale->get_stride().empty()) {
            auto const& scale_dim = scale->get_dim();
            std::vector<int64_t> stride_order;
            CHECK_CUDNN_FRONTEND_ERROR(
                detail::generate_stride_order_preserving_format(X->get_stride(), scale_dim.size(), stride_order));
            scale->set_stride(detail::generate_stride(scale_dim, stride_order));
        }

        auto bias = attributes.inputs[Layernorm_attributes::input_names::BIAS];
        // Only infer dims and strides if user did not set them
        if (bias->get_dim().empty()) {
            bias->set_dim(scale_bias_dim);
        }
        if (bias->get_stride().empty()) {
            auto const& bias_dim = bias->get_dim();
            std::vector<int64_t> stride_order;
            CHECK_CUDNN_FRONTEND_ERROR(
                detail::generate_stride_order_preserving_format(X->get_stride(), bias_dim.size(), stride_order));
            bias->set_stride(detail::generate_stride(bias_dim, stride_order));
        }

        if (attributes.forward_phase == NormFwdPhase_t::TRAINING) {
            // stats dim is x where scale == 1 else 1
            auto stats_dim = X->get_dim();
            for (size_t i = 0; i < stats_dim.size(); i++) {
                if (scale->get_dim()[i] != 1) {
                    stats_dim[i] = 1;
                }
            }

            auto mean = attributes.outputs[Layernorm_attributes::output_names::MEAN];
            // Only infer dims and strides if user did not set them
            if (mean->get_dim().empty()) {
                mean->set_dim(stats_dim);
            }
            if (mean->get_stride().empty()) {
                auto const& mean_dim = mean->get_dim();
                std::vector<int64_t> stride_order;
                CHECK_CUDNN_FRONTEND_ERROR(
                    detail::generate_stride_order_preserving_format(X->get_stride(), mean_dim.size(), stride_order));
                mean->set_stride(detail::generate_stride(mean_dim, stride_order));
            }

            auto inv_var = attributes.outputs[Layernorm_attributes::output_names::INV_VARIANCE];
            // Only infer dims and strides if user did not set them
            if (inv_var->get_dim().empty()) {
                inv_var->set_dim(stats_dim);
            }
            if (inv_var->get_stride().empty()) {
                auto const& inv_var_dim = inv_var->get_dim();
                std::vector<int64_t> stride_order;
                CHECK_CUDNN_FRONTEND_ERROR(
                    detail::generate_stride_order_preserving_format(X->get_stride(), inv_var_dim.size(), stride_order));
                inv_var->set_stride(detail::generate_stride(inv_var_dim, stride_order));
            }
        }

        // Set scalar tensors
        std::vector<int64_t> ones(X->get_dim().size(), 1);
        auto infer_scalar_tensors = [&ones](std::shared_ptr<Tensor_attributes>& T) {
            // Only infer dims and strides if user did not set them
            if (T->get_dim().empty()) {
                T->set_dim(ones);
            }
            if (T->get_stride().empty()) {
                T->set_stride(ones);
            }
        };
        infer_scalar_tensors(attributes.inputs[Layernorm_attributes::input_names::EPSILON]);

        return {error_code_t::OK, ""};
    }

    error_t
    pre_validate_node() const override final {
        CUDNN_FE_LOG_LABEL_ENDL("INFO: " << "Validating LayerNormNode " << attributes.name << "...");

        // Norm forward phase should be set
        RETURN_CUDNN_FRONTEND_ERROR_IF(attributes.forward_phase == NormFwdPhase_t::NOT_SET,
                                       error_code_t::ATTRIBUTE_NOT_SET,
                                       "Forward phase not set of layernorm node.");

        return {error_code_t::OK, ""};
    }

    error_t
    create_cudnn_operations(
        std::unordered_set<Tensor_attributes::uid_t>& uids_involved_in_operations,
        std::vector<std::shared_ptr<cudnn_frontend::Operation>>& operations,
        managed_backend_descriptor_t& raw_operations,
        std::unordered_map<int64_t, std::shared_ptr<cudnn_frontend::Tensor>>& tensors) const override final {
        CUDNN_FRONTEND_UNUSED(raw_operations);
        CUDNN_FE_LOG_LABEL_ENDL("INFO: " << "Building LayerNormNode operations " << attributes.name << "...");

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

#ifndef CUDNN_FRONTEND_SKIP_JSON_LIB
    virtual void
    serialize(json& j) const override final {
        j = attributes;
        j.update(R"( {"tag": "LAYER_NORM"})"_json);
    }
#endif
};

}  // namespace graph

}  // namespace cudnn_frontend