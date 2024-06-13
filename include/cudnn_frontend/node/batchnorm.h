#pragma once

#include "../../cudnn_frontend_Heuristics.h"
#include "../../cudnn_frontend_Logging.h"

#include "../graph_helpers.h"
#include "../node_interface.h"

namespace cudnn_frontend {

namespace graph {
class BatchNormNode : public NodeCRTP<BatchNormNode> {
   public:
    Batchnorm_attributes attributes;

    BatchNormNode(Batchnorm_attributes&& attributes_, detail::Context const& context)
        : NodeCRTP(context), attributes(std::move(attributes_)) {}

    Type
    getType() override final {
        return Type::BATCHNORM;
    }

    error_t
    expand_and_infer_properties_node() override final {
        getLogger() << "[cudnn_frontend] INFO: Inferencing properties for batchnorm node " << attributes.name << "..."
                    << std::endl;

        attributes.fill_from_context(context);

        auto X                  = attributes.inputs[Batchnorm_attributes::input_names::X];
        auto const x_tensor_dim = X->get_dim();

        auto Y            = attributes.outputs[Batchnorm_attributes::output_names::Y];
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

        // Set channel length tensors
        auto infer_per_channel_tensors = [&x_tensor_dim](std::shared_ptr<Tensor_attributes>& T) {
            auto tensor_dim = T->get_dim();
            // Only infer dims and strides if user did not set them
            if (tensor_dim.empty()) {
                tensor_dim.resize(x_tensor_dim.size(), 1);
                tensor_dim[1] = x_tensor_dim[1];
                T->set_dim(tensor_dim);
            }
            if (T->get_stride().empty()) {
                auto const& T_dim = T->get_dim();
                // Default to NHWC
                auto const& stride_order = detail::generate_NHWC_stride_order(T_dim.size());
                T->set_stride(detail::generate_stride(T_dim, stride_order));
            }
        };
        infer_per_channel_tensors(attributes.outputs[Batchnorm_attributes::output_names::MEAN]);
        infer_per_channel_tensors(attributes.outputs[Batchnorm_attributes::output_names::INV_VARIANCE]);
        infer_per_channel_tensors(attributes.outputs[Batchnorm_attributes::output_names::NEXT_RUNNING_MEAN]);
        infer_per_channel_tensors(attributes.outputs[Batchnorm_attributes::output_names::NEXT_RUNNING_VAR]);

        return {error_code_t::OK, ""};
    }

    error_t
    pre_validate_node() const override final {
        getLogger() << "[cudnn_frontend] INFO: " << "Validating BatchNormNode " << attributes.name << "..."
                    << std::endl;

        // Ensure all needed input output tensors are valid
        CUDNN_FE_VALIDATE_INPUT_TENSOR(Batchnorm_attributes::input_names::X);
        CUDNN_FE_VALIDATE_INPUT_TENSOR(Batchnorm_attributes::input_names::SCALE);
        CUDNN_FE_VALIDATE_INPUT_TENSOR(Batchnorm_attributes::input_names::BIAS);
        CUDNN_FE_VALIDATE_INPUT_TENSOR(Batchnorm_attributes::input_names::PREV_RUNNING_MEAN);
        CUDNN_FE_VALIDATE_INPUT_TENSOR(Batchnorm_attributes::input_names::PREV_RUNNING_VAR);
        CUDNN_FE_VALIDATE_INPUT_TENSOR(Batchnorm_attributes::input_names::EPSILON);
        CUDNN_FE_VALIDATE_INPUT_TENSOR(Batchnorm_attributes::input_names::MOMENTUM);

        CUDNN_FE_VALIDATE_OUTPUT_TENSOR(Batchnorm_attributes::output_names::Y);
        CUDNN_FE_VALIDATE_OUTPUT_TENSOR(Batchnorm_attributes::output_names::MEAN);
        CUDNN_FE_VALIDATE_OUTPUT_TENSOR(Batchnorm_attributes::output_names::INV_VARIANCE);
        CUDNN_FE_VALIDATE_OUTPUT_TENSOR(Batchnorm_attributes::output_names::NEXT_RUNNING_MEAN);
        CUDNN_FE_VALIDATE_OUTPUT_TENSOR(Batchnorm_attributes::output_names::NEXT_RUNNING_VAR);

        // Validate inputs
        // The iteration over graph happens in topological order, so previous nodes should have set input tensor
        // properties, if the user did not set them initially.
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
    create_cudnn_operations(
        std::unordered_set<uid_t>& uids_involved_in_operations,
        std::vector<std::shared_ptr<cudnn_frontend::Operation>>& operations,
        std::unordered_map<int64_t, std::shared_ptr<cudnn_frontend::Tensor>>& tensors) const override final {
        getLogger() << "[cudnn_frontend] INFO: " << "Building BatchNormNode operations " << attributes.name << "..."
                    << std::endl;

        std::vector<cudnn_frontend::Tensor> peer_stats;
        for (auto const& peer_stat : attributes.peer_stats) {
            peer_stats.emplace_back(std::move(*(tensors[peer_stat->get_uid()])));
        }

        auto&& batchnorm_operation_builder =
            cudnn_frontend::OperationBuilder(DescriptorType_t::OPERATION_NORM_FORWARD_DESCRIPTOR);

        batchnorm_operation_builder.setNormalizationMode(NormMode_t::BATCH_NORM)
            .setNormFwdPhase(NormFwdPhase_t::TRAINING);

        CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(X, Batchnorm_attributes::input_names::X);
        batchnorm_operation_builder.setxDesc(*(tensors[X->second->get_uid()]));

        CUDNN_FE_VALIDATE_AND_ASSIGN_OUTPUT_TENSOR(MEAN, Batchnorm_attributes::output_names::MEAN);
        CUDNN_FE_VALIDATE_AND_ASSIGN_OUTPUT_TENSOR(INV_VARIANCE, Batchnorm_attributes::output_names::INV_VARIANCE);
        batchnorm_operation_builder.setSavedMeanAndInvVar(*(tensors[MEAN->second->get_uid()]),
                                                          *(tensors[INV_VARIANCE->second->get_uid()]));

        CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(SCALE, Batchnorm_attributes::input_names::SCALE);
        CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(BIAS, Batchnorm_attributes::input_names::BIAS);
        batchnorm_operation_builder.setScaleAndBias(*(tensors[SCALE->second->get_uid()]),
                                                    *(tensors[BIAS->second->get_uid()]));

        CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(PREV_RUNNING_MEAN,
                                                  Batchnorm_attributes::input_names::PREV_RUNNING_MEAN);
        CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(PREV_RUNNING_VAR,
                                                  Batchnorm_attributes::input_names::PREV_RUNNING_VAR);
        batchnorm_operation_builder.setPrevRunningMeanAndVar(*(tensors[PREV_RUNNING_MEAN->second->get_uid()]),
                                                             *(tensors[PREV_RUNNING_VAR->second->get_uid()]));

        CUDNN_FE_VALIDATE_AND_ASSIGN_OUTPUT_TENSOR(NEXT_RUNNING_MEAN,
                                                   Batchnorm_attributes::output_names::NEXT_RUNNING_MEAN);
        CUDNN_FE_VALIDATE_AND_ASSIGN_OUTPUT_TENSOR(NEXT_RUNNING_VAR,
                                                   Batchnorm_attributes::output_names::NEXT_RUNNING_VAR);
        batchnorm_operation_builder.setNextRunningMeanAndVar(*(tensors[NEXT_RUNNING_MEAN->second->get_uid()]),
                                                             *(tensors[NEXT_RUNNING_VAR->second->get_uid()]));

        CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(EPSILON, Batchnorm_attributes::input_names::EPSILON);
        batchnorm_operation_builder.setEpsilonTensor(*(tensors[EPSILON->second->get_uid()]));

        CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(MOMENTUM, Batchnorm_attributes::input_names::MOMENTUM);
        batchnorm_operation_builder.setExpDecayFactorTensor(*(tensors[MOMENTUM->second->get_uid()]));

        CUDNN_FE_VALIDATE_AND_ASSIGN_OUTPUT_TENSOR(Y, Batchnorm_attributes::output_names::Y);
        batchnorm_operation_builder.setyDesc(*(tensors[Y->second->get_uid()]));

        batchnorm_operation_builder.setPeerStatTensor(peer_stats);

#ifdef NV_CUDNN_DISABLE_EXCEPTION
        // disable exception macro is defined. Calling build will not throw.
        // Check status of desc and return error.
        auto operation = batchnorm_operation_builder.build();
        RETURN_CUDNN_FRONTEND_ERROR_IF(operation.get_status() != CUDNN_STATUS_SUCCESS,
                                       error_code_t::CUDNN_BACKEND_API_FAILED,
                                       operation.get_error());
        operations.push_back(std::make_shared<Operation_v8>(std::move(operation)));
#else
        // build() can throw
        // wrap in try catch
        try {
            auto operation = batchnorm_operation_builder.build();
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
        j.update(R"( {"tag": "BATCHNORM"})"_json);
    }
#endif
};

}  // namespace graph

}  // namespace cudnn_frontend