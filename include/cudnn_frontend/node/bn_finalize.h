#pragma once

#include "../../cudnn_frontend_Heuristics.h"
#include "../../cudnn_frontend_Logging.h"

#include "../graph_helpers.h"
#include "../node_interface.h"

namespace cudnn_frontend {

namespace graph {

class BatchNormFinalizeNode : public NodeCRTP<BatchNormFinalizeNode> {
   public:
    BN_finalize_attributes attributes;

    BatchNormFinalizeNode(BN_finalize_attributes&& attributes_, detail::Context const& context)
        : NodeCRTP(context), attributes(std::move(attributes_)) {}

    Type
    getType() override final {
        return Type::BN_FINALIZE;
    }

    error_t
    pre_validate_node() const override final {
        CHECK_CUDNN_FRONTEND_ERROR(attributes.validate_inputs());

        return {error_code_t::OK, ""};
    }

    error_t
    expand_and_infer_properties_node() override final {
        getLogger() << "[cudnn_frontend] INFO: Inferencing properties for batchnorm finalize node  " << attributes.name
                    << "..." << std::endl;

        attributes.fill_from_context(context);

        auto SUM                  = attributes.inputs[BN_finalize_attributes::input_names::SUM];
        auto const sum_tensor_dim = SUM->get_dim();

        // Set channel length tensors
        auto infer_per_channel_tensors = [&sum_tensor_dim](std::shared_ptr<Tensor_attributes>& T) {
            auto tensor_dim = T->get_dim();
            // Only infer dims and strides if user did not set them
            if (tensor_dim.empty()) {
                tensor_dim = sum_tensor_dim;
                T->set_dim(tensor_dim);
            }
            if (T->get_stride().empty()) {
                auto const& T_dim = T->get_dim();
                // Default to NHWC
                auto const& stride_order = detail::generate_NHWC_stride_order(T_dim.size());
                T->set_stride(detail::generate_stride(T_dim, stride_order));
            }
        };
        infer_per_channel_tensors(attributes.outputs[BN_finalize_attributes::output_names::EQ_BIAS]);
        infer_per_channel_tensors(attributes.outputs[BN_finalize_attributes::output_names::EQ_SCALE]);
        infer_per_channel_tensors(attributes.outputs[BN_finalize_attributes::output_names::MEAN]);
        infer_per_channel_tensors(attributes.outputs[BN_finalize_attributes::output_names::INV_VARIANCE]);
        infer_per_channel_tensors(attributes.outputs[BN_finalize_attributes::output_names::NEXT_RUNNING_MEAN]);
        infer_per_channel_tensors(attributes.outputs[BN_finalize_attributes::output_names::NEXT_RUNNING_VAR]);

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
        getLogger() << "[cudnn_frontend] INFO: " << "Building BatchNormFinalizeNode operations " << attributes.name
                    << "..." << std::endl;

        // Create the batchnorm operation.
        auto&& batchnorm_operation_builder =
            cudnn_frontend::OperationBuilder(DescriptorType_t::OPERATION_BN_FINALIZE_STATISTICS_DESCRIPTOR);
        batchnorm_operation_builder.setComputeType(CUDNN_DATA_FLOAT)
            .setBNFinalizeMode(CUDNN_BN_FINALIZE_STATISTICS_TRAINING);

        CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(SUM, BN_finalize_attributes::input_names::SUM);
        batchnorm_operation_builder.setSumDesc(*(tensors.at(SUM->second->get_uid())));

        CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(SQ_SUM, BN_finalize_attributes::input_names::SQ_SUM);
        batchnorm_operation_builder.setSqSumDesc(*(tensors.at(SQ_SUM->second->get_uid())));

        CUDNN_FE_VALIDATE_AND_ASSIGN_OUTPUT_TENSOR(EQ_SCALE, BN_finalize_attributes::output_names::EQ_SCALE);
        CUDNN_FE_VALIDATE_AND_ASSIGN_OUTPUT_TENSOR(EQ_BIAS, BN_finalize_attributes::output_names::EQ_BIAS);
        batchnorm_operation_builder.setEqScaleAndBias(*(tensors.at(EQ_SCALE->second->get_uid())),
                                                      *(tensors.at(EQ_BIAS->second->get_uid())));

        CUDNN_FE_VALIDATE_AND_ASSIGN_OUTPUT_TENSOR(MEAN, BN_finalize_attributes::output_names::MEAN);
        CUDNN_FE_VALIDATE_AND_ASSIGN_OUTPUT_TENSOR(INV_VARIANCE, BN_finalize_attributes::output_names::INV_VARIANCE);
        batchnorm_operation_builder.setSavedMeanAndInvVar(*(tensors.at(MEAN->second->get_uid())),
                                                          *(tensors.at(INV_VARIANCE->second->get_uid())));

        CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(SCALE, BN_finalize_attributes::input_names::SCALE);
        CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(BIAS, BN_finalize_attributes::input_names::BIAS);
        batchnorm_operation_builder.setScaleAndBias(*(tensors.at(SCALE->second->get_uid())),
                                                    *(tensors.at(BIAS->second->get_uid())));

        CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(PREV_RUNNING_MEAN,
                                                  BN_finalize_attributes::input_names::PREV_RUNNING_MEAN);
        CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(PREV_RUNNING_VAR,
                                                  BN_finalize_attributes::input_names::PREV_RUNNING_VAR);
        batchnorm_operation_builder.setPrevRunningMeanAndVar(*(tensors.at(PREV_RUNNING_MEAN->second->get_uid())),
                                                             *(tensors.at(PREV_RUNNING_VAR->second->get_uid())));

        CUDNN_FE_VALIDATE_AND_ASSIGN_OUTPUT_TENSOR(NEXT_RUNNING_MEAN,
                                                   BN_finalize_attributes::output_names::NEXT_RUNNING_MEAN);
        CUDNN_FE_VALIDATE_AND_ASSIGN_OUTPUT_TENSOR(NEXT_RUNNING_VAR,
                                                   BN_finalize_attributes::output_names::NEXT_RUNNING_VAR);
        batchnorm_operation_builder.setNextRunningMeanAndVar(*(tensors.at(NEXT_RUNNING_MEAN->second->get_uid())),
                                                             *(tensors.at(NEXT_RUNNING_VAR->second->get_uid())));

        CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(EPSILON, BN_finalize_attributes::input_names::EPSILON);
        batchnorm_operation_builder.setEpsilonTensor(*(tensors.at(EPSILON->second->get_uid())));

        CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(MOMENTUM, BN_finalize_attributes::input_names::MOMENTUM);
        batchnorm_operation_builder.setExpDecayFactorTensor(*(tensors.at(MOMENTUM->second->get_uid())));

        CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(ACCUM_COUNT, BN_finalize_attributes::input_names::ACCUM_COUNT);
        batchnorm_operation_builder.setAccumCountTensor(*(tensors.at(ACCUM_COUNT->second->get_uid())));

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
        j.update(R"( {"tag": "BN_FINALIZE"})"_json);
    }
#endif
};

}  // namespace graph

}  // namespace cudnn_frontend