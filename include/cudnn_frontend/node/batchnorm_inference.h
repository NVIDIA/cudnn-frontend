#pragma once

#include "../../cudnn_frontend_Heuristics.h"
#include "../../cudnn_frontend_Logging.h"

#include "../graph_helpers.h"
#include "../node_interface.h"

namespace cudnn_frontend {

namespace graph {
class BatchnormInferenceNode : public NodeCRTP<BatchnormInferenceNode> {
   public:
    Batchnorm_inference_attributes attributes;

    BatchnormInferenceNode(Batchnorm_inference_attributes&& attributes_, detail::Context const& context)
        : NodeCRTP(context), attributes(std::move(attributes_)) {}

    Type
    getType() override final {
        return Type::BATCHNORM_INFERENCE;
    }

    error_t
    infer_properties_node() override final {
        CUDNN_FE_LOG_LABEL_ENDL("INFO: Inferencing properties for batchnorm inference node " << attributes.name
                                                                                             << "...");

        attributes.fill_from_context(context);

        auto X = attributes.inputs[Batchnorm_inference_attributes::input_names::X];
        auto Y = attributes.outputs[Batchnorm_inference_attributes::output_names::Y];
        // Only infer dims and strides if user did not set them
        if (Y->get_dim().empty()) {
            Y->set_dim(X->get_dim());
        }
        if (Y->get_stride().empty()) {
            Y->set_stride(X->get_stride());
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
        CUDNN_FE_LOG_LABEL_ENDL("INFO: Building BatchnormInferenceNode operations " << attributes.name << "...");

        auto&& batchnorm_operation_builder =
            cudnn_frontend::OperationBuilder(DescriptorType_t::OPERATION_NORM_FORWARD_DESCRIPTOR);
        batchnorm_operation_builder.setNormalizationMode(NormMode_t::BATCH_NORM)
            .setNormFwdPhase(NormFwdPhase_t::INFERENCE);

        CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(X, Batchnorm_inference_attributes::input_names::X);
        batchnorm_operation_builder.setxDesc(*(tensors.at(X->second->get_uid())));

        CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(MEAN, Batchnorm_inference_attributes::input_names::MEAN);
        CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(INV_VARIANCE,
                                                  Batchnorm_inference_attributes::input_names::INV_VARIANCE);
        batchnorm_operation_builder.setSavedMeanAndInvVar(*(tensors.at(MEAN->second->get_uid())),
                                                          *(tensors.at(INV_VARIANCE->second->get_uid())));

        CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(SCALE, Batchnorm_inference_attributes::input_names::SCALE);
        CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(BIAS, Batchnorm_inference_attributes::input_names::BIAS);
        batchnorm_operation_builder.setScaleAndBias(*(tensors.at(SCALE->second->get_uid())),
                                                    *(tensors.at(BIAS->second->get_uid())));

        CUDNN_FE_VALIDATE_AND_ASSIGN_OUTPUT_TENSOR(Y, Batchnorm_inference_attributes::output_names::Y);
        batchnorm_operation_builder.setyDesc(*(tensors.at(Y->second->get_uid())));

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
        j.update(R"( {"tag": "BATCHNORM_INFERENCE"})"_json);
    }
#endif
};

}  // namespace graph

}  // namespace cudnn_frontend