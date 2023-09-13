#pragma once

#include "../../cudnn_frontend_Heuristics.h"
#include "../../cudnn_frontend_Logging.h"

#include "../cudnn_frontend_graph_helpers.h"
#include "../cudnn_frontend_node_interface.h"

namespace cudnn_frontend {

namespace graph {
class BatchnormInferenceNode : public INode {
   public:
    Batchnorm_inference_attributes attributes;

    BatchnormInferenceNode(Batchnorm_inference_attributes&& attributes_, detail::Context const& context)
        : INode(context), attributes(std::move(attributes_)) {}

    Type
    getType() override final {
        return Type::BATCHNORM_INFERENCE;
    }

    error_t
    infer_properties_node() override final {
        getLogger() << "[cudnn_frontend] INFO: Inferencing properties for batchnorm inference node " << attributes.name
                    << "..." << std::endl;

        attributes.fill_from_context(context);

        auto X                  = attributes.inputs.X;
        auto const x_tensor_dim = X->get_dim();

        auto Y            = attributes.outputs.Y;
        auto y_tensor_dim = Y->get_dim();
        // Only infer dims and strides if user did not set them
        if (y_tensor_dim.empty()) {
            y_tensor_dim.resize(x_tensor_dim.size());
            Y->set_dim(x_tensor_dim);
        }
        if (Y->get_stride().empty()) {
            Y->set_stride(detail::generate_stride(Y->get_dim()));
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
                T->set_stride(detail::generate_stride(T->get_dim()));
            }
        };
        infer_per_channel_tensors(attributes.inputs.MEAN);
        infer_per_channel_tensors(attributes.inputs.INV_VARIANCE);
        infer_per_channel_tensors(attributes.inputs.SCALE);
        infer_per_channel_tensors(attributes.inputs.BIAS);

        return {error_code_t::OK, ""};
    }

    error_t
    validate_node() const override final {
        getLogger() << "[cudnn_frontend] INFO: "
                    << "Validating BatchnormInferenceNode " << attributes.name << "..." << std::endl;

        return {error_code_t::OK, ""};
    }

    error_t
    assign_uids_node() override final {
        attributes.inputs.X->set_uid(ICudnn::create_new_uid());
        attributes.inputs.SCALE->set_uid(ICudnn::create_new_uid());
        attributes.inputs.BIAS->set_uid(ICudnn::create_new_uid());
        attributes.inputs.MEAN->set_uid(ICudnn::create_new_uid());
        attributes.inputs.INV_VARIANCE->set_uid(ICudnn::create_new_uid());
        attributes.outputs.Y->set_uid(ICudnn::create_new_uid());

        return {error_code_t::OK, ""};
    }

    error_t
    createTensors() override final {
        getLogger() << "[cudnn_frontend] INFO: "
                    << "Building BatchnormInferenceNode tensors " << attributes.name << "..." << std::endl;

        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(attributes.inputs.X));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(attributes.inputs.SCALE));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(attributes.inputs.BIAS));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(attributes.inputs.MEAN));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(attributes.inputs.INV_VARIANCE));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(attributes.outputs.Y));
        return {error_code_t::OK, ""};
    }

    error_t
    createOperations() override final {
        getLogger() << "[cudnn_frontend] INFO: "
                    << "Building BatchnormInferenceNode operations " << attributes.name << "..." << std::endl;

#ifndef NV_CUDNN_DISABLE_EXCEPTION
        try {
#endif

            auto batchnorm_operation =
                cudnn_frontend::OperationBuilder(DescriptorType_t::OPERATION_NORM_FORWARD_DESCRIPTOR)
                    .setNormalizationMode(NormMode_t::BATCH_NORM)
                    .setNormFwdPhase(NormFwdPhase_t::INFERENCE)
                    .setxDesc(*(tensors.at(attributes.inputs.X->get_uid())))
                    .setSavedMeanAndInvVar(*(tensors.at(attributes.inputs.MEAN->get_uid())),
                                           *(tensors.at(attributes.inputs.INV_VARIANCE->get_uid())))
                    .setScaleAndBias(*(tensors.at(attributes.inputs.SCALE->get_uid())),
                                     *(tensors.at(attributes.inputs.BIAS->get_uid())))
                    .setyDesc(*(tensors.at(attributes.outputs.Y->get_uid())))
                    .build();

            // Push all real tensors as required for operation execution.
            auto tensors_involved_in_operation = {attributes.inputs.X,
                                                  attributes.inputs.SCALE,
                                                  attributes.inputs.BIAS,
                                                  attributes.inputs.MEAN,
                                                  attributes.inputs.INV_VARIANCE,
                                                  attributes.outputs.Y};

            std::vector<uid_t> uids_in_operation;
            for (auto const& tensor : tensors_involved_in_operation) {
                if (tensor && tensor->get_is_virtual() == false) {
                    uids_in_operation.push_back(tensor->get_uid());
                }
            }

            operations.push_back({std::move(batchnorm_operation), std::move(uids_in_operation)});

#ifndef NV_CUDNN_DISABLE_EXCEPTION
        } catch (cudnn_frontend::cudnnException& e) {
            throw cudnnException(e.what(), e.getCudnnStatus());
        }
#endif

        return {error_code_t::OK, ""};
    }

    virtual void
    serialize(json& j) const override final {
        j = attributes;
    }
};

}  // namespace graph

}  // namespace cudnn_frontend