#pragma once

#include "../../cudnn_frontend_Heuristics.h"
#include "../../cudnn_frontend_Logging.h"

#include "../graph_helpers.h"
#include "../node_interface.h"

namespace cudnn_frontend {

namespace graph {

class BatchNormFinalizeNode : public INode {
    BN_finalize_attributes options;

   public:
    BatchNormFinalizeNode(BN_finalize_attributes&& options_, detail::Context const& context)
        : INode(context), options(std::move(options_)) {}

    Type
    getType() override final {
        return Type::BN_FINALIZE;
    }

    error_t
    infer_properties_node() override final {
        getLogger() << "[cudnn_frontend] INFO: Inferencing properties for batchnorm finalize node  " << options.name
                    << "..." << std::endl;

        options.fill_from_context(context);

        auto SUM                  = options.inputs.SUM;
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
        infer_per_channel_tensors(options.inputs.SQ_SUM);
        infer_per_channel_tensors(options.inputs.SCALE);
        infer_per_channel_tensors(options.inputs.BIAS);
        infer_per_channel_tensors(options.inputs.PREV_RUNNING_MEAN);
        infer_per_channel_tensors(options.inputs.PREV_RUNNING_VAR);
        infer_per_channel_tensors(options.outputs.EQ_BIAS);
        infer_per_channel_tensors(options.outputs.EQ_SCALE);
        infer_per_channel_tensors(options.outputs.MEAN);
        infer_per_channel_tensors(options.outputs.INV_VARIANCE);
        infer_per_channel_tensors(options.outputs.NEXT_RUNNING_MEAN);
        infer_per_channel_tensors(options.outputs.NEXT_RUNNING_VAR);

        // Set scalars
        auto infer_scalars = [&sum_tensor_dim](std::shared_ptr<Tensor_attributes>& T) {
            auto tensor_dim = T->get_dim();
            // Only infer dims and strides if user did not set them
            if (tensor_dim.empty()) {
                tensor_dim.resize(sum_tensor_dim.size(), 1);
                T->set_dim(tensor_dim);
            }
            if (T->get_stride().empty()) {
                auto const& T_dim = T->get_dim();
                // Default to NHWC
                auto const& stride_order = detail::generate_NHWC_stride_order(T_dim.size());
                T->set_stride(detail::generate_stride(T_dim, stride_order));
            }
        };
        infer_scalars(options.inputs.EPSILON);
        infer_scalars(options.inputs.MOMENTUM);
        infer_scalars(options.inputs.ACCUM_COUNT);

        return {error_code_t::OK, ""};
    }

    error_t
    assign_uids_node() override final {
        options.inputs.SUM->set_uid(ICudnn::create_new_uid());
        options.inputs.SQ_SUM->set_uid(ICudnn::create_new_uid());
        options.inputs.SCALE->set_uid(ICudnn::create_new_uid());
        options.inputs.BIAS->set_uid(ICudnn::create_new_uid());
        options.inputs.PREV_RUNNING_MEAN->set_uid(ICudnn::create_new_uid());
        options.inputs.PREV_RUNNING_VAR->set_uid(ICudnn::create_new_uid());
        options.inputs.EPSILON->set_uid(ICudnn::create_new_uid());
        options.inputs.MOMENTUM->set_uid(ICudnn::create_new_uid());
        options.inputs.ACCUM_COUNT->set_uid(ICudnn::create_new_uid());
        options.outputs.EQ_BIAS->set_uid(ICudnn::create_new_uid());
        options.outputs.EQ_SCALE->set_uid(ICudnn::create_new_uid());
        options.outputs.MEAN->set_uid(ICudnn::create_new_uid());
        options.outputs.INV_VARIANCE->set_uid(ICudnn::create_new_uid());
        options.outputs.NEXT_RUNNING_MEAN->set_uid(ICudnn::create_new_uid());
        options.outputs.NEXT_RUNNING_VAR->set_uid(ICudnn::create_new_uid());
        return {error_code_t::OK, ""};
    }

    error_t
    createTensors() override final {
        getLogger() << "[cudnn_frontend] INFO: "
                    << "Building BatchNormFinalizeNode tensors " << options.name << "..." << std::endl;

        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.inputs.SUM));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.inputs.SQ_SUM));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.inputs.SCALE));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.inputs.BIAS));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.inputs.PREV_RUNNING_MEAN));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.inputs.PREV_RUNNING_VAR));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.inputs.EPSILON));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.inputs.MOMENTUM));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.inputs.ACCUM_COUNT));

        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.outputs.EQ_BIAS));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.outputs.EQ_SCALE));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.outputs.MEAN));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.outputs.INV_VARIANCE));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.outputs.NEXT_RUNNING_MEAN));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.outputs.NEXT_RUNNING_VAR));

        return {error_code_t::OK, ""};
    }

    error_t
    createOperations() override final {
        getLogger() << "[cudnn_frontend] INFO: "
                    << "Building BatchNormFinalizeNode operations " << options.name << "..." << std::endl;

#ifndef NV_CUDNN_DISABLE_EXCEPTION
        try {
#endif

            // Create the batchnorm operation.
            auto batchnorm_operation =
                cudnn_frontend::OperationBuilder(DescriptorType_t::OPERATION_BN_FINALIZE_STATISTICS_DESCRIPTOR)
                    .setComputeType(CUDNN_DATA_FLOAT)
                    .setBNFinalizeMode(CUDNN_BN_FINALIZE_STATISTICS_TRAINING)
                    .setSumDesc(*(tensors.at(options.inputs.SUM->get_uid())))
                    .setSqSumDesc(*(tensors.at(options.inputs.SQ_SUM->get_uid())))
                    .setEqScaleAndBias(*(tensors.at(options.outputs.EQ_SCALE->get_uid())),
                                       *(tensors.at(options.outputs.EQ_BIAS->get_uid())))
                    .setSavedMeanAndInvVar(*(tensors.at(options.outputs.MEAN->get_uid())),
                                           *(tensors.at(options.outputs.INV_VARIANCE->get_uid())))
                    .setScaleAndBias(*(tensors.at(options.inputs.SCALE->get_uid())),
                                     *(tensors.at(options.inputs.BIAS->get_uid())))
                    .setPrevRunningMeanAndVar(*(tensors.at(options.inputs.PREV_RUNNING_MEAN->get_uid())),
                                              *(tensors.at(options.inputs.PREV_RUNNING_VAR->get_uid())))
                    .setNextRunningMeanAndVar(*(tensors.at(options.outputs.NEXT_RUNNING_MEAN->get_uid())),
                                              *(tensors.at(options.outputs.NEXT_RUNNING_VAR->get_uid())))
                    .setEpsilonTensor(*(tensors.at(options.inputs.EPSILON->get_uid())))
                    .setExpDecayFactorTensor(*(tensors.at(options.inputs.MOMENTUM->get_uid())))
                    .setAccumCountTensor(*(tensors.at(options.inputs.ACCUM_COUNT->get_uid())))
                    .build();

            // Push all real tensors as required for operation execution.
            auto const& tensors_involved_in_operation = {options.inputs.SUM,
                                                         options.inputs.SQ_SUM,
                                                         options.inputs.PREV_RUNNING_MEAN,
                                                         options.inputs.PREV_RUNNING_VAR,
                                                         options.inputs.EPSILON,
                                                         options.inputs.MOMENTUM,
                                                         options.inputs.ACCUM_COUNT,
                                                         options.inputs.SCALE,
                                                         options.inputs.BIAS,
                                                         options.outputs.EQ_BIAS,
                                                         options.outputs.EQ_SCALE,
                                                         options.outputs.MEAN,
                                                         options.outputs.INV_VARIANCE,
                                                         options.outputs.NEXT_RUNNING_MEAN,
                                                         options.outputs.NEXT_RUNNING_VAR};

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
        j = options;
    }
};

}  // namespace graph

}  // namespace cudnn_frontend