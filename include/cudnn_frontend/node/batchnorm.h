#pragma once

#include "../../cudnn_frontend_Heuristics.h"
#include "../../cudnn_frontend_Logging.h"

#include "../cudnn_frontend_graph_helpers.h"
#include "../cudnn_frontend_node_interface.h"

namespace cudnn_frontend {

namespace graph {
class BatchNormNode : public INode {
   public:
    Batchnorm_attributes options;

    BatchNormNode(Batchnorm_attributes&& options_, detail::Context const& context)
        : INode(context), options(std::move(options_)) {}

    Type
    getType() override final {
        return Type::BATCHNORM;
    }

    error_t
    infer_properties_node() override final {
        getLogger() << "[cudnn_frontend] INFO: Inferencing properties for batchnorm node " << options.name << "..."
                    << std::endl;

        options.fill_from_context(context);

        auto X                  = options.inputs.X;
        auto const x_tensor_dim = X->get_dim();

        auto Y            = options.outputs.Y;
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
        infer_per_channel_tensors(options.outputs.MEAN);
        infer_per_channel_tensors(options.outputs.INV_VARIANCE);
        infer_per_channel_tensors(options.outputs.NEXT_RUNNING_MEAN);
        infer_per_channel_tensors(options.outputs.NEXT_RUNNING_VAR);
        infer_per_channel_tensors(options.inputs.PREV_RUNNING_MEAN);
        infer_per_channel_tensors(options.inputs.PREV_RUNNING_VAR);
        infer_per_channel_tensors(options.inputs.SCALE);
        infer_per_channel_tensors(options.inputs.BIAS);

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
        infer_scalar_tensors(options.inputs.EPSILON);
        infer_scalar_tensors(options.inputs.MOMENTUM);

        for (auto const& peer_stat : options.inputs.peer_stats) {
            if (peer_stat->get_stride().empty()) {
                auto const& peer_stat_dim = peer_stat->get_dim();
                // Default to NHWC
                auto const& stride_order = detail::generate_NHWC_stride_order(peer_stat_dim.size());
                peer_stat->set_stride(detail::generate_stride(peer_stat_dim, stride_order));
            }
        }

        return {error_code_t::OK, ""};
    }

    error_t
    validate_node() const override final {
        getLogger() << "[cudnn_frontend] INFO: "
                    << "Validating BatchNormNode " << options.name << "..." << std::endl;

        // Norm forward phase should be set
        RETURN_CUDNN_FRONTEND_ERROR_IF(options.forward_phase == NormFwdPhase_t::NOT_SET,
                                       error_code_t::ATTRIBUTE_NOT_SET,
                                       "Forward phase not set of batchnorm node.");

        return {error_code_t::OK, ""};
    }

    error_t
    assign_uids_node() override final {
        options.inputs.X->set_uid(ICudnn::create_new_uid());
        options.inputs.SCALE->set_uid(ICudnn::create_new_uid());
        options.inputs.BIAS->set_uid(ICudnn::create_new_uid());
        options.inputs.PREV_RUNNING_MEAN->set_uid(ICudnn::create_new_uid());
        options.inputs.PREV_RUNNING_VAR->set_uid(ICudnn::create_new_uid());
        options.inputs.EPSILON->set_uid(ICudnn::create_new_uid());
        options.inputs.MOMENTUM->set_uid(ICudnn::create_new_uid());
        options.outputs.Y->set_uid(ICudnn::create_new_uid());
        options.outputs.MEAN->set_uid(ICudnn::create_new_uid());
        options.outputs.INV_VARIANCE->set_uid(ICudnn::create_new_uid());
        options.outputs.NEXT_RUNNING_MEAN->set_uid(ICudnn::create_new_uid());
        options.outputs.NEXT_RUNNING_VAR->set_uid(ICudnn::create_new_uid());
        for (auto const& peer_stat : options.inputs.peer_stats) {
            peer_stat->set_uid(ICudnn::create_new_uid());
        }
        return {error_code_t::OK, ""};
    }

    error_t
    createTensors() override final {
        getLogger() << "[cudnn_frontend] INFO: "
                    << "Building BatchNormNode tensors " << options.name << "..." << std::endl;

        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.inputs.X));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.inputs.PREV_RUNNING_MEAN));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.inputs.PREV_RUNNING_VAR));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.inputs.EPSILON));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.inputs.MOMENTUM));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.inputs.SCALE));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.inputs.BIAS));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.outputs.Y));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.outputs.MEAN));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.outputs.INV_VARIANCE));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.outputs.NEXT_RUNNING_MEAN));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.outputs.NEXT_RUNNING_VAR));
        for (auto const& peer_stat : options.inputs.peer_stats) {
            CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(peer_stat));
        }
        return {error_code_t::OK, ""};
    }

    error_t
    createOperations() override final {
        getLogger() << "[cudnn_frontend] INFO: "
                    << "Building BatchNormNode operations " << options.name << "..." << std::endl;

#ifndef NV_CUDNN_DISABLE_EXCEPTION
        try {
#endif

            std::vector<cudnn_frontend::Tensor> peer_stats;
            for (auto const& peer_stat : options.inputs.peer_stats) {
                peer_stats.emplace_back(std::move(*(tensors.at(peer_stat->get_uid()))));
            }

            auto batchnorm_operation =
                cudnn_frontend::OperationBuilder(DescriptorType_t::OPERATION_NORM_FORWARD_DESCRIPTOR)
                    .setNormalizationMode(NormMode_t::BATCH_NORM)
                    .setNormFwdPhase(options.forward_phase)
                    .setxDesc(*(tensors.at(options.inputs.X->get_uid())))
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
                    .setyDesc(*(tensors.at(options.outputs.Y->get_uid())))
                    .setPeerStatTensor(peer_stats)
                    .build();

            // Push all real tensors as required for operation execution.
            std::vector<std::shared_ptr<Tensor_attributes>> tensors_involved_in_operation = {
                options.inputs.X,
                options.inputs.PREV_RUNNING_MEAN,
                options.inputs.PREV_RUNNING_VAR,
                options.inputs.EPSILON,
                options.inputs.MOMENTUM,
                options.inputs.SCALE,
                options.inputs.BIAS,
                options.outputs.Y,
                options.outputs.MEAN,
                options.outputs.INV_VARIANCE,
                options.outputs.NEXT_RUNNING_MEAN,
                options.outputs.NEXT_RUNNING_VAR};
            for (auto const& peer_stat : options.inputs.peer_stats) {
                tensors_involved_in_operation.push_back(peer_stat);
            }

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