#pragma once

#include "../../cudnn_frontend_Heuristics.h"
#include "../../cudnn_frontend_Logging.h"

#include "../graph_helpers.h"
#include "../node_interface.h"

namespace cudnn_frontend {

namespace graph {

class DBNNode : public INode {
   public:
    Batchnorm_backward_attributes options;

    DBNNode(Batchnorm_backward_attributes&& options_, detail::Context const& context)
        : INode(context), options(std::move(options_)) {}

    Type
    getType() override final {
        return Type::DBN;
    }

    error_t
    validate_node() const override final {
        getLogger() << "[cudnn_frontend] INFO: "
                    << "Validating DBNNode " << options.name << "..." << std::endl;

        RETURN_CUDNN_FRONTEND_ERROR_IF(
            !(options.inputs.MEAN) && !(options.inputs.INV_VARIANCE) && !(options.inputs.EPSILON),
            error_code_t::ATTRIBUTE_NOT_SET,
            "Either saved mean/inv_variance or epsilon required.");

        return {error_code_t::OK, ""};
    }

    error_t
    infer_properties_node() override final {
        getLogger() << "[cudnn_frontend] INFO: Inferencing properties for DBN node " << options.name << "..."
                    << std::endl;

        options.fill_from_context(context);

        // TODO: Only inferencing from X works today.
        auto X                  = options.inputs.X;
        auto const x_tensor_dim = X->get_dim();

        auto DY            = options.inputs.DY;
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

        auto DX            = options.outputs.DX;
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
        infer_per_channel_tensors(options.inputs.MEAN);
        infer_per_channel_tensors(options.inputs.INV_VARIANCE);
        infer_per_channel_tensors(options.inputs.SCALE);
        infer_per_channel_tensors(options.outputs.DSCALE);
        infer_per_channel_tensors(options.outputs.DBIAS);

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
    assign_uids_node() override final {
        options.inputs.X->set_uid(ICudnn::create_new_uid());
        options.inputs.DY->set_uid(ICudnn::create_new_uid());
        options.inputs.SCALE->set_uid(ICudnn::create_new_uid());
        options.inputs.MEAN->set_uid(ICudnn::create_new_uid());
        options.inputs.INV_VARIANCE->set_uid(ICudnn::create_new_uid());
        // epsilon->set_uid(ICudnn::create_new_uid());
        options.outputs.DX->set_uid(ICudnn::create_new_uid());
        options.outputs.DSCALE->set_uid(ICudnn::create_new_uid());
        options.outputs.DBIAS->set_uid(ICudnn::create_new_uid());
        for (auto const& peer_stat : options.inputs.peer_stats) {
            peer_stat->set_uid(ICudnn::create_new_uid());
        }
        return {error_code_t::OK, ""};
    }

    error_t
    createTensors() override final {
        getLogger() << "[cudnn_frontend] INFO: "
                    << "Building DBNNode tensors " << options.name << "..." << std::endl;

        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.inputs.X));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.inputs.DY));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.inputs.SCALE));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.inputs.MEAN));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.inputs.INV_VARIANCE));
        // CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(epsilon));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.outputs.DX));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.outputs.DSCALE));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.outputs.DBIAS));
        for (auto const& peer_stat : options.inputs.peer_stats) {
            CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(peer_stat));
        }

        return {error_code_t::OK, ""};
    }

    error_t
    createOperations() override final {
        getLogger() << "[cudnn_frontend] INFO: "
                    << "Building DBNNode operations " << options.name << "..." << std::endl;

#ifndef NV_CUDNN_DISABLE_EXCEPTION
        try {
#endif

            std::vector<cudnn_frontend::Tensor> peer_stats;
            for (auto const& peer_stat : options.inputs.peer_stats) {
                peer_stats.emplace_back(std::move(*(tensors.at(peer_stat->get_uid()))));
            }

            // Create the DBN operation.
            auto DBN_operation = cudnn_frontend::OperationBuilder(DescriptorType_t::OPERATION_NORM_BACKWARD_DESCRIPTOR)
                                     .setNormalizationMode(NormMode_t::BATCH_NORM)
                                     .setxDesc(*(tensors.at(options.inputs.X->get_uid())))
                                     .setdyDesc(*(tensors.at(options.inputs.DY->get_uid())))
                                     .setScale(*(tensors.at(options.inputs.SCALE->get_uid())))
                                     .setSavedMeanAndInvVar(*(tensors.at(options.inputs.MEAN->get_uid())),
                                                            *(tensors.at(options.inputs.INV_VARIANCE->get_uid())))
                                     .setDScaleAndDBias(*(tensors.at(options.outputs.DSCALE->get_uid())),
                                                        *(tensors.at(options.outputs.DBIAS->get_uid())))
                                     // .setEpsilonTensor(*(tensors.at(epsilon->get_uid())))
                                     .setdxDesc(*(tensors.at(options.outputs.DX->get_uid())))
                                     .setPeerStatTensor(peer_stats)
                                     .build();

            // Push all real tensors as required for operation execution.
            std::vector<std::shared_ptr<Tensor_attributes>> tensors_involved_in_operation = {options.inputs.X,
                                                                                             options.inputs.DY,
                                                                                             options.inputs.SCALE,
                                                                                             options.inputs.MEAN,
                                                                                             options.inputs.INV_VARIANCE
                                                                                             // , epsilon
                                                                                             ,
                                                                                             options.outputs.DX,
                                                                                             options.outputs.DSCALE,
                                                                                             options.outputs.DBIAS};

            for (auto const& peer_stat : options.inputs.peer_stats) {
                tensors_involved_in_operation.push_back(peer_stat);
            }

            std::vector<uid_t> uids_in_operation;
            for (auto const& tensor : tensors_involved_in_operation) {
                if (tensor && tensor->get_is_virtual() == false) {
                    uids_in_operation.push_back(tensor->get_uid());
                }
            }

            operations.push_back({std::move(DBN_operation), std::move(uids_in_operation)});

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