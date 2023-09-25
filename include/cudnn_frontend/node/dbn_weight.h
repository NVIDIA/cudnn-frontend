#pragma once

#include "../../cudnn_frontend_Heuristics.h"
#include "../../cudnn_frontend_Logging.h"

#include "../cudnn_frontend_graph_helpers.h"
#include "../cudnn_frontend_node_interface.h"

namespace cudnn_frontend {

namespace graph {

class DBNWeightNode : public INode {
    DBN_weight_attributes options;

   public:
    DBNWeightNode(DBN_weight_attributes&& options_, detail::Context const& context)
        : INode(context), options(std::move(options_)) {}

    Type
    getType() override final {
        return Type::DBN_WEIGHT;
    }

    error_t
    infer_properties_node() override final {
        getLogger() << "[cudnn_frontend] INFO: Inferencing properties for batchnorm finalize node " << options.name
                    << "..." << std::endl;

        options.fill_from_context(context);

        // TODO: Only inferencing from DY works today.
        auto DY                  = options.inputs.DY;
        auto const dy_tensor_dim = DY->get_dim();

        auto X            = options.inputs.X;
        auto x_tensor_dim = X->get_dim();
        // Only infer dims and strides if user did not set them
        if (x_tensor_dim.empty()) {
            x_tensor_dim.resize(dy_tensor_dim.size());
            X->set_dim(dy_tensor_dim);
        }
        if (X->get_stride().empty()) {
            auto const& X_dim = X->get_dim();
            // Default to NHWC
            auto const& stride_order = detail::generate_NHWC_stride_order(X_dim.size());
            X->set_stride(detail::generate_stride(X_dim, stride_order));
        }

        // Set channel length tensors
        auto infer_per_channel_tensors = [&dy_tensor_dim](std::shared_ptr<Tensor_attributes> const& T) {
            auto tensor_dim = T->get_dim();
            // Only infer dims and strides if user did not set them
            if (T->get_dim().empty()) {
                tensor_dim.resize(dy_tensor_dim.size(), 1);
                tensor_dim[1] = dy_tensor_dim[1];
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
        infer_per_channel_tensors(options.outputs.DBIAS);
        infer_per_channel_tensors(options.outputs.DSCALE);
        infer_per_channel_tensors(options.outputs.EQ_BIAS);
        infer_per_channel_tensors(options.outputs.EQ_SCALE_DY);
        infer_per_channel_tensors(options.outputs.EQ_SCALE_X);

        return {error_code_t::OK, ""};
    }

    error_t
    assign_uids_node() override final {
        options.inputs.X->set_uid(ICudnn::create_new_uid());
        options.inputs.DY->set_uid(ICudnn::create_new_uid());
        options.inputs.SCALE->set_uid(ICudnn::create_new_uid());
        options.inputs.MEAN->set_uid(ICudnn::create_new_uid());
        options.inputs.INV_VARIANCE->set_uid(ICudnn::create_new_uid());
        options.outputs.DSCALE->set_uid(ICudnn::create_new_uid());
        options.outputs.DBIAS->set_uid(ICudnn::create_new_uid());
        options.outputs.EQ_SCALE_DY->set_uid(ICudnn::create_new_uid());
        options.outputs.EQ_SCALE_X->set_uid(ICudnn::create_new_uid());
        options.outputs.EQ_BIAS->set_uid(ICudnn::create_new_uid());
        return {error_code_t::OK, ""};
    }

    error_t
    createTensors() override final {
        getLogger() << "[cudnn_frontend] INFO: "
                    << "Building DBNWeightNode tensors " << options.name << "..." << std::endl;

        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.inputs.X));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.inputs.DY));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.inputs.MEAN));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.inputs.INV_VARIANCE));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.inputs.SCALE));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.outputs.DSCALE));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.outputs.DBIAS));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.outputs.EQ_BIAS));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.outputs.EQ_SCALE_DY));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.outputs.EQ_SCALE_X));

        return {error_code_t::OK, ""};
    }

    error_t
    createOperations() override final {
        getLogger() << "[cudnn_frontend] INFO: "
                    << "Building DBNWeightNode operations " << options.name << "..." << std::endl;

#ifndef NV_CUDNN_DISABLE_EXCEPTION
        try {
#endif

            // Create the batchnorm operation.
            auto batchnorm_operation =
                cudnn_frontend::OperationBuilder(DescriptorType_t::OPERATION_BN_BWD_WEIGHTS_DESCRIPTOR)
                    .setComputeType(CUDNN_DATA_FLOAT)
                    .setEqScalesAndBias(*(tensors.at(options.outputs.EQ_SCALE_DY->get_uid())),
                                        *(tensors.at(options.outputs.EQ_SCALE_X->get_uid())),
                                        *(tensors.at(options.outputs.EQ_BIAS->get_uid())))
                    .setSavedMeanAndInvVar(*(tensors.at(options.inputs.MEAN->get_uid())),
                                           *(tensors.at(options.inputs.INV_VARIANCE->get_uid())))
                    .setScale(*(tensors.at(options.inputs.SCALE->get_uid())))
                    .setxDesc(*(tensors.at(options.inputs.X->get_uid())))
                    .setdyDesc(*(tensors.at(options.inputs.DY->get_uid())))
                    .setDScaleAndDBias(*(tensors.at(options.outputs.DSCALE->get_uid())),
                                       *(tensors.at(options.outputs.DBIAS->get_uid())))
                    .build();

            // Push all real tensors as required for operation execution.
            auto const& tensors_involved_in_operation = {options.inputs.X,
                                                         options.inputs.DY,
                                                         options.inputs.MEAN,
                                                         options.inputs.INV_VARIANCE,
                                                         options.inputs.SCALE,
                                                         options.outputs.DBIAS,
                                                         options.outputs.DSCALE,
                                                         options.outputs.EQ_BIAS,
                                                         options.outputs.EQ_SCALE_DY,
                                                         options.outputs.EQ_SCALE_X};

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