#pragma once

#include "../../cudnn_frontend_Heuristics.h"
#include "../../cudnn_frontend_Logging.h"

#include "../cudnn_frontend_graph_helpers.h"
#include "../cudnn_frontend_node_interface.h"

namespace cudnn_frontend {

namespace graph {

class DLNNode : public INode {
   public:
    Layernorm_backward_attributes options;

    DLNNode(Layernorm_backward_attributes&& options_, detail::Context const& context)
        : INode(context), options(std::move(options_)) {}

    Type
    getType() override final {
        return Type::DLN;
    }

    error_t
    validate_node() const override final {
        getLogger() << "[cudnn_frontend] INFO: "
                    << "Validating DLNNode " << options.name << "..." << std::endl;

        if (!(options.inputs.MEAN) && !(options.inputs.INV_VARIANCE) && !(options.inputs.EPSILON) &&
            !(options.inputs.SCALE)) {
            auto status         = error_code_t::ATTRIBUTE_NOT_SET;
            std::string message = "[cudnn_frontend] ERROR: Either saved mean/inv_variance/scale or epsilon required.";
            return {status, message};
        }

        return {error_code_t::OK, ""};
    }

    error_t
    infer_properties_node() override final {
        getLogger() << "[cudnn_frontend] INFO: Inferencing properties for DLN node " << options.name << "..."
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
            DY->set_stride(detail::generate_stride(DY->get_dim()));
        }

        auto DX            = options.outputs.DX;
        auto dx_tensor_dim = DX->get_dim();
        // Only infer dims and strides if user did not set them
        if (dx_tensor_dim.empty()) {
            dx_tensor_dim.resize(x_tensor_dim.size());
            DX->set_dim(x_tensor_dim);
        }
        if (DX->get_stride().empty()) {
            DX->set_stride(detail::generate_stride(DX->get_dim()));
        }

        auto scale_bias_dim = X->get_dim();
        scale_bias_dim[0]   = 1;

        auto stats_dim = X->get_dim();
        for (size_t i = 1; i < stats_dim.size(); i++) {
            stats_dim[i] = 1;
        }

        auto mean = options.inputs.MEAN;
        if (mean->get_dim().empty()) {
            mean->set_dim(stats_dim);
        }
        if (mean->get_stride().empty()) {
            mean->set_stride(detail::generate_stride(mean->get_dim()));
        }

        auto inv_var = options.inputs.INV_VARIANCE;
        if (inv_var->get_dim().empty()) {
            inv_var->set_dim(stats_dim);
        }
        if (inv_var->get_stride().empty()) {
            inv_var->set_stride(detail::generate_stride(inv_var->get_dim()));
        }

        // Set channel length tensors
        auto infer_scale_bias_tensors = [&scale_bias_dim](std::shared_ptr<Tensor_attributes>& T) {
            auto tensor_dim = T->get_dim();
            // Only infer dims and strides if user did not set them
            if (tensor_dim.empty()) {
                T->set_dim(scale_bias_dim);
            }
            if (T->get_stride().empty()) {
                T->set_stride(detail::generate_stride(T->get_dim()));
            }
        };

        infer_scale_bias_tensors(options.inputs.SCALE);
        infer_scale_bias_tensors(options.outputs.DSCALE);
        infer_scale_bias_tensors(options.outputs.DBIAS);

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
        return {error_code_t::OK, ""};
    }

    error_t
    createTensors() override final {
        getLogger() << "[cudnn_frontend] INFO: "
                    << "Building DLNNode tensors " << options.name << "..." << std::endl;

        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.inputs.X));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.inputs.DY));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.inputs.SCALE));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.inputs.MEAN));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.inputs.INV_VARIANCE));
        // CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(epsilon));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.outputs.DX));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.outputs.DSCALE));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.outputs.DBIAS));

        return {error_code_t::OK, ""};
    }

    error_t
    createOperations() override final {
        getLogger() << "[cudnn_frontend] INFO: "
                    << "Building DLNNode operations " << options.name << "..." << std::endl;

#ifndef NV_CUDNN_DISABLE_EXCEPTION
        try {
#endif

            // Create the DLN operation.
            auto DLN_operation = cudnn_frontend::OperationBuilder(DescriptorType_t::OPERATION_NORM_BACKWARD_DESCRIPTOR)
                                     .setNormalizationMode(NormMode_t::LAYER_NORM)
                                     .setxDesc(*(tensors.at(options.inputs.X->get_uid())))
                                     .setdyDesc(*(tensors.at(options.inputs.DY->get_uid())))
                                     .setScale(*(tensors.at(options.inputs.SCALE->get_uid())))
                                     .setSavedMeanAndInvVar(*(tensors.at(options.inputs.MEAN->get_uid())),
                                                            *(tensors.at(options.inputs.INV_VARIANCE->get_uid())))
                                     .setDScaleAndDBias(*(tensors.at(options.outputs.DSCALE->get_uid())),
                                                        *(tensors.at(options.outputs.DBIAS->get_uid())))
                                     // .setEpsilonTensor(*(tensors.at(epsilon->get_uid())))
                                     .setdxDesc(*(tensors.at(options.outputs.DX->get_uid())))
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

            std::vector<uid_t> uids_in_operation;
            for (auto const& tensor : tensors_involved_in_operation) {
                if (tensor && tensor->get_is_virtual() == false) {
                    uids_in_operation.push_back(tensor->get_uid());
                }
            }

            operations.push_back({std::move(DLN_operation), std::move(uids_in_operation)});

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