#pragma once

#include "../../cudnn_frontend_Heuristics.h"
#include "../../cudnn_frontend_Logging.h"

#include "../cudnn_frontend_graph_helpers.h"
#include "../cudnn_frontend_node_interface.h"

namespace cudnn_frontend {

namespace graph {
class LayerNormNode : public INode {
   public:
    Layernorm_attributes options;

    LayerNormNode(Layernorm_attributes&& options_, detail::Context const& context)
        : INode(context), options(std::move(options_)) {}

    Type
    getType() override final {
        return Type::LAYERNORM;
    }

    error_t
    infer_properties_node() override final {
        getLogger() << "[cudnn_frontend] INFO: Inferencing properties for layernorm node " << options.name << "..."
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

        // scale_bias   dim is 1,c,h,w
        // mean inv_var dim is n,1,1,1
        auto scale_bias_dim = X->get_dim();
        scale_bias_dim[0]   = 1;

        auto stats_dim = X->get_dim();
        for (size_t i = 1; i < stats_dim.size(); i++) {
            stats_dim[i] = 1;
        }

        auto scale = options.inputs.SCALE;
        if (scale->get_dim().empty()) {
            scale->set_dim(scale_bias_dim);
        }
        if (scale->get_stride().empty()) {
            auto const& scale_dim = scale->get_dim();
            // Default to NHWC
            auto const& stride_order = detail::generate_NHWC_stride_order(scale_dim.size());
            scale->set_stride(detail::generate_stride(scale_dim, stride_order));
        }

        auto bias = options.inputs.BIAS;
        if (bias->get_dim().empty()) {
            bias->set_dim(scale_bias_dim);
        }
        if (bias->get_stride().empty()) {
            auto const& bias_dim = bias->get_dim();
            // Default to NHWC
            auto const& stride_order = detail::generate_NHWC_stride_order(bias_dim.size());
            bias->set_stride(detail::generate_stride(bias_dim, stride_order));
        }

        if (options.forward_phase == NormFwdPhase_t::TRAINING) {
            auto mean = options.outputs.MEAN;
            if (mean->get_dim().empty()) {
                mean->set_dim(stats_dim);
            }
            if (mean->get_stride().empty()) {
                auto const& mean_dim = mean->get_dim();
                // Default to NHWC
                auto const& stride_order = detail::generate_NHWC_stride_order(mean_dim.size());
                mean->set_stride(detail::generate_stride(mean_dim, stride_order));
            }

            auto inv_var = options.outputs.INV_VARIANCE;
            if (inv_var->get_dim().empty()) {
                inv_var->set_dim(stats_dim);
            }
            if (inv_var->get_stride().empty()) {
                auto const& inv_var_dim = inv_var->get_dim();
                // Default to NHWC
                auto const& stride_order = detail::generate_NHWC_stride_order(inv_var_dim.size());
                inv_var->set_stride(detail::generate_stride(inv_var_dim, stride_order));
            }
        }

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

        return {error_code_t::OK, ""};
    }

    error_t
    validate_node() const override final {
        getLogger() << "[cudnn_frontend] INFO: "
                    << "Validating LayerNormNode " << options.name << "..." << std::endl;

        // Norm forward phase should be set
        RETURN_CUDNN_FRONTEND_ERROR_IF(options.forward_phase == NormFwdPhase_t::NOT_SET,
                                       error_code_t::ATTRIBUTE_NOT_SET,
                                       "Forward phase not set of layernorm node.");

        return {error_code_t::OK, ""};
    }

    error_t
    assign_uids_node() override final {
        options.inputs.X->set_uid(ICudnn::create_new_uid());
        options.inputs.SCALE->set_uid(ICudnn::create_new_uid());
        options.inputs.BIAS->set_uid(ICudnn::create_new_uid());
        options.inputs.EPSILON->set_uid(ICudnn::create_new_uid());
        options.outputs.Y->set_uid(ICudnn::create_new_uid());
        if (options.forward_phase == NormFwdPhase_t::TRAINING) {
            options.outputs.MEAN->set_uid(ICudnn::create_new_uid());
            options.outputs.INV_VARIANCE->set_uid(ICudnn::create_new_uid());
        }
        return {error_code_t::OK, ""};
    }

    error_t
    createTensors() override final {
        getLogger() << "[cudnn_frontend] INFO: "
                    << "Building LayerNormNode tensors " << options.name << "..." << std::endl;

        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.inputs.X));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.inputs.EPSILON));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.inputs.SCALE));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.inputs.BIAS));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.outputs.Y));
        if (options.forward_phase == NormFwdPhase_t::TRAINING) {
            CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.outputs.MEAN));
            CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.outputs.INV_VARIANCE));
        }
        return {error_code_t::OK, ""};
    }
    error_t
    createOperations() override final {
        getLogger() << "[cudnn_frontend] INFO: "
                    << "Building LayerNormNode operations " << options.name << "..." << std::endl;

#ifndef NV_CUDNN_DISABLE_EXCEPTION
        try {
#endif
            // Push all real tensors as required for operation execution.
            std::vector<std::shared_ptr<Tensor_attributes>> tensors_involved_in_operation = {
                options.inputs.X, options.inputs.EPSILON, options.inputs.SCALE, options.inputs.BIAS, options.outputs.Y};

            if (options.forward_phase == NormFwdPhase_t::TRAINING) {
                tensors_involved_in_operation.push_back(options.outputs.MEAN);
                tensors_involved_in_operation.push_back(options.outputs.INV_VARIANCE);
            }

            std::vector<uid_t> uids_in_operation;
            for (auto const& tensor : tensors_involved_in_operation) {
                if (tensor && tensor->get_is_virtual() == false) {
                    uids_in_operation.push_back(tensor->get_uid());
                }
            }

            if (options.forward_phase == NormFwdPhase_t::TRAINING) {
                auto layernorm_operation =
                    cudnn_frontend::OperationBuilder(DescriptorType_t::OPERATION_NORM_FORWARD_DESCRIPTOR)
                        .setNormalizationMode(NormMode_t::LAYER_NORM)
                        .setNormFwdPhase(options.forward_phase)
                        .setxDesc(*(tensors.at(options.inputs.X->get_uid())))
                        .setSavedMeanAndInvVar(*(tensors.at(options.outputs.MEAN->get_uid())),
                                               *(tensors.at(options.outputs.INV_VARIANCE->get_uid())))
                        .setScaleAndBias(*(tensors.at(options.inputs.SCALE->get_uid())),
                                         *(tensors.at(options.inputs.BIAS->get_uid())))
                        .setEpsilonTensor(*(tensors.at(options.inputs.EPSILON->get_uid())))
                        .setyDesc(*(tensors.at(options.outputs.Y->get_uid())))
                        .build();
                operations.push_back({std::move(layernorm_operation), std::move(uids_in_operation)});
            } else {
                auto layernorm_operation =
                    cudnn_frontend::OperationBuilder(DescriptorType_t::OPERATION_NORM_FORWARD_DESCRIPTOR)
                        .setNormalizationMode(NormMode_t::LAYER_NORM)
                        .setNormFwdPhase(options.forward_phase)
                        .setxDesc(*(tensors.at(options.inputs.X->get_uid())))
                        .setScaleAndBias(*(tensors.at(options.inputs.SCALE->get_uid())),
                                         *(tensors.at(options.inputs.BIAS->get_uid())))
                        .setEpsilonTensor(*(tensors.at(options.inputs.EPSILON->get_uid())))
                        .setyDesc(*(tensors.at(options.outputs.Y->get_uid())))
                        .build();
                operations.push_back({std::move(layernorm_operation), std::move(uids_in_operation)});
            }
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