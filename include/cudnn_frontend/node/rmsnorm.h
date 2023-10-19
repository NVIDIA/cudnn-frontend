#pragma once

#include "../../cudnn_frontend_Heuristics.h"
#include "../../cudnn_frontend_Logging.h"

#include "../graph_helpers.h"
#include "../node_interface.h"

namespace cudnn_frontend {

namespace graph {
class RMSNormNode : public INode {
   public:
    Rmsnorm_attributes options;

    RMSNormNode(Rmsnorm_attributes&& options_, detail::Context const& context)
        : INode(context), options(std::move(options_)) {}

    Type
    getType() override final {
        return Type::RMSNORM;
    }

    error_t
    infer_properties_node() override final {
        getLogger() << "[cudnn_frontend] INFO: Inferencing properties for rmsnorm node " << options.name << "..."
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
        auto infer_norm_apply_tensors = [&x_tensor_dim](std::shared_ptr<Tensor_attributes>& T) {
            auto tensor_dim = T->get_dim();
            // Only infer dims and strides if user did not set them
            if (tensor_dim.empty()) {
                tensor_dim    = x_tensor_dim;
                tensor_dim[0] = 1;
                T->set_dim(tensor_dim);
            }
            if (T->get_stride().empty()) {
                auto const& T_dim = T->get_dim();
                // Default to NHWC
                auto const& stride_order = detail::generate_NHWC_stride_order(T_dim.size());
                T->set_stride(detail::generate_stride(T_dim, stride_order));
            }
        };
        infer_norm_apply_tensors(options.inputs.SCALE);
        if (options.inputs.BIAS) {
            infer_norm_apply_tensors(options.inputs.BIAS);
        }

        if (options.forward_phase == NormFwdPhase_t::TRAINING) {
            auto inv_var = options.outputs.INV_VARIANCE;
            if (auto inv_var_dim = inv_var->get_dim(); inv_var_dim.empty()) {
                inv_var_dim.resize(x_tensor_dim.size(), 1);
                inv_var_dim[0] = x_tensor_dim[0];
                inv_var->set_dim(inv_var_dim);
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
                    << "Validating RMSNormNode " << options.name << "..." << std::endl;

        // Norm forward phase should be set
        RETURN_CUDNN_FRONTEND_ERROR_IF(options.forward_phase == NormFwdPhase_t::NOT_SET,
                                       error_code_t::ATTRIBUTE_NOT_SET,
                                       "Forward phase not set of rmsnorm node.");

        return {error_code_t::OK, ""};
    }

    error_t
    assign_uids_node() override final {
        options.inputs.X->set_uid(ICudnn::create_new_uid());
        options.inputs.SCALE->set_uid(ICudnn::create_new_uid());
        if (options.inputs.BIAS) options.inputs.BIAS->set_uid(ICudnn::create_new_uid());
        options.inputs.EPSILON->set_uid(ICudnn::create_new_uid());
        options.outputs.Y->set_uid(ICudnn::create_new_uid());
        if (options.forward_phase == NormFwdPhase_t::TRAINING) {
            options.outputs.INV_VARIANCE->set_uid(ICudnn::create_new_uid());
        }
        return {error_code_t::OK, ""};
    }

    error_t
    createTensors() override final {
        getLogger() << "[cudnn_frontend] INFO: "
                    << "Building RMSNormNode tensors " << options.name << "..." << std::endl;

        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.inputs.X));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.inputs.EPSILON));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.inputs.SCALE));
        if (options.inputs.BIAS) CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.inputs.BIAS));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.outputs.Y));
        if (options.forward_phase == NormFwdPhase_t::TRAINING) {
            CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.outputs.INV_VARIANCE));
        }
        return {error_code_t::OK, ""};
    }
    error_t
    createOperations() override final {
        getLogger() << "[cudnn_frontend] INFO: "
                    << "Building RMSNormNode operations " << options.name << "..." << std::endl;

#ifndef NV_CUDNN_DISABLE_EXCEPTION
        try {
#endif
            // Push all real tensors as required for operation execution.
            auto tensors_involved_in_operation = {options.inputs.X,
                                                  options.inputs.EPSILON,
                                                  options.inputs.SCALE,
                                                  options.inputs.BIAS,
                                                  options.outputs.Y,
                                                  options.outputs.INV_VARIANCE};

            std::vector<uid_t> uids_in_operation;
            for (auto const& tensor : tensors_involved_in_operation) {
                if (tensor && tensor->get_is_virtual() == false) {
                    uids_in_operation.push_back(tensor->get_uid());
                }
            }

            if (options.inputs.BIAS) {
                if (options.forward_phase == NormFwdPhase_t::TRAINING) {
                    auto rmsnorm_operation =
                        cudnn_frontend::OperationBuilder(DescriptorType_t::OPERATION_NORM_FORWARD_DESCRIPTOR)
                            .setNormalizationMode(NormMode_t::RMS_NORM)
                            .setNormFwdPhase(options.forward_phase)
                            .setxDesc(*(tensors.at(options.inputs.X->get_uid())))
                            .setSavedInvVar(*(tensors.at(options.outputs.INV_VARIANCE->get_uid())))
                            .setScaleAndBias(*(tensors.at(options.inputs.SCALE->get_uid())),
                                             *(tensors.at(options.inputs.BIAS->get_uid())))
                            .setEpsilonTensor(*(tensors.at(options.inputs.EPSILON->get_uid())))
                            .setyDesc(*(tensors.at(options.outputs.Y->get_uid())))
                            .build();
                    operations.push_back({std::move(rmsnorm_operation), std::move(uids_in_operation)});
                } else {
                    auto rmsnorm_operation =
                        cudnn_frontend::OperationBuilder(DescriptorType_t::OPERATION_NORM_FORWARD_DESCRIPTOR)
                            .setNormalizationMode(NormMode_t::RMS_NORM)
                            .setNormFwdPhase(options.forward_phase)
                            .setxDesc(*(tensors.at(options.inputs.X->get_uid())))
                            .setScaleAndBias(*(tensors.at(options.inputs.SCALE->get_uid())),
                                             *(tensors.at(options.inputs.BIAS->get_uid())))
                            .setEpsilonTensor(*(tensors.at(options.inputs.EPSILON->get_uid())))
                            .setyDesc(*(tensors.at(options.outputs.Y->get_uid())))
                            .build();
                    operations.push_back({std::move(rmsnorm_operation), std::move(uids_in_operation)});
                }
            } else {
                if (options.forward_phase == NormFwdPhase_t::TRAINING) {
                    auto rmsnorm_operation =
                        cudnn_frontend::OperationBuilder(DescriptorType_t::OPERATION_NORM_FORWARD_DESCRIPTOR)
                            .setNormalizationMode(NormMode_t::RMS_NORM)
                            .setNormFwdPhase(options.forward_phase)
                            .setxDesc(*(tensors.at(options.inputs.X->get_uid())))
                            .setSavedInvVar(*(tensors.at(options.outputs.INV_VARIANCE->get_uid())))
                            .setScale(*(tensors.at(options.inputs.SCALE->get_uid())))
                            .setEpsilonTensor(*(tensors.at(options.inputs.EPSILON->get_uid())))
                            .setyDesc(*(tensors.at(options.outputs.Y->get_uid())))
                            .build();
                    operations.push_back({std::move(rmsnorm_operation), std::move(uids_in_operation)});
                } else {
                    auto rmsnorm_operation =
                        cudnn_frontend::OperationBuilder(DescriptorType_t::OPERATION_NORM_FORWARD_DESCRIPTOR)
                            .setNormalizationMode(NormMode_t::RMS_NORM)
                            .setNormFwdPhase(options.forward_phase)
                            .setxDesc(*(tensors.at(options.inputs.X->get_uid())))
                            .setScale(*(tensors.at(options.inputs.SCALE->get_uid())))
                            .setEpsilonTensor(*(tensors.at(options.inputs.EPSILON->get_uid())))
                            .setyDesc(*(tensors.at(options.outputs.Y->get_uid())))
                            .build();
                    operations.push_back({std::move(rmsnorm_operation), std::move(uids_in_operation)});
                }
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

class DRMSNormNode : public INode {
   public:
    Rmsnorm_backward_attributes options;

    DRMSNormNode(Rmsnorm_backward_attributes&& options_, detail::Context const& context)
        : INode(context), options(std::move(options_)) {}

    Type
    getType() override final {
        return Type::DRMSNorm;
    }

    error_t
    validate_node() const override final {
        getLogger() << "[cudnn_frontend] INFO: "
                    << "Validating DRMSNormNode node " << options.name << "..." << std::endl;

        RETURN_CUDNN_FRONTEND_ERROR_IF(options.use_dbias.has_value() == false,
                                       error_code_t::ATTRIBUTE_NOT_SET,
                                       "DRMSNormNode node needs has_bias(bool) to be called.");

        return {error_code_t::OK, ""};
    }

    error_t
    infer_properties_node() override final {
        getLogger() << "[cudnn_frontend] INFO: Inferencing properties for DRMSNorm node " << options.name << "..."
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

        auto scale_bias_dim = X->get_dim();
        scale_bias_dim[0]   = 1;

        auto stats_dim = X->get_dim();
        for (size_t i = 1; i < stats_dim.size(); i++) {
            stats_dim[i] = 1;
        }

        auto inv_var = options.inputs.INV_VARIANCE;
        if (inv_var->get_dim().empty()) {
            inv_var->set_dim(stats_dim);
        }
        if (inv_var->get_stride().empty()) {
            auto const& inv_var_dim = inv_var->get_dim();
            // Default to NHWC
            auto const& stride_order = detail::generate_NHWC_stride_order(inv_var_dim.size());
            inv_var->set_stride(detail::generate_stride(inv_var_dim, stride_order));
        }

        // Set channel length tensors
        auto infer_scale_bias_tensors = [&scale_bias_dim](std::shared_ptr<Tensor_attributes>& T) {
            auto tensor_dim = T->get_dim();
            // Only infer dims and strides if user did not set them
            if (tensor_dim.empty()) {
                T->set_dim(scale_bias_dim);
            }
            if (T->get_stride().empty()) {
                auto const& T_dim = T->get_dim();
                // Default to NHWC
                auto const& stride_order = detail::generate_NHWC_stride_order(T_dim.size());
                T->set_stride(detail::generate_stride(T_dim, stride_order));
            }
        };

        infer_scale_bias_tensors(options.inputs.SCALE);
        infer_scale_bias_tensors(options.outputs.DSCALE);
        if (options.use_dbias.value()) {
            infer_scale_bias_tensors(options.outputs.DBIAS);
        }

        return {error_code_t::OK, ""};
    }

    error_t
    assign_uids_node() override final {
        options.inputs.X->set_uid(ICudnn::create_new_uid());
        options.inputs.DY->set_uid(ICudnn::create_new_uid());
        options.inputs.SCALE->set_uid(ICudnn::create_new_uid());
        options.inputs.INV_VARIANCE->set_uid(ICudnn::create_new_uid());
        options.outputs.DX->set_uid(ICudnn::create_new_uid());
        options.outputs.DSCALE->set_uid(ICudnn::create_new_uid());
        if (options.use_dbias.value()) {
            options.outputs.DBIAS->set_uid(ICudnn::create_new_uid());
        }
        return {error_code_t::OK, ""};
    }

    error_t
    createTensors() override final {
        getLogger() << "[cudnn_frontend] INFO: "
                    << "Building DRMSNormNode tensors " << options.name << "..." << std::endl;

        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.inputs.X));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.inputs.DY));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.inputs.SCALE));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.inputs.INV_VARIANCE));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.outputs.DX));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.outputs.DSCALE));
        if (options.use_dbias.value()) {
            CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.outputs.DBIAS));
        }

        return {error_code_t::OK, ""};
    }

    error_t
    createOperations() override final {
        getLogger() << "[cudnn_frontend] INFO: "
                    << "Building DRMSNormNode operations " << options.name << "..." << std::endl;

#ifndef NV_CUDNN_DISABLE_EXCEPTION
        try {
#endif

            // Push all real tensors as required for operation execution.
            std::vector<std::shared_ptr<Tensor_attributes>> tensors_involved_in_operation = {
                options.inputs.X,
                options.inputs.DY,
                options.inputs.SCALE,
                options.inputs.INV_VARIANCE,
                options.outputs.DX,
                options.outputs.DSCALE,
                options.outputs.DBIAS};

            std::vector<uid_t> uids_in_operation;
            for (auto const& tensor : tensors_involved_in_operation) {
                if (tensor && tensor->get_is_virtual() == false) {
                    uids_in_operation.push_back(tensor->get_uid());
                }
            }

            if (options.use_dbias.value()) {
                // Create the DRMSNorm operation.
                auto DRMSNorm_operation =
                    cudnn_frontend::OperationBuilder(DescriptorType_t::OPERATION_NORM_BACKWARD_DESCRIPTOR)
                        .setNormalizationMode(NormMode_t::RMS_NORM)
                        .setxDesc(*(tensors.at(options.inputs.X->get_uid())))
                        .setdyDesc(*(tensors.at(options.inputs.DY->get_uid())))
                        .setScale(*(tensors.at(options.inputs.SCALE->get_uid())))
                        .setSavedInvVar(*(tensors.at(options.inputs.INV_VARIANCE->get_uid())))
                        .setDScaleAndDBias(*(tensors.at(options.outputs.DSCALE->get_uid())),
                                           *(tensors.at(options.outputs.DBIAS->get_uid())))
                        .setdxDesc(*(tensors.at(options.outputs.DX->get_uid())))
                        .build();
                operations.push_back({std::move(DRMSNorm_operation), std::move(uids_in_operation)});
            } else {
                // Create the DRMSNorm operation.
                auto DRMSNorm_operation =
                    cudnn_frontend::OperationBuilder(DescriptorType_t::OPERATION_NORM_BACKWARD_DESCRIPTOR)
                        .setNormalizationMode(NormMode_t::RMS_NORM)
                        .setxDesc(*(tensors.at(options.inputs.X->get_uid())))
                        .setdyDesc(*(tensors.at(options.inputs.DY->get_uid())))
                        .setScale(*(tensors.at(options.inputs.SCALE->get_uid())))
                        .setSavedInvVar(*(tensors.at(options.inputs.INV_VARIANCE->get_uid())))
                        .setDScale(*(tensors.at(options.outputs.DSCALE->get_uid())))
                        .setdxDesc(*(tensors.at(options.outputs.DX->get_uid())))
                        .build();
                operations.push_back({std::move(DRMSNorm_operation), std::move(uids_in_operation)});
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