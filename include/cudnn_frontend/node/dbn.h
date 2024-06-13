#pragma once

#include "../../cudnn_frontend_Heuristics.h"
#include "../../cudnn_frontend_Logging.h"

#include "../graph_helpers.h"
#include "../node_interface.h"

namespace cudnn_frontend {

namespace graph {

class DBNNode : public NodeCRTP<DBNNode> {
   public:
    Batchnorm_backward_attributes attributes;

    DBNNode(Batchnorm_backward_attributes&& attributes_, detail::Context const& context)
        : NodeCRTP(context), attributes(std::move(attributes_)) {}

    Type
    getType() override final {
        return Type::DBN;
    }

    error_t
    pre_validate_node() const override final {
        getLogger() << "[cudnn_frontend] INFO: " << "Validating DBNNode " << attributes.name << "..." << std::endl;

        CHECK_CUDNN_FRONTEND_ERROR(attributes.validate_inputs());
        return {error_code_t::OK, ""};
    }

    error_t
    expand_and_infer_properties_node() override final {
        getLogger() << "[cudnn_frontend] INFO: Inferencing properties for DBN node " << attributes.name << "..."
                    << std::endl;

        attributes.fill_from_context(context);

        // TODO: Only inferencing from X works today.
        auto X                  = attributes.inputs[Batchnorm_backward_attributes::input_names::X];
        auto const x_tensor_dim = X->get_dim();

        auto DX            = attributes.outputs[Batchnorm_backward_attributes::output_names::DX];
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
        infer_per_channel_tensors(attributes.outputs[Batchnorm_backward_attributes::output_names::DSCALE]);
        infer_per_channel_tensors(attributes.outputs[Batchnorm_backward_attributes::output_names::DBIAS]);
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
        getLogger() << "[cudnn_frontend] INFO: " << "Building DBNNode operations " << attributes.name << "..."
                    << std::endl;

        std::vector<cudnn_frontend::Tensor> peer_stats;
        for (auto const& peer_stat : attributes.peer_stats) {
            peer_stats.emplace_back(std::move(*(tensors.at(peer_stat->get_uid()))));
        }

        // Create the DBN operation.
        auto&& DBN_operation_builder =
            cudnn_frontend::OperationBuilder(DescriptorType_t::OPERATION_NORM_BACKWARD_DESCRIPTOR);

        DBN_operation_builder.setNormalizationMode(NormMode_t::BATCH_NORM);

        CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(X, Batchnorm_backward_attributes::input_names::X);
        DBN_operation_builder.setxDesc(*(tensors.at(X->second->get_uid())));

        CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(DY, Batchnorm_backward_attributes::input_names::DY);
        DBN_operation_builder.setdyDesc(*(tensors.at(DY->second->get_uid())));

        CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(SCALE, Batchnorm_backward_attributes::input_names::SCALE);
        DBN_operation_builder.setScale(*(tensors.at(SCALE->second->get_uid())));

        CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(MEAN, Batchnorm_backward_attributes::input_names::MEAN);
        CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(INV_VARIANCE,
                                                  Batchnorm_backward_attributes::input_names::INV_VARIANCE);
        DBN_operation_builder.setSavedMeanAndInvVar(*(tensors.at(MEAN->second->get_uid())),
                                                    *(tensors.at(INV_VARIANCE->second->get_uid())));

        CUDNN_FE_VALIDATE_AND_ASSIGN_OUTPUT_TENSOR(DSCALE, Batchnorm_backward_attributes::output_names::DSCALE);
        CUDNN_FE_VALIDATE_AND_ASSIGN_OUTPUT_TENSOR(DBIAS, Batchnorm_backward_attributes::output_names::DBIAS);
        DBN_operation_builder.setDScaleAndDBias(*(tensors.at(DSCALE->second->get_uid())),
                                                *(tensors.at(DBIAS->second->get_uid())));

        CUDNN_FE_VALIDATE_AND_ASSIGN_OUTPUT_TENSOR(DX, Batchnorm_backward_attributes::output_names::DX);
        DBN_operation_builder.setdxDesc(*(tensors.at(DX->second->get_uid())));

        DBN_operation_builder.setPeerStatTensor(peer_stats);

#ifdef NV_CUDNN_DISABLE_EXCEPTION
        // disable exception macro is defined. Calling build will not throw.
        // Check status of desc and return error.
        auto operation = DBN_operation_builder.build();
        RETURN_CUDNN_FRONTEND_ERROR_IF(operation.get_status() != CUDNN_STATUS_SUCCESS,
                                       error_code_t::CUDNN_BACKEND_API_FAILED,
                                       operation.get_error());
        operations.push_back(std::make_shared<Operation_v8>(std::move(operation)));
#else
        // build() can throw
        // wrap in try catch
        try {
            auto operation = DBN_operation_builder.build();
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
        j.update(R"( {"tag": "DBN"})"_json);
    }
#endif
};

}  // namespace graph

}  // namespace cudnn_frontend