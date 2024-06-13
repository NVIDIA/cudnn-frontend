#pragma once

#include "../../cudnn_frontend_Heuristics.h"
#include "../../cudnn_frontend_Logging.h"

#include "../graph_helpers.h"
#include "../node_interface.h"

namespace cudnn_frontend {

namespace graph {

class DBNWeightNode : public NodeCRTP<DBNWeightNode> {
   public:
    DBN_weight_attributes attributes;

    DBNWeightNode(DBN_weight_attributes&& attributes_, detail::Context const& context)
        : NodeCRTP(context), attributes(std::move(attributes_)) {}

    Type
    getType() override final {
        return Type::DBN_WEIGHT;
    }

    error_t
    pre_validate_node() const override final {
        CHECK_CUDNN_FRONTEND_ERROR(attributes.validate_inputs());

        return {error_code_t::OK, ""};
    }

    error_t
    expand_and_infer_properties_node() override final {
        getLogger() << "[cudnn_frontend] INFO: Inferencing properties for batchnorm finalize node " << attributes.name
                    << "..." << std::endl;

        attributes.fill_from_context(context);

        // TODO: Only inferencing from DY works today.
        auto DY                  = attributes.inputs[DBN_weight_attributes::input_names::DY];
        auto const dy_tensor_dim = DY->get_dim();

        auto X            = attributes.inputs[DBN_weight_attributes::input_names::X];
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
        infer_per_channel_tensors(attributes.outputs[DBN_weight_attributes::output_names::DBIAS]);
        infer_per_channel_tensors(attributes.outputs[DBN_weight_attributes::output_names::DSCALE]);
        infer_per_channel_tensors(attributes.outputs[DBN_weight_attributes::output_names::EQ_BIAS]);
        infer_per_channel_tensors(attributes.outputs[DBN_weight_attributes::output_names::EQ_SCALE_DY]);
        infer_per_channel_tensors(attributes.outputs[DBN_weight_attributes::output_names::EQ_SCALE_X]);

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
        getLogger() << "[cudnn_frontend] INFO: " << "Building DBNWeightNode operations " << attributes.name << "..."
                    << std::endl;

        // Create the batchnorm operation.
        auto&& batchnorm_operation_builder =
            cudnn_frontend::OperationBuilder(DescriptorType_t::OPERATION_BN_BWD_WEIGHTS_DESCRIPTOR);

        batchnorm_operation_builder.setComputeType(CUDNN_DATA_FLOAT);

        CUDNN_FE_VALIDATE_AND_ASSIGN_OUTPUT_TENSOR(EQ_SCALE_DY, DBN_weight_attributes::output_names::EQ_SCALE_DY);
        CUDNN_FE_VALIDATE_AND_ASSIGN_OUTPUT_TENSOR(EQ_SCALE_X, DBN_weight_attributes::output_names::EQ_SCALE_X);
        CUDNN_FE_VALIDATE_AND_ASSIGN_OUTPUT_TENSOR(EQ_BIAS, DBN_weight_attributes::output_names::EQ_BIAS);
        batchnorm_operation_builder.setEqScalesAndBias(*(tensors.at(EQ_SCALE_DY->second->get_uid())),
                                                       *(tensors.at(EQ_SCALE_X->second->get_uid())),
                                                       *(tensors.at(EQ_BIAS->second->get_uid())));

        CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(MEAN, DBN_weight_attributes::input_names::MEAN);
        CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(INV_VARIANCE, DBN_weight_attributes::input_names::INV_VARIANCE);
        batchnorm_operation_builder.setSavedMeanAndInvVar(*(tensors.at(MEAN->second->get_uid())),
                                                          *(tensors.at(INV_VARIANCE->second->get_uid())));

        CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(SCALE, DBN_weight_attributes::input_names::SCALE);
        batchnorm_operation_builder.setScale(*(tensors.at(SCALE->second->get_uid())));

        CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(X, DBN_weight_attributes::input_names::X);
        batchnorm_operation_builder.setxDesc(*(tensors.at(X->second->get_uid())));

        CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(DY, DBN_weight_attributes::input_names::DY);
        batchnorm_operation_builder.setdyDesc(*(tensors.at(DY->second->get_uid())));

        CUDNN_FE_VALIDATE_AND_ASSIGN_OUTPUT_TENSOR(DSCALE, DBN_weight_attributes::output_names::DSCALE);
        CUDNN_FE_VALIDATE_AND_ASSIGN_OUTPUT_TENSOR(DBIAS, DBN_weight_attributes::output_names::DBIAS);
        batchnorm_operation_builder.setDScaleAndDBias(*(tensors.at(DSCALE->second->get_uid())),
                                                      *(tensors.at(DBIAS->second->get_uid())));

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
        j.update(R"( {"tag": "DBN_WEIGHT"})"_json);
    }
#endif
};

}  // namespace graph

}  // namespace cudnn_frontend