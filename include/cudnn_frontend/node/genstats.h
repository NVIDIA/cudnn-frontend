#pragma once

#include "../../cudnn_frontend_Logging.h"

#include "../graph_helpers.h"
#include "../node_interface.h"

namespace cudnn_frontend {

namespace graph {

class GenstatsNode : public NodeCRTP<GenstatsNode> {
   public:
    Genstats_attributes attributes;

    GenstatsNode(Genstats_attributes&& attributes_, detail::Context const& context)
        : NodeCRTP(context), attributes(std::move(attributes_)) {}

    Type
    getType() override final {
        return Type::GENSTATS;
    }

    error_t
    infer_properties_node() override final {
        attributes.fill_from_context(context);

        // Only inferrencing from X works today.
        auto X      = attributes.inputs[Genstats_attributes::input_names::X];
        auto SUM    = attributes.outputs[Genstats_attributes::output_names::SUM];
        auto SQ_SUM = attributes.outputs[Genstats_attributes::output_names::SQ_SUM];

        auto const x_tensor_dim = X->get_dim();
        auto sum_tensor_dim     = SUM->get_dim();
        auto sq_sum_tensor_dim  = SQ_SUM->get_dim();

        // Only infer dims and strides if user did not set them
        if (sum_tensor_dim.empty()) {
            sum_tensor_dim.resize(x_tensor_dim.size(), 1);
            sum_tensor_dim[1] = x_tensor_dim[1];
            SUM->set_dim(sum_tensor_dim);
        }
        if (SUM->get_stride().empty()) {
            auto const& SUM_dim = SUM->get_dim();
            // Default to NHWC
            auto const& stride_order = detail::generate_NHWC_stride_order(SUM_dim.size());
            SUM->set_stride(detail::generate_stride(SUM_dim, stride_order));
        }

        // Only infer dims and strides if user did not set them
        if (sq_sum_tensor_dim.empty()) {
            sq_sum_tensor_dim.resize(x_tensor_dim.size(), 1);
            sq_sum_tensor_dim[1] = x_tensor_dim[1];
            SQ_SUM->set_dim(sq_sum_tensor_dim);
        }
        if (SQ_SUM->get_stride().empty()) {
            auto const& SQ_SUM_dim = SQ_SUM->get_dim();
            // Default to NHWC
            auto const& stride_order = detail::generate_NHWC_stride_order(SQ_SUM_dim.size());
            SQ_SUM->set_stride(detail::generate_stride(SQ_SUM_dim, stride_order));
        }

        return {error_code_t::OK, ""};
    }

    error_t
    create_cudnn_operations(
        std::unordered_set<uid_t>& uids_involved_in_operations,
        std::vector<std::shared_ptr<cudnn_frontend::Operation>>& operations,
        managed_backend_descriptor_t& raw_operations,
        std::unordered_map<int64_t, std::shared_ptr<cudnn_frontend::Tensor>>& tensors) const override final {
        CUDNN_FRONTEND_UNUSED(raw_operations);
        CUDNN_FE_LOG_LABEL_ENDL("INFO: " << "Building GenstatsNode operations " << attributes.name << "...");

        auto&& genstats_operation_builder =
            cudnn_frontend::OperationBuilder(DescriptorType_t::OPERATION_GEN_STATS_DESCRIPTOR);

        CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(X, Genstats_attributes::input_names::X);
        genstats_operation_builder.setxDesc(*(tensors.at(X->second->get_uid())));

        genstats_operation_builder.setGenStatsMode(CUDNN_GENSTATS_SUM_SQSUM);

        CUDNN_FE_VALIDATE_AND_ASSIGN_OUTPUT_TENSOR(SUM, Genstats_attributes::output_names::SUM);
        genstats_operation_builder.setSumDesc(*(tensors.at(SUM->second->get_uid())));

        CUDNN_FE_VALIDATE_AND_ASSIGN_OUTPUT_TENSOR(SQ_SUM, Genstats_attributes::output_names::SQ_SUM);
        genstats_operation_builder.setSqSumDesc(*(tensors.at(SQ_SUM->second->get_uid())));
#ifdef NV_CUDNN_DISABLE_EXCEPTION
        // disable exception macro is defined. Calling build will not throw.
        // Check status of desc and return error.
        auto operation = genstats_operation_builder.build();
        RETURN_CUDNN_FRONTEND_ERROR_IF(operation.get_status() != CUDNN_STATUS_SUCCESS,
                                       error_code_t::CUDNN_BACKEND_API_FAILED,
                                       operation.get_error());
        operations.push_back(std::make_shared<Operation_v8>(std::move(operation)));
#else
        // build() can throw
        // wrap in try catch
        try {
            auto operation = genstats_operation_builder.build();
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
        j.update(R"( {"tag": "GENSTATS"})"_json);
    }
#endif
};

}  // namespace graph

}  // namespace cudnn_frontend