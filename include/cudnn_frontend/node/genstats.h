#pragma once

#include "../../cudnn_frontend_Logging.h"

#include "../graph_helpers.h"
#include "../node_interface.h"

namespace cudnn_frontend {

namespace graph {

class GenstatsNode : public INode {
    Genstats_attributes attributes;

   public:
    GenstatsNode(Genstats_attributes&& attributes_, detail::Context const& context)
        : INode(context), attributes(std::move(attributes_)) {}

    Type
    getType() override final {
        return Type::GENSTATS;
    }

    error_t
    pre_validate_node() const override final {
        CHECK_CUDNN_FRONTEND_ERROR(attributes.validate_inputs());

        return {error_code_t::OK, ""};
    }

    error_t
    expand_and_infer_properties() override final {
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
    post_validate_node() const override final {
        // Validate outputs
        // All properties of output tensors should have been set now.
        CHECK_CUDNN_FRONTEND_ERROR(attributes.validate_outputs());

        return {error_code_t::OK, ""};
    }

    error_t
    create_cudnn_tensors(int64_t& uid, std::unordered_map<int64_t, std::shared_ptr<cudnn_frontend::Tensor>>& tensors)
        const override final {
        getLogger() << "[cudnn_frontend] INFO: "
                    << "Building GenstatsNode tensors " << attributes.name << "..." << std::endl;

        for (auto const& [name, tensor] : attributes.inputs) {
            (void)name;
            if (tensor) {
                CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(tensor, uid, tensors));
            }
        }
        for (auto const& [name, tensor] : attributes.outputs) {
            (void)name;
            if (tensor) {
                CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(tensor, uid, tensors));
            }
        }
        return {error_code_t::OK, ""};
    }

    error_t
    create_cudnn_operations(
        std::unordered_set<uid_t>& uids_involved_in_operations,
        std::vector<cudnn_frontend::Operation_v8>& operations,
        std::unordered_map<int64_t, std::shared_ptr<cudnn_frontend::Tensor>>& tensors) const override final {
        getLogger() << "[cudnn_frontend] INFO: "
                    << "Building GenstatsNode operations " << attributes.name << "..." << std::endl;

#ifndef NV_CUDNN_DISABLE_EXCEPTION
        try {
#endif

            auto&& genstats_operation_builder =
                cudnn_frontend::OperationBuilder(DescriptorType_t::OPERATION_GEN_STATS_DESCRIPTOR);

            CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(X, Genstats_attributes::input_names::X);
            genstats_operation_builder.setxDesc(*(tensors.at(X->second->get_uid())));

            genstats_operation_builder.setGenStatsMode(CUDNN_GENSTATS_SUM_SQSUM);

            CUDNN_FE_VALIDATE_AND_ASSIGN_OUTPUT_TENSOR(SUM, Genstats_attributes::output_names::SUM);
            genstats_operation_builder.setSumDesc(*(tensors.at(SUM->second->get_uid())));

            CUDNN_FE_VALIDATE_AND_ASSIGN_OUTPUT_TENSOR(SQ_SUM, Genstats_attributes::output_names::SQ_SUM);
            genstats_operation_builder.setSqSumDesc(*(tensors.at(SQ_SUM->second->get_uid())));

            operations.push_back(std::move(genstats_operation_builder.build()));

#ifndef NV_CUDNN_DISABLE_EXCEPTION
        } catch (cudnn_frontend::cudnnException& e) {
            throw cudnnException(e.what(), e.getCudnnStatus());
        }
#endif

        auto const& non_virtual_uids = attributes.get_non_virtual_uids();
        uids_involved_in_operations.insert(non_virtual_uids.begin(), non_virtual_uids.end());
        return {error_code_t::OK, ""};
    }

    virtual void
    serialize(json& j) const override final {
        j = attributes;
    }
};

}  // namespace graph

}  // namespace cudnn_frontend