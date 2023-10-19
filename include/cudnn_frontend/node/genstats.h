#pragma once

#include "../../cudnn_frontend_Logging.h"

#include "../graph_helpers.h"
#include "../node_interface.h"

namespace cudnn_frontend {

namespace graph {

class GenstatsNode : public INode {
    Genstats_attributes options;

   public:
    GenstatsNode(Genstats_attributes&& options_, detail::Context const& context)
        : INode(context), options(std::move(options_)) {}

    Type
    getType() override final {
        return Type::GENSTATS;
    }

    error_t
    infer_properties_node() override final {
        options.fill_from_context(context);

        // Only inferrencing from X works today.
        auto X      = options.inputs.X;
        auto SUM    = options.outputs.SUM;
        auto SQ_SUM = options.outputs.SQ_SUM;

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
    assign_uids_node() override final {
        options.inputs.X->set_uid(ICudnn::create_new_uid());
        options.outputs.SUM->set_uid(ICudnn::create_new_uid());
        options.outputs.SQ_SUM->set_uid(ICudnn::create_new_uid());
        return {error_code_t::OK, ""};
    }

    error_t
    createTensors() override final {
        getLogger() << "[cudnn_frontend] INFO: "
                    << "Building GenstatsNode tensors " << options.name << "..." << std::endl;

        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.inputs.X));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.outputs.SUM));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.outputs.SQ_SUM));

        return {error_code_t::OK, ""};
    }

    error_t
    createOperations() override final {
        getLogger() << "[cudnn_frontend] INFO: "
                    << "Building GenstatsNode operations " << options.name << "..." << std::endl;

#ifndef NV_CUDNN_DISABLE_EXCEPTION
        try {
#endif

            auto genstats_operation = cudnn_frontend::OperationBuilder(DescriptorType_t::OPERATION_GEN_STATS_DESCRIPTOR)
                                          .setxDesc(*(tensors.at(options.inputs.X->get_uid())))
                                          .setGenStatsMode(CUDNN_GENSTATS_SUM_SQSUM)
                                          .setSumDesc(*(tensors.at(options.outputs.SUM->get_uid())))
                                          .setSqSumDesc(*(tensors.at(options.outputs.SQ_SUM->get_uid())))
                                          .build();

            // Push all real tensors as required for operation execution.
            auto const& tensors_involved_in_operation = {options.inputs.X, options.outputs.SUM, options.outputs.SQ_SUM};

            std::vector<uid_t> uids_in_operation;
            for (auto const& tensor : tensors_involved_in_operation) {
                if (tensor && tensor->get_is_virtual() == false) {
                    uids_in_operation.push_back(tensor->get_uid());
                }
            }

            operations.push_back({std::move(genstats_operation), std::move(uids_in_operation)});

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