#pragma once

#include "../../cudnn_frontend_ReductionDesc.h"
#include "../../cudnn_frontend_Logging.h"

#include "../cudnn_frontend_graph_helpers.h"
#include "../cudnn_frontend_node_interface.h"

namespace cudnn_frontend::graph {

class ReductionNode : public INode {
    Reduction_attributes options;

   public:
    ReductionNode(Reduction_attributes&& options_, detail::Context const& context)
        : INode(context), options(std::move(options_)) {}

    Type
    getType() override final {
        return Type::REDUCTION;
    }

    error_t
    validate_node() const override final {
        getLogger() << "[cudnn_frontend] INFO: "
                    << "Validating reduction node " << options.name << "..." << std::endl;

        RETURN_CUDNN_FRONTEND_ERROR_IF(
            !(options.inputs.X), error_code_t::ATTRIBUTE_NOT_SET, "reduction input not set.");

        RETURN_CUDNN_FRONTEND_ERROR_IF(!(options.outputs.Y), error_code_t::ATTRIBUTE_NOT_SET, "reduction Y not set.");

        return {error_code_t::OK, ""};
    }

    error_t
    infer_properties_node() override final {
        getLogger() << "[cudnn_frontend] INFO: Inferrencing properties for reduction node " << options.name << "..."
                    << std::endl;

        options.fill_from_context(context);

        // Only inferrencing from IN_0 to OUT_0 works today.
        auto x_tensor = options.inputs.X;
        auto y_tensor = options.outputs.Y;

        auto const& x_tensor_dim = x_tensor->get_dim();
        auto y_tensor_dim        = y_tensor->get_dim();
        // Only infer dims and strides if user did not set them
        if (y_tensor_dim.empty()) {
            y_tensor->set_dim(x_tensor_dim);
        }
        if (y_tensor->get_stride().empty()) {
            auto const& y_dim = y_tensor->get_dim();
            // Default to NHWC
            auto const& stride_order = detail::generate_NHWC_stride_order(y_dim.size());
            y_tensor->set_stride(detail::generate_stride(y_dim, stride_order));
        }

        return {error_code_t::OK, ""};
    }

    error_t
    assign_uids_node() override final {
        options.inputs.X->set_uid(ICudnn::create_new_uid());
        options.outputs.Y->set_uid(ICudnn::create_new_uid());
        return {error_code_t::OK, ""};
    }

    error_t
    createTensors() override final {
        getLogger() << "[cudnn_frontend] INFO: "
                    << "Building ReductionNode tensors " << options.name << "..." << std::endl;

        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.inputs.X));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.outputs.Y));

        return {error_code_t::OK, ""};
    }

    error_t
    createOperations() override final {
        getLogger() << "[cudnn_frontend] INFO: "
                    << "Building ReductionNode operations " << options.name << "..." << std::endl;

#ifndef NV_CUDNN_DISABLE_EXCEPTION
        try {
#endif

            auto reduction_descriptor = cudnn_frontend::ReductionDescBuilder()
                                            .setComputeType(options.get_compute_data_type())
                                            .setReductionOp(options.get_mode().value())
                                            .build();

            auto reduction_operation = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR)
                                           .setxDesc(*(tensors.at(options.inputs.X->get_uid())))
                                           .setyDesc(*(tensors.at(options.outputs.Y->get_uid())))
                                           .setreductionDesc(reduction_descriptor)
                                           .build();

            // Push all real tensors as required for operation execution.
            auto const& tensors_involved_in_operation = {options.inputs.X, options.outputs.Y};

            std::vector<uid_t> uids_in_operation;
            for (auto const& tensor : tensors_involved_in_operation) {
                if (tensor && tensor->get_is_virtual() == false) {
                    uids_in_operation.push_back(tensor->get_uid());
                }
            }

            operations.push_back({std::move(reduction_operation), std::move(uids_in_operation)});

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

}  // namespace cudnn_frontend::graph