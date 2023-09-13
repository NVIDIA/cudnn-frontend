#pragma once

#include "../../cudnn_frontend_Logging.h"

#include "../cudnn_frontend_graph_helpers.h"
#include "../cudnn_frontend_node_interface.h"

namespace cudnn_frontend::graph {

class ReshapeNode : public INode {
    Reshape_attributes options;

   public:
    ReshapeNode(Reshape_attributes&& options_, detail::Context const& context)
        : INode(context), options(std::move(options_)) {}

    Type
    getType() override final {
        return Type::RESHAPE;
    }

    error_t
    validate_node() const override final {
        getLogger() << "[cudnn_frontend] INFO: "
                    << "Validating ReshapeNode " << options.name << "..." << std::endl;

        if (nullptr == options.inputs.X) {
            auto status         = error_code_t::ATTRIBUTE_NOT_SET;
            std::string message = "[cudnn_frontend] ERROR: reshape input not set.";
            return {status, message};
        }
        if (nullptr == options.outputs.Y) {
            auto status         = error_code_t::ATTRIBUTE_NOT_SET;
            std::string message = "[cudnn_frontend] ERROR: reshape output not set.";
            return {status, message};
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
    infer_properties_node() override final {
        getLogger() << "[cudnn_frontend] INFO: Inferrencing properties for reshape node " << options.name << "..."
                    << std::endl;

        auto y_tensor = options.outputs.Y;

        options.fill_from_context(context);

        // If user does not set shape and layout of the output tensor,
        // Get it from node attributes
        // If layout is not set, generate the strides from layout

        if (y_tensor->get_dim().empty() && options.get_dim().size()) {
            y_tensor->set_dim(options.dim);
        }

        if (y_tensor->get_stride().empty()) {
            if (options.get_stride().size()) {
                y_tensor->set_stride(options.get_stride());
            } else {
                y_tensor->set_stride(detail::generate_stride(y_tensor->get_dim()));
            }
        }

        if (y_tensor->get_dim().empty() || y_tensor->get_stride().empty()) {
            return {error_code_t::SHAPE_DEDUCTION_FAILED, "Reshape node output shape deduction failed"};
        }

        return {error_code_t::OK, ""};
    }

    error_t
    createTensors() override final {
        getLogger() << "[cudnn_frontend] INFO: "
                    << "Building Reshape tensors " << options.name << "..." << std::endl;

        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.inputs.X));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.outputs.Y));

        return {error_code_t::OK, ""};
    }

    error_t
    createOperations() override final {
        getLogger() << "[cudnn_frontend] INFO: "
                    << "Building ReshapeNode operations " << options.name << "..." << std::endl;

#ifndef NV_CUDNN_DISABLE_EXCEPTION
        try {
#endif
            // Push all real tensors as required for operation execution.
            auto const& tensors_involved_in_operation = {options.inputs.X, options.outputs.Y};

            auto reshape_op = cudnn_frontend::OperationBuilder(DescriptorType_t::OPERATION_RESHAPE_DESCRIPTOR)
                                  .setxDesc(*(tensors.at(options.inputs.X->get_uid())))
                                  .setyDesc(*(tensors.at(options.outputs.Y->get_uid())))
                                  .build();

            std::vector<uid_t> uids_in_operation;
            for (auto const& tensor : tensors_involved_in_operation) {
                if (tensor && tensor->get_is_virtual() == false) {
                    uids_in_operation.push_back(tensor->get_uid());
                }
            }
            operations.push_back({std::move(reshape_op), std::move(uids_in_operation)});

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