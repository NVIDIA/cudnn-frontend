#pragma once

#include "../cudnn_frontend_ConvDesc.h"
#include "../cudnn_frontend_Heuristics.h"
#include "../cudnn_frontend_Logging.h"

#include "cudnn_frontend_graph_helpers.h"
#include "cudnn_frontend_node_interface.h"

namespace cudnn_frontend::graph {

class ConvolutionNode : public INode {
   public:
    Conv_fprop_attributes options;

    ConvolutionNode(Conv_fprop_attributes&& options_, detail::Context const& context)
        : INode(context), options(std::move(options_)) {}

    Type
    getType() override final {
        return Type::CONVOLUTION;
    }

    error_t
    infer_properties_node() override final {
        getLogger() << "[cudnn_frontend] INFO: Inferrencing properties for conv node " << options.name << "..."
                    << std::endl;

        options.fill_from_context(context);

        // TODO: Only inferrencing from (X, W) -> Y works today.
        auto X = options.inputs.X;
        auto W = options.inputs.W;
        auto Y = options.outputs.Y;

        auto const x_tensor_dim = X->get_dim();
        auto const w_tensor_dim = W->get_dim();
        auto y_tensor_dim       = Y->get_dim();

        // Only infer dims and strides if user did not set them
        if (y_tensor_dim.empty()) {
            y_tensor_dim.resize(x_tensor_dim.size());
            auto const& padding  = options.get_padding();
            auto const& stride   = options.get_stride();
            auto const& dilation = options.get_dilation();
            // N
            y_tensor_dim[0] = x_tensor_dim[0];
            // PQ
            for (size_t dim = 2; dim < x_tensor_dim.size(); ++dim) {
                y_tensor_dim[dim] =
                    1 + (x_tensor_dim[dim] - dilation[dim - 2] * (w_tensor_dim[dim] - 1) - 1 + 2 * padding[dim - 2]) /
                            stride[dim - 2];
            }
            // K
            y_tensor_dim[1] = w_tensor_dim[0];
            Y->set_dim(y_tensor_dim);
        }
        if (Y->get_stride().empty()) {
            Y->set_stride(detail::generate_stride(Y->get_dim()));
        }

        return {error_code_t::OK, ""};
    }

    error_t
    assign_uids_node() override final {
        options.inputs.X->set_uid(ICudnn::create_new_uid());
        options.inputs.W->set_uid(ICudnn::create_new_uid());
        options.outputs.Y->set_uid(ICudnn::create_new_uid());
        return {error_code_t::OK, ""};
    }

    error_t
    createTensors() override final {
        getLogger() << "[cudnn_frontend] INFO: "
                    << "Building ConvolutionNode tensors " << options.name << "..." << std::endl;

        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.inputs.X));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.inputs.W));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.outputs.Y));

        return {error_code_t::OK, ""};
    }

    error_t
    createOperations() override final {
        getLogger() << "[cudnn_frontend] INFO: "
                    << "Building ConvolutionNode operations " << options.name << "..." << std::endl;

#ifndef NV_CUDNN_DISABLE_EXCEPTION
        try {
#endif

            // convolution descriptor
            int64_t const spatial_dim_count = options.get_padding().size();
            auto convolution_descriptor     = cudnn_frontend::ConvDescBuilder()
                                              .setComputeType(options.get_compute_data_type())
                                              .setMathMode(CUDNN_CROSS_CORRELATION)
                                              .setSpatialDimCount(spatial_dim_count)
                                              .setSpatialStride(spatial_dim_count, options.get_stride().data())
                                              .setPrePadding(spatial_dim_count, options.get_padding().data())
                                              .setPostPadding(spatial_dim_count, options.get_padding().data())
                                              .setDilation(spatial_dim_count, options.get_dilation().data())
                                              .build();

            // Create the convolution operation.
            auto convolution_operation =
                cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)
                    .setxDesc(*(tensors.at(options.inputs.X->get_uid())))
                    .setwDesc(*(tensors.at(options.inputs.W->get_uid())))
                    .setyDesc(*(tensors.at(options.outputs.Y->get_uid())))
                    .setcDesc(convolution_descriptor)
                    .setAlpha(1.f)
                    .setBeta(0.f)
                    .build();

            // Push all real tensors as required for operation execution.
            auto const& tensors_involved_in_operation = {options.inputs.X, options.inputs.W, options.outputs.Y};

            std::vector<uid_t> uids_in_operation;
            for (auto const& tensor : tensors_involved_in_operation) {
                if (tensor && tensor->get_is_virtual() == false) {
                    uids_in_operation.push_back(tensor->get_uid());
                }
            }

            operations.push_back({std::move(convolution_operation), std::move(uids_in_operation)});

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