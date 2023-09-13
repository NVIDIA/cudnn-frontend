#pragma once

#include "../../cudnn_frontend_PointWiseDesc.h"
#include "../../cudnn_frontend_Heuristics.h"
#include "../../cudnn_frontend_Logging.h"

#include "../cudnn_frontend_graph_helpers.h"
#include "../cudnn_frontend_node_interface.h"

namespace cudnn_frontend::graph {

class PointwiseNode : public INode {
   public:
    Pointwise_attributes options;

    PointwiseNode(Pointwise_attributes&& options_, detail::Context const& context)
        : INode(context), options(std::move(options_)) {}

    Type
    getType() override final {
        return Type::POINTWISE;
    }

    error_t
    validate_node() const override final {
        getLogger() << "[cudnn_frontend] INFO: "
                    << "Validating pointwise node " << options.name << "..." << std::endl;

        if (options.mode == PointwiseMode_t::NOT_SET) {
            auto status         = error_code_t::ATTRIBUTE_NOT_SET;
            std::string message = "[cudnn_frontend] ERROR: pointwise mode not set.";
            getLogger() << message << std::endl;
            return {status, message};
        }

        if (!(options.inputs.IN_0)) {
            auto status         = error_code_t::ATTRIBUTE_NOT_SET;
            std::string message = "[cudnn_frontend] ERROR: pointwise input IN_0 not set.";
            getLogger() << message << std::endl;
            return {status, message};
        }

        auto const port_count = get_pointwise_mode_port_count(options.mode);
        if (port_count >= 3) {
            if (!(options.inputs.IN_1)) {
                auto status         = error_code_t::ATTRIBUTE_NOT_SET;
                std::string message = "[cudnn_frontend] ERROR: pointwise input IN_1 not set.";
                getLogger() << message << std::endl;
                return {status, message};
            }
        }

        if (port_count >= 4) {
            if (!(options.inputs.IN_2)) {
                auto status         = error_code_t::ATTRIBUTE_NOT_SET;
                std::string message = "[cudnn_frontend] ERROR: pointwise input IN_2 not set.";
                getLogger() << message << std::endl;
                return {status, message};
            }
        }

        if (!(options.outputs.OUT_0)) {
            auto status         = error_code_t::ATTRIBUTE_NOT_SET;
            std::string message = "[cudnn_frontend] ERROR: pointwise output OUT_0 not set in " + options.get_name();
            getLogger() << message << std::endl;
            return {status, message};
        }

        return {error_code_t::OK, ""};
    }

    error_t
    infer_properties_node() override final {
        getLogger() << "[cudnn_frontend] INFO: Inferrencing properties for pointwise node " << options.name << "..."
                    << std::endl;

        options.fill_from_context(context);

        // Only inferrencing from IN_0 to OUT_0 works today.
        auto in_0_tensor  = options.inputs.IN_0;
        auto out_0_tensor = options.outputs.OUT_0;

        auto out_0_tensor_dim = out_0_tensor->get_dim();
        // Only infer dims and strides if user did not set them
        if (out_0_tensor_dim.empty()) {
            out_0_tensor->set_dim(in_0_tensor->get_dim());
        }
        // Special case here where input strides are being copied over
        if (out_0_tensor->get_stride().empty()) {
            out_0_tensor->set_stride(in_0_tensor->get_stride());
        }

        return {error_code_t::OK, ""};
    }

    error_t
    assign_uids_node() override final {
        options.inputs.IN_0->set_uid(ICudnn::create_new_uid());
        if (options.inputs.IN_1) options.inputs.IN_1->set_uid(ICudnn::create_new_uid());
        if (options.inputs.IN_2) options.inputs.IN_2->set_uid(ICudnn::create_new_uid());
        options.outputs.OUT_0->set_uid(ICudnn::create_new_uid());
        return {error_code_t::OK, ""};
    }

    error_t
    createTensors() override final {
        getLogger() << "[cudnn_frontend] INFO: "
                    << "Building PointwiseNode " << options.name << " tensors X:" << std::endl;
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.inputs.IN_0));

        auto const port_count = get_pointwise_mode_port_count(options.mode);
        if (port_count >= 3) {
            getLogger() << "[cudnn_frontend] INFO: "
                        << "Building PointwiseNode " << options.name << " tensors B:" << std::endl;
            CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.inputs.IN_1));
        }
        if (port_count >= 4) {
            getLogger() << "[cudnn_frontend] INFO: "
                        << "Building PointwiseNode " << options.name << " tensors T:" << std::endl;
            CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.inputs.IN_2));
        }

        getLogger() << "[cudnn_frontend] INFO: "
                    << "Building PointwiseNode " << options.name << " tensors Y:" << std::endl;
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.outputs.OUT_0));

        return {error_code_t::OK, ""};
    }

    error_t
    createOperations() override final {
        getLogger() << "[cudnn_frontend] INFO: "
                    << "Building PointwiseNode operations " << options.name << "..." << std::endl;

#ifndef NV_CUDNN_DISABLE_EXCEPTION
        try {
#endif

            // Push all real tensors as required for operation execution.
            auto const& tensors_involved_in_operation = {
                options.inputs.IN_0, options.inputs.IN_1, options.inputs.IN_2, options.outputs.OUT_0};

            auto pointwise_descriptor = cudnn_frontend::PointwiseDescBuilder()
                                            .setAxis(options.get_axis().value_or(-1))
                                            .setReluLowerClipSlope(options.relu_lower_clip_slope.value_or(0.0))
                                            .setComputeType(options.get_compute_data_type())
                                            .setMode(options.mode)
                                            .build();

            auto const port_count = get_pointwise_mode_port_count(options.mode);
            if (port_count == 4) {
                auto pointwise_operation =
                    cudnn_frontend::OperationBuilder(DescriptorType_t::OPERATION_POINTWISE_DESCRIPTOR)
                        .setxDesc(*(tensors.at(options.inputs.IN_0->get_uid())))
                        .setbDesc(*(tensors.at(options.inputs.IN_1->get_uid())))
                        .settDesc(*(tensors.at(options.inputs.IN_2->get_uid())))
                        .setyDesc(*(tensors.at(options.outputs.OUT_0->get_uid())))
                        .setpwDesc(pointwise_descriptor)
                        .build();
                std::vector<uid_t> uids_in_operation;
                for (auto const& tensor : tensors_involved_in_operation) {
                    if (tensor && tensor->get_is_virtual() == false) {
                        uids_in_operation.push_back(tensor->get_uid());
                    }
                }

                operations.push_back({std::move(pointwise_operation), std::move(uids_in_operation)});
            } else if (port_count == 3) {
                if (options.mode == PointwiseMode_t::RELU_BWD) {
                    auto pointwise_operation =
                        cudnn_frontend::OperationBuilder(DescriptorType_t::OPERATION_POINTWISE_DESCRIPTOR)
                            .setdyDesc(*(tensors.at(options.inputs.IN_0->get_uid())))
                            .setxDesc(*(tensors.at(options.inputs.IN_1->get_uid())))
                            .setdxDesc(*(tensors.at(options.outputs.OUT_0->get_uid())))
                            .setpwDesc(pointwise_descriptor)
                            .build();
                    std::vector<uid_t> uids_in_operation;
                    for (auto const& tensor : tensors_involved_in_operation) {
                        if (tensor && tensor->get_is_virtual() == false) {
                            uids_in_operation.push_back(tensor->get_uid());
                        }
                    }

                    operations.push_back({std::move(pointwise_operation), std::move(uids_in_operation)});
                } else {
                    auto pointwise_operation =
                        cudnn_frontend::OperationBuilder(DescriptorType_t::OPERATION_POINTWISE_DESCRIPTOR)
                            .setxDesc(*(tensors.at(options.inputs.IN_0->get_uid())))
                            .setbDesc(*(tensors.at(options.inputs.IN_1->get_uid())))
                            .setyDesc(*(tensors.at(options.outputs.OUT_0->get_uid())))
                            .setpwDesc(pointwise_descriptor)
                            .build();
                    std::vector<uid_t> uids_in_operation;
                    for (auto const& tensor : tensors_involved_in_operation) {
                        if (tensor && tensor->get_is_virtual() == false) {
                            uids_in_operation.push_back(tensor->get_uid());
                        }
                    }

                    operations.push_back({std::move(pointwise_operation), std::move(uids_in_operation)});
                }
            } else if (port_count == 2) {
                auto pointwise_operation =
                    cudnn_frontend::OperationBuilder(DescriptorType_t::OPERATION_POINTWISE_DESCRIPTOR)
                        .setxDesc(*(tensors.at(options.inputs.IN_0->get_uid())))
                        .setyDesc(*(tensors.at(options.outputs.OUT_0->get_uid())))
                        .setpwDesc(pointwise_descriptor)
                        .build();
                std::vector<uid_t> uids_in_operation;
                for (auto const& tensor : tensors_involved_in_operation) {
                    if (tensor && tensor->get_is_virtual() == false) {
                        uids_in_operation.push_back(tensor->get_uid());
                    }
                }

                operations.push_back({std::move(pointwise_operation), std::move(uids_in_operation)});
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

}  // namespace cudnn_frontend::graph