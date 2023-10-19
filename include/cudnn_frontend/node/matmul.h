#pragma once

#include "../../cudnn_frontend_MatMulDesc.h"
#include "../../cudnn_frontend_Heuristics.h"
#include "../../cudnn_frontend_Logging.h"

#include "../graph_helpers.h"
#include "../node_interface.h"

namespace cudnn_frontend::graph {

class MatmulNode : public INode {
    Matmul_attributes options;

   public:
    MatmulNode(Matmul_attributes&& options_, detail::Context const& context)
        : INode(context), options(std::move(options_)) {}

    Type
    getType() override final {
        return Type::MATMUL;
    }

    error_t
    validate_node() const override final {
        getLogger() << "[cudnn_frontend] INFO: "
                    << "Validating matmul node " << options.name << "..." << std::endl;

        RETURN_CUDNN_FRONTEND_ERROR_IF(!(options.inputs.A), error_code_t::ATTRIBUTE_NOT_SET, "matmul A not set.");

        RETURN_CUDNN_FRONTEND_ERROR_IF(!(options.inputs.B), error_code_t::ATTRIBUTE_NOT_SET, "matmul B not set.");

        RETURN_CUDNN_FRONTEND_ERROR_IF(!(options.outputs.C), error_code_t::ATTRIBUTE_NOT_SET, "matmul C not set.");

        return {error_code_t::OK, ""};
    }

    error_t
    infer_properties_node() override final {
        getLogger() << "[cudnn_frontend] INFO: Inferrencing properties for matmul node " << options.name << "..."
                    << std::endl;

        options.fill_from_context(context);

        // Only inferrencing from (A, B) -> C works today.
        auto a_tensor = options.inputs.A;
        auto b_tensor = options.inputs.B;
        auto c_tensor = options.outputs.C;

        auto const a_tensor_dim = a_tensor->get_dim();
        auto const b_tensor_dim = b_tensor->get_dim();
        auto c_tensor_dim       = c_tensor->get_dim();

        // Only infer dims and strides if user did not set them
        if (c_tensor_dim.empty()) {
            c_tensor_dim.resize(a_tensor_dim.size());
            c_tensor_dim[0] = a_tensor_dim[0];  // B
            c_tensor_dim[1] = a_tensor_dim[1];  // M
            c_tensor_dim[2] = b_tensor_dim[2];  // N
            c_tensor->set_dim(c_tensor_dim);
        }
        if (c_tensor->get_stride().empty()) {
            auto const& c_dim = c_tensor->get_dim();
            // Default to Col major
            auto const& stride_order = detail::generate_row_major_stride_order(c_dim.size());
            c_tensor->set_stride(detail::generate_stride(c_dim, stride_order));
        }

        return {error_code_t::OK, ""};
    }

    error_t
    assign_uids_node() override final {
        options.inputs.A->set_uid(ICudnn::create_new_uid());
        options.inputs.B->set_uid(ICudnn::create_new_uid());
        if (options.inputs.M_override) options.inputs.M_override->set_uid(ICudnn::create_new_uid());
        if (options.inputs.N_override) options.inputs.N_override->set_uid(ICudnn::create_new_uid());
        if (options.inputs.K_override) options.inputs.K_override->set_uid(ICudnn::create_new_uid());
        options.outputs.C->set_uid(ICudnn::create_new_uid());
        return {error_code_t::OK, ""};
    }

    error_t
    createTensors() override final {
        getLogger() << "[cudnn_frontend] INFO: "
                    << "Building MatmulNode tensors " << options.name << "..." << std::endl;

        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.inputs.A));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.inputs.B));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.outputs.C));

        for (auto const& tensor : {options.inputs.M_override, options.inputs.N_override, options.inputs.K_override}) {
            if (tensor) {
                CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(tensor));
            }
        }

        return {error_code_t::OK, ""};
    }

    error_t
    createOperations() override final {
        getLogger() << "[cudnn_frontend] INFO: "
                    << "Building MatmulNode operations " << options.name << "..." << std::endl;

#ifndef NV_CUDNN_DISABLE_EXCEPTION
        try {
#endif

            // Push all real tensors as required for operation execution.
            auto const& tensors_involved_in_operation = {options.inputs.A,
                                                         options.inputs.B,
                                                         options.inputs.M_override,
                                                         options.inputs.N_override,
                                                         options.inputs.K_override,
                                                         options.outputs.C};

            // matmul descriptor
            auto matmul_descriptor =
                cudnn_frontend::MatMulDescBuilder().setComputeType(options.get_compute_data_type()).build();

            if (options.inputs.N_override) {
                // Create the matmul operation.
                auto matmul_operation = cudnn_frontend::OperationBuilder(DescriptorType_t::OPERATION_MATMUL_DESCRIPTOR)
                                            .setaMatDesc(*tensors.at(options.inputs.A->get_uid()))
                                            .setbMatDesc(*tensors.at(options.inputs.B->get_uid()))
                                            .setcMatDesc(*tensors.at(options.outputs.C->get_uid()))
                                            .setmatmulDesc(matmul_descriptor)
                                            .setmOverrideDesc(*tensors.at(options.inputs.M_override->get_uid()))
                                            .setnOverrideDesc(*tensors.at(options.inputs.N_override->get_uid()))
                                            .build();
                std::vector<uid_t> uids_in_operation;
                for (auto const& tensor : tensors_involved_in_operation) {
                    if (tensor && tensor->get_is_virtual() == false) {
                        uids_in_operation.push_back(tensor->get_uid());
                    }
                }

                operations.push_back({std::move(matmul_operation), std::move(uids_in_operation)});
            } else if (options.inputs.K_override) {
                // Create the matmul operation.
                auto matmul_operation = cudnn_frontend::OperationBuilder(DescriptorType_t::OPERATION_MATMUL_DESCRIPTOR)
                                            .setaMatDesc(*tensors.at(options.inputs.A->get_uid()))
                                            .setbMatDesc(*tensors.at(options.inputs.B->get_uid()))
                                            .setcMatDesc(*tensors.at(options.outputs.C->get_uid()))
                                            .setmatmulDesc(matmul_descriptor)
                                            .setmOverrideDesc(*tensors.at(options.inputs.M_override->get_uid()))
                                            .setkOverrideDesc(*tensors.at(options.inputs.K_override->get_uid()))
                                            .build();
                std::vector<uid_t> uids_in_operation;
                for (auto const& tensor : tensors_involved_in_operation) {
                    if (tensor && tensor->get_is_virtual() == false) {
                        uids_in_operation.push_back(tensor->get_uid());
                    }
                }

                operations.push_back({std::move(matmul_operation), std::move(uids_in_operation)});
            } else {
                // Create the matmul operation.
                auto matmul_operation = cudnn_frontend::OperationBuilder(DescriptorType_t::OPERATION_MATMUL_DESCRIPTOR)
                                            .setaMatDesc(*tensors.at(options.inputs.A->get_uid()))
                                            .setbMatDesc(*tensors.at(options.inputs.B->get_uid()))
                                            .setcMatDesc(*tensors.at(options.outputs.C->get_uid()))
                                            .setmatmulDesc(matmul_descriptor)
                                            .build();
                std::vector<uid_t> uids_in_operation;
                for (auto const& tensor : tensors_involved_in_operation) {
                    if (tensor && tensor->get_is_virtual() == false) {
                        uids_in_operation.push_back(tensor->get_uid());
                    }
                }

                operations.push_back({std::move(matmul_operation), std::move(uids_in_operation)});
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