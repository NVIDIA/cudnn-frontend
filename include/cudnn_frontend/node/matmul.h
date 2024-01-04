#pragma once

#include "../../cudnn_frontend_MatMulDesc.h"
#include "../../cudnn_frontend_Heuristics.h"
#include "../../cudnn_frontend_Logging.h"

#include "../graph_helpers.h"
#include "../node_interface.h"

namespace cudnn_frontend::graph {

class MatmulNode : public INode {
    Matmul_attributes attributes;

   public:
    MatmulNode(Matmul_attributes&& attributes_, detail::Context const& context)
        : INode(context), attributes(std::move(attributes_)) {}

    Type
    getType() override final {
        return Type::MATMUL;
    }

    error_t
    pre_validate_node() const override final {
        getLogger() << "[cudnn_frontend] INFO: "
                    << "Validating matmul node " << attributes.name << "..." << std::endl;

        CUDNN_FE_VALIDATE_INPUT_TENSOR(Matmul_attributes::input_names::A);
        CUDNN_FE_VALIDATE_INPUT_TENSOR(Matmul_attributes::input_names::B);
        CUDNN_FE_VALIDATE_OUTPUT_TENSOR(Matmul_attributes::output_names::C);

        CHECK_CUDNN_FRONTEND_ERROR(attributes.validate_inputs());

        return {error_code_t::OK, ""};
    }

    error_t
    expand_and_infer_properties() override final {
        getLogger() << "[cudnn_frontend] INFO: Inferrencing properties for matmul node " << attributes.name << "..."
                    << std::endl;

        attributes.fill_from_context(context);

        // Only inferrencing from (A, B) -> C works today.
        auto a_tensor = attributes.inputs[Matmul_attributes::input_names::A];
        auto b_tensor = attributes.inputs[Matmul_attributes::input_names::B];
        auto c_tensor = attributes.outputs[Matmul_attributes::output_names::C];

        auto const a_tensor_dim = a_tensor->get_dim();
        auto const b_tensor_dim = b_tensor->get_dim();
        auto c_tensor_dim       = c_tensor->get_dim();

        // Only infer dims and strides if user did not set them
        if (c_tensor_dim.empty()) {
            c_tensor_dim.resize(a_tensor_dim.size());
            if (a_tensor_dim.size() == 4) {
                c_tensor_dim[0] = a_tensor_dim[0];  // B
                c_tensor_dim[1] = a_tensor_dim[1];  // H
                c_tensor_dim[2] = a_tensor_dim[2];  // M
                c_tensor_dim[3] = b_tensor_dim[3];  // N
            } else {
                c_tensor_dim[0] = a_tensor_dim[0];  // B
                c_tensor_dim[1] = a_tensor_dim[1];  // M
                c_tensor_dim[2] = b_tensor_dim[2];  // N
            }
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
                    << "Building MatmulNode tensors " << attributes.name << "..." << std::endl;

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
        std::vector<std::shared_ptr<cudnn_frontend::Operation>>& operations,
        std::unordered_map<int64_t, std::shared_ptr<cudnn_frontend::Tensor>>& tensors) const override final {
        getLogger() << "[cudnn_frontend] INFO: "
                    << "Building MatmulNode operations " << attributes.name << "..." << std::endl;

#ifndef NV_CUDNN_DISABLE_EXCEPTION
        try {
#endif

            // matmul descriptor
            auto matmul_descriptor = cudnn_frontend::MatMulDescBuilder()
                                         .setComputeType(attributes.compute_data_type)
                                         .setPaddingValue(attributes.padding_value)
                                         .build();

            auto&& matmul_operation_builder =
                cudnn_frontend::OperationBuilder(DescriptorType_t::OPERATION_MATMUL_DESCRIPTOR);

            CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(A, Matmul_attributes::input_names::A);
            matmul_operation_builder.setaMatDesc(*tensors.at(A->second->get_uid()));

            CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(B, Matmul_attributes::input_names::B);
            matmul_operation_builder.setbMatDesc(*tensors.at(B->second->get_uid()));

            CUDNN_FE_VALIDATE_AND_ASSIGN_OUTPUT_TENSOR(C, Matmul_attributes::output_names::C);
            matmul_operation_builder.setcMatDesc(*tensors.at(C->second->get_uid()));
            matmul_operation_builder.setmatmulDesc(matmul_descriptor);

            auto M_override = attributes.inputs.find(Matmul_attributes::input_names::M_override);
            if ((M_override != attributes.inputs.end()) && (M_override->second != nullptr)) {
                matmul_operation_builder.setmOverrideDesc(*tensors.at(M_override->second->get_uid()));
            }

            auto N_override = attributes.inputs.find(Matmul_attributes::input_names::N_override);
            if ((N_override != attributes.inputs.end()) && (N_override->second != nullptr)) {
                matmul_operation_builder.setnOverrideDesc(*tensors.at(N_override->second->get_uid()));
            }

            auto K_override = attributes.inputs.find(Matmul_attributes::input_names::K_override);
            if ((K_override != attributes.inputs.end()) && (K_override->second != nullptr)) {
                matmul_operation_builder.setkOverrideDesc(*tensors.at(K_override->second->get_uid()));
            }

            auto operation = matmul_operation_builder.build();

            operations.push_back(std::make_shared<Operation_v8>(std::move(operation)));

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

}  // namespace cudnn_frontend::graph