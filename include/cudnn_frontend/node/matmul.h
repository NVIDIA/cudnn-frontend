#pragma once

#include "../../cudnn_frontend_MatMulDesc.h"
#include "../../cudnn_frontend_Heuristics.h"
#include "../../cudnn_frontend_Logging.h"

#include "../graph_helpers.h"
#include "../node_interface.h"

namespace cudnn_frontend::graph {

class MatmulNode : public NodeCRTP<MatmulNode> {
   public:
    Matmul_attributes attributes;

    MatmulNode(Matmul_attributes&& attributes_, detail::Context const& context)
        : NodeCRTP(context), attributes(std::move(attributes_)) {}

    Type
    getType() override final {
        return Type::MATMUL;
    }

    error_t
    infer_properties_node() override final {
        CUDNN_FE_LOG_LABEL_ENDL("INFO: Inferrencing properties for matmul node " << attributes.name << "...");

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
            // CHECK_CUDNN_FRONTEND_ERROR(detail::generate_matmul_output_dim(a_tensor_dim, b_tensor_dim, c_tensor_dim));

            c_tensor_dim.resize(a_tensor_dim.size());
            int64_t gemm_start_dim           = a_tensor_dim.size() - 2;
            c_tensor_dim[gemm_start_dim]     = a_tensor_dim[gemm_start_dim];      // M
            c_tensor_dim[gemm_start_dim + 1] = b_tensor_dim[gemm_start_dim + 1];  // N

            // Broadcast the batches
            for (int64_t i = 0; i < gemm_start_dim; ++i) {
                c_tensor_dim[i] = std::max(a_tensor_dim[i], b_tensor_dim[i]);
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
    create_cudnn_operations(
        std::unordered_set<uid_t>& uids_involved_in_operations,
        std::vector<std::shared_ptr<cudnn_frontend::Operation>>& operations,
        managed_backend_descriptor_t& raw_operations,
        std::unordered_map<int64_t, std::shared_ptr<cudnn_frontend::Tensor>>& tensors) const override final {
        CUDNN_FRONTEND_UNUSED(raw_operations);
        CUDNN_FE_LOG_LABEL_ENDL("INFO: " << "Building MatmulNode operations " << attributes.name << "...");

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

#ifdef NV_CUDNN_DISABLE_EXCEPTION
        // disable exception macro is defined. Calling build will not throw.
        // Check status of desc and return error.
        auto operation = matmul_operation_builder.build();
        RETURN_CUDNN_FRONTEND_ERROR_IF(operation.get_status() != CUDNN_STATUS_SUCCESS,
                                       error_code_t::CUDNN_BACKEND_API_FAILED,
                                       operation.get_error());
        operations.push_back(std::make_shared<Operation_v8>(std::move(operation)));
#else
        // build() can throw
        // wrap in try catch
        try {
            auto operation = matmul_operation_builder.build();
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
        j.update(R"( {"tag": "MATMUL"})"_json);
    }
#endif
};

inline void
INode::matmul(std::shared_ptr<Tensor_attributes> a,
              std::shared_ptr<Tensor_attributes> b,
              Matmul_attributes attributes,
              std::shared_ptr<Tensor_attributes> c) {
    attributes.inputs[Matmul_attributes::input_names::A]   = a;
    attributes.inputs[Matmul_attributes::input_names::B]   = b;
    attributes.outputs[Matmul_attributes::output_names::C] = c;
    sub_nodes.emplace_back(std::make_unique<MatmulNode>(std::move(attributes), context));
}

inline std::shared_ptr<Tensor_attributes>
INode::matmul(std::shared_ptr<Tensor_attributes> a,
              std::shared_ptr<Tensor_attributes> b,
              Matmul_attributes attributes) {
    if (attributes.name.empty()) {
        attributes.name += std::to_string(sub_nodes.size());
    }
    attributes.inputs[Matmul_attributes::input_names::A] = a;
    attributes.inputs[Matmul_attributes::input_names::B] = b;

    if (a->get_name().empty()) {
        a->set_name(attributes.name + "::A");
    };
    if (b->get_name().empty()) {
        b->set_name(attributes.name + "::B");
    };

    auto m_override = attributes.inputs.find(Matmul_attributes::input_names::M_override);
    auto n_override = attributes.inputs.find(Matmul_attributes::input_names::N_override);
    auto k_override = attributes.inputs.find(Matmul_attributes::input_names::K_override);

    if (m_override != attributes.inputs.end()) {
        auto tensor = m_override->second;
        if (tensor && tensor->get_name().empty()) {
            tensor->set_name(attributes.name + "::M_override");
        }
    }
    if (n_override != attributes.inputs.end()) {
        auto tensor = n_override->second;
        if (tensor && tensor->get_name().empty()) {
            tensor->set_name(attributes.name + "::N_override");
        }
    }
    if (k_override != attributes.inputs.end()) {
        auto tensor = k_override->second;
        if (tensor && tensor->get_name().empty()) {
            tensor->set_name(attributes.name + "::K_override");
        }
    }

    auto C = attributes.outputs[Matmul_attributes::output_names::C] = output_tensor(attributes.name + "::C");

    sub_nodes.emplace_back(std::make_unique<MatmulNode>(std::move(attributes), context));
    return C;
}

}  // namespace cudnn_frontend::graph