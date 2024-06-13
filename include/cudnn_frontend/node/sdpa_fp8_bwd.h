#pragma once

#include "../../cudnn_frontend_Heuristics.h"
#include "../../cudnn_frontend_Logging.h"

#include "../graph_helpers.h"
#include "../node_interface.h"

#include "matmul_fp8.h"
#include "pointwise.h"
#include "reduction.h"
#include "softmax.h"

namespace cudnn_frontend::graph {

class SDPAFP8BackwardNode : public NodeCRTP<SDPAFP8BackwardNode> {
    using input_names  = SDPA_fp8_backward_attributes::input_names;
    using output_names = SDPA_fp8_backward_attributes::output_names;

   public:
    SDPA_fp8_backward_attributes attributes;

    SDPAFP8BackwardNode(SDPA_fp8_backward_attributes&& attributes_, detail::Context const& context)
        : NodeCRTP(context), attributes(std::move(attributes_)) {}

    Type
    getType() override final {
        return Type::COMPOSITE;
    }

    error_t
    pre_validate_node() const override final {
        getLogger() << "[cudnn_frontend] INFO: " << "Validating SDPAFP8BackwardNode " << attributes.name << "..."
                    << std::endl;

        RETURN_CUDNN_FRONTEND_ERROR_IF(detail::get_backend_version() < 90100,
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "sdpa fp8 backward operation is only supported starting cudnn 9.1.0. Please "
                                       "consider upgrading your current version.");

        cudaDeviceProp prop;
        int device;
        CHECK_CUDA_ERROR(detail::cuda_get_device(&device));
        CHECK_CUDA_ERROR(detail::cuda_get_device_properties(&prop, device));
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            prop.major < 9,
            error_code_t::GRAPH_NOT_SUPPORTED,
            "sdpa fp8 forward operation is only supported on Hopper architecture and newer. Please "
            "consider using a newer architecture.");

        // check that Q, K, V, O, stats, dO, dQ, dK, dV tensors has been assigned
        // check that dim and strides has been assigned and last stride is 1
#define CUDNN_FE_SDPA_VALIDATE_DIM_STRIDE(port, port_map)                                                       \
    {                                                                                                           \
        std::shared_ptr<Tensor_attributes> tensor_ptr = port_map.at(port);                                      \
        RETURN_CUDNN_FRONTEND_ERROR_IF(tensor_ptr->get_dim().size() != 4,                                       \
                                       error_code_t::ATTRIBUTE_NOT_SET,                                         \
                                       "The dim for " + std::string(#port) + " is invalid");                    \
        RETURN_CUDNN_FRONTEND_ERROR_IF(tensor_ptr->get_stride().size() != 4,                                    \
                                       error_code_t::ATTRIBUTE_NOT_SET,                                         \
                                       "The stride for " + std::string(#port) + " is invalid");                 \
        RETURN_CUDNN_FRONTEND_ERROR_IF(                                                                         \
            tensor_ptr->get_stride()[3] != 1,                                                                   \
            error_code_t::GRAPH_NOT_SUPPORTED,                                                                  \
            "The stride for the last dimension corresponding to the embedding size per head should be 1 for " + \
                std::string(#port));                                                                            \
    }

        CUDNN_FE_VALIDATE_INPUT_TENSOR(input_names::Q);
        CUDNN_FE_SDPA_VALIDATE_DIM_STRIDE(input_names::Q, attributes.inputs);
        CUDNN_FE_VALIDATE_INPUT_TENSOR(input_names::K);
        CUDNN_FE_SDPA_VALIDATE_DIM_STRIDE(input_names::K, attributes.inputs);
        CUDNN_FE_VALIDATE_INPUT_TENSOR(input_names::V);
        CUDNN_FE_SDPA_VALIDATE_DIM_STRIDE(input_names::V, attributes.inputs);
        CUDNN_FE_VALIDATE_INPUT_TENSOR(input_names::O);
        CUDNN_FE_SDPA_VALIDATE_DIM_STRIDE(input_names::O, attributes.inputs);
        CUDNN_FE_VALIDATE_INPUT_TENSOR(input_names::Stats);
        CUDNN_FE_SDPA_VALIDATE_DIM_STRIDE(input_names::Stats, attributes.inputs);
        CUDNN_FE_VALIDATE_INPUT_TENSOR(input_names::dO);
        CUDNN_FE_SDPA_VALIDATE_DIM_STRIDE(input_names::dO, attributes.inputs);
        CUDNN_FE_VALIDATE_OUTPUT_TENSOR(output_names::dQ);
        CUDNN_FE_SDPA_VALIDATE_DIM_STRIDE(output_names::dQ, attributes.outputs);
        CUDNN_FE_VALIDATE_OUTPUT_TENSOR(output_names::dK);
        CUDNN_FE_SDPA_VALIDATE_DIM_STRIDE(output_names::dK, attributes.outputs);
        CUDNN_FE_VALIDATE_OUTPUT_TENSOR(output_names::dV);
        CUDNN_FE_SDPA_VALIDATE_DIM_STRIDE(output_names::dV, attributes.outputs);

#undef CUDNN_FE_SDPA_VALIDATE_DIM_STRIDE

        // validate options for attn_scale
        auto const& attn_scale    = attributes.inputs.find(input_names::Attn_scale);
        bool const has_attn_scale = (attn_scale != attributes.inputs.end()) && (attn_scale->second != nullptr);
        RETURN_CUDNN_FRONTEND_ERROR_IF(has_attn_scale && attributes.attn_scale_value.has_value(),
                                       error_code_t::ATTRIBUTE_NOT_SET,
                                       "attn_scale with tensor and value cannot be set at the same time.");

        CHECK_CUDNN_FRONTEND_ERROR(attributes.validate_inputs());
        return {error_code_t::OK, ""};
    }

    error_t
    expand_and_infer_properties_node() override final {
        getLogger() << "[cudnn_frontend] INFO: Inferrencing properties for Scaled_dot_product_flash_attention node  "
                    << attributes.name << "..." << std::endl;

        attributes.fill_from_context(context);

        // Gather dim to fill properties of virtual tensors
        auto const& q_dim = attributes.inputs[input_names::Q]->get_dim();
        auto b            = q_dim[0];
        auto h_q          = q_dim[1];
        auto s_q          = q_dim[2];
        // auto d_qk         = q_dim[3];
        auto const& k_dim = attributes.inputs[input_names::K]->get_dim();
        // auto h_k          = k_dim[1];
        auto s_kv = k_dim[2];
        // auto const& v_dim = attributes.inputs[input_names::V]->get_dim();
        // auto h_v          = v_dim[1];
        // auto d_v          = v_dim[3];

        // cuDNN frontend API attention requires Q, K, V where
        // Q = {b, h_q, s_q, d_qk}
        // K = {b, h_k, s_kv, d_qk}
        // V = {b, h_v, s_kv, d_v}
        // but cuDNN backend API attention requires Q, KT, VT
        // Q = {b, h_q, s_q, d_qk}
        // KT = {b, h_k, d_qk, s_kv}
        // VT = {b, h_v, d_v, s_kv}
        // So the code below maps the K->KT and V->VT
        std::vector<int64_t> temp_vec;

        temp_vec = attributes.inputs[input_names::K]->get_dim();
        std::swap(temp_vec[2], temp_vec[3]);
        attributes.inputs[input_names::K]->set_dim(temp_vec);

        temp_vec = attributes.inputs[input_names::K]->get_stride();
        std::swap(temp_vec[2], temp_vec[3]);
        attributes.inputs[input_names::K]->set_stride(temp_vec);

        temp_vec = attributes.inputs[input_names::V]->get_dim();
        std::swap(temp_vec[2], temp_vec[3]);
        attributes.inputs[input_names::V]->set_dim(temp_vec);

        temp_vec = attributes.inputs[input_names::V]->get_stride();
        std::swap(temp_vec[2], temp_vec[3]);
        attributes.inputs[input_names::V]->set_stride(temp_vec);

        auto mul_attributes = Pointwise_attributes().set_mode(PointwiseMode_t::MUL);

        //// dO * O
        mul_attributes.set_name("mul_dO_O");
        auto last_output =
            pointwise(attributes.inputs[input_names::dO], attributes.inputs[input_names::O], mul_attributes);

        // reduce(dO)
        last_output =
            reduction(last_output, Reduction_attributes().set_name("reduce_dO").set_mode(ReductionMode_t::ADD));
        last_output->set_dim({b, h_q, s_q, 1}).set_stride({h_q * s_q, s_q, 1, 1});

        // Descale dO
        mul_attributes.set_name("descale_dO");
        last_output = pointwise(last_output, attributes.inputs.at(input_names::Descale_dO), mul_attributes);

        // Descale O
        mul_attributes.set_name("descale_O");
        auto softmax_sum = pointwise(last_output, attributes.inputs.at(input_names::Descale_O), mul_attributes);

        //// Q * K
        auto bmm_Q_K_attributes = Matmul_attributes().set_name("bmm_Q_K");
        auto last_dV = matmul(attributes.inputs[input_names::Q], attributes.inputs[input_names::K], bmm_Q_K_attributes);

        //// Optional Attn scale
        // In case user provided a scalar value, do a fused scalar.
        if (attributes.attn_scale_value.has_value()) {
            attributes.inputs[input_names::Attn_scale] =
                std::make_shared<Tensor_attributes>(attributes.attn_scale_value.value());
        }

        // If attn scale present, add a pointwise mul node
        if (attributes.inputs[input_names::Attn_scale]) {
            mul_attributes.set_name("attn_scale");
            last_dV = pointwise(last_dV, attributes.inputs[input_names::Attn_scale], mul_attributes);
        }

        //// Descales
        // Descale Q
        mul_attributes.set_name("descale_q");
        last_dV = pointwise(last_dV, attributes.inputs.at(input_names::Descale_Q), mul_attributes);

        // Descale K
        mul_attributes.set_name("descale_k");
        last_dV = pointwise(last_dV, attributes.inputs.at(input_names::Descale_K), mul_attributes);

        //// Optional causal masking
        if (attributes.causal_mask) {
            auto row_index_attributes =
                Pointwise_attributes().set_name("gen_row_index").set_mode(PointwiseMode_t::GEN_INDEX).set_axis(2);
            auto const& row_index_output = pointwise(last_dV, row_index_attributes);

            auto col_index_attributes =
                Pointwise_attributes().set_name("gen_col_index").set_mode(PointwiseMode_t::GEN_INDEX).set_axis(3);
            auto const& col_index_output = pointwise(last_dV, col_index_attributes);

            auto greater_than_attributes = Pointwise_attributes()
                                               .set_name("row_greater_than_col")
                                               .set_mode(PointwiseMode_t::CMP_GE)
                                               .set_compute_data_type(DataType_t::BOOLEAN);
            auto const& row_greater_than_col_output =
                pointwise(row_index_output, col_index_output, greater_than_attributes);
            row_greater_than_col_output->set_data_type(DataType_t::BOOLEAN);

            // Lower attributes to binary select attributes
            auto negative_inf_causal = std::make_shared<Tensor_attributes>(std::numeric_limits<float>::lowest());

            auto binary_select_attributes =
                Pointwise_attributes().set_name("binary_select").set_mode(PointwiseMode_t::BINARY_SELECT);
            last_dV = pointwise(last_dV, negative_inf_causal, row_greater_than_col_output, binary_select_attributes);
        }

        //// Apply Softmax
        // last_dV = last_dV - stats
        last_dV = pointwise(last_dV,
                            attributes.inputs[input_names::Stats],
                            Pointwise_attributes().set_name("sub_dV_Stats").set_mode(PointwiseMode_t::SUB));

        // last_dV = exp(last_dV)
        last_dV    = pointwise(last_dV, Pointwise_attributes().set_name("exp_dV").set_mode(PointwiseMode_t::EXP));
        auto exp_S = last_dV;

        // Scale S
        mul_attributes.set_name("scale_S");
        last_dV = pointwise(last_dV, attributes.inputs.at(input_names::Scale_S), mul_attributes);
        last_dV->set_data_type(attributes.inputs.at(input_names::Q)->get_data_type());

        // Reshape S
        last_dV = reshape(last_dV, Reshape_attributes().set_name("S_transpose"));
        last_dV->set_name("S_T").set_dim({b, h_q, s_kv, s_q}).set_stride({h_q * s_q * s_kv, s_q * s_kv, 1, s_kv});
        last_dV->set_data_type(attributes.inputs[input_names::Q]->get_data_type());

        //// S_T * dO
        // Special non-functional-style call. Needed because output already created and provided to user.
        matmul_fp8(last_dV,
                   attributes.inputs[input_names::dO],
                   attributes.inputs[input_names::Descale_S],
                   attributes.inputs[input_names::Descale_dO],
                   attributes.inputs[input_names::Scale_dV],
                   Matmul_fp8_attributes().set_name("bmm_S_T_dO"),
                   attributes.outputs[output_names::dV],
                   attributes.outputs[output_names::Amax_dV]);

        //// dO * V_T
        auto bmm_dO_V_T_attributes = Matmul_attributes().set_name("bmm_dO_V_T");
        last_output =
            matmul(attributes.inputs[input_names::dO], attributes.inputs[input_names::V], bmm_dO_V_T_attributes);

        //// Descales
        // Descale dO
        mul_attributes.set_name("descale_dO");
        last_output = pointwise(last_output, attributes.inputs.at(input_names::Descale_dO), mul_attributes);

        // Descale V
        mul_attributes.set_name("descale_V");
        last_output = pointwise(last_output, attributes.inputs.at(input_names::Descale_V), mul_attributes);

        // dP = last_output - softmax_sum
        auto dP = pointwise(last_output,
                            softmax_sum,
                            Pointwise_attributes().set_name("sub_dP_softmax_sum").set_mode(PointwiseMode_t::SUB));

        // dP = dP * exp_S
        mul_attributes.set_name("mul_dP_exp_S");
        dP = pointwise(dP, exp_S, mul_attributes);

        // (optional) dP = dP * attn_scale
        if (attributes.inputs[input_names::Attn_scale]) {
            mul_attributes.set_name("mul_dS_attn_scale");
            dP = pointwise(dP, attributes.inputs[input_names::Attn_scale], mul_attributes);
        }

        // Amax dP
        auto amax_attributes = Reduction_attributes().set_name("amax_dP").set_mode(ReductionMode_t::AMAX);
        // Special non-functional-style call. Needed because output already created and provided to user.
        reduction(dP, amax_attributes, attributes.outputs.at(output_names::Amax_dP));

        // Scale dP
        mul_attributes.set_name("scale_dP");
        dP = pointwise(dP, attributes.inputs.at(input_names::Scale_dP), mul_attributes);
        dP->set_data_type(attributes.inputs.at(input_names::dO)->get_data_type());

        //// dP * K
        auto const& kt_dim    = attributes.inputs[input_names::K]->get_dim();
        auto const& kt_stride = attributes.inputs[input_names::K]->get_stride();

        auto K = reshape(attributes.inputs[input_names::K], Reshape_attributes().set_name("reshape_K"));
        K->set_dim({kt_dim[0], kt_dim[1], kt_dim[3], kt_dim[2]})
            .set_stride({kt_stride[0], kt_stride[1], kt_stride[3], kt_stride[2]});

        auto bmm_dP_K_attributes = Matmul_fp8_attributes().set_name("bmm_dP_K");
        // Special non-functional-style call. Needed because output already created and provided to user.
        matmul_fp8(dP,
                   K,
                   attributes.inputs[input_names::Descale_dP],
                   attributes.inputs[input_names::Descale_K],
                   attributes.inputs[input_names::Scale_dQ],
                   bmm_dP_K_attributes,
                   attributes.outputs[output_names::dQ],
                   attributes.outputs[output_names::Amax_dQ]);

        //// dP.T * Q
        auto dP_T_attributes = Reshape_attributes().set_name("dP_T");
        auto dP_T            = reshape(dP, dP_T_attributes);
        dP_T->set_data_type(attributes.inputs.at(input_names::dO)->get_data_type());
        dP_T->set_name("dP_T").set_dim({b, h_q, s_kv, s_q}).set_stride({h_q * s_q * s_kv, s_q * s_kv, 1, s_kv});

        auto bmm_dP_T_Q_attributes = Matmul_fp8_attributes().set_name("bmm_dP_T_Q");
        // Special non-functional-style call. Needed because output already created and provided to user.
        matmul_fp8(dP_T,
                   attributes.inputs[input_names::Q],
                   attributes.inputs[input_names::Descale_dP],
                   attributes.inputs[input_names::Descale_Q],
                   attributes.inputs[input_names::Scale_dK],
                   bmm_dP_T_Q_attributes,
                   attributes.outputs[output_names::dK],
                   attributes.outputs[output_names::Amax_dK]);

        return {error_code_t::OK, ""};
    }

    error_t
    post_validate_node() const override final {
        // Validate outputs
        // All properties of output tensors should have been set now.
        CHECK_CUDNN_FRONTEND_ERROR(attributes.validate_outputs());

        return {error_code_t::OK, ""};
    }

#ifndef CUDNN_FRONTEND_SKIP_JSON_LIB
    virtual void
    serialize(json& j) const override final {
        j = attributes;
        j.update(R"({"tag": "SDPA_FP8_BWD"})"_json);
    }
#endif
};

}  // namespace cudnn_frontend::graph