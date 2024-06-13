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

class SDPAFP8Node : public NodeCRTP<SDPAFP8Node> {
    using input_names  = SDPA_fp8_attributes::input_names;
    using output_names = SDPA_fp8_attributes::output_names;

   public:
    SDPA_fp8_attributes attributes;

    SDPAFP8Node(SDPA_fp8_attributes&& attributes_, detail::Context const& context)
        : NodeCRTP(context), attributes(std::move(attributes_)) {}

    Type
    getType() override final {
        return Type::COMPOSITE;
    }

    error_t
    pre_validate_node() const override final {
        getLogger() << "[cudnn_frontend] INFO: " << "Validating SDPAFP8Node " << attributes.name << "..." << std::endl;

        RETURN_CUDNN_FRONTEND_ERROR_IF(detail::get_backend_version() < 90100,
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "sdpa fp8 forward operation is only supported starting cudnn 9.1.0. Please "
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

        // check that Q, K, V, O tensors has been assigned
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
        CUDNN_FE_VALIDATE_OUTPUT_TENSOR(output_names::O);
        CUDNN_FE_SDPA_VALIDATE_DIM_STRIDE(output_names::O, attributes.outputs);

        // validate options for is_inference and stats tensor
        RETURN_CUDNN_FRONTEND_ERROR_IF(attributes.is_inference.has_value() == false,
                                       error_code_t::ATTRIBUTE_NOT_SET,
                                       "is_inference attribute not set");

        if (attributes.is_inference.value() == false) {
            CUDNN_FE_VALIDATE_OUTPUT_TENSOR(output_names::Stats);
        }

#undef CUDNN_FE_SDPA_VALIDATE_DIM_STRIDE

        CHECK_CUDNN_FRONTEND_ERROR(attributes.validate_inputs());

        int64_t d_qk = attributes.inputs.at(input_names::Q)->get_dim()[3];
        int64_t d_v  = attributes.inputs.at(input_names::V)->get_dim()[3];

        if (detail::get_backend_version() >= 90101) {
            RETURN_CUDNN_FRONTEND_ERROR_IF(
                (d_qk > 256) || (d_qk % 8 != 0) || (d_v > 256) || (d_v % 8 != 0),
                error_code_t::GRAPH_NOT_SUPPORTED,
                "Num hidden_dim shoud be less than 256 and hidden_dim should be multiple of 8");
        } else {
            RETURN_CUDNN_FRONTEND_ERROR_IF(
                (d_qk > 128) || (d_qk % 8 != 0) || (d_v > 128) || (d_v % 8 != 0),
                error_code_t::GRAPH_NOT_SUPPORTED,
                "Num hidden_dim shoud be less than 128 and hidden_dim should be multiple of 8");
        }
        return {error_code_t::OK, ""};
    }

    error_t
    expand_and_infer_properties_node() override final {
        getLogger() << "[cudnn_frontend] INFO: Inferrencing properties for Scaled_dot_product_flash_attention node  "
                    << attributes.name << "..." << std::endl;

        // DO NOT REMOVE
        // input data type is needed for:
        // - aType of bmm2
        attributes.fill_from_context(context);

        // Gather dim to fill properties of virtual tensors
        // auto const& q_dim = attributes.inputs[input_names::Q]->get_dim();
        // auto b            = q_dim[0];
        // auto h            = q_dim[1];
        // auto s_q          = q_dim[2];
        // auto const& k_dim = attributes.inputs[input_names::K]->get_dim();
        // auto s_kv         = k_dim[2];

        // cuDNN frontend API attention requires Q, K, V where
        // Q = {b, h_q, s_q, d_qk}
        // K = {b, h_k, s_kv, d_qk}
        // V = {b, h_v, s_kv, d_v}
        // but cuDNN backend API attention requires Q, KT, V
        // Q = {b, h_q, s_q, d_qk}
        // KT = {b, h_k, d_qk, s_kv}
        // V = {b, h_v, s_kv, d_v}
        // So the code below maps the K->KT
        std::vector<int64_t> temp_vec;

        temp_vec = attributes.inputs[input_names::K]->get_dim();
        std::swap(temp_vec[2], temp_vec[3]);
        attributes.inputs[input_names::K]->set_dim(temp_vec);

        temp_vec = attributes.inputs[input_names::K]->get_stride();
        std::swap(temp_vec[2], temp_vec[3]);
        attributes.inputs[input_names::K]->set_stride(temp_vec);

        // This tensor tracks the main chain of data flow
        std::shared_ptr<Tensor_attributes> last_output;
        auto mul_attributes = Pointwise_attributes().set_mode(PointwiseMode_t::MUL);

        //// Q * K
        auto bmm1_attributes = Matmul_attributes().set_name("bmm1").set_padding(0.0);
        last_output = matmul(attributes.inputs[input_names::Q], attributes.inputs[input_names::K], bmm1_attributes);

        //// Optional Attn scale
        // In case user provided a scalar value, do a fused scalar.
        if (attributes.attn_scale_value.has_value()) {
            attributes.inputs[input_names::Attn_scale] =
                std::make_shared<Tensor_attributes>(attributes.attn_scale_value.value());
        }

        // If attn scale present, add a pointwise mul node
        if (attributes.inputs[input_names::Attn_scale]) {
            mul_attributes.set_name("attn_scale");
            auto const& attn_scale_output =
                pointwise(last_output, attributes.inputs[input_names::Attn_scale], mul_attributes);
            last_output = attn_scale_output;
        }

        //// Descales
        // Descale Q
        mul_attributes.set_name("descale_q");
        last_output = pointwise(last_output, attributes.inputs.at(input_names::Descale_Q), mul_attributes);

        // Descale K
        mul_attributes.set_name("descale_k");
        last_output = pointwise(last_output, attributes.inputs.at(input_names::Descale_K), mul_attributes);

        //// Optional causal masking
        if (attributes.causal_mask) {
            auto row_index_attributes =
                Pointwise_attributes().set_name("gen_row_index").set_mode(PointwiseMode_t::GEN_INDEX).set_axis(2);
            auto const& row_index_output = pointwise(last_output, row_index_attributes);

            auto col_index_attributes =
                Pointwise_attributes().set_name("gen_col_index").set_mode(PointwiseMode_t::GEN_INDEX).set_axis(3);
            auto const& col_index_output = pointwise(last_output, col_index_attributes);

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
            auto const& causal_mask_output =
                pointwise(last_output, negative_inf_causal, row_greater_than_col_output, binary_select_attributes);
            last_output = causal_mask_output;
        }

        //// Softmax
        // softmax output, S, is always virtual.
        auto softmax_output = std::make_shared<Tensor_attributes>();
        softmax_output->set_is_virtual(true);

        // Create virtual stats if inference step otherwise output.Stats should be provided by user.
        auto softmax_stats = attributes.outputs[output_names::Stats];
        if (attributes.is_inference.value() == true) {
            softmax_stats = std::make_shared<Tensor_attributes>();
            softmax_stats->set_is_virtual(true);
        }

        auto softmax_attributes =
            Softmax_attributes().set_name("softmax").has_stats(true).has_M_Zinv(false);  // As this is flash attention
        // Special non-functional-style call. Needed because output already created and provided to user.
        softmax(last_output, softmax_attributes, softmax_output, softmax_stats);
        last_output = softmax_output;

        // Amax S
        auto amax_attributes = Reduction_attributes().set_name("amax_s").set_mode(ReductionMode_t::AMAX);
        // Special non-functional-style call. Needed because output already created and provided to user.
        reduction(last_output, amax_attributes, attributes.outputs.at(output_names::Amax_S));

        // Scale S
        mul_attributes.set_name("scale_s");
        last_output = pointwise(last_output, attributes.inputs.at(input_names::Scale_S), mul_attributes);
        last_output->set_data_type(attributes.inputs.at(input_names::Q)->get_data_type());

        //// S * V
        auto bmm2_attributes = Matmul_fp8_attributes().set_name("bmm2");
        // Special non-functional-style call. Needed because output already created and provided to user.
        matmul_fp8(last_output,
                   attributes.inputs.at(input_names::V),
                   attributes.inputs.at(input_names::Descale_S),
                   attributes.inputs.at(input_names::Descale_V),
                   attributes.inputs.at(input_names::Scale_O),
                   bmm2_attributes,
                   attributes.outputs.at(output_names::O),
                   attributes.outputs.at(output_names::Amax_O));

        return {error_code_t::OK, ""};
    }

    error_t
    post_validate_node() const override final {
#define CUDNN_FE_VALIDATE_STRIDE(port, port_map)                                                                \
    {                                                                                                           \
        auto const& t = port_map.find(port);                                                                    \
        RETURN_CUDNN_FRONTEND_ERROR_IF(                                                                         \
            t->second->get_stride().back() != 1,                                                                \
            error_code_t::GRAPH_NOT_SUPPORTED,                                                                  \
            "The stride for the last dimension corresponding to the embedding size per head should be 1 for " + \
                std::string(#port));                                                                            \
    }

        CUDNN_FE_VALIDATE_STRIDE(output_names::O, attributes.outputs);

#undef CUDNN_FE_VALIDATE_STRIDE

        // Validate outputs
        // All properties of output tensors should have been set now.
        CHECK_CUDNN_FRONTEND_ERROR(attributes.validate_outputs());

        return {error_code_t::OK, ""};
    }

#ifndef CUDNN_FRONTEND_SKIP_JSON_LIB
    virtual void
    serialize(json& j) const override final {
        j = attributes;
        j.update(R"({"tag": "SDPA_FP8_FWD"})"_json);
    }
#endif
};

}  // namespace cudnn_frontend::graph