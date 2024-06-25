#pragma once

#include "../../cudnn_frontend_Heuristics.h"
#include "../../cudnn_frontend_Logging.h"

#include "../graph_helpers.h"
#include "../node_interface.h"

#include "matmul.h"
#include "pointwise.h"
#include "rng.h"
#include "softmax.h"

namespace cudnn_frontend::graph {

class SDPANode : public NodeCRTP<SDPANode> {
    using input_names  = SDPA_attributes::input_names;
    using output_names = SDPA_attributes::output_names;

    std::shared_ptr<Tensor_attributes> rng_output;
    std::shared_ptr<Tensor_attributes> alibi_slopes;
    int64_t alibi_slopes_size = 0;

   public:
    SDPA_attributes attributes;

    SDPANode(SDPA_attributes&& attributes_, detail::Context const& context)
        : NodeCRTP(context), attributes(std::move(attributes_)) {}

    Type
    getType() override final {
        return Type::COMPOSITE;
    }

    error_t
    pre_validate_node() const override final {
        getLogger() << "[cudnn_frontend] INFO: " << "Validating SDPANode " << attributes.name << "..." << std::endl;

        // check that Q, K, V, O tensors has been assigned
        // check that dim and strides has been assigned and last stride is 1
#define CUDNN_FE_SDPA_VALIDATE_DIM_STRIDE(port, port_map)                                                        \
    {                                                                                                            \
        std::shared_ptr<Tensor_attributes> tensor_ptr = port_map.at(port);                                       \
        RETURN_CUDNN_FRONTEND_ERROR_IF(tensor_ptr->get_dim().size() != 4,                                        \
                                       error_code_t::ATTRIBUTE_NOT_SET,                                          \
                                       "The dim for " + std::string(#port) + " is invalid");                     \
        RETURN_CUDNN_FRONTEND_ERROR_IF(tensor_ptr->get_stride().size() != 4,                                     \
                                       error_code_t::ATTRIBUTE_NOT_SET,                                          \
                                       "The stride for " + std::string(#port) + " is invalid");                  \
        RETURN_CUDNN_FRONTEND_ERROR_IF(                                                                          \
            tensor_ptr->get_stride()[3] != 1,                                                                    \
            error_code_t::GRAPH_NOT_SUPPORTED,                                                                   \
            "The stride for the last dimension corresponding to the embedding size per head should be 1 for " +  \
                std::string(#port));                                                                             \
        RETURN_CUDNN_FRONTEND_ERROR_IF(                                                                          \
            tensor_ptr->get_stride()[2] == 0,                                                                    \
            error_code_t::GRAPH_NOT_SUPPORTED,                                                                   \
            "The stride for the dimension corresponding to the sequence lengths per head should not be 0 for " + \
                std::string(#port));                                                                             \
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

        // validate backend limitations for the operation
        // clang-format off
        int64_t s_q  = attributes.inputs.at(input_names::Q)->get_dim()[2];
        int64_t s_kv = attributes.inputs.at(input_names::K)->get_dim()[2];
        int64_t h_q  = attributes.inputs.at(input_names::Q)->get_dim()[1];
        int64_t h_k  = attributes.inputs.at(input_names::K)->get_dim()[1];
        int64_t h_v  = attributes.inputs.at(input_names::V)->get_dim()[1];
        int64_t d_qk = attributes.inputs.at(input_names::Q)->get_dim()[3];
        int64_t d_v  = attributes.inputs.at(input_names::V)->get_dim()[3];

        bool const is_ragged = attributes.inputs.at(input_names::Q)->get_ragged_offset() ||
                               attributes.inputs.at(input_names::K)->get_ragged_offset() ||
                               attributes.inputs.at(input_names::V)->get_ragged_offset() ||
                               attributes.outputs.at(output_names::O)->get_ragged_offset();

        auto const& bias_mask = attributes.inputs.find(input_names::Bias);
        bool const is_bias   = (bias_mask != attributes.inputs.end() && bias_mask->second != nullptr);

        auto const& dropout_mask     = attributes.inputs.find(input_names::Dropout_mask);
        bool const is_dropout_custom = (dropout_mask != attributes.inputs.end()) && (dropout_mask->second != nullptr);
        bool const is_dropout        = attributes.dropout_probability.has_value() || is_dropout_custom;

        // validation TODO:
        //    - validate stats has valid dims

        // validate basic dimension requirements
        RETURN_CUDNN_FRONTEND_ERROR_IF((d_qk > 256) || (d_qk % 8 != 0) || (d_v > 256) || (d_v % 8 != 0),
                                        error_code_t::GRAPH_NOT_SUPPORTED,
                                        "hidden_dim shoud be less than 256 and hidden_dim should be multiple of 8");

        RETURN_CUDNN_FRONTEND_ERROR_IF((h_q % h_k != 0) || (h_q % h_v != 0),
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "For group-query attention, number of heads for key and query must be a factor of number of heads for query");

        // validate options for attn_scale
        auto const& attn_scale    = attributes.inputs.find(input_names::Attn_scale);
        bool const has_attn_scale = (attn_scale != attributes.inputs.end()) && (attn_scale->second != nullptr);
        RETURN_CUDNN_FRONTEND_ERROR_IF(has_attn_scale && attributes.attn_scale_value.has_value(),
                                       error_code_t::ATTRIBUTE_NOT_SET,
                                       "attn_scale with tensor and value cannot be set at the same time.");

        // validate options for bias mask
        RETURN_CUDNN_FRONTEND_ERROR_IF(is_bias && (bias_mask->second->get_data_type() == DataType_t::BOOLEAN),
                                        error_code_t::GRAPH_NOT_SUPPORTED,
                                        "Bias mask data type cannot be boolean");

        // validate options for padding mask
        auto const& seq_len_q     = attributes.inputs.find(input_names::SEQ_LEN_Q);
        bool const has_seq_len_q  = (seq_len_q != attributes.inputs.end()) && (seq_len_q->second != nullptr);
        auto const& seq_len_kv    = attributes.inputs.find(input_names::SEQ_LEN_KV);
        bool const has_seq_len_kv = (seq_len_kv != attributes.inputs.end()) && (seq_len_kv->second != nullptr);
        RETURN_CUDNN_FRONTEND_ERROR_IF(attributes.padding_mask && (!has_seq_len_q || !has_seq_len_kv),
                                       error_code_t::ATTRIBUTE_NOT_SET,
                                       "Padding mask requires seq_len_q and seq_len_kv to be set.");
        RETURN_CUDNN_FRONTEND_ERROR_IF((!attributes.padding_mask) && (has_seq_len_q || has_seq_len_kv),
                                       error_code_t::ATTRIBUTE_NOT_SET,
                                       "seq_len_q and seq_len_kv needs to be set only if padding mask is enabled.");

        // validate options for bottom right causal mask
        RETURN_CUDNN_FRONTEND_ERROR_IF(attributes.causal_mask && attributes.causal_mask_bottom_right,
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "Bottom right causal mask and causal mask cannot be both enabled");

        RETURN_CUDNN_FRONTEND_ERROR_IF(attributes.causal_mask_bottom_right && s_q > s_kv,
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "Bottom right causal mask does not support s_q > s_kv. Please virtually slice the Q tensor and pass it as s_q == s_kv");

        RETURN_CUDNN_FRONTEND_ERROR_IF(attributes.causal_mask_bottom_right && (is_bias || attributes.alibi_mask || is_ragged || attributes.padding_mask || is_dropout),
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "Bottom right causal mask is only supported with is_bias=False, is_alibi=False, is_ragged=False, padding_mask=False, is_dropout=False");

        RETURN_CUDNN_FRONTEND_ERROR_IF(attributes.causal_mask_bottom_right && ((s_q % 64 != 0) || (s_kv % 64 != 0)),
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "Bottom right causal mask is only supported with s_q multiple of 64, and s_kv multiple of 64");

        // validate options for sliding window length
        RETURN_CUDNN_FRONTEND_ERROR_IF(attributes.sliding_window_length.has_value() && attributes.sliding_window_length.value() < 0,
                                       error_code_t::INVALID_VALUE,
                                       "Sliding window length should be greater than or equals to zero when set.");


        RETURN_CUDNN_FRONTEND_ERROR_IF(attributes.sliding_window_length.has_value() && (attributes.padding_mask || !attributes.causal_mask || is_dropout || is_bias || is_ragged),
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "Sliding window attention is only supported with padding_mask=False, causal_mask=True, is_dropout=False, is_bias=False, is_ragged=False");

        // validate options for dropout mask
        RETURN_CUDNN_FRONTEND_ERROR_IF(attributes.dropout_probability.has_value() && is_dropout_custom,
                                       error_code_t::ATTRIBUTE_NOT_SET,
                                       "Using both, custom dropout mask and internal-mask generation using dropout probability, is ill-formed.");

        RETURN_CUDNN_FRONTEND_ERROR_IF(attributes.dropout_probability.has_value() && attributes.dropout_probability.value() == 1.0,
                                       error_code_t::ATTRIBUTE_NOT_SET,
                                       "Dropout probability cannot be 1 as corresponding scale wont be well formed.");

        // version specific validation
        RETURN_CUDNN_FRONTEND_ERROR_IF(detail::get_backend_version() < 8906 && ((s_kv % 64 != 0) || (d_qk % 64 != 0)),
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "For cuDNN version below 8.9.6, s_kv not a multiple of 64 or d not a multiple of 64 is not supported");

        RETURN_CUDNN_FRONTEND_ERROR_IF(detail::get_backend_version() < 8907 && (s_kv % 64 != 0) && (!(attributes.padding_mask)),
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "For cuDNN version below 8.9.7, s_kv not a multiple of 64 is not supported");

        RETURN_CUDNN_FRONTEND_ERROR_IF(detail::get_backend_version() < 90000 && ((s_q % 64 != 0) || (s_kv % 64 != 0)) && (attributes.padding_mask || is_dropout),
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "For cuDNN version below 9.0.0, s_q/s_kv not a multiple of 64 with padding/dropout mask is not supported");

        RETURN_CUDNN_FRONTEND_ERROR_IF(detail::get_backend_version() < 90000 && ((d_qk > 128) || (d_qk % 8 != 0) || (d_v > 128) || (d_v % 8 != 0)),
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "For cuDNN version below 9.0.0, hidden_dim shoud be less than 128 and hidden_dim should be multiple of 8");

        RETURN_CUDNN_FRONTEND_ERROR_IF(detail::get_backend_version() < 90200 && attributes.sliding_window_length.has_value(),
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "For cuDNN version below 9.2.0, sliding window attention is not supported");


        // validate that datatype is set for the graph
        RETURN_CUDNN_FRONTEND_ERROR_IF(context.get_intermediate_data_type() == DataType_t::NOT_SET,
                                       error_code_t::ATTRIBUTE_NOT_SET,
                                       "Intermediate tensor data type needs to be set as internal tensors require it.");
        // clang-format on

        CHECK_CUDNN_FRONTEND_ERROR(attributes.validate_inputs());
        return {error_code_t::OK, ""};
    }

    error_t
    expand_and_infer_properties_node() override final {
        getLogger() << "[cudnn_frontend] INFO: Inferrencing properties for Scaled_dot_product_flash_attention node  "
                    << attributes.name << "..." << std::endl;

        // DO NOT REMOVE
        // input data type is needed for:
        // - aType of bmm2
        // - dropout scale in pre 8.9.3
        attributes.fill_from_context(context);

        // Gather dim to fill properties of virtual tensors
        auto const& q_dim = attributes.inputs[input_names::Q]->get_dim();
        auto b            = q_dim[0];
        auto h            = q_dim[1];
        auto s_q          = q_dim[2];
        auto const& k_dim = attributes.inputs[input_names::K]->get_dim();
        auto s_kv         = k_dim[2];

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

        std::shared_ptr<Tensor_attributes> last_output;

        auto bmm1_attributes = Matmul_attributes()
                                   .set_name("bmm1")
                                   .set_m_override(attributes.inputs[input_names::SEQ_LEN_Q])
                                   .set_n_override(attributes.inputs[input_names::SEQ_LEN_KV]);

        if (attributes.padding_mask) {
            bmm1_attributes.set_padding(0.0);
        }

        auto const& bmm1_output =
            matmul(attributes.inputs[input_names::Q], attributes.inputs[input_names::K], bmm1_attributes);
        // Setting dim and strides as pointwise op wont have knowledge of how to do it for mha.
        bmm1_output->set_dim({b, h, s_q, s_kv}).set_stride({h * s_q * s_kv, s_q * s_kv, s_kv, 1});
        last_output = bmm1_output;

        // Optional scale
        if (attributes.attn_scale_value.has_value()) {
            attributes.inputs[input_names::Attn_scale] =
                std::make_shared<Tensor_attributes>(attributes.attn_scale_value.value());
        }
        if (attributes.inputs[input_names::Attn_scale]) {
            Pointwise_attributes scale_attributes;
            scale_attributes.set_name("attn_scale").set_mode(PointwiseMode_t::MUL);
            auto const& attn_scale_output =
                pointwise(last_output, attributes.inputs[input_names::Attn_scale], scale_attributes);
            last_output = attn_scale_output;
        }

        // Optional bias
        if (attributes.inputs[input_names::Bias]) {
            auto add_attributes     = Pointwise_attributes().set_name("bias").set_mode(PointwiseMode_t::ADD);
            auto const& bias_output = pointwise(last_output, attributes.inputs[input_names::Bias], add_attributes);
            last_output             = bias_output;
        }

        if (attributes.alibi_mask) {
            auto row_index_attributes = Pointwise_attributes()
                                            .set_name("gen_row_index")
                                            .set_mode(PointwiseMode_t::GEN_INDEX)
                                            .set_axis(2)
                                            .set_compute_data_type(DataType_t::INT32);
            auto const& row_index_output = pointwise(last_output, row_index_attributes);
            row_index_output->set_data_type(DataType_t::INT32);

            auto col_index_attributes = Pointwise_attributes()
                                            .set_name("gen_col_index")
                                            .set_mode(PointwiseMode_t::GEN_INDEX)
                                            .set_axis(3)
                                            .set_compute_data_type(DataType_t::INT32);
            auto const& col_index_output = pointwise(last_output, col_index_attributes);
            col_index_output->set_data_type(DataType_t::INT32);

            auto sub_attributes = Pointwise_attributes()
                                      .set_name("sub")
                                      .set_mode(PointwiseMode_t::SUB)
                                      .set_compute_data_type(DataType_t::INT32);
            auto const& sub_output = pointwise(col_index_output, row_index_output, sub_attributes);
            sub_output->set_data_type(DataType_t::INT32);

            // Multiply by alibi slope
            alibi_slopes = std::make_shared<Tensor_attributes>();
            alibi_slopes->set_dim({1, h, 1, 1})
                .set_stride({h, 1, 1, 1})
                // Hard code data type float as FE itself will compute and place in variant pack later
                .set_data_type(DataType_t::FLOAT);
            alibi_slopes_size = h * sizeof(float);

            auto mul_attributes    = Pointwise_attributes().set_name("mul").set_mode(PointwiseMode_t::MUL);
            auto const& alibi_mask = pointwise(sub_output, alibi_slopes, mul_attributes);

            // Add alibi_mask
            auto add_attributes    = Pointwise_attributes().set_name("add").set_mode(PointwiseMode_t::ADD);
            auto const& add_output = pointwise(last_output, alibi_mask, add_attributes);
            last_output            = add_output;
        }

        // There are two cases of applying padding mask
        // 1. when actual seq_len is less than max_seq_len
        if (attributes.padding_mask) {
            auto row_index_attributes = Pointwise_attributes()
                                            .set_name("gen_row_index")
                                            .set_mode(PointwiseMode_t::GEN_INDEX)
                                            .set_axis(2)
                                            .set_compute_data_type(DataType_t::INT32);
            auto const& row_index_output = pointwise(last_output, row_index_attributes);
            row_index_output->set_data_type(DataType_t::INT32);

            auto col_index_attributes = Pointwise_attributes()
                                            .set_name("gen_col_index")
                                            .set_mode(PointwiseMode_t::GEN_INDEX)
                                            .set_axis(3)
                                            .set_compute_data_type(DataType_t::INT32);
            auto const& col_index_output = pointwise(last_output, col_index_attributes);
            col_index_output->set_data_type(DataType_t::INT32);

            auto row_less_seq_q_attributes = Pointwise_attributes()
                                                 .set_name("row_less_seq_q")
                                                 .set_mode(PointwiseMode_t::CMP_LT)
                                                 .set_compute_data_type(DataType_t::INT32);
            auto const& row_less_seq_q_output =
                pointwise(row_index_output, attributes.inputs[input_names::SEQ_LEN_Q], row_less_seq_q_attributes);
            row_less_seq_q_output->set_data_type(DataType_t::INT32);

            auto col_less_seq_kv_attributes = Pointwise_attributes()
                                                  .set_name("col_less_seq_kv")
                                                  .set_mode(PointwiseMode_t::CMP_LT)
                                                  .set_compute_data_type(DataType_t::INT32);
            auto const& col_less_seq_kv_output =
                pointwise(col_index_output, attributes.inputs[input_names::SEQ_LEN_KV], col_less_seq_kv_attributes);
            col_less_seq_kv_output->set_data_type(DataType_t::INT32);

            auto logical_and_attributes = Pointwise_attributes()
                                              .set_name("logical_and")
                                              .set_mode(PointwiseMode_t::LOGICAL_AND)
                                              .set_compute_data_type(DataType_t::BOOLEAN);
            auto const& logical_and_output =
                pointwise(row_less_seq_q_output, col_less_seq_kv_output, logical_and_attributes);
            logical_and_output->set_data_type(DataType_t::BOOLEAN);

            // Lower attributes to binary select attributes
            auto negative_inf_padding = std::make_shared<Tensor_attributes>(std::numeric_limits<float>::lowest());

            auto binary_select_attributes =
                Pointwise_attributes().set_name("binary_select").set_mode(PointwiseMode_t::BINARY_SELECT);
            auto const& padding_mask_output =
                pointwise(last_output, negative_inf_padding, logical_and_output, binary_select_attributes);
            last_output = padding_mask_output;
        }

        // 2. (bug in cudnn backend) no padding with max_seq_len%64!=0
        if ((s_kv % 64 != 0) && (!(attributes.padding_mask)) && (detail::get_backend_version() < 90000)) {
            auto col_index_attributes =
                Pointwise_attributes().set_name("gen_col_index").set_mode(PointwiseMode_t::GEN_INDEX).set_axis(3);
            auto col_index_output = pointwise(last_output, col_index_attributes);
            // scalar seq_kv only needs to be passed in case there in no padding mask and seq_kv is not multiple of 64.
            // Also future versions of cudnn will not need it, hence tensor is pre-fixed with WAR.
            auto WAR_scalar_max_seq_kv = std::make_shared<Tensor_attributes>(static_cast<int32_t>(s_kv));

            auto col_less_seq_kv_attributes =
                Pointwise_attributes().set_name("col_less_seq_kv").set_mode(PointwiseMode_t::CMP_LT);
            auto col_less_seq_kv_output =
                pointwise(col_index_output, WAR_scalar_max_seq_kv, col_less_seq_kv_attributes);

            // Lower attributes to binary select attributes
            auto negative_inf_padding = std::make_shared<Tensor_attributes>(std::numeric_limits<float>::lowest());
            auto binary_select_attributes =
                Pointwise_attributes().set_name("binary_select").set_mode(PointwiseMode_t::BINARY_SELECT);
            auto padding_mask_output =
                pointwise(last_output, negative_inf_padding, col_less_seq_kv_output, binary_select_attributes);
            last_output = padding_mask_output;
        }

        if (attributes.causal_mask || attributes.causal_mask_bottom_right) {
            std::shared_ptr<Tensor_attributes> row_index;

            row_index = pointwise(last_output,
                                  Pointwise_attributes()
                                      .set_name("gen_row_idx_causal")
                                      .set_mode(PointwiseMode_t::GEN_INDEX)
                                      .set_axis(2)
                                      .set_compute_data_type(DataType_t::INT32));
            row_index->set_data_type(DataType_t::INT32);

            if (attributes.causal_mask_bottom_right) {
                if (attributes.inputs[input_names::SEQ_LEN_KV]) {
                    row_index = pointwise(row_index,
                                          attributes.inputs[input_names::SEQ_LEN_KV],
                                          Pointwise_attributes()
                                              .set_name("row_idx_add_skv")
                                              .set_mode(PointwiseMode_t::ADD)
                                              .set_compute_data_type(DataType_t::INT32));
                } else {
                    row_index = pointwise(row_index,
                                          std::make_shared<Tensor_attributes>(static_cast<int32_t>(s_kv)),
                                          Pointwise_attributes()
                                              .set_name("row_idx_add_skv")
                                              .set_mode(PointwiseMode_t::ADD)
                                              .set_compute_data_type(DataType_t::INT32));
                }
                row_index->set_data_type(DataType_t::INT32);

                if (attributes.inputs[input_names::SEQ_LEN_Q]) {
                    row_index = pointwise(row_index,
                                          attributes.inputs[input_names::SEQ_LEN_Q],
                                          Pointwise_attributes()
                                              .set_name("row_idx_add_sq_sub_sq")
                                              .set_mode(PointwiseMode_t::SUB)
                                              .set_compute_data_type(DataType_t::INT32));
                } else {
                    row_index = pointwise(row_index,
                                          std::make_shared<Tensor_attributes>(static_cast<int32_t>(s_q)),
                                          Pointwise_attributes()
                                              .set_name("row_idx_add_sq_sub_sq")
                                              .set_mode(PointwiseMode_t::SUB)
                                              .set_compute_data_type(DataType_t::INT32));
                }
                row_index->set_data_type(DataType_t::INT32);
            }

            auto const& col_index = pointwise(last_output,
                                              Pointwise_attributes()
                                                  .set_name("gen_col_idx_causal")
                                                  .set_mode(PointwiseMode_t::GEN_INDEX)
                                                  .set_axis(3)
                                                  .set_compute_data_type(DataType_t::INT32));
            col_index->set_data_type(DataType_t::INT32);

            auto const& bool_mask = pointwise(row_index,
                                              col_index,
                                              Pointwise_attributes()
                                                  .set_name("row_greater_than_col")
                                                  .set_mode(PointwiseMode_t::CMP_GE)
                                                  .set_compute_data_type(DataType_t::BOOLEAN));
            bool_mask->set_data_type(DataType_t::BOOLEAN);

            last_output =
                pointwise(last_output,
                          std::make_shared<Tensor_attributes>(std::numeric_limits<float>::lowest()),
                          bool_mask,
                          Pointwise_attributes().set_name("binary_select").set_mode(PointwiseMode_t::BINARY_SELECT));
        }

        if (attributes.sliding_window_length.has_value()) {
            auto row_index_attributes =
                Pointwise_attributes().set_name("gen_row_index").set_mode(PointwiseMode_t::GEN_INDEX).set_axis(2);
            auto const& row_index_output = pointwise(last_output, row_index_attributes);

            auto col_index_attributes =
                Pointwise_attributes().set_name("gen_col_index").set_mode(PointwiseMode_t::GEN_INDEX).set_axis(3);
            auto const& col_index_output = pointwise(last_output, col_index_attributes);

            // sliding window length parameter should be of float type
            auto const& sliding_window_length =
                std::make_shared<Tensor_attributes>((float)attributes.sliding_window_length.value());

            auto add_col_attributes = Pointwise_attributes()
                                          .set_name("add_window_len")
                                          .set_mode(PointwiseMode_t::ADD)
                                          .set_compute_data_type(DataType_t::FLOAT)
                                          .set_axis(3);

            auto const& col_index_lower_output = pointwise(col_index_output, sliding_window_length, add_col_attributes);

            auto greater_than_attributes = Pointwise_attributes()
                                               .set_name("greaterthan_row<col+ws")
                                               .set_mode(PointwiseMode_t::CMP_GT)
                                               .set_compute_data_type(DataType_t::BOOLEAN);

            auto const& row_lesser_than_col_ws_output =
                pointwise(col_index_lower_output, row_index_output, greater_than_attributes);

            row_lesser_than_col_ws_output->set_data_type(DataType_t::BOOLEAN);

            // Lower attributes to binary select attributes
            auto negative_inf_swa = std::make_shared<Tensor_attributes>(-1024.0f * 1024.0f * 1024.0f);

            auto binary_select_attributes =
                Pointwise_attributes().set_name("binary_select").set_mode(PointwiseMode_t::BINARY_SELECT);

            auto const& swa_mask_output =
                pointwise(last_output, negative_inf_swa, row_lesser_than_col_ws_output, binary_select_attributes);

            last_output = swa_mask_output;
        }

        // Lower attributes to softmax attributes
        auto softmax_output = std::make_shared<Tensor_attributes>();
        softmax_output->set_is_virtual(true);

        // Create a virtual output for stats if inference step otherwise output.Stats is already set
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

        // Two cases for training: dropout present or not
        bool dropout_present = false;
        if (attributes.dropout_probability.has_value()) {
            dropout_present = true;
            // Special case: Skip dropout when 0.0 probability. Only do for 8.9.3 and up as rng isn't optional earlier.
            if (detail::get_backend_version() > 8902 && attributes.dropout_probability.value() == 0.0) {
                dropout_present = false;
            }
        } else if (attributes.inputs[input_names::Dropout_mask]) {
            dropout_present = true;
        }

        if (dropout_present) {
            if (attributes.outputs[output_names::RNG_DUMP] != nullptr) {
                rng_output = attributes.outputs[output_names::RNG_DUMP];
                rng(attributes.inputs[input_names::Seed],
                    attributes.inputs[input_names::Offset],
                    Rng_attributes()
                        .set_name("rng")
                        .set_distribution(RngDistribution_t::BERNOULLI)
                        .set_bernoulli_probability(1.0 - attributes.dropout_probability.value()),
                    rng_output);
            } else {
                rng_output = rng(attributes.inputs[input_names::Seed],
                                 attributes.inputs[input_names::Offset],
                                 Rng_attributes()
                                     .set_name("rng")
                                     .set_distribution(RngDistribution_t::BERNOULLI)
                                     .set_bernoulli_probability(1.0 - attributes.dropout_probability.value()));
                rng_output
                    // Hard coding dim and strides as rng output can no inputs to infer it from.
                    ->set_dim({b, h, s_q, s_kv})
                    .set_stride({h * s_q * s_kv, s_q * s_kv, s_kv, 1});
            }

            auto mask_attributes = Pointwise_attributes().set_name("dropout_mask_mul").set_mode(PointwiseMode_t::MUL);
            auto const& dropout_mask_output = pointwise(last_output, rng_output, mask_attributes);
            last_output                     = dropout_mask_output;

            std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> dropout_scale = nullptr;

            if (detail::get_backend_version() < 8903) {
                half dropout_scale_value = __float2half(1.0f / (1.0f - attributes.dropout_probability.value()));
                dropout_scale            = std::make_shared<Tensor_attributes>(dropout_scale_value);
            } else {
                float dropout_scale_value = (1.0f / (1.0f - attributes.dropout_probability.value()));
                dropout_scale             = std::make_shared<Tensor_attributes>(dropout_scale_value);
            }

            auto dropout_scale_attributes =
                Pointwise_attributes().set_name("dropout_scale").set_mode(PointwiseMode_t::MUL);
            auto const& dropout_scale_output = pointwise(last_output, dropout_scale, dropout_scale_attributes);
            last_output                      = dropout_scale_output;
        }

        // Lower attributes to bmm2 attributes
        // Requirement by cudnn backend to take in bmm2 aType as i/o type.
        last_output->set_data_type(attributes.inputs[input_names::Q]->get_data_type());

        auto const& seq_len_q  = attributes.inputs[input_names::SEQ_LEN_Q];
        auto const& seq_len_kv = attributes.inputs[input_names::SEQ_LEN_KV];
        auto const& V          = attributes.inputs[input_names::V];
        auto const& O          = attributes.outputs[output_names::O];
        auto bmm2_attributes =
            Matmul_attributes().set_name("bmm2").set_m_override(seq_len_q).set_k_override(seq_len_kv);
        // Special non-functional-style call. Needed because output already created and provided to user.
        matmul(last_output, V, bmm2_attributes, O);

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

    virtual int64_t
    get_fe_workspace_size_node() const override final {
        int64_t size = 0;

        // align alibi slopes memory to 16 bytes
        size += ((alibi_slopes_size + 15) / 16 * 16);

        return size;
    }

    virtual error_t
    workspace_modifications_tensors_(
        std::unordered_map<uid_t, std::tuple<int64_t, int64_t, std::vector<float>>>& workspace_modifications,
        int64_t& offset) const override final {
        if (attributes.alibi_mask) {
            CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(Q, input_names::Q);
            int64_t const h_q     = Q->second->get_dim()[1];
            auto alibi_slopes_vec = detail::get_abili_slope(h_q);
            workspace_modifications.emplace(alibi_slopes->get_uid(), std::make_tuple(0, offset, alibi_slopes_vec));
            int64_t alibi_slopes_size_padded = ((alibi_slopes_size + 15) / 16 * 16);
            offset                           = offset + alibi_slopes_size_padded;
        }
        return {error_code_t::OK, ""};
    }

#ifndef CUDNN_FRONTEND_SKIP_JSON_LIB
    virtual void
    serialize(json& j) const override final {
        j = attributes;
        j.update(R"({"tag": "SDPA_FWD"})"_json);
    }
#endif
};

class SDPABackwardNode : public NodeCRTP<SDPABackwardNode> {
    using input_names  = SDPA_backward_attributes::input_names;
    using output_names = SDPA_backward_attributes::output_names;

   private:
    // non-virtual node gpu tensors
    std::shared_ptr<Tensor_attributes> dQ_accum;
    int64_t dQ_accum_size = 0;
    std::shared_ptr<Tensor_attributes> softmax_sum;
    int64_t softmax_sum_size = 0;
    std::shared_ptr<Tensor_attributes> alibi_slopes;
    int64_t alibi_slopes_size = 0;

   public:
    SDPA_backward_attributes attributes;

    SDPABackwardNode(SDPA_backward_attributes&& attributes_, detail::Context const& context)
        : NodeCRTP(context), attributes(std::move(attributes_)) {}

    Type
    getType() override final {
        return Type::COMPOSITE;
    }

    error_t
    pre_validate_node() const override final {
        getLogger() << "[cudnn_frontend] INFO: " << "Validating SDPABackwardNode" << attributes.name << "..."
                    << std::endl;

        // check that Q, K, V, O, stats, dO, dQ, dK, dV tensors has been assigned
        // check that dim and strides has been assigned and last stride is 1
#define CUDNN_FE_SDPA_VALIDATE_DIM_STRIDE(port, port_map)                                                        \
    {                                                                                                            \
        std::shared_ptr<Tensor_attributes> tensor_ptr = port_map.at(port);                                       \
        RETURN_CUDNN_FRONTEND_ERROR_IF(tensor_ptr->get_dim().size() != 4,                                        \
                                       error_code_t::ATTRIBUTE_NOT_SET,                                          \
                                       "The dim for " + std::string(#port) + " is invalid");                     \
        RETURN_CUDNN_FRONTEND_ERROR_IF(tensor_ptr->get_stride().size() != 4,                                     \
                                       error_code_t::ATTRIBUTE_NOT_SET,                                          \
                                       "The stride for " + std::string(#port) + " is invalid");                  \
        RETURN_CUDNN_FRONTEND_ERROR_IF(                                                                          \
            tensor_ptr->get_stride()[3] != 1,                                                                    \
            error_code_t::GRAPH_NOT_SUPPORTED,                                                                   \
            "The stride for the last dimension corresponding to the embedding size per head should be 1 for " +  \
                std::string(#port));                                                                             \
        RETURN_CUDNN_FRONTEND_ERROR_IF(                                                                          \
            tensor_ptr->get_stride()[2] == 0,                                                                    \
            error_code_t::GRAPH_NOT_SUPPORTED,                                                                   \
            "The stride for the dimension corresponding to the sequence lengths per head should not be 0 for " + \
                std::string(#port));                                                                             \
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

        // validate backend limitations for the operation
        // clang-format off
        int64_t s_q  = attributes.inputs.at(input_names::Q)->get_dim()[2];
        int64_t s_kv = attributes.inputs.at(input_names::V)->get_dim()[2];
        int64_t h_q  = attributes.inputs.at(input_names::Q)->get_dim()[1];
        int64_t h_k  = attributes.inputs.at(input_names::K)->get_dim()[1];
        int64_t h_v  = attributes.inputs.at(input_names::V)->get_dim()[1];
        int64_t d_qk = attributes.inputs.at(input_names::Q)->get_dim()[3];
        int64_t d_v  = attributes.inputs.at(input_names::V)->get_dim()[3];

        bool const is_ragged = attributes.inputs.at(input_names::Q)->get_ragged_offset() ||
                               attributes.inputs.at(input_names::K)->get_ragged_offset() ||
                               attributes.inputs.at(input_names::V)->get_ragged_offset() ||
                               attributes.inputs.at(input_names::O)->get_ragged_offset();

        auto const& bias_mask = attributes.inputs.find(input_names::Bias);
        bool const is_bias   = (bias_mask != attributes.inputs.end() && bias_mask->second != nullptr);

        auto const& dropout_mask     = attributes.inputs.find(input_names::Dropout_mask);
        bool const is_dropout_custom = (dropout_mask != attributes.inputs.end()) && (dropout_mask->second != nullptr);
        bool const is_dropout        = attributes.dropout_probability.has_value() || is_dropout_custom;

        // validation TODO:
        //    - validate stats has valid dims
        //    - validate Q and dQ have the same dims

        // validate basic dimension requirements
        RETURN_CUDNN_FRONTEND_ERROR_IF((d_qk > 128) || (d_qk % 8 != 0) || (d_v > 128) || (d_v % 8 != 0),
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "Num hidden_dim shoud be less than 128 and hidden_dim should be multiple of 8");

        RETURN_CUDNN_FRONTEND_ERROR_IF((h_q % h_k != 0) || (h_q % h_v != 0),
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "For group-query attention, number of heads for key and query must be a factor of number of heads for query");

        // validate options for attn_scale
        auto const& attn_scale    = attributes.inputs.find(input_names::Attn_scale);
        bool const has_attn_scale = (attn_scale != attributes.inputs.end()) && (attn_scale->second != nullptr);
        RETURN_CUDNN_FRONTEND_ERROR_IF(has_attn_scale && attributes.attn_scale_value.has_value(),
                                       error_code_t::ATTRIBUTE_NOT_SET,
                                       "attn_scale with tensor and value cannot be set at the same time.");

        // validate options for bias mask
        RETURN_CUDNN_FRONTEND_ERROR_IF(is_bias && (bias_mask->second->get_data_type() == DataType_t::BOOLEAN),
                                        error_code_t::GRAPH_NOT_SUPPORTED,
                                        "Bias mask data type cannot be boolean");

        // validate options for padding mask
        auto const& seq_len_q     = attributes.inputs.find(input_names::SEQ_LEN_Q);
        bool const has_seq_len_q  = (seq_len_q != attributes.inputs.end()) && (seq_len_q->second != nullptr);
        auto const& seq_len_kv    = attributes.inputs.find(input_names::SEQ_LEN_KV);
        bool const has_seq_len_kv = (seq_len_kv != attributes.inputs.end()) && (seq_len_kv->second != nullptr);
        RETURN_CUDNN_FRONTEND_ERROR_IF(attributes.padding_mask && (!has_seq_len_q || !has_seq_len_kv),
                                       error_code_t::ATTRIBUTE_NOT_SET,
                                       "Padding mask requires seq_len_q and seq_len_kv to be set.");
        RETURN_CUDNN_FRONTEND_ERROR_IF((!attributes.padding_mask) && (has_seq_len_q || has_seq_len_kv),
                                       error_code_t::ATTRIBUTE_NOT_SET,
                                       "seq_len_q and seq_len_kv needs to be set only if padding mask is enabled.");

        // validate options for bottom right causal mask
        RETURN_CUDNN_FRONTEND_ERROR_IF(attributes.causal_mask && attributes.causal_mask_bottom_right,
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "Bottom right causal mask and causal mask cannot be both enabled");

        RETURN_CUDNN_FRONTEND_ERROR_IF(attributes.causal_mask_bottom_right && s_q > s_kv,
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "Bottom right causal mask does not support s_q > s_kv. Please virtually slice the Q tensor and pass it as s_q == s_kv");

        RETURN_CUDNN_FRONTEND_ERROR_IF(attributes.causal_mask_bottom_right && (is_bias || attributes.alibi_mask || is_ragged || attributes.padding_mask || is_dropout),
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "Bottom right causal mask is only supported with is_bias=False, is_alibi=False, is_ragged=False, padding_mask=False, is_dropout=False");

        RETURN_CUDNN_FRONTEND_ERROR_IF(attributes.causal_mask_bottom_right && ((s_q % 64 != 0) || (s_kv % 64 != 0)),
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "Bottom right causal mask is only supported with s_q multiple of 64, and s_kv multiple of 64");

        // validate options for sliding window length
        RETURN_CUDNN_FRONTEND_ERROR_IF(attributes.sliding_window_length.has_value() && attributes.sliding_window_length.value() < 0,
                                       error_code_t::INVALID_VALUE,
                                       "Sliding window length should be greater than or equals to zero when set.");


        RETURN_CUDNN_FRONTEND_ERROR_IF(attributes.sliding_window_length.has_value() && (attributes.padding_mask || !attributes.causal_mask || is_dropout || is_bias || is_ragged),
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "Sliding window attention is only supported with padding_mask=False, causal_mask=True, is_dropout=False, is_bias=False, is_ragged=False");

        // validate options for dropout mask
        RETURN_CUDNN_FRONTEND_ERROR_IF(attributes.dropout_probability.has_value() && is_dropout_custom,
                                       error_code_t::ATTRIBUTE_NOT_SET,
                                       "Using both, custom dropout mask and internal-mask generation using dropout probability, is ill-formed.");

        RETURN_CUDNN_FRONTEND_ERROR_IF(attributes.dropout_probability.has_value() && attributes.dropout_probability.value() == 1.0,
                                       error_code_t::ATTRIBUTE_NOT_SET,
                                       "Dropout probability cannot be 1 as corresponding scale wont be well formed.");

        // version specific validation
        RETURN_CUDNN_FRONTEND_ERROR_IF(detail::get_backend_version() < 8906 && ((s_kv % 64 != 0) || (d_qk % 64 != 0)),
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "For cuDNN version below 8.9.6, s_kv not a multiple of 64 or d not a multiple of 64 is not supported");

        RETURN_CUDNN_FRONTEND_ERROR_IF(detail::get_backend_version() < 8907 && (s_kv % 64 != 0) && (!(attributes.padding_mask)),
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "For cuDNN version below 8.9.7, s_kv not a multiple of 64 is not supported");

        RETURN_CUDNN_FRONTEND_ERROR_IF(detail::get_backend_version() < 90000 && ((s_q % 64 != 0) || (s_kv % 64 != 0)) && (attributes.padding_mask || is_dropout),
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "For cuDNN version below 9.0.0, s_q/s_kv not a multiple of 64 with padding/dropout mask is not supported");

        RETURN_CUDNN_FRONTEND_ERROR_IF(detail::get_backend_version() < 90000 && (s_q < 64),
                                       error_code_t::GRAPH_NOT_SUPPORTED,
            "                          Sequence length must be greater than or equal to 64 for cudnn version prior to v9.0.0");

        RETURN_CUDNN_FRONTEND_ERROR_IF(detail::get_backend_version() < 90200 && attributes.sliding_window_length.has_value(),
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "For cuDNN version below 9.2.0, sliding window attention is not supported");

        // validate that datatype is set for the graph
        RETURN_CUDNN_FRONTEND_ERROR_IF(context.get_intermediate_data_type() == DataType_t::NOT_SET,
                                       error_code_t::ATTRIBUTE_NOT_SET,
                                       "Intermediate tensor data type needs to be set as internal tensors require it.");
        // clang-format on

        CHECK_CUDNN_FRONTEND_ERROR(attributes.validate_inputs());
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
    expand_and_infer_properties_node() override final {
        getLogger() << "[cudnn_frontend] INFO: Inferrencing properties for SDPABackwardNode " << attributes.name
                    << "..." << std::endl;

        attributes.fill_from_context(context);

        // Gather dim to fill properties of virtual tensors
        auto const& q_dim = attributes.inputs[input_names::Q]->get_dim();
        auto b            = q_dim[0];
        auto h_q          = q_dim[1];
        auto s_q          = q_dim[2];
        auto d_qk         = q_dim[3];
        auto const& k_dim = attributes.inputs[input_names::K]->get_dim();
        auto h_k          = k_dim[1];
        auto s_kv         = k_dim[2];
        auto const& v_dim = attributes.inputs[input_names::V]->get_dim();
        auto h_v          = v_dim[1];
        auto d_v          = v_dim[3];

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

        std::shared_ptr<Tensor_attributes> last_output, exp_s_output, dS_output, rng_output;

        // --------------Initialize and create tensors before creating nodes--------------------
        // one_tensor is needed for non-dropout graphs
        // one_tensor is passed by the node
        auto one_tensor = std::make_shared<Tensor_attributes>(1.0f);

        if (attributes.attn_scale_value.has_value()) {
            attributes.inputs[input_names::Attn_scale] =
                std::make_shared<Tensor_attributes>(attributes.attn_scale_value.value());
        }

        // alibi_slopes is passed by the node
        if (attributes.alibi_mask) {
            alibi_slopes = std::make_shared<Tensor_attributes>();
            alibi_slopes->set_is_virtual(false);
            alibi_slopes->set_dim({1, h_q, 1, 1}).set_stride({h_q, h_q, 1, 1});
            alibi_slopes->set_data_type(DataType_t::FLOAT);
            alibi_slopes_size = h_q * sizeof(float);
        }

        // if dropout_prob is used, then the node passes scale and scale inverse
        // if dropout_mask is used, then the user passes scale and scale_inverse
        bool is_dropout_prob = (attributes.dropout_probability.has_value());
        bool is_dropout_mask = (attributes.inputs[input_names::Dropout_mask] != nullptr);
        if (is_dropout_prob) {
            float dropout_scale_value     = 1.0f / (1.0f - attributes.dropout_probability.value());
            float dropout_scale_inv_value = (1.0f - attributes.dropout_probability.value());

            attributes.inputs[input_names::Dropout_scale] = std::make_shared<Tensor_attributes>(dropout_scale_value);
            attributes.inputs[input_names::Dropout_scale_inv] =
                std::make_shared<Tensor_attributes>(dropout_scale_inv_value);
        }

        // ---------------------input tensor workarounds---------------------------

        bool use_workspace_opt = false;

        if (detail::get_backend_version() >= 8905 && detail::get_backend_version() < 90000) {
            // workspace optimization is enabled by default when:
            //   8.9.5 <= cudnn version < 9.0.0
            //   device >= hopper
            //   batch * num_heads * seq_len_q * seq_len_kv * 2 <= dP workspace limit
            //
            // This following environment variable allows you to control the dP workspace limit.
            // From cuDNN version 9.0.0, this option is obsolete will be ignored.
            // CUDNN_FRONTEND_ATTN_DP_WORKSPACE_LIMIT=unset  - enable workspace opt. until the default 256MB limit.
            // CUDNN_FRONTEND_ATTN_DP_WORKSPACE_LIMIT=-1     - always enable workspace opt.
            // CUDNN_FRONTEND_ATTN_DP_WORKSPACE_LIMIT=0      - always disable workspace opt.
            // CUDNN_FRONTEND_ATTN_DP_WORKSPACE_LIMIT=n      - enable workspace opt. until the n byte limit
            struct cudaDeviceProp prop;
            CHECK_CUDA_ERROR(detail::cuda_get_device_properties(&prop, 0));

            // hopper or above
            if (prop.major >= 9) {
                // default upper limit for workspace 256MB
                int64_t max_dp_workspace_bytes = 256 * 1024 * 1024;

                // allow setting the upper limit with envvars
                char* env_dp_workspace_limit_char = std::getenv("CUDNN_FRONTEND_ATTN_DP_WORKSPACE_LIMIT");
                if (env_dp_workspace_limit_char) {
                    try {
                        std::string env_dp_workspace_limit_str(env_dp_workspace_limit_char);
                        max_dp_workspace_bytes = static_cast<int64_t>(std::stoll(env_dp_workspace_limit_str));
                    } catch (...) {
                        RETURN_CUDNN_FRONTEND_ERROR_IF(true,
                                                       error_code_t::ATTRIBUTE_NOT_SET,
                                                       "Invalid argument for CUDNN_FRONTEND_ATTN_DP_WORKSPACE_LIMIT "
                                                       "(int64_t; in bytes)");
                    }
                }

                int64_t workspace_s_q               = ((s_q + 64 - 1) / 64) * 64;
                int64_t workspace_s_kv              = ((s_kv + 64 - 1) / 64) * 64;
                int64_t required_dp_workspace_bytes = b * h_q * workspace_s_q * workspace_s_kv * 2;

                if (max_dp_workspace_bytes == -1) {
                    use_workspace_opt = true;
                } else if (max_dp_workspace_bytes == 0) {
                    use_workspace_opt = false;
                } else {
                    use_workspace_opt = (required_dp_workspace_bytes <= max_dp_workspace_bytes);
                }
            }
        }

        // Force dP workspace implementation if:
        //  - dBias is enabled (dBias is only supported on workspace implementation)
        //  - the user force requests deterministic algorithm
        if (attributes.outputs[output_names::dBias] || attributes.is_deterministic_algorithm) {
            use_workspace_opt = true;
        }

        // non-virtual dQ_accum is how the backend API signals workspace optimization
        if (!use_workspace_opt) {
            dQ_accum = std::make_shared<Tensor_attributes>();
            dQ_accum->set_is_virtual(false);
            dQ_accum->set_dim({b, h_q, s_q, d_qk}).set_stride({h_q * s_q * d_qk, s_q * d_qk, d_qk, 1});
            dQ_accum->set_data_type(DataType_t::FLOAT).set_reordering_type(TensorReordering_t::F16x16);
            dQ_accum_size = b * h_q * s_q * d_qk * sizeof(float);
        }

        // --------------RNG node--------------------

        if (is_dropout_prob) {
            if (attributes.outputs[output_names::RNG_DUMP] != nullptr) {
                rng_output = attributes.outputs[output_names::RNG_DUMP];
                rng(attributes.inputs[input_names::Seed],
                    attributes.inputs[input_names::Offset],
                    Rng_attributes()
                        .set_name("rng")
                        .set_distribution(RngDistribution_t::BERNOULLI)
                        .set_bernoulli_probability(1.0f - attributes.dropout_probability.value()),
                    rng_output);
            } else {
                rng_output = rng(attributes.inputs[input_names::Seed],
                                 attributes.inputs[input_names::Offset],
                                 Rng_attributes()
                                     .set_name("rng")
                                     .set_distribution(RngDistribution_t::BERNOULLI)
                                     .set_bernoulli_probability(1.0f - attributes.dropout_probability.value()));
                rng_output->set_dim({b, h_q, s_q, s_kv}).set_stride({h_q * s_q * s_kv, s_q * s_kv, s_kv, 1});
            }
        } else if (is_dropout_mask) {
            rng_output = attributes.inputs[input_names::Dropout_mask];
        }

        // --------------"dO * o => softmax_sum" chain--------------------

        // last_output = dO * O
        last_output = pointwise(attributes.inputs[input_names::dO],
                                attributes.inputs[input_names::O],
                                Pointwise_attributes().set_name("mul_dO_O").set_mode(PointwiseMode_t::MUL));
        last_output->set_dim({b, h_q, s_q, d_v}).set_stride({h_q * s_q * d_v, s_q * d_v, h_q * d_v, 1});

        // last_output = reduce(last_output, "b hq sq dv -> b hq sq 1")
        last_output =
            reduction(last_output, Reduction_attributes().set_name("reduce_dO_o").set_mode(ReductionMode_t::ADD));
        last_output->set_dim({b, h_q, s_q, 1}).set_stride({h_q * s_q, s_q, 1, 1});

        // softmax_sum = last_output * dropout_scale
        last_output = pointwise(last_output,
                                attributes.inputs[input_names::Dropout_scale_inv]
                                    ? attributes.inputs[input_names::Dropout_scale_inv]
                                    : one_tensor,
                                Pointwise_attributes().set_name("scale_dropout_inv").set_mode(PointwiseMode_t::MUL));

        softmax_sum = last_output;

        // --------------"Q @ KT => exp_softmax => dV" chain--------------------

        // s = einsum(q, k, "b hq sq dqk, b (hk g) skv dqk -> b hq sq skv", g=hq//hk)
        last_output = matmul(attributes.inputs[input_names::Q],
                             attributes.inputs[input_names::K],
                             Matmul_attributes()
                                 .set_name("matmul_Q_KT")
                                 .set_m_override(attributes.inputs[input_names::SEQ_LEN_Q])
                                 .set_n_override(attributes.inputs[input_names::SEQ_LEN_KV]));
        last_output->set_dim({b, h_q, s_q, s_kv}).set_stride({h_q * s_q * s_kv, s_q * s_kv, s_kv, 1});

        // last_output = last_output * attention_scale
        if (attributes.inputs[input_names::Attn_scale]) {
            last_output = pointwise(last_output,
                                    attributes.inputs[input_names::Attn_scale],
                                    Pointwise_attributes().set_name("mul_s_attn_scale").set_mode(PointwiseMode_t::MUL));
        }

        // (optional) last_output = last_output + bias
        if (attributes.inputs[input_names::Bias]) {
            last_output = pointwise(last_output,
                                    attributes.inputs[input_names::Bias],
                                    Pointwise_attributes().set_name("add_bias").set_mode(PointwiseMode_t::ADD));
        }

        // (optional) last_output = last_output + alibi_mask
        if (attributes.alibi_mask) {
            auto row_idx_output = pointwise(last_output,
                                            Pointwise_attributes()
                                                .set_name("gen_row_idx_alibi")
                                                .set_mode(PointwiseMode_t::GEN_INDEX)
                                                .set_axis(2)
                                                .set_compute_data_type(DataType_t::INT32));
            row_idx_output->set_data_type(DataType_t::INT32);

            auto col_idx_output = pointwise(last_output,
                                            Pointwise_attributes()
                                                .set_name("gen_col_idx_alibi")
                                                .set_mode(PointwiseMode_t::GEN_INDEX)
                                                .set_axis(3)
                                                .set_compute_data_type(DataType_t::INT32));
            col_idx_output->set_data_type(DataType_t::INT32);

            auto sub_idx_output = pointwise(col_idx_output,
                                            row_idx_output,
                                            Pointwise_attributes()
                                                .set_name("sub_col_row_alibi")
                                                .set_mode(PointwiseMode_t::SUB)
                                                .set_compute_data_type(DataType_t::INT32));
            sub_idx_output->set_data_type(DataType_t::INT32);

            auto alibi_mask_output =
                pointwise(sub_idx_output,
                          alibi_slopes,
                          Pointwise_attributes().set_name("mul_slope_alibi").set_mode(PointwiseMode_t::MUL));

            last_output = pointwise(last_output,
                                    alibi_mask_output,
                                    Pointwise_attributes().set_name("add_alibi").set_mode(PointwiseMode_t::ADD));
        }

        // (optional) Apply padding mask
        if (attributes.padding_mask) {
            auto row_idx_output = pointwise(last_output,
                                            Pointwise_attributes()
                                                .set_name("gen_row_idx_padding")
                                                .set_mode(PointwiseMode_t::GEN_INDEX)
                                                .set_axis(2)
                                                .set_compute_data_type(DataType_t::INT32));
            row_idx_output->set_data_type(DataType_t::INT32);

            auto col_idx_output = pointwise(last_output,
                                            Pointwise_attributes()
                                                .set_name("gen_col_idx_padding")
                                                .set_mode(PointwiseMode_t::GEN_INDEX)
                                                .set_axis(3)
                                                .set_compute_data_type(DataType_t::INT32));
            col_idx_output->set_data_type(DataType_t::INT32);

            auto row_mask_output = pointwise(row_idx_output,
                                             attributes.inputs[input_names::SEQ_LEN_Q],
                                             Pointwise_attributes()
                                                 .set_name("lt_row_sq_padding")
                                                 .set_mode(PointwiseMode_t::CMP_LT)
                                                 .set_compute_data_type(DataType_t::BOOLEAN));
            row_mask_output->set_data_type(DataType_t::BOOLEAN);

            auto col_mask_output = pointwise(col_idx_output,
                                             attributes.inputs[input_names::SEQ_LEN_KV],
                                             Pointwise_attributes()
                                                 .set_name("lt_col_skv_padding")
                                                 .set_mode(PointwiseMode_t::CMP_LT)
                                                 .set_compute_data_type(DataType_t::BOOLEAN));
            col_mask_output->set_data_type(DataType_t::BOOLEAN);

            auto padding_mask_output = pointwise(row_mask_output,
                                                 col_mask_output,
                                                 Pointwise_attributes()
                                                     .set_name("and_row_col_padding")
                                                     .set_mode(PointwiseMode_t::LOGICAL_AND)
                                                     .set_compute_data_type(DataType_t::BOOLEAN));
            padding_mask_output->set_data_type(DataType_t::BOOLEAN);
            auto negative_inf_padding = std::make_shared<Tensor_attributes>(std::numeric_limits<float>::lowest());

            last_output =
                pointwise(last_output,
                          negative_inf_padding,
                          padding_mask_output,
                          Pointwise_attributes().set_name("select_padding").set_mode(PointwiseMode_t::BINARY_SELECT));
        }

        if (attributes.causal_mask || attributes.causal_mask_bottom_right) {
            std::shared_ptr<Tensor_attributes> row_index;

            row_index = pointwise(last_output,
                                  Pointwise_attributes()
                                      .set_name("gen_row_idx_causal")
                                      .set_mode(PointwiseMode_t::GEN_INDEX)
                                      .set_axis(2)
                                      .set_compute_data_type(DataType_t::INT32));
            row_index->set_data_type(DataType_t::INT32);

            if (attributes.causal_mask_bottom_right) {
                if (attributes.inputs[input_names::SEQ_LEN_KV]) {
                    row_index = pointwise(row_index,
                                          attributes.inputs[input_names::SEQ_LEN_KV],
                                          Pointwise_attributes()
                                              .set_name("row_idx_add_skv")
                                              .set_mode(PointwiseMode_t::ADD)
                                              .set_compute_data_type(DataType_t::INT32));
                } else {
                    row_index = pointwise(row_index,
                                          std::make_shared<Tensor_attributes>(static_cast<int32_t>(s_kv)),
                                          Pointwise_attributes()
                                              .set_name("row_idx_add_skv")
                                              .set_mode(PointwiseMode_t::ADD)
                                              .set_compute_data_type(DataType_t::INT32));
                }
                row_index->set_data_type(DataType_t::INT32);

                if (attributes.inputs[input_names::SEQ_LEN_Q]) {
                    row_index = pointwise(row_index,
                                          attributes.inputs[input_names::SEQ_LEN_Q],
                                          Pointwise_attributes()
                                              .set_name("row_idx_add_sq_sub_sq")
                                              .set_mode(PointwiseMode_t::SUB)
                                              .set_compute_data_type(DataType_t::INT32));
                } else {
                    row_index = pointwise(row_index,
                                          std::make_shared<Tensor_attributes>(static_cast<int32_t>(s_q)),
                                          Pointwise_attributes()
                                              .set_name("row_idx_add_sq_sub_sq")
                                              .set_mode(PointwiseMode_t::SUB)
                                              .set_compute_data_type(DataType_t::INT32));
                }
                row_index->set_data_type(DataType_t::INT32);
            }

            auto const& col_index = pointwise(last_output,
                                              Pointwise_attributes()
                                                  .set_name("gen_col_idx_causal")
                                                  .set_mode(PointwiseMode_t::GEN_INDEX)
                                                  .set_axis(3)
                                                  .set_compute_data_type(DataType_t::INT32));
            col_index->set_data_type(DataType_t::INT32);

            auto const& bool_mask = pointwise(row_index,
                                              col_index,
                                              Pointwise_attributes()
                                                  .set_name("row_greater_than_col")
                                                  .set_mode(PointwiseMode_t::CMP_GE)
                                                  .set_compute_data_type(DataType_t::BOOLEAN));
            bool_mask->set_data_type(DataType_t::BOOLEAN);

            last_output =
                pointwise(last_output,
                          std::make_shared<Tensor_attributes>(std::numeric_limits<float>::lowest()),
                          bool_mask,
                          Pointwise_attributes().set_name("binary_select").set_mode(PointwiseMode_t::BINARY_SELECT));
        }

        // last_output = last_output - stats
        last_output = pointwise(last_output,
                                attributes.inputs[input_names::Stats],
                                Pointwise_attributes().set_name("sub_s_m").set_mode(PointwiseMode_t::SUB));

        // WAR for bug 4475073 by explicitly putting the padding value again after the stats have been loaded
        if (attributes.padding_mask && detail::get_backend_version() >= 90000) {
            auto row_idx_output = pointwise(last_output,
                                            Pointwise_attributes()
                                                .set_name("gen_row_idx_2nd_padding")
                                                .set_mode(PointwiseMode_t::GEN_INDEX)
                                                .set_axis(2)
                                                .set_compute_data_type(DataType_t::INT32));
            row_idx_output->set_data_type(DataType_t::INT32);

            auto col_idx_output = pointwise(last_output,
                                            Pointwise_attributes()
                                                .set_name("gen_col_idx_2nd_padding")
                                                .set_mode(PointwiseMode_t::GEN_INDEX)
                                                .set_axis(3)
                                                .set_compute_data_type(DataType_t::INT32));
            col_idx_output->set_data_type(DataType_t::INT32);

            auto row_mask_output = pointwise(row_idx_output,
                                             attributes.inputs[input_names::SEQ_LEN_Q],
                                             Pointwise_attributes()
                                                 .set_name("lt_row_sq_2nd_padding")
                                                 .set_mode(PointwiseMode_t::CMP_LT)
                                                 .set_compute_data_type(DataType_t::BOOLEAN));
            row_mask_output->set_data_type(DataType_t::BOOLEAN);

            auto col_mask_output = pointwise(col_idx_output,
                                             attributes.inputs[input_names::SEQ_LEN_KV],
                                             Pointwise_attributes()
                                                 .set_name("lt_col_skv_2nd_padding")
                                                 .set_mode(PointwiseMode_t::CMP_LT)
                                                 .set_compute_data_type(DataType_t::BOOLEAN));
            col_mask_output->set_data_type(DataType_t::BOOLEAN);

            auto padding_mask_output = pointwise(row_mask_output,
                                                 col_mask_output,
                                                 Pointwise_attributes()
                                                     .set_name("and_row_col_2nd_padding")
                                                     .set_mode(PointwiseMode_t::LOGICAL_AND)
                                                     .set_compute_data_type(DataType_t::BOOLEAN));
            padding_mask_output->set_data_type(DataType_t::BOOLEAN);
            auto negative_inf_padding = std::make_shared<Tensor_attributes>(std::numeric_limits<float>::lowest());

            last_output = pointwise(
                last_output,
                negative_inf_padding,
                padding_mask_output,
                Pointwise_attributes().set_name("select_2nd_padding").set_mode(PointwiseMode_t::BINARY_SELECT));
        }

        if (attributes.sliding_window_length.has_value()) {
            auto row_index_attributes =
                Pointwise_attributes().set_name("gen_row_index").set_mode(PointwiseMode_t::GEN_INDEX).set_axis(2);
            auto const& row_index_output = pointwise(last_output, row_index_attributes);

            auto col_index_attributes =
                Pointwise_attributes().set_name("gen_col_index").set_mode(PointwiseMode_t::GEN_INDEX).set_axis(3);
            auto const& col_index_output = pointwise(last_output, col_index_attributes);

            // sliding window length parameter should be of float type
            auto const& sliding_window_length =
                std::make_shared<Tensor_attributes>((float)attributes.sliding_window_length.value());

            auto add_col_attributes = Pointwise_attributes()
                                          .set_name("add_window_len")
                                          .set_mode(PointwiseMode_t::ADD)
                                          .set_compute_data_type(DataType_t::FLOAT)
                                          .set_axis(3);

            auto const& col_index_lower_output = pointwise(col_index_output, sliding_window_length, add_col_attributes);

            auto greater_than_attributes = Pointwise_attributes()
                                               .set_name("greaterthan_row<col+ws")
                                               .set_mode(PointwiseMode_t::CMP_GT)
                                               .set_compute_data_type(DataType_t::BOOLEAN);

            auto const& row_lesser_than_col_ws_output =
                pointwise(col_index_lower_output, row_index_output, greater_than_attributes);

            row_lesser_than_col_ws_output->set_data_type(DataType_t::BOOLEAN);

            // Lower attributes to binary select attributes
            auto negative_inf_swa = std::make_shared<Tensor_attributes>(std::numeric_limits<float>::lowest());

            auto binary_select_attributes =
                Pointwise_attributes().set_name("binary_select").set_mode(PointwiseMode_t::BINARY_SELECT);

            auto const& swa_mask_output =
                pointwise(last_output, negative_inf_swa, row_lesser_than_col_ws_output, binary_select_attributes);

            last_output = swa_mask_output;
        }

        // last_output = exp(last_output)
        last_output = pointwise(last_output, Pointwise_attributes().set_name("exp_s").set_mode(PointwiseMode_t::EXP));

        exp_s_output = last_output;

        // (optional) last_output = last_output * dropout rng_output
        if (is_dropout_prob || is_dropout_mask) {
            last_output =
                pointwise(last_output,
                          rng_output,
                          Pointwise_attributes().set_name("mul_p_dropout_mask").set_mode(PointwiseMode_t::MUL));
        }

        // (optional) last_output = last_output * dropout_scale
        if (attributes.inputs[input_names::Dropout_scale]) {
            last_output =
                pointwise(last_output,
                          attributes.inputs[input_names::Dropout_scale],
                          Pointwise_attributes().set_name("mul_p_dropout_scale").set_mode(PointwiseMode_t::MUL));
        }

        // dV = einsum(p, dO, "b hq sq skv", "b hq sq dv -> b hq skv dv")
        // if GQA, then dV = reduce(dV, "b (hv g) skv dv -> b hv skv dv", g=hq//hv)
        // as reshape + matmul
        last_output = reshape(last_output, Reshape_attributes().set_name("reshape_p"));
        last_output->set_dim({b, h_q, s_kv, s_q}).set_stride({h_q * s_q * s_kv, s_q * s_kv, 1, s_kv});
        last_output->set_data_type(attributes.inputs[input_names::Q]->get_data_type());

        if (h_q == h_v) {
            // for MHA
            matmul(last_output,
                   attributes.inputs[input_names::dO],
                   Matmul_attributes()
                       .set_name("matmul_pT_dO")
                       .set_m_override(attributes.inputs[input_names::SEQ_LEN_KV])
                       .set_k_override(attributes.inputs[input_names::SEQ_LEN_Q]),
                   attributes.outputs[output_names::dV]);
        } else {
            // for GQA and MQA
            last_output = matmul(last_output,
                                 attributes.inputs[input_names::dO],
                                 Matmul_attributes()
                                     .set_name("matmul_pT_dO")
                                     .set_m_override(attributes.inputs[input_names::SEQ_LEN_KV])
                                     .set_k_override(attributes.inputs[input_names::SEQ_LEN_Q]));
            last_output->set_dim({b, h_q, s_kv, d_v}).set_stride({h_q * s_kv * d_v, s_kv * d_v, d_v, 1});
            last_output->set_data_type(attributes.inputs[input_names::Q]->get_data_type());
            reduction(last_output,
                      Reduction_attributes().set_name("red_dV_head").set_mode(ReductionMode_t::ADD),
                      attributes.outputs[output_names::dV]);
        }

        // --------------"dO @ VT => dS_output => dK" chain--------------------

        // dP = einsum(dO, v, "b hq sq dv, b (hv g) skv dv -> b hq sq skv", g=hq//hv)
        last_output = matmul(attributes.inputs[input_names::dO],
                             attributes.inputs[input_names::V],
                             Matmul_attributes()
                                 .set_name("matmul_dO_VT")
                                 .set_m_override(attributes.inputs[input_names::SEQ_LEN_Q])
                                 .set_n_override(attributes.inputs[input_names::SEQ_LEN_KV]));
        last_output->set_dim({b, h_q, s_q, s_kv}).set_stride({h_q * s_q * s_kv, s_q * s_kv, s_kv, 1});

        // last_output = last_output(dP) * mask
        last_output = pointwise(last_output,
                                (is_dropout_prob || is_dropout_mask) ? rng_output : one_tensor,
                                Pointwise_attributes().set_name("dP_dropout_mask").set_mode(PointwiseMode_t::MUL));

        // last_output = last_output - softmax_sum
        last_output = pointwise(last_output,
                                softmax_sum,
                                Pointwise_attributes().set_name("sub_dP_softmax_sum").set_mode(PointwiseMode_t::SUB));

        // last_output = last_output * exp_s_output
        last_output = pointwise(
            last_output, exp_s_output, Pointwise_attributes().set_name("mul_dP_exp_s").set_mode(PointwiseMode_t::MUL));

        // (optional) last_output = last_output * dropout_scale
        if (attributes.inputs[input_names::Dropout_scale]) {
            last_output =
                pointwise(last_output,
                          attributes.inputs[input_names::Dropout_scale],
                          Pointwise_attributes().set_name("mul_dS_dropout_scale").set_mode(PointwiseMode_t::MUL));
        }

        if (attributes.outputs[output_names::dBias]) {
            reduction(last_output,
                      Reduction_attributes().set_name("red_dP_dBias").set_mode(ReductionMode_t::ADD),
                      attributes.outputs[output_names::dBias]);
        }

        // (optional) last_output = last_output * bmm_scale
        if (attributes.inputs[input_names::Attn_scale]) {
            last_output =
                pointwise(last_output,
                          attributes.inputs[input_names::Attn_scale],
                          Pointwise_attributes().set_name("mul_dS_attn_scale").set_mode(PointwiseMode_t::MUL));
        }

        dS_output = last_output;

        // dK = einsum(dS, Q, "b hq sq skv", "b hq sq dqk -> b hq skv dqk")
        // if GQA, then dK = reduce(dK, "b (hk g) skv dqk -> b hk skv dqk", hq//hk)
        // as reshape + matmul
        last_output = reshape(last_output, Reshape_attributes().set_name("reshape_dS"));
        last_output->set_dim({b, h_q, s_kv, s_q}).set_stride({h_q * s_q * s_kv, s_q * s_kv, 1, s_kv});
        last_output->set_data_type(attributes.inputs[input_names::Q]->get_data_type());

        if (h_q == h_k) {
            // for MHA
            matmul(last_output,
                   attributes.inputs[input_names::Q],
                   Matmul_attributes()
                       .set_name("matmul_dST_Q")
                       .set_m_override(attributes.inputs[input_names::SEQ_LEN_KV])
                       .set_k_override(attributes.inputs[input_names::SEQ_LEN_Q]),
                   attributes.outputs[output_names::dK]);
        } else {
            // for GQA and MQA
            last_output = matmul(last_output,
                                 attributes.inputs[input_names::Q],
                                 Matmul_attributes()
                                     .set_name("matmul_dST_Q")
                                     .set_m_override(attributes.inputs[input_names::SEQ_LEN_KV])
                                     .set_k_override(attributes.inputs[input_names::SEQ_LEN_Q]));
            last_output->set_dim({b, h_q, s_kv, d_qk}).set_stride({h_q * s_kv * d_qk, s_kv * d_qk, d_qk, 1});
            last_output->set_data_type(attributes.inputs[input_names::Q]->get_data_type());
            reduction(last_output,
                      Reduction_attributes().set_name("red_dK_head").set_mode(ReductionMode_t::ADD),
                      attributes.outputs[output_names::dK]);
        }

        // --------------"dp_scaled @ K => dQ" chain--------------------

        auto const& kt_dim    = attributes.inputs[input_names::K]->get_dim();
        auto const& kt_stride = attributes.inputs[input_names::K]->get_stride();

        // dQ = einsum(dS, K, "b hq sq skv, b (hk g) skv dqk -> b hq sq dqk", g=hq//hk)
        // as reshape + matmul
        last_output = reshape(attributes.inputs[input_names::K], Reshape_attributes().set_name("reshape_k"));
        last_output->set_dim({kt_dim[0], kt_dim[1], kt_dim[3], kt_dim[2]})
            .set_stride({kt_stride[0], kt_stride[1], kt_stride[3], kt_stride[2]});

        if (attributes.inputs[input_names::K]->get_ragged_offset() != nullptr) {
            last_output->set_ragged_offset(attributes.inputs[input_names::K]->get_ragged_offset());
        }

        matmul(dS_output,
               last_output,
               Matmul_attributes()
                   .set_name("matmul_dS_K")
                   .set_m_override(attributes.inputs[input_names::SEQ_LEN_Q])
                   .set_k_override(attributes.inputs[input_names::SEQ_LEN_KV]),
               (dQ_accum) ? dQ_accum : attributes.outputs[output_names::dQ]);

        if (dQ_accum) {
            pointwise(dQ_accum,
                      Pointwise_attributes().set_name("identity_dQ").set_mode(PointwiseMode_t::IDENTITY),
                      attributes.outputs[output_names::dQ]);
        }

        // ---------------------output tensor workarounds---------------------------

        // non-virtual softmax_sum is required for below cuDNN 8.9.5
        // non-virtual softmax_sum is passed by the node
        if (detail::get_backend_version() < 8905) {
            softmax_sum->set_is_virtual(false);
            softmax_sum->set_dim({b, h_q, s_q, 1});
            softmax_sum->set_data_type(DataType_t::FLOAT);
            softmax_sum_size = b * h_q * s_q * sizeof(float);
        }

        return {error_code_t::OK, ""};
    }

    virtual int64_t
    get_fe_workspace_size_node() const override final {
        int64_t size = 0;

        size += ((alibi_slopes_size + 15) / 16 * 16);  // align alibi slopes memory to 16 bytes
        size += dQ_accum_size;
        size += softmax_sum_size;

        return size;
    }

    virtual error_t
    workspace_modifications_tensors_(
        std::unordered_map<uid_t, std::tuple<int64_t, int64_t, std::vector<float>>>& workspace_modifications,
        int64_t& offset) const override final {
        if (attributes.alibi_mask) {
            CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(Q, input_names::Q);
            int64_t const h_q     = Q->second->get_dim()[1];
            auto alibi_slopes_vec = detail::get_abili_slope(h_q);
            workspace_modifications.emplace(alibi_slopes->get_uid(), std::make_tuple(0, offset, alibi_slopes_vec));
            int64_t alibi_slopes_size_padded = ((alibi_slopes_size + 15) / 16 * 16);
            offset                           = offset + alibi_slopes_size_padded;
        }

        if (dQ_accum && !dQ_accum->get_is_virtual()) {
            std::vector<float> f_vec = {(float)dQ_accum_size};
            workspace_modifications.emplace(dQ_accum->get_uid(), std::make_tuple(1, offset, f_vec));
            offset = offset + dQ_accum_size;
        }

        if (softmax_sum && !softmax_sum->get_is_virtual()) {
            // There is no requirement for softmax_sum to be memset to 0
            std::vector<float> f_vec = {};
            workspace_modifications.emplace(softmax_sum->get_uid(), std::make_tuple(2, offset, f_vec));
        }

        return {error_code_t::OK, ""};
    }

#ifndef CUDNN_FRONTEND_SKIP_JSON_LIB
    virtual void
    serialize(json& j) const override final {
        j = attributes;
        j.update(R"({"tag": "SDPA_BWD"})"_json);
    }
#endif
};

}  // namespace cudnn_frontend::graph
