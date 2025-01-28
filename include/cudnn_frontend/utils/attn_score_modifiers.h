#pragma once

#include "../graph_interface.h"

namespace cudnn_frontend::graph::attn::score_modifiers {

[[maybe_unused]] inline std::shared_ptr<Tensor_attributes>
causal_mask(std::shared_ptr<Graph> graph, std::shared_ptr<Tensor_attributes> attention_score) {
    return sliding_window_mask(graph,
                               attention_score,
                               DiagonalAlignment_t::TOP_LEFT,
                               {},       // no left bound specified
                               0,        // right bound = 0
                               0,        // s_q does not matter for causal mask
                               0,        // s_kv does not matter for causal mask
                               nullptr,  // s_q does not matter for causal mask
                               nullptr   // s_kv does not matter for causal mask
    );
}

[[maybe_unused]] inline std::shared_ptr<Tensor_attributes>
causal_mask_bottom_right(std::shared_ptr<Graph> graph,
                         std::shared_ptr<Tensor_attributes> attention_score,
                         int64_t s_q,
                         int64_t s_kv,
                         std::shared_ptr<Tensor_attributes> seq_len_q,
                         std::shared_ptr<Tensor_attributes> seq_len_kv) {
    return sliding_window_mask(graph,
                               attention_score,
                               DiagonalAlignment_t::BOTTOM_RIGHT,
                               {},                    // no left bound specified
                               0,                     // right bound = 0
                               s_q,                   // s_q dimension (max Q sequence length)
                               s_kv,                  // s_kv dimension (max KV sequence length)
                               std::move(seq_len_q),  // Actuall Q sequence lengths
                               std::move(seq_len_kv)  // Actual KV sequence lengths
    );
}

[[maybe_unused]] inline std::shared_ptr<Tensor_attributes>
padding_mask(std::shared_ptr<Graph> graph,
             std::shared_ptr<Tensor_attributes> attention_score,
             std::shared_ptr<Tensor_attributes> seq_len_kv,
             std::shared_ptr<Tensor_attributes> seq_len_q) {
    auto row_idx_output = graph->pointwise(attention_score,
                                           Pointwise_attributes()
                                               .set_name("gen_row_idx_padding")
                                               .set_mode(PointwiseMode_t::GEN_INDEX)
                                               .set_axis(2)
                                               .set_compute_data_type(DataType_t::INT32));
    row_idx_output->set_data_type(DataType_t::INT32);

    auto col_idx_output = graph->pointwise(attention_score,
                                           Pointwise_attributes()
                                               .set_name("gen_col_idx_padding")
                                               .set_mode(PointwiseMode_t::GEN_INDEX)
                                               .set_axis(3)
                                               .set_compute_data_type(DataType_t::INT32));
    col_idx_output->set_data_type(DataType_t::INT32);

    auto row_mask_output = graph->pointwise(row_idx_output,
                                            seq_len_q,
                                            Pointwise_attributes()
                                                .set_name("lt_row_sq_padding")
                                                .set_mode(PointwiseMode_t::CMP_LT)
                                                .set_compute_data_type(DataType_t::BOOLEAN));
    row_mask_output->set_data_type(DataType_t::BOOLEAN);

    auto col_mask_output = graph->pointwise(col_idx_output,
                                            seq_len_kv,
                                            Pointwise_attributes()
                                                .set_name("lt_col_skv_padding")
                                                .set_mode(PointwiseMode_t::CMP_LT)
                                                .set_compute_data_type(DataType_t::BOOLEAN));
    col_mask_output->set_data_type(DataType_t::BOOLEAN);

    auto padding_mask_output = graph->pointwise(row_mask_output,
                                                col_mask_output,
                                                Pointwise_attributes()
                                                    .set_name("and_row_col_padding")
                                                    .set_mode(PointwiseMode_t::LOGICAL_AND)
                                                    .set_compute_data_type(DataType_t::BOOLEAN));
    padding_mask_output->set_data_type(DataType_t::BOOLEAN);
    auto negative_inf_padding = std::make_shared<Tensor_attributes>(std::numeric_limits<float>::lowest());

    auto after_padding_mask =
        graph->pointwise(attention_score,
                         negative_inf_padding,
                         padding_mask_output,
                         Pointwise_attributes().set_name("select_padding").set_mode(PointwiseMode_t::BINARY_SELECT));

    return after_padding_mask;
}

[[maybe_unused]] inline std::shared_ptr<Tensor_attributes>
alibi_mask(std::shared_ptr<Graph> graph,
           std::shared_ptr<Tensor_attributes> attention_score,
           std::shared_ptr<Tensor_attributes>& alibi_slopes,
           int64_t query_heads,
           int64_t& alibi_slopes_size) {
    auto row_idx_output = graph->pointwise(attention_score,
                                           Pointwise_attributes()
                                               .set_name("gen_row_idx_alibi")
                                               .set_mode(PointwiseMode_t::GEN_INDEX)
                                               .set_axis(2)
                                               .set_compute_data_type(DataType_t::INT32));
    row_idx_output->set_data_type(DataType_t::INT32);

    auto col_idx_output = graph->pointwise(attention_score,
                                           Pointwise_attributes()
                                               .set_name("gen_col_idx_alibi")
                                               .set_mode(PointwiseMode_t::GEN_INDEX)
                                               .set_axis(3)
                                               .set_compute_data_type(DataType_t::INT32));
    col_idx_output->set_data_type(DataType_t::INT32);

    auto sub_idx_output = graph->pointwise(col_idx_output,
                                           row_idx_output,
                                           Pointwise_attributes()
                                               .set_name("sub_col_row_alibi")
                                               .set_mode(PointwiseMode_t::SUB)
                                               .set_compute_data_type(DataType_t::INT32));
    sub_idx_output->set_data_type(DataType_t::INT32);

    // Multiply by alibi slope
    alibi_slopes = std::make_shared<Tensor_attributes>();
    alibi_slopes->set_dim({1, query_heads, 1, 1}).set_stride({query_heads, 1, 1, 1}).set_data_type(DataType_t::FLOAT);
    alibi_slopes_size = query_heads * sizeof(float);

    auto alibi_mask_output =
        graph->pointwise(sub_idx_output,
                         alibi_slopes,
                         Pointwise_attributes().set_name("mul_slope_alibi").set_mode(PointwiseMode_t::MUL));

    auto after_alibi_mask =
        graph->pointwise(attention_score,
                         alibi_mask_output,
                         Pointwise_attributes().set_name("add_alibi").set_mode(PointwiseMode_t::ADD));
    return after_alibi_mask;
}

[[maybe_unused]] inline std::shared_ptr<Tensor_attributes>
bias(std::shared_ptr<Graph> graph,
     std::shared_ptr<Tensor_attributes> attention_score,
     std::shared_ptr<Tensor_attributes> bias) {
    auto bias_out = graph->pointwise(
        attention_score, bias, Pointwise_attributes().set_name("bias_add").set_mode(PointwiseMode_t::ADD));

    return bias_out;
}

[[maybe_unused]] inline std::shared_ptr<Tensor_attributes>
sliding_window_mask(std::shared_ptr<Graph> graph,
                    std::shared_ptr<Tensor_attributes> attention_score,
                    DiagonalAlignment_t diagonal_alignment,
                    std::optional<int64_t> left_bound,
                    std::optional<int64_t> right_bound,
                    int64_t s_q,
                    int64_t s_kv,
                    std::shared_ptr<Tensor_attributes> s_q_ptr,
                    std::shared_ptr<Tensor_attributes> s_kv_ptr) {
    std::shared_ptr<Tensor_attributes> return_mask = attention_score;

    // Note: the right and left bound subtrees can be constructed in different ways as well that yield functionally
    // correct results. However, for performance reasons in the cuDNN backend they are organized as they are. Be
    // cautious of performance when editting.

    // Set the right bound
    if (right_bound.has_value()) {
        auto row_index_attributes =
            Pointwise_attributes().set_name("gen_row_index").set_mode(PointwiseMode_t::GEN_INDEX).set_axis(2);
        std::shared_ptr<Tensor_attributes> row_index = graph->pointwise(attention_score, row_index_attributes);
        row_index->set_data_type(DataType_t::INT32);

        auto col_index_attributes =
            Pointwise_attributes().set_name("gen_col_index").set_mode(PointwiseMode_t::GEN_INDEX).set_axis(3);
        std::shared_ptr<Tensor_attributes> col_index = graph->pointwise(attention_score, col_index_attributes);
        col_index->set_data_type(DataType_t::INT32);

        if (diagonal_alignment == DiagonalAlignment_t::BOTTOM_RIGHT) {
            row_index = graph->pointwise(
                row_index,
                // Use actual sequence lengths if they are provided
                s_kv_ptr != nullptr ? s_kv_ptr : std::make_shared<Tensor_attributes>(static_cast<int32_t>(s_kv)),
                Pointwise_attributes()
                    .set_name("row_idx_add_skv")
                    .set_mode(PointwiseMode_t::ADD)
                    .set_compute_data_type(DataType_t::INT32));

            row_index->set_data_type(DataType_t::INT32);

            row_index = graph->pointwise(
                row_index,
                // Use actual sequence lengths if they are provided
                s_q_ptr != nullptr ? s_q_ptr : std::make_shared<Tensor_attributes>(static_cast<int32_t>(s_q)),
                Pointwise_attributes()
                    .set_name("row_idx_add_skv_sub_sq")
                    .set_mode(PointwiseMode_t::SUB)
                    .set_compute_data_type(DataType_t::INT32));
            row_index->set_data_type(DataType_t::INT32);
        }

        // Shift the diagonal in case there is a non-zero right bound
        if (right_bound.value() != 0) {
            row_index = graph->pointwise(row_index,
                                         std::make_shared<Tensor_attributes>(static_cast<int32_t>(right_bound.value())),
                                         Pointwise_attributes()
                                             .set_name("row_idx_add_skv_sub_sq_add_right_bound")
                                             .set_mode(PointwiseMode_t::ADD)
                                             .set_compute_data_type(DataType_t::INT32));
            row_index->set_data_type(DataType_t::INT32);
        }

        auto const& bool_mask = graph->pointwise(row_index,
                                                 col_index,
                                                 Pointwise_attributes()
                                                     .set_name("row_greater_than_col")
                                                     .set_mode(PointwiseMode_t::CMP_GE)
                                                     .set_compute_data_type(DataType_t::BOOLEAN));
        bool_mask->set_data_type(DataType_t::BOOLEAN);

        return_mask =
            graph->pointwise(attention_score,
                             std::make_shared<Tensor_attributes>(std::numeric_limits<float>::lowest()),
                             bool_mask,
                             Pointwise_attributes().set_name("binary_select").set_mode(PointwiseMode_t::BINARY_SELECT));
    }

    // Set the left bound
    if (left_bound.has_value()) {
        auto row_index_attributes =
            Pointwise_attributes().set_name("gen_row_index").set_mode(PointwiseMode_t::GEN_INDEX).set_axis(2);
        std::shared_ptr<Tensor_attributes> row_index_output = graph->pointwise(return_mask, row_index_attributes);

        auto col_index_attributes =
            Pointwise_attributes().set_name("gen_col_index").set_mode(PointwiseMode_t::GEN_INDEX).set_axis(3);
        std::shared_ptr<Tensor_attributes> col_index_output = graph->pointwise(return_mask, col_index_attributes);
        // When the diagonal is top left aligned: setup a graph so we can compare column + window_size > row
        // All elements for which column + window_size > row, will be retained, all others will be masked out
        // Note that here and following sections, row refers to the s_q index and column refers to the s_kv index in
        // the s_q x s_kv masking matrix
        if (diagonal_alignment == DiagonalAlignment_t::TOP_LEFT) {
            // sliding window length parameter should be of float type
            auto sliding_window_length = std::make_shared<Tensor_attributes>((float)left_bound.value());
            auto add_col_attributes    = Pointwise_attributes()
                                          .set_name("col+window")
                                          .set_mode(PointwiseMode_t::ADD)
                                          .set_compute_data_type(DataType_t::FLOAT)
                                          .set_axis(3);

            col_index_output = graph->pointwise(col_index_output, sliding_window_length, add_col_attributes);
        }
        // With bottom right diagonal alignment, we need to shift the diagonal.
        // Setup a graph so we can compare column + window_size - (s_kv - s_q) > row
        // Optimization with fixed sequence lengths: single pointwise addition for the left-hand of the comparison
        // Again, all elements satisfying the comparison will be retained.
        else if (s_kv_ptr == nullptr && s_q_ptr == nullptr) {
            auto sliding_window_length = std::make_shared<Tensor_attributes>((float)(left_bound.value() - s_kv + s_q));
            auto add_col_attributes    = Pointwise_attributes()
                                          .set_name("col+window-skv+sq")
                                          .set_mode(PointwiseMode_t::ADD)
                                          .set_compute_data_type(DataType_t::FLOAT)
                                          .set_axis(3);

            col_index_output = graph->pointwise(col_index_output, sliding_window_length, add_col_attributes);
        }
        // With bottom right diagonal alignment: general case when at least one of Q and KV have variable sequence
        // lengths.
        // Setup a graph so we can compare column + window_size - (s_k[i] - s_q[i]) > row  for each batch i
        // Also here, all elements satisfying the comparison will be retained.
        else {
            col_index_output->set_data_type(DataType_t::INT32);
            row_index_output->set_data_type(DataType_t::INT32);

            auto sliding_window_length = std::make_shared<Tensor_attributes>((int32_t)left_bound.value());
            auto add_col_attributes    = Pointwise_attributes()
                                          .set_name("col+window")
                                          .set_mode(PointwiseMode_t::ADD)
                                          .set_compute_data_type(DataType_t::INT32)
                                          .set_axis(3);

            col_index_output = graph->pointwise(col_index_output, sliding_window_length, add_col_attributes);
            col_index_output->set_data_type(DataType_t::INT32);

            if (s_kv_ptr) {
                col_index_output = graph->pointwise(col_index_output,
                                                    s_kv_ptr,
                                                    Pointwise_attributes()
                                                        .set_name("col+window-skv")
                                                        .set_mode(PointwiseMode_t::SUB)
                                                        .set_compute_data_type(DataType_t::INT32));
            } else {
                col_index_output = graph->pointwise(col_index_output,
                                                    std::make_shared<Tensor_attributes>(static_cast<int32_t>(s_kv)),
                                                    Pointwise_attributes()
                                                        .set_name("col+window-skv")
                                                        .set_mode(PointwiseMode_t::SUB)
                                                        .set_compute_data_type(DataType_t::INT32));
            }
            col_index_output->set_data_type(DataType_t::INT32);

            if (s_q_ptr) {
                col_index_output = graph->pointwise(col_index_output,
                                                    s_q_ptr,
                                                    Pointwise_attributes()
                                                        .set_name("col+window-skv+sq")
                                                        .set_mode(PointwiseMode_t::ADD)
                                                        .set_compute_data_type(DataType_t::INT32));
            } else {
                col_index_output = graph->pointwise(col_index_output,
                                                    std::make_shared<Tensor_attributes>(static_cast<int32_t>(s_q)),
                                                    Pointwise_attributes()
                                                        .set_name("col+window-skv+sq")
                                                        .set_mode(PointwiseMode_t::ADD)
                                                        .set_compute_data_type(DataType_t::INT32));
            }
            col_index_output->set_data_type(DataType_t::INT32);
        }

        auto greater_than_attributes =
            Pointwise_attributes().set_mode(PointwiseMode_t::CMP_GT).set_compute_data_type(DataType_t::BOOLEAN);

        if (diagonal_alignment == DiagonalAlignment_t::TOP_LEFT) {
            greater_than_attributes.set_name("col+ws>row");
        } else {
            greater_than_attributes.set_name("col+window-skv+sq>row");
        }

        auto swa_comparison_output = graph->pointwise(col_index_output, row_index_output, greater_than_attributes);
        swa_comparison_output->set_data_type(DataType_t::BOOLEAN);

        // Lower attributes to binary select attributes
        auto negative_inf_swa = std::make_shared<Tensor_attributes>(-1024.0f * 1024.0f * 1024.0f);

        auto binary_select_attributes =
            Pointwise_attributes().set_name("binary_select").set_mode(PointwiseMode_t::BINARY_SELECT);

        return_mask = graph->pointwise(return_mask, negative_inf_swa, swa_comparison_output, binary_select_attributes);
    }
    return return_mask;
}

class Softcap {
   private:
    // saved tensors in fprop to be used in bprop

    std::shared_ptr<Tensor_attributes> before_tanh_activation;

   public:
    std::shared_ptr<Tensor_attributes>
    forward(std::shared_ptr<Graph> graph,
            std::shared_ptr<Tensor_attributes> attention_score,
            std::shared_ptr<Tensor_attributes> soft_cap_scalar) {
        before_tanh_activation =
            graph->pointwise(attention_score,
                             soft_cap_scalar,
                             Pointwise_attributes().set_name("div_by_soft_cap").set_mode(PointwiseMode_t::DIV));

        auto tanh_out = graph->pointwise(
            before_tanh_activation, Pointwise_attributes().set_name("activation").set_mode(PointwiseMode_t::TANH_FWD));

        auto out = graph->pointwise(tanh_out,
                                    soft_cap_scalar,
                                    Pointwise_attributes().set_name("mul_by_soft_cap").set_mode(PointwiseMode_t::MUL));

        return out;
    }

    std::shared_ptr<Tensor_attributes>
    backward(std::shared_ptr<Graph> graph,
             std::shared_ptr<Tensor_attributes> attention_score,
             std::shared_ptr<Tensor_attributes> soft_cap_scalar) {
        auto mul_out =
            graph->pointwise(attention_score,
                             soft_cap_scalar,
                             Pointwise_attributes().set_name("mul_by_soft_cap_bprop").set_mode(PointwiseMode_t::MUL));

        auto tanh_out = graph->pointwise(mul_out,
                                         before_tanh_activation,
                                         Pointwise_attributes().set_name("dtanh").set_mode(PointwiseMode_t::TANH_BWD));

        auto out =
            graph->pointwise(tanh_out,
                             soft_cap_scalar,
                             Pointwise_attributes().set_name("div_by_soft_cap_bprop").set_mode(PointwiseMode_t::DIV));

        return out;
    }
};

}  // namespace cudnn_frontend::graph::attn::score_modifiers