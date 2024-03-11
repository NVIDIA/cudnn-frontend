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

class ScaledDotProductAttentionNode : public NodeCRTP<ScaledDotProductAttentionNode> {
   public:
    std::shared_ptr<Tensor_attributes> negative_inf;

    Scaled_dot_product_attention_attributes options;

    ScaledDotProductAttentionNode(Scaled_dot_product_attention_attributes&& options_, detail::Context const& context)
        : NodeCRTP(context), options(std::move(options_)) {}

    Type
    getType() override final {
        return Type::COMPOSITE;
    }

    error_t
    pre_validate_node() const override final {
        getLogger() << "[cudnn_frontend] INFO: "
                    << "Validating ScaledDotProductAttentionNode " << options.name << "..." << std::endl;

        if (options.is_inference.has_value() == false) {
            auto status         = error_code_t::ATTRIBUTE_NOT_SET;
            std::string message = "is_infernece attribute not set.";
            return {status, message};
        }

        if (options.dropout_probability.has_value() && options.dropout_probability.value() == 1) {
            auto status         = error_code_t::ATTRIBUTE_NOT_SET;
            std::string message = "Dropout probability cannot be 1 as corresponding scale wont be well formed.";
            return {status, message};
        }

        if (options.dropout_probability.has_value() && options.inputs.Dropout_mask) {
            auto status         = error_code_t::ATTRIBUTE_NOT_SET;
            std::string message = "Both, dropout probability and custom dropout mask, cannot be set together.";
            return {status, message};
        }

        return {error_code_t::OK, ""};
    }

    error_t
    expand_and_infer_properties_node() override final {
        getLogger() << "[cudnn_frontend] INFO: Inferrencing properties for Scaled_dot_product_attention node "
                    << options.name << "..." << std::endl;

        options.fill_from_context(context);

        // Gather dims to fill properties of virtual tensors
        auto const& q_dim = options.inputs.Q->get_dim();
        auto b            = q_dim[0];
        auto h            = q_dim[1];
        auto s_q          = q_dim[2];
        auto d            = q_dim[3];
        auto const& k_dim = options.inputs.K->get_dim();
        auto s_kv         = k_dim[3];

        std::shared_ptr<Tensor_attributes> last_output;

        // User does not create tensor for scale k, so create it internally
        negative_inf = std::make_shared<Tensor_attributes>();
        negative_inf->set_dim({1, 1, 1, 1})
            .set_stride({1, 1, 1, 1})
            .set_is_pass_by_value(true)
            .set_data_type(DataType_t::FLOAT);

        // Optional scale
        if (options.inputs.Attn_scale) {
            // Lower options to scale options
            Pointwise_attributes scale_attributes;
            scale_attributes.set_mode(PointwiseMode_t::MUL).set_name("attn_scale");
            scale_attributes.inputs.IN_0 = options.inputs.K;
            scale_attributes.inputs.IN_1 = options.inputs.Attn_scale;
            last_output = scale_attributes.outputs.OUT_0 = std::make_shared<Tensor_attributes>();
            scale_attributes.outputs.OUT_0->set_is_virtual(true);
            auto scale_node = std::make_unique<PointwiseNode>(std::move(scale_attributes), context);
            sub_nodes.emplace_back(std::move(scale_node));

            // Requirement by cudnn backend to take in bmm1 bType as i/o type.
            last_output->set_data_type(options.inputs.K->get_data_type());
        } else {
            last_output = options.inputs.K;
        }

        // Lower options to bmm1 options
        Matmul_attributes bmm1_attributes;
        bmm1_attributes.set_name("bmm1");
        bmm1_attributes.inputs.A          = options.inputs.Q;
        bmm1_attributes.inputs.B          = last_output;
        bmm1_attributes.inputs.M_override = options.inputs.SEQ_LEN_Q;
        bmm1_attributes.inputs.N_override = options.inputs.SEQ_LEN_KV;
        last_output = bmm1_attributes.outputs.C = std::make_shared<Tensor_attributes>();
        // Set dims and strides for output of bmm1 as user never sets them
        last_output->set_is_virtual(true)
            .set_dim({b, h, s_q, s_kv})
            .set_stride({h * s_q * s_kv, s_q * s_kv, s_kv, 1})
            .fill_from_context(context);

        auto bmm1_node = std::make_unique<MatmulNode>(std::move(bmm1_attributes), context);
        sub_nodes.emplace_back(std::move(bmm1_node));

        if (options.inputs.Bias) {
            // Lower options to add options
            Pointwise_attributes add_attributes;
            add_attributes.set_name("bias");
            add_attributes.set_mode(PointwiseMode_t::ADD);
            add_attributes.inputs.IN_0 = last_output;
            add_attributes.inputs.IN_1 = options.inputs.Bias;
            last_output = add_attributes.outputs.OUT_0 = std::make_shared<Tensor_attributes>();
            add_attributes.outputs.OUT_0->set_is_virtual(true);
            auto add_node = std::make_unique<PointwiseNode>(std::move(add_attributes), context);
            sub_nodes.emplace_back(std::move(add_node));
        }

        if (options.padding_mask) {
            // Lower options to generate row index options
            Pointwise_attributes row_index_attributes;
            row_index_attributes.set_name("gen_row_index");
            row_index_attributes.set_mode(PointwiseMode_t::GEN_INDEX).set_axis(2);
            row_index_attributes.inputs.IN_0 = last_output;
            auto row_index = row_index_attributes.outputs.OUT_0 = std::make_shared<Tensor_attributes>();
            row_index_attributes.outputs.OUT_0->set_is_virtual(true);
            auto row_index_node = std::make_unique<PointwiseNode>(std::move(row_index_attributes), context);
            sub_nodes.emplace_back(std::move(row_index_node));

            // Lower options to generate col index options
            Pointwise_attributes col_index_attributes;
            col_index_attributes.set_name("gen_col_index");
            col_index_attributes.set_mode(PointwiseMode_t::GEN_INDEX).set_axis(3);
            col_index_attributes.inputs.IN_0 = last_output;
            auto col_index = col_index_attributes.outputs.OUT_0 = std::make_shared<Tensor_attributes>();
            col_index_attributes.outputs.OUT_0->set_is_virtual(true);
            auto col_index_node = std::make_unique<PointwiseNode>(std::move(col_index_attributes), context);
            sub_nodes.emplace_back(std::move(col_index_node));

            // Lower options to less than row options
            Pointwise_attributes less_than_row_attributes;
            less_than_row_attributes.set_name("cmp_less_than_row");
            less_than_row_attributes.set_mode(PointwiseMode_t::CMP_LT);
            less_than_row_attributes.inputs.IN_0 = row_index;
            less_than_row_attributes.inputs.IN_1 = options.inputs.SEQ_LEN_Q;
            auto less_than_row = less_than_row_attributes.outputs.OUT_0 = std::make_shared<Tensor_attributes>();
            less_than_row_attributes.outputs.OUT_0->set_is_virtual(true);
            auto less_than_row_node = std::make_unique<PointwiseNode>(std::move(less_than_row_attributes), context);
            sub_nodes.emplace_back(std::move(less_than_row_node));

            // Lower options to less than col options
            Pointwise_attributes less_than_col_attributes;
            less_than_col_attributes.set_name("cmp_less_than_col");
            less_than_col_attributes.set_mode(PointwiseMode_t::CMP_LT);
            less_than_col_attributes.inputs.IN_0 = col_index;
            less_than_col_attributes.inputs.IN_1 = options.inputs.SEQ_LEN_KV;
            auto less_than_col = less_than_col_attributes.outputs.OUT_0 = std::make_shared<Tensor_attributes>();
            less_than_col_attributes.outputs.OUT_0->set_is_virtual(true);
            auto less_than_col_node = std::make_unique<PointwiseNode>(std::move(less_than_col_attributes), context);
            sub_nodes.emplace_back(std::move(less_than_col_node));

            // Lower options to logical and options
            Pointwise_attributes logical_and_attributes;
            logical_and_attributes.set_name("logical_and");
            logical_and_attributes.set_mode(PointwiseMode_t::LOGICAL_AND).set_compute_data_type(DataType_t::BOOLEAN);
            logical_and_attributes.inputs.IN_0 = less_than_row;
            logical_and_attributes.inputs.IN_1 = less_than_col;
            auto mask = logical_and_attributes.outputs.OUT_0 = std::make_shared<Tensor_attributes>();
            logical_and_attributes.outputs.OUT_0->set_is_virtual(true);
            auto logical_and_node = std::make_unique<PointwiseNode>(std::move(logical_and_attributes), context);
            sub_nodes.emplace_back(std::move(logical_and_node));

            if (options.causal_mask) {
                // Lower options to greater than options
                Pointwise_attributes greater_than_attributes;
                greater_than_attributes.set_name("row_greater_than_col");
                greater_than_attributes.set_mode(PointwiseMode_t::CMP_GE);
                greater_than_attributes.inputs.IN_0 = row_index;
                greater_than_attributes.inputs.IN_1 = col_index;
                auto row_greater_col = greater_than_attributes.outputs.OUT_0 = std::make_shared<Tensor_attributes>();
                greater_than_attributes.outputs.OUT_0->set_is_virtual(true);
                auto greater_than_node = std::make_unique<PointwiseNode>(std::move(greater_than_attributes), context);
                sub_nodes.emplace_back(std::move(greater_than_node));

                // Lower options to logical and options
                Pointwise_attributes logical_and_attributes_1;
                logical_and_attributes_1.set_name("logical_and");
                logical_and_attributes_1.set_mode(PointwiseMode_t::LOGICAL_AND)
                    .set_compute_data_type(DataType_t::BOOLEAN);
                logical_and_attributes_1.inputs.IN_0 = mask;
                logical_and_attributes_1.inputs.IN_1 = row_greater_col;
                mask = logical_and_attributes_1.outputs.OUT_0 = std::make_shared<Tensor_attributes>();
                logical_and_attributes_1.outputs.OUT_0->set_is_virtual(true);
                auto logical_and_node_1 = std::make_unique<PointwiseNode>(std::move(logical_and_attributes_1), context);
                sub_nodes.emplace_back(std::move(logical_and_node_1));
            }

            // Lower options to binary select options
            Pointwise_attributes binary_select_attributes;
            binary_select_attributes.set_name("binary_select");
            binary_select_attributes.set_mode(PointwiseMode_t::BINARY_SELECT);
            binary_select_attributes.inputs.IN_0 = last_output;
            binary_select_attributes.inputs.IN_1 = negative_inf;
            binary_select_attributes.inputs.IN_2 = mask;
            last_output = binary_select_attributes.outputs.OUT_0 = std::make_shared<Tensor_attributes>();
            binary_select_attributes.outputs.OUT_0->set_is_virtual(true);
            auto binary_select_node = std::make_unique<PointwiseNode>(std::move(binary_select_attributes), context);
            sub_nodes.emplace_back(std::move(binary_select_node));
        }

        // Lower options to softmax options
        Softmax_attributes softmax_attributes;
        softmax_attributes.set_name("softmax");
        softmax_attributes.use_stats = false;  // As this is non-flash attention
        softmax_attributes.inputs.P  = last_output;
        // Use tensor provided by Graph when real S
        if (options.is_inference.value() == true) {
            last_output = softmax_attributes.outputs.S = std::make_shared<Tensor_attributes>();
            softmax_attributes.outputs.S->set_is_virtual(true);
            auto softmax_node = std::make_unique<SoftmaxNode>(std::move(softmax_attributes), context);
            sub_nodes.emplace_back(std::move(softmax_node));
        } else {
            // Two cases for training: dropout present or not
            bool const dropout_present = options.dropout_probability.has_value() || options.inputs.Dropout_mask;
            if (dropout_present) {
                last_output = softmax_attributes.outputs.S = std::make_shared<Tensor_attributes>();
                softmax_attributes.outputs.S->set_is_virtual(true);
                auto softmax_node = std::make_unique<SoftmaxNode>(std::move(softmax_attributes), context);
                sub_nodes.emplace_back(std::move(softmax_node));

                std::shared_ptr<Tensor_attributes> mask_output;
                if (options.dropout_probability.has_value()) {
                    auto const p = options.dropout_probability.value();

                    // Lower options to rng options
                    Rng_attributes rng_attributes;
                    rng_attributes.set_name("rng");
                    rng_attributes.set_distribution(RngDistribution_t::BERNOULLI)
                        .set_seed(options.seed)
                        .set_bernoulli_probability(p);
                    mask_output = rng_attributes.outputs.Y = std::make_shared<Tensor_attributes>();
                    rng_attributes.outputs.Y->set_is_virtual(true);
                    auto rng_node = std::make_unique<RngNode>(std::move(rng_attributes), context);
                    sub_nodes.emplace_back(std::move(rng_node));

                    // kickstarting rng Y and subsequant MUL infer_properties
                    mask_output->set_dim({b, h, s_q, s_kv}).set_stride({h * s_q * s_kv, s_q * s_kv, s_kv, 1});

                    // Compute dropout scale
                    options.dropout_scale = (1.f / (1.f - p));
                } else {
                    mask_output = options.inputs.Dropout_mask;
                }

                // Lower options to mask options
                Pointwise_attributes mask_attributes;
                mask_attributes.set_name("dropout_mask_mul");
                mask_attributes.set_mode(PointwiseMode_t::MUL);
                mask_attributes.inputs.IN_0 = last_output;
                mask_attributes.inputs.IN_1 = mask_output;
                last_output = mask_attributes.outputs.OUT_0 = options.outputs.S;
                auto mask_node = std::make_unique<PointwiseNode>(std::move(mask_attributes), context);
                sub_nodes.emplace_back(std::move(mask_node));

            } else {
                last_output = softmax_attributes.outputs.S = options.outputs.S;
                auto softmax_node = std::make_unique<SoftmaxNode>(std::move(softmax_attributes), context);
                sub_nodes.emplace_back(std::move(softmax_node));
            }

            // Requirement by cudnn backend as output is a special swizzled format.
            last_output->set_reordering_type(cudnn_frontend::TensorReordering_t::F16x16);
        }

        // Inference or not, dropout or not, always put a scale.
        // Default value 1.f. Will have no perf impact
        // Lower options to dropout_scale options
        if ((options.is_inference.value() == true) || (options.dropout_probability.has_value())) {
            // Data type is i/o type
            options.inputs.Dropout_scale = std::make_shared<Tensor_attributes>();
            options.inputs.Dropout_scale->set_dim({1, 1, 1, 1})
                .set_stride({1, 1, 1, 1})
                .set_is_pass_by_value(true)
                .set_data_type(options.inputs.Q->get_data_type());
        }
        Pointwise_attributes dropout_scale_attributes;
        dropout_scale_attributes.set_name("dropout_scale");
        dropout_scale_attributes.set_mode(PointwiseMode_t::MUL);
        dropout_scale_attributes.inputs.IN_0 = last_output;
        dropout_scale_attributes.inputs.IN_1 = options.inputs.Dropout_scale;
        last_output = dropout_scale_attributes.outputs.OUT_0 = std::make_shared<Tensor_attributes>();
        dropout_scale_attributes.outputs.OUT_0->set_is_virtual(true);
        auto dropout_scale_node = std::make_unique<PointwiseNode>(std::move(dropout_scale_attributes), context);
        sub_nodes.emplace_back(std::move(dropout_scale_node));

        // Lower options to bmm2 options
        Matmul_attributes bmm2_attributes;
        bmm2_attributes.set_name("bmm2");
        // Requirement by cudnn backend to take in bmm2 aType as i/o type.
        last_output->set_data_type(DataType_t::HALF);
        bmm2_attributes.inputs.A          = last_output;
        bmm2_attributes.inputs.B          = options.inputs.V;
        bmm2_attributes.inputs.M_override = options.inputs.SEQ_LEN_Q;
        bmm2_attributes.inputs.K_override = options.inputs.SEQ_LEN_KV;
        bmm2_attributes.outputs.C         = options.outputs.O;
        auto bmm2_node                    = std::make_unique<MatmulNode>(std::move(bmm2_attributes), context);
        sub_nodes.emplace_back(std::move(bmm2_node));

        // Set dims and strides if user did not
        if (options.outputs.O->get_dim().empty()) {
            // TODO: mha node needs to set it?
            options.outputs.O->set_dim({b, h, s_q, d}).set_stride({s_q * h * d, d, h * d, 1});
        }

        return {error_code_t::OK, ""};
    }

    virtual error_t
    pass_by_value_tensors_(std::map<uid_t, pass_by_values_t>& tensor_to_pass_by_value) const override final {
        half dropout_scale_value = options.dropout_scale;
        tensor_to_pass_by_value.emplace(options.inputs.Dropout_scale->get_uid(), dropout_scale_value);

        float negative_inf_value = std::numeric_limits<float>::min();
        tensor_to_pass_by_value.emplace(negative_inf->get_uid(), negative_inf_value);

        return {error_code_t::OK, ""};
    }
};

}  // namespace cudnn_frontend::graph