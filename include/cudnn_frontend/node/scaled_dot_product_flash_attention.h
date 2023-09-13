#pragma once

#include "../../cudnn_frontend_Heuristics.h"
#include "../../cudnn_frontend_Logging.h"

#include "../cudnn_frontend_graph_helpers.h"
#include "../cudnn_frontend_node_interface.h"

#include "matmul.h"
#include "pointwise.h"
#include "rng.h"
#include "softmax.h"

namespace cudnn_frontend::graph {

class ScaledDotProductFlashAttentionNode : public INode {
    std::shared_ptr<Tensor_attributes> rng_output;
    std::shared_ptr<Tensor_attributes> dropout_scale;
    std::shared_ptr<Tensor_attributes> negative_inf_causal;
    std::shared_ptr<Tensor_attributes> negative_inf_padding;
    std::shared_ptr<Tensor_attributes> alibi_slopes;

   public:
    Scaled_dot_product_flash_attention_attributes options;

    ScaledDotProductFlashAttentionNode(Scaled_dot_product_flash_attention_attributes&& options_,
                                       detail::Context const& context)
        : INode(context), options(std::move(options_)) {}

    Type
    getType() override final {
        return Type::COMPOSITE;
    }

    error_t
    validate_node() const override final {
        getLogger() << "[cudnn_frontend] INFO: "
                    << "Validating ScaledDotProductFlashAttentionNode " << options.name << "..." << std::endl;

        if (options.is_inference.has_value() == false) {
            auto status         = error_code_t::ATTRIBUTE_NOT_SET;
            std::string message = "[cudnn_frontend] ERROR: is_infernece attribute not set.";
            getLogger() << message << std::endl;
            return {status, message};
        }

        if (options.dropout_probability.has_value() && options.inputs.Dropout_mask) {
            auto status = error_code_t::ATTRIBUTE_NOT_SET;
            std::string message =
                "[cudnn_frontend] ERROR: Using both, custom dropout mask and internal-mask generation using dropout "
                "probability, is ill-formed.";
            getLogger() << message << std::endl;
            return {status, message};
        }

        if (options.dropout_probability.has_value() && options.dropout_probability.value() == 1.0) {
            auto status = error_code_t::ATTRIBUTE_NOT_SET;
            std::string message =
                "[cudnn_frontend] ERROR: Dropout probability cannot be 1 as corresponding scale wont be well formed.";
            getLogger() << message << std::endl;
            return {status, message};
        }

        if (context.get_intermediate_data_type() == DataType_t::NOT_SET) {
            auto status = error_code_t::ATTRIBUTE_NOT_SET;
            std::string message =
                "[cudnn_frontend] ERROR: Intermediate tensor data type needs to be set as internal tensors require it.";
            getLogger() << message << std::endl;
            return {status, message};
        }

        if (options.padding_mask && (!(options.inputs.SEQ_LEN_Q) || !(options.inputs.SEQ_LEN_KV))) {
            auto status         = error_code_t::ATTRIBUTE_NOT_SET;
            std::string message = "[cudnn_frontend] ERROR: Padding mask requires seq_len_q and seq_len_kv to be set.";
            getLogger() << message << std::endl;
            return {status, message};
        }

        if ((!options.padding_mask) && (options.inputs.SEQ_LEN_Q || options.inputs.SEQ_LEN_KV)) {
            auto status = error_code_t::ATTRIBUTE_NOT_SET;
            std::string message =
                "[cudnn_frontend] ERROR: seq_len_q and seq_len_kv needs to be set only if padding mask is enabled.";
            getLogger() << message << std::endl;
            return {status, message};
        }

        return {error_code_t::OK, ""};
    }

    error_t
    infer_properties_node() override final {
        getLogger() << "[cudnn_frontend] INFO: Inferrencing properties for Scaled_dot_product_flash_attention node  "
                    << options.name << "..." << std::endl;

        // DO NOT REMOVE
        // input data type is needed for:
        // - aType of bmm2
        // - dropout scale in pre 8.9.3
        options.fill_from_context(context);

        // Gather dims to fill properties of virtual tensors
        auto const& q_dim = options.inputs.Q->get_dim();
        auto b            = q_dim[0];
        auto h            = q_dim[1];
        auto s_q          = q_dim[2];
        auto const& k_dim = options.inputs.K->get_dim();
        auto s_kv         = k_dim[3];
        auto const& v_dim = options.inputs.V->get_dim();
        auto d_v          = v_dim[3];

        std::shared_ptr<Tensor_attributes> last_output;

        // Lower options to bmm1 options
        auto bmm1_output = std::make_shared<Tensor_attributes>();
        bmm1_output
            ->set_is_virtual(true)
            // Setting dims and strides as pointwise op wont have knowledge of how to do it for mha.
            .set_dim({b, h, s_q, s_kv})
            .set_stride({h * s_q * s_kv, s_q * s_kv, s_kv, 1});

        Matmul_attributes bmm1_attributes;
        bmm1_attributes.set_name("bmm1");
        bmm1_attributes.inputs.A          = options.inputs.Q;
        bmm1_attributes.inputs.B          = options.inputs.K;
        bmm1_attributes.inputs.M_override = options.inputs.SEQ_LEN_Q;
        bmm1_attributes.inputs.N_override = options.inputs.SEQ_LEN_KV;
        last_output = bmm1_attributes.outputs.C = bmm1_output;
        auto bmm1_node                          = std::make_unique<MatmulNode>(std::move(bmm1_attributes), context);
        sub_nodes.emplace_back(std::move(bmm1_node));

        // Optional scale
        if (options.inputs.Attn_scale) {
            // Lower options to scale options
            auto attn_scale_output = std::make_shared<Tensor_attributes>();
            attn_scale_output->set_is_virtual(true);

            Pointwise_attributes scale_attributes;
            scale_attributes.set_name("attn_scale");
            scale_attributes.set_mode(PointwiseMode_t::MUL);
            scale_attributes.inputs.IN_0 = last_output;
            scale_attributes.inputs.IN_1 = options.inputs.Attn_scale;
            last_output = scale_attributes.outputs.OUT_0 = attn_scale_output;
            auto scale_node = std::make_unique<PointwiseNode>(std::move(scale_attributes), context);
            sub_nodes.emplace_back(std::move(scale_node));
        }

        // Optional bias
        if (options.inputs.Bias) {
            // Lower options to add options
            auto bias_output = std::make_shared<Tensor_attributes>();
            bias_output->set_is_virtual(true);

            Pointwise_attributes add_attributes;
            add_attributes.set_name("bias");
            add_attributes.set_mode(PointwiseMode_t::ADD);
            add_attributes.inputs.IN_0 = last_output;
            add_attributes.inputs.IN_1 = options.inputs.Bias;
            last_output = add_attributes.outputs.OUT_0 = bias_output;
            auto add_node = std::make_unique<PointwiseNode>(std::move(add_attributes), context);
            sub_nodes.emplace_back(std::move(add_node));
        }

        if (options.alibi_mask) {
            // Lower options to generate row index options
            auto row_index_output = std::make_shared<Tensor_attributes>();
            row_index_output->set_is_virtual(true).set_data_type(DataType_t::INT32);

            Pointwise_attributes row_index_attributes;
            row_index_attributes.set_name("gen_row_index")
                .set_mode(PointwiseMode_t::GEN_INDEX)
                .set_axis(2)
                .set_compute_data_type(DataType_t::INT32);
            row_index_attributes.inputs.IN_0   = last_output;
            row_index_attributes.outputs.OUT_0 = row_index_output;
            auto row_index_node = std::make_unique<PointwiseNode>(std::move(row_index_attributes), context);
            sub_nodes.emplace_back(std::move(row_index_node));

            // Lower options to generate col index options
            auto col_index_output = std::make_shared<Tensor_attributes>();
            col_index_output->set_is_virtual(true).set_data_type(DataType_t::INT32);

            Pointwise_attributes col_index_attributes;
            col_index_attributes.set_name("gen_col_index")
                .set_mode(PointwiseMode_t::GEN_INDEX)
                .set_axis(3)
                .set_compute_data_type(DataType_t::INT32);
            col_index_attributes.inputs.IN_0   = last_output;
            col_index_attributes.outputs.OUT_0 = col_index_output;
            auto col_index_node = std::make_unique<PointwiseNode>(std::move(col_index_attributes), context);
            sub_nodes.emplace_back(std::move(col_index_node));

            // Lower options to sub options
            auto sub_output = std::make_shared<Tensor_attributes>();
            sub_output->set_is_virtual(true).set_data_type(DataType_t::INT32);

            Pointwise_attributes sub_attributes;
            sub_attributes.set_name("sub").set_mode(PointwiseMode_t::SUB).set_compute_data_type(DataType_t::INT32);
            sub_attributes.inputs.IN_0   = col_index_output;
            sub_attributes.inputs.IN_1   = row_index_output;
            sub_attributes.outputs.OUT_0 = sub_output;
            auto sub_node                = std::make_unique<PointwiseNode>(std::move(sub_attributes), context);
            sub_nodes.emplace_back(std::move(sub_node));

            // Multiply by alibi slope
            alibi_slopes = std::make_shared<Tensor_attributes>();
            alibi_slopes->set_dim({1, h, 1, 1})
                .set_stride({h, 1, 1, 1})
                // Hard code data type float as FE itself will compute and place in variant pack later
                .set_data_type(DataType_t::FLOAT);

            auto alibi_mask = std::make_shared<Tensor_attributes>();
            alibi_mask->set_is_virtual(true);

            Pointwise_attributes mul_attributes;
            mul_attributes.set_name("mul").set_mode(PointwiseMode_t::MUL);
            mul_attributes.inputs.IN_0   = sub_output;
            mul_attributes.inputs.IN_1   = alibi_slopes;
            mul_attributes.outputs.OUT_0 = alibi_mask;
            auto mul_node                = std::make_unique<PointwiseNode>(std::move(mul_attributes), context);
            sub_nodes.emplace_back(std::move(mul_node));

            // Add alibi_mask
            auto add_output = std::make_shared<Tensor_attributes>();
            add_output->set_is_virtual(true);

            Pointwise_attributes add_attributes;
            add_attributes.set_name("add").set_mode(PointwiseMode_t::ADD);
            add_attributes.inputs.IN_0 = last_output;
            add_attributes.inputs.IN_1 = alibi_mask;
            last_output = add_attributes.outputs.OUT_0 = add_output;
            auto add_node = std::make_unique<PointwiseNode>(std::move(add_attributes), context);
            sub_nodes.emplace_back(std::move(add_node));
        }

        if (options.padding_mask) {
            // Lower options to generate row index options
            auto row_index_output = std::make_shared<Tensor_attributes>();
            row_index_output->set_is_virtual(true).set_data_type(DataType_t::INT32);

            Pointwise_attributes row_index_attributes;
            row_index_attributes.set_name("gen_row_index")
                .set_mode(PointwiseMode_t::GEN_INDEX)
                .set_axis(2)
                .set_compute_data_type(DataType_t::INT32);
            row_index_attributes.inputs.IN_0   = last_output;
            row_index_attributes.outputs.OUT_0 = row_index_output;
            auto row_index_node = std::make_unique<PointwiseNode>(std::move(row_index_attributes), context);
            sub_nodes.emplace_back(std::move(row_index_node));

            // Lower options to generate col index options
            auto col_index_output = std::make_shared<Tensor_attributes>();
            col_index_output->set_is_virtual(true).set_data_type(DataType_t::INT32);

            Pointwise_attributes col_index_attributes;
            col_index_attributes.set_name("gen_col_index")
                .set_mode(PointwiseMode_t::GEN_INDEX)
                .set_axis(3)
                .set_compute_data_type(DataType_t::INT32);
            col_index_attributes.inputs.IN_0   = last_output;
            col_index_attributes.outputs.OUT_0 = col_index_output;
            auto col_index_node = std::make_unique<PointwiseNode>(std::move(col_index_attributes), context);
            sub_nodes.emplace_back(std::move(col_index_node));

            // Create operation for row index less than seq_q
            auto row_less_seq_q_output = std::make_shared<Tensor_attributes>();
            row_less_seq_q_output->set_is_virtual(true).set_data_type(DataType_t::INT32);

            Pointwise_attributes row_less_seq_q_attributes;
            row_less_seq_q_attributes.set_name("row_less_seq_q")
                .set_mode(PointwiseMode_t::CMP_LT)
                .set_compute_data_type(DataType_t::INT32);
            row_less_seq_q_attributes.inputs.IN_0   = row_index_output;
            row_less_seq_q_attributes.inputs.IN_1   = options.inputs.SEQ_LEN_Q;
            row_less_seq_q_attributes.outputs.OUT_0 = row_less_seq_q_output;
            auto row_less_seq_q_node = std::make_unique<PointwiseNode>(std::move(row_less_seq_q_attributes), context);
            sub_nodes.emplace_back(std::move(row_less_seq_q_node));

            // Create operation for col index less than seq_q
            auto col_less_seq_kv_output = std::make_shared<Tensor_attributes>();
            col_less_seq_kv_output->set_is_virtual(true).set_data_type(DataType_t::INT32);

            Pointwise_attributes col_less_seq_kv_attributes;
            col_less_seq_kv_attributes.set_name("col_less_seq_kv")
                .set_mode(PointwiseMode_t::CMP_LT)
                .set_compute_data_type(DataType_t::INT32);
            col_less_seq_kv_attributes.inputs.IN_0   = col_index_output;
            col_less_seq_kv_attributes.inputs.IN_1   = options.inputs.SEQ_LEN_KV;
            col_less_seq_kv_attributes.outputs.OUT_0 = col_less_seq_kv_output;
            auto col_less_seq_kv_node = std::make_unique<PointwiseNode>(std::move(col_less_seq_kv_attributes), context);
            sub_nodes.emplace_back(std::move(col_less_seq_kv_node));

            // Create operation logical AND that creates padding mask
            auto logical_and_output = std::make_shared<Tensor_attributes>();
            logical_and_output->set_is_virtual(true).set_data_type(DataType_t::BOOLEAN);

            Pointwise_attributes logical_and_attributes;
            logical_and_attributes.set_name("logical_and")
                .set_mode(PointwiseMode_t::LOGICAL_AND)
                .set_compute_data_type(DataType_t::BOOLEAN);
            logical_and_attributes.inputs.IN_0   = row_less_seq_q_output;
            logical_and_attributes.inputs.IN_1   = col_less_seq_kv_output;
            logical_and_attributes.outputs.OUT_0 = logical_and_output;
            auto logical_and_node = std::make_unique<PointwiseNode>(std::move(logical_and_attributes), context);
            sub_nodes.emplace_back(std::move(logical_and_node));

            // Lower options to binary select options
            negative_inf_padding = std::make_shared<Tensor_attributes>();
            negative_inf_padding->set_dim({1, 1, 1, 1})
                .set_stride({1, 1, 1, 1})
                .set_is_pass_by_value(true)
                // Hard code data type float as FE itself will place FLOAT_MIN in variant pack later
                .set_data_type(DataType_t::FLOAT);

            auto padding_mask_output = std::make_shared<Tensor_attributes>();
            padding_mask_output->set_is_virtual(true);

            Pointwise_attributes binary_select_attributes;
            binary_select_attributes.set_name("binary_select");
            binary_select_attributes.set_mode(PointwiseMode_t::BINARY_SELECT);
            binary_select_attributes.inputs.IN_0 = last_output;
            binary_select_attributes.inputs.IN_1 = negative_inf_padding;
            binary_select_attributes.inputs.IN_2 = logical_and_output;
            last_output = binary_select_attributes.outputs.OUT_0 = padding_mask_output;
            auto binary_select_node = std::make_unique<PointwiseNode>(std::move(binary_select_attributes), context);
            sub_nodes.emplace_back(std::move(binary_select_node));
        }

        if (options.causal_mask) {
            // Lower options to generate row index options
            auto row_index_output = std::make_shared<Tensor_attributes>();
            row_index_output->set_is_virtual(true);

            Pointwise_attributes row_index_attributes;
            row_index_attributes.set_name("gen_row_index");
            row_index_attributes.set_mode(PointwiseMode_t::GEN_INDEX).set_axis(2);
            row_index_attributes.inputs.IN_0   = last_output;
            row_index_attributes.outputs.OUT_0 = row_index_output;
            auto row_index_node = std::make_unique<PointwiseNode>(std::move(row_index_attributes), context);
            sub_nodes.emplace_back(std::move(row_index_node));

            // Lower options to generate col index options
            auto col_index_output = std::make_shared<Tensor_attributes>();
            col_index_output->set_is_virtual(true);

            Pointwise_attributes col_index_attributes;
            col_index_attributes.set_name("gen_col_index");
            col_index_attributes.set_mode(PointwiseMode_t::GEN_INDEX).set_axis(3);
            col_index_attributes.inputs.IN_0   = last_output;
            col_index_attributes.outputs.OUT_0 = col_index_output;
            auto col_index_node = std::make_unique<PointwiseNode>(std::move(col_index_attributes), context);
            sub_nodes.emplace_back(std::move(col_index_node));

            // Lower options to greater than options
            auto row_greater_than_col_output = std::make_shared<Tensor_attributes>();
            row_greater_than_col_output
                ->set_is_virtual(true)
                // Hard coding data type
                .set_data_type(DataType_t::BOOLEAN);

            Pointwise_attributes greater_than_attributes;
            greater_than_attributes.set_name("row_greater_than_col");
            greater_than_attributes.set_mode(PointwiseMode_t::CMP_GE).set_compute_data_type(DataType_t::BOOLEAN);
            greater_than_attributes.inputs.IN_0   = row_index_output;
            greater_than_attributes.inputs.IN_1   = col_index_output;
            greater_than_attributes.outputs.OUT_0 = row_greater_than_col_output;
            auto greater_than_node = std::make_unique<PointwiseNode>(std::move(greater_than_attributes), context);
            sub_nodes.emplace_back(std::move(greater_than_node));

            // Lower options to binary select options
            negative_inf_causal = std::make_shared<Tensor_attributes>();
            negative_inf_causal->set_dim({1, 1, 1, 1})
                .set_stride({1, 1, 1, 1})
                .set_is_pass_by_value(true)
                // Hard code data type float as FE itself will place FLOAT_MIN in variant pack later
                .set_data_type(DataType_t::FLOAT);

            auto causal_mask_output = std::make_shared<Tensor_attributes>();
            causal_mask_output->set_is_virtual(true);

            Pointwise_attributes binary_select_attributes;
            binary_select_attributes.set_name("binary_select");
            binary_select_attributes.set_mode(PointwiseMode_t::BINARY_SELECT);
            binary_select_attributes.inputs.IN_0 = last_output;
            binary_select_attributes.inputs.IN_1 = negative_inf_causal;
            binary_select_attributes.inputs.IN_2 = row_greater_than_col_output;
            last_output = binary_select_attributes.outputs.OUT_0 = causal_mask_output;
            auto binary_select_node = std::make_unique<PointwiseNode>(std::move(binary_select_attributes), context);
            sub_nodes.emplace_back(std::move(binary_select_node));
        }

        // Lower options to softmax options
        auto softmax_output = std::make_shared<Tensor_attributes>();
        softmax_output->set_is_virtual(true);

        Softmax_attributes softmax_attributes;
        softmax_attributes.set_name("softmax");
        softmax_attributes.use_stats = true;  // As this is flash attention
        softmax_attributes.inputs.P  = last_output;
        last_output = softmax_attributes.outputs.S = softmax_output;
        softmax_attributes.outputs.Stats           = options.outputs.Stats;
        auto softmax_node = std::make_unique<SoftmaxNode>(std::move(softmax_attributes), context);
        sub_nodes.emplace_back(std::move(softmax_node));

        // Two cases for training: dropout present or not
        // Special case: Skip dropout when 0.0 probability
        bool dropout_present = (options.dropout_probability.has_value() && options.dropout_probability.value() != 0.0);
        dropout_present = dropout_present || options.inputs.Dropout_mask;

        if (dropout_present) {
            // Lower options to rng options
            auto rng_output = std::make_shared<Tensor_attributes>();
            rng_output
                ->set_is_virtual(true)
                // Hard coding dims and strides as rng output can no inputs to infer it from.
                .set_dim({b, h, s_q, s_kv})
                .set_stride({h * s_q * s_kv, s_q * s_kv, s_kv, 1});

            Rng_attributes rng_attributes;
            rng_attributes.set_name("rng");
            rng_attributes.set_distribution(RngDistribution_t::BERNOULLI)
                .set_bernoulli_probability(1.0 -
                                           options.dropout_probability.value());  // As user sets dropout probability
            rng_attributes.inputs.Seed   = options.inputs.Seed;
            rng_attributes.inputs.Offset = options.inputs.Offset;
            rng_attributes.outputs.Y     = rng_output;
            auto rng_node                = std::make_unique<RngNode>(std::move(rng_attributes), context);
            sub_nodes.emplace_back(std::move(rng_node));

            // Lower options to mask options
            auto dropout_mask_output = std::make_shared<Tensor_attributes>();
            dropout_mask_output->set_is_virtual(true);

            Pointwise_attributes mask_attributes;
            mask_attributes.set_name("dropout_mask_mul");
            mask_attributes.set_mode(PointwiseMode_t::MUL);
            mask_attributes.inputs.IN_0 = last_output;
            mask_attributes.inputs.IN_1 = rng_output;
            last_output = mask_attributes.outputs.OUT_0 = dropout_mask_output;
            auto mask_node = std::make_unique<PointwiseNode>(std::move(mask_attributes), context);
            sub_nodes.emplace_back(std::move(mask_node));

            // Lower options to dropout_scale options
            auto dropout_scale_output = std::make_shared<Tensor_attributes>();
            dropout_scale_output->set_is_virtual(true);

            dropout_scale = std::make_shared<Tensor_attributes>();
            dropout_scale->set_dim({1, 1, 1, 1})
                .set_stride({1, 1, 1, 1})
                .set_is_pass_by_value(true)
// Hard code data type input type as FE itself will place value in variant pack later
#if CUDNN_VERSION < 8903
                .set_data_type(options.inputs.Q->get_data_type());
#else
                .set_data_type(DataType_t::FLOAT);
#endif

            Pointwise_attributes dropout_scale_attributes;
            dropout_scale_attributes.set_name("dropout_scale");
            dropout_scale_attributes.set_mode(PointwiseMode_t::MUL);
            dropout_scale_attributes.inputs.IN_0 = last_output;
            dropout_scale_attributes.inputs.IN_1 = dropout_scale;
            last_output = dropout_scale_attributes.outputs.OUT_0 = dropout_scale_output;
            auto dropout_scale_node = std::make_unique<PointwiseNode>(std::move(dropout_scale_attributes), context);
            sub_nodes.emplace_back(std::move(dropout_scale_node));
        }

        // Lower options to bmm2 options
        // Requirement by cudnn backend to take in bmm2 aType as i/o type.
        last_output->set_data_type(options.inputs.Q->get_data_type());

        Matmul_attributes bmm2_attributes;
        bmm2_attributes.set_name("bmm2");
        bmm2_attributes.inputs.A          = last_output;
        bmm2_attributes.inputs.B          = options.inputs.V;
        bmm2_attributes.outputs.C         = options.outputs.O;
        bmm2_attributes.inputs.M_override = options.inputs.SEQ_LEN_Q;
        bmm2_attributes.inputs.K_override = options.inputs.SEQ_LEN_KV;
        auto bmm2_node                    = std::make_unique<MatmulNode>(std::move(bmm2_attributes), context);
        sub_nodes.emplace_back(std::move(bmm2_node));

        // Set dims if user did not
        if (options.outputs.O->get_dim().empty()) {
            options.outputs.O->set_dim({b, h, s_q, d_v});
        }
        if (options.outputs.O->get_stride().empty()) {
            auto const O_dim = options.outputs.O->get_dim();
            options.outputs.O->set_stride({O_dim[3] * O_dim[2] * O_dim[1], O_dim[3] * O_dim[2], O_dim[3], 1});
        }

        return {error_code_t::OK, ""};
    }

    virtual int64_t
    get_fe_workspace_size_node() const override final {
        int64_t const h = options.inputs.Q->get_dim()[1];
        return h * sizeof(float);
    }

    virtual error_t
    pass_by_value_tensors_(
        std::unordered_map<std::shared_ptr<Tensor_attributes>, pass_by_values_t>& tensor_to_pass_by_value,
        void* node_workspace) override {
        if (options.dropout_probability.has_value()) {
#if CUDNN_VERSION < 8903
            half dropout_scale_value = (1.f / (1.0 - options.dropout_probability.value()));
#else
            float dropout_scale_value = (1.f / (1.0 - options.dropout_probability.value()));
#endif
            tensor_to_pass_by_value.emplace(dropout_scale, dropout_scale_value);
        }

        if (options.padding_mask) {
            float negative_inf_value = std::numeric_limits<float>::min();
            tensor_to_pass_by_value.emplace(negative_inf_padding, negative_inf_value);
        }

        if (options.causal_mask) {
            float negative_inf_value = std::numeric_limits<float>::min();
            tensor_to_pass_by_value.emplace(negative_inf_causal, negative_inf_value);
        }

        if (options.alibi_mask) {
            int64_t const h            = options.inputs.Q->get_dim()[1];
            auto h_alibi_slopes_vector = detail::get_abili_slope(h);

            cudaMemcpy(node_workspace, h_alibi_slopes_vector.data(), h * sizeof(float), cudaMemcpyHostToDevice);
            tensor_to_pass_by_value.emplace(alibi_slopes, node_workspace);
        }

        return {error_code_t::OK, ""};
    }
};

}  // namespace cudnn_frontend::graph