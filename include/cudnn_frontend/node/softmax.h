#pragma once

#include "../../cudnn_frontend_Heuristics.h"
#include "../../cudnn_frontend_Logging.h"

#include "../cudnn_frontend_graph_helpers.h"
#include "../cudnn_frontend_node_interface.h"

#include "pointwise.h"
#include "reduction.h"

namespace cudnn_frontend::graph {

class SoftmaxNode : public INode {
   public:
    Softmax_attributes options;

    SoftmaxNode(Softmax_attributes&& options_, detail::Context const& context)
        : INode(context), options(std::move(options_)) {}

    Type
    getType() override final {
        return Type::COMPOSITE;
    }

    error_t
    validate_node() const override final {
        getLogger() << "[cudnn_frontend] INFO: "
                    << "Validating SoftmaxNode " << options.name << "..." << std::endl;

        RETURN_CUDNN_FRONTEND_ERROR_IF(
            options.use_stats.has_value() == false, error_code_t::ATTRIBUTE_NOT_SET, "use_stats attribute not set.");
        return {error_code_t::OK, ""};
    }

    error_t
    infer_properties_node() override final {
        getLogger() << "[cudnn_frontend] INFO: Inferrencing properties for Softmax node " << options.name << "."
                    << std::endl;

        options.fill_from_context(context);

        // Fill properties of virtual tensors
        auto const& p_dim = options.inputs.P->get_dim();
        auto b            = p_dim[0];
        auto h            = p_dim[1];
        auto s_q          = p_dim[2];

        // Lower options to max options
        auto max_output = std::make_shared<Tensor_attributes>();
        max_output
            ->set_is_virtual(true)
            // Reduction today has no dim inferencing logic today. Hence, hardcoding output dim here.
            .set_dim({b, h, s_q, 1})
            .set_stride({h * s_q, s_q, 1, 1});

        auto max_options = Reduction_attributes();
        max_options.set_name("max");
        max_options.set_mode(ReductionMode_t::MAX);
        max_options.inputs.X  = options.inputs.P;
        max_options.outputs.Y = max_output;
        auto max_node         = std::make_unique<ReductionNode>(std::move(max_options), context);
        sub_nodes.emplace_back(std::move(max_node));

        // Lower options to sub options
        auto sub_output = std::make_shared<Tensor_attributes>();
        sub_output->set_is_virtual(true);

        Pointwise_attributes sub_options;
        sub_options.set_name("sub");
        sub_options.set_mode(PointwiseMode_t::SUB);
        sub_options.inputs.IN_0   = options.inputs.P;
        sub_options.inputs.IN_1   = max_output;
        sub_options.outputs.OUT_0 = sub_output;
        auto sub_node             = std::make_unique<PointwiseNode>(std::move(sub_options), context);
        sub_nodes.emplace_back(std::move(sub_node));

        // Lower options to exp options
        auto exp_output = std::make_shared<Tensor_attributes>();
        exp_output->set_is_virtual(true);

        Pointwise_attributes exp_options;
        exp_options.set_name("exp");
        exp_options.set_mode(PointwiseMode_t::EXP);
        exp_options.inputs.IN_0   = sub_output;
        exp_options.outputs.OUT_0 = exp_output;
        auto exp_node             = std::make_unique<PointwiseNode>(std::move(exp_options), context);
        sub_nodes.emplace_back(std::move(exp_node));

        // Lower options to reduce sum options
        auto sum_output = std::make_shared<Tensor_attributes>();
        sum_output
            ->set_is_virtual(true)
            // Reduction today has no dim inferencing logic today. Hence, hardcoding output dim here.
            .set_dim({b, h, s_q, 1})
            .set_stride({h * s_q, s_q, 1, 1});

        auto sum_options = Reduction_attributes();
        sum_options.set_name("sum");
        sum_options.set_mode(ReductionMode_t::ADD);
        sum_options.inputs.X  = exp_output;
        sum_options.outputs.Y = sum_output;
        auto sum_node         = std::make_unique<ReductionNode>(std::move(sum_options), context);
        sub_nodes.emplace_back(std::move(sum_node));

        // Another path to add when in flash attention mode.
        if (options.use_stats.value()) {
            // Lower options to log options
            auto log_output = std::make_shared<Tensor_attributes>();
            log_output->set_is_virtual(true);

            auto log_options = Pointwise_attributes();
            log_options.set_name("log");
            log_options.set_mode(PointwiseMode_t::LOG);
            log_options.inputs.IN_0   = sum_output;
            log_options.outputs.OUT_0 = log_output;
            auto log_node             = std::make_unique<PointwiseNode>(std::move(log_options), context);
            sub_nodes.emplace_back(std::move(log_node));

            // Lower options to add options
            auto add_options = Pointwise_attributes();
            add_options.set_name("add");
            add_options.set_mode(PointwiseMode_t::ADD);
            add_options.inputs.IN_0   = max_output;
            add_options.inputs.IN_1   = log_output;
            add_options.outputs.OUT_0 = options.outputs.Stats;
            auto add_node             = std::make_unique<PointwiseNode>(std::move(add_options), context);
            sub_nodes.emplace_back(std::move(add_node));
        }

        // Lower options to div options
        auto div_options = Pointwise_attributes();
        div_options.set_name("div");
        div_options.set_mode(PointwiseMode_t::DIV);
        div_options.inputs.IN_0   = exp_output;
        div_options.inputs.IN_1   = sum_output;
        div_options.outputs.OUT_0 = options.outputs.S;
        auto div_node             = std::make_unique<PointwiseNode>(std::move(div_options), context);
        sub_nodes.emplace_back(std::move(div_node));

        return {error_code_t::OK, ""};
    }
};
}  // namespace cudnn_frontend::graph