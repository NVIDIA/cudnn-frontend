#pragma once

#include "../../cudnn_frontend_Heuristics.h"
#include "../../cudnn_frontend_Logging.h"

#include "../graph_helpers.h"
#include "../node_interface.h"

#include "pointwise.h"
#include "reduction.h"

namespace cudnn_frontend::graph {

class SoftmaxNode : public NodeCRTP<SoftmaxNode> {
   public:
    Softmax_attributes attributes;

    SoftmaxNode(Softmax_attributes&& attributes_, detail::Context const& context)
        : NodeCRTP(context), attributes(std::move(attributes_)) {}

    Type
    getType() override final {
        return Type::COMPOSITE;
    }

    error_t
    pre_validate_node() const override final {
        CUDNN_FE_LOG_LABEL_ENDL("INFO: Validating SoftmaxNode " << attributes.name);

        return {error_code_t::OK, ""};
    }

    error_t
    infer_properties_node() override final {
        return {error_code_t::OK, ""};
    }

    error_t
    expand_node() override final {
        CUDNN_FE_LOG_LABEL_ENDL("INFO:     Inferrencing properties for Softmax node " << attributes.name);

        attributes.fill_from_context(context);

        // Fill properties of virtual tensors
        auto const p_dim = attributes.inputs[Softmax_attributes::input_names::P]->get_dim();
        auto b           = p_dim[0];
        auto h           = p_dim[1];
        auto s_q         = p_dim[2];

        auto max_output = attributes.outputs[Softmax_attributes::output_names::Max];
        if (max_output == nullptr) {
            max_output = std::make_shared<Tensor_attributes>();
            max_output->set_is_virtual(true);
        }
        //////////////// TODO //////////////////////////
        // Check Stride (Before setting dimension?)
        if (max_output->get_dim().empty()) {
            max_output->set_dim({b, h, s_q, 1});
        }
        if (max_output->get_stride().empty()) {
            max_output->set_stride({h * s_q, s_q, 1, 1});
        }

        auto max_attributes = Reduction_attributes().set_name("Max").set_mode(ReductionMode_t::MAX);
        // If sink tensor is present, we also need to take a pointwise max with sink
        if (attributes.inputs.find(Softmax_attributes::input_names::SINK) != attributes.inputs.end()) {
            auto s_max = reduction(attributes.inputs[Softmax_attributes::input_names::P], max_attributes);
            s_max->set_name("s_max");

            auto sink_tensor     = attributes.inputs[Softmax_attributes::input_names::SINK];
            auto sink_attributes = Pointwise_attributes().set_name("max_sink").set_mode(PointwiseMode_t::MAX);
            pointwise(s_max, sink_tensor, sink_attributes, max_output);
        } else {
            // Special non-functional-style call. Needed because output already created and provided to user.
            reduction(attributes.inputs[Softmax_attributes::input_names::P], max_attributes, max_output);
        }

        auto sub_attributes = Pointwise_attributes().set_name("sub").set_mode(PointwiseMode_t::SUB);
        auto const& sub_output =
            pointwise(attributes.inputs[Softmax_attributes::input_names::P], max_output, sub_attributes);
        sub_output->set_name("sub_M");

        auto exp_attributes    = Pointwise_attributes().set_name("exp").set_mode(PointwiseMode_t::EXP);
        auto const& exp_output = pointwise(sub_output, exp_attributes);
        exp_output->set_name("exp_sub_M");

        auto sum_output = attributes.outputs[Softmax_attributes::output_names::Sum_exp];
        if (sum_output == nullptr) {
            sum_output = std::make_shared<Tensor_attributes>();
            sum_output->set_is_virtual(true);
        }
        sum_output->set_name("SumExp");
        if (sum_output->get_dim().empty()) {
            sum_output->set_dim({b, h, s_q, 1});
        }
        if (sum_output->get_stride().empty()) {
            sum_output->set_stride({h * s_q, s_q, 1, 1});
        }
        auto sum_attributes = Reduction_attributes().set_name("sum").set_mode(ReductionMode_t::ADD);
        // If sink tensor is present, also subtract it and take its exp
        if (attributes.inputs.find(Softmax_attributes::input_names::SINK) != attributes.inputs.end()) {
            auto sink_tensor = attributes.inputs[Softmax_attributes::input_names::SINK];
            auto sub_sink    = pointwise(sink_tensor, max_output, sub_attributes);
            sub_sink->set_name("sub_sink");

            auto exp_sink = pointwise(sub_sink, exp_attributes);
            exp_sink->set_name("exp_sink");

            auto temp_sum = reduction(exp_output, sum_attributes);
            temp_sum->set_name("SumExp_elements").set_dim({b, h, s_q, 1}).set_stride({h * s_q, s_q, 1, 1});

            auto add_attributes = Pointwise_attributes().set_name("add_sink").set_mode(PointwiseMode_t::ADD);
            pointwise(temp_sum, exp_sink, add_attributes, sum_output);
        } else {
            reduction(exp_output, sum_attributes, sum_output);
        }

        // WAR when:
        // - softmax stats in not requested
        // - max and sum_exp are not requested
        if (attributes.outputs[Softmax_attributes::output_names::Stats] == nullptr &&
            attributes.outputs[Softmax_attributes::output_names::Max] == nullptr &&
            attributes.outputs[Softmax_attributes::output_names::Sum_exp] == nullptr) {
            auto softmax_stats = std::make_shared<Tensor_attributes>();
            softmax_stats->set_is_virtual(true);
            attributes.outputs[Softmax_attributes::output_names::Stats] = softmax_stats;
        }

        if (attributes.outputs.find(Softmax_attributes::output_names::Stats) != attributes.outputs.end() &&
            attributes.outputs[Softmax_attributes::output_names::Stats] != nullptr) {
            auto log_attributes    = Pointwise_attributes().set_name("log").set_mode(PointwiseMode_t::LOG);
            auto const& log_output = pointwise(sum_output, log_attributes);
            log_output->set_dim({b, h, s_q, 1}).set_stride({h * s_q, s_q, 1, 1});

            auto add_attributes = Pointwise_attributes().set_name("add").set_mode(PointwiseMode_t::ADD);
            // Special non-functional-style call. Needed because output already created and provided to user.
            pointwise(
                max_output, log_output, add_attributes, attributes.outputs[Softmax_attributes::output_names::Stats]);
        }

        auto div_attributes = Pointwise_attributes().set_name("div").set_mode(PointwiseMode_t::DIV);
        // Special non-functional-style call. Needed because output already created and provided to user.
        pointwise(exp_output, sum_output, div_attributes, attributes.outputs[Softmax_attributes::output_names::S]);

        return {error_code_t::OK, ""};
    }

#ifndef CUDNN_FRONTEND_SKIP_JSON_LIB
    virtual void
    serialize(json& j) const override final {
        j = attributes;
    }
#endif
};

inline void
INode::softmax(std::shared_ptr<Tensor_attributes> p,
               Softmax_attributes attributes,
               std::shared_ptr<Tensor_attributes> s,
               std::shared_ptr<Tensor_attributes> stats,
               std::shared_ptr<Tensor_attributes> max,
               std::shared_ptr<Tensor_attributes> sum_exp) {
    attributes.inputs[Softmax_attributes::input_names::P]         = p;
    attributes.outputs[Softmax_attributes::output_names::S]       = s;
    attributes.outputs[Softmax_attributes::output_names::Stats]   = stats;
    attributes.outputs[Softmax_attributes::output_names::Max]     = max;
    attributes.outputs[Softmax_attributes::output_names::Sum_exp] = sum_exp;
    sub_nodes.emplace_back(std::make_unique<SoftmaxNode>(std::move(attributes), context));
}

}  // namespace cudnn_frontend::graph