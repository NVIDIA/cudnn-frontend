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
        CUDNN_FE_LOG_LABEL_ENDL("INFO: Validating SoftmaxNode " << attributes.name << "...");

        RETURN_CUDNN_FRONTEND_ERROR_IF(
            attributes.use_stats.has_value() == false, error_code_t::ATTRIBUTE_NOT_SET, "use_stats attribute not set.");

        RETURN_CUDNN_FRONTEND_ERROR_IF(attributes.use_M_Zinv.has_value() == false,
                                       error_code_t::ATTRIBUTE_NOT_SET,
                                       "use_M_Zinv attribute not set.");

        return {error_code_t::OK, ""};
    }

    error_t
    infer_properties_node() override final {
        return {error_code_t::OK, ""};
    }

    error_t
    expand_node() override final {
        CUDNN_FE_LOG_LABEL_ENDL("INFO: Inferrencing properties for Softmax node " << attributes.name << ".");

        attributes.fill_from_context(context);

        // Fill properties of virtual tensors
        auto const p_dim = attributes.inputs[Softmax_attributes::input_names::P]->get_dim();
        auto b           = p_dim[0];
        auto h           = p_dim[1];
        auto s_q         = p_dim[2];

        auto max_output = attributes.outputs[Softmax_attributes::output_names::M];
        if (!attributes.use_M_Zinv.value()) {
            max_output = std::make_shared<Tensor_attributes>();
            max_output->set_name("M").set_is_virtual(true);
        }
        //////////////// TODO //////////////////////////
        // Check Stride (Before setting dimension?)
        max_output->set_dim({b, h, s_q, 1}).set_stride({h * s_q, s_q, 1, 1});

        auto max_attributes = Reduction_attributes().set_name("M").set_mode(ReductionMode_t::MAX);
        // Special non-functional-style call. Needed because output already created and provided to user.
        reduction(attributes.inputs[Softmax_attributes::input_names::P], max_attributes, max_output);

        auto sub_attributes = Pointwise_attributes().set_name("sub").set_mode(PointwiseMode_t::SUB);
        auto const& sub_output =
            pointwise(attributes.inputs[Softmax_attributes::input_names::P], max_output, sub_attributes);
        sub_output->set_name("sub_M");

        auto exp_attributes    = Pointwise_attributes().set_name("exp").set_mode(PointwiseMode_t::EXP);
        auto const& exp_output = pointwise(sub_output, exp_attributes);
        exp_output->set_name("exp_sub_M");

        auto sum_attributes    = Reduction_attributes().set_name("sum").set_mode(ReductionMode_t::ADD);
        auto const& sum_output = reduction(exp_output, sum_attributes);
        sum_output->set_name("Z").set_dim({b, h, s_q, 1}).set_stride({h * s_q, s_q, 1, 1});

        // Another path to add when in flash attention mode.
        if (attributes.use_stats.value()) {
            auto log_attributes    = Pointwise_attributes().set_name("log").set_mode(PointwiseMode_t::LOG);
            auto const& log_output = pointwise(sum_output, log_attributes);
            log_output->set_dim({b, h, s_q, 1}).set_stride({h * s_q, s_q, 1, 1});

            auto add_attributes = Pointwise_attributes().set_name("add").set_mode(PointwiseMode_t::ADD);
            // Special non-functional-style call. Needed because output already created and provided to user.
            pointwise(
                max_output, log_output, add_attributes, attributes.outputs[Softmax_attributes::output_names::Stats]);
        }

        if (attributes.use_M_Zinv.value()) {
            auto reciprocal_attributes =
                Pointwise_attributes().set_name("reciprocal").set_mode(PointwiseMode_t::RECIPROCAL);
            // Special non-functional-style call. Needed because output already created and provided to user.
            attributes.outputs[Softmax_attributes::output_names::Zinv]
                ->set_dim({b, h, s_q, 1})
                .set_stride({h * s_q, s_q, 1, 1});
            pointwise(sum_output, reciprocal_attributes, attributes.outputs[Softmax_attributes::output_names::Zinv]);
        }

        if (!attributes.use_M_Zinv.value()) {
            auto div_attributes = Pointwise_attributes().set_name("div").set_mode(PointwiseMode_t::DIV);
            // Special non-functional-style call. Needed because output already created and provided to user.
            pointwise(exp_output, sum_output, div_attributes, attributes.outputs[Softmax_attributes::output_names::S]);
        } else {
            auto mul_attributes = Pointwise_attributes().set_name("mul").set_mode(PointwiseMode_t::MUL);
            // Special non-functional-style call. Needed because output already created and provided to user.
            pointwise(exp_output,
                      attributes.outputs[Softmax_attributes::output_names::Zinv],
                      mul_attributes,
                      attributes.outputs[Softmax_attributes::output_names::S]);
        }

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
               std::shared_ptr<Tensor_attributes> stats) {
    attributes.inputs[Softmax_attributes::input_names::P]       = p;
    attributes.outputs[Softmax_attributes::output_names::S]     = s;
    attributes.outputs[Softmax_attributes::output_names::Stats] = stats;
    sub_nodes.emplace_back(std::make_unique<SoftmaxNode>(std::move(attributes), context));
}

inline void
INode::softmax(std::shared_ptr<Tensor_attributes> p,
               Softmax_attributes attributes,
               std::shared_ptr<Tensor_attributes> s,
               std::shared_ptr<Tensor_attributes> m,
               std::shared_ptr<Tensor_attributes> zinv) {
    attributes.inputs[Softmax_attributes::input_names::P]      = p;
    attributes.outputs[Softmax_attributes::output_names::S]    = s;
    attributes.outputs[Softmax_attributes::output_names::M]    = m;
    attributes.outputs[Softmax_attributes::output_names::Zinv] = zinv;
    sub_nodes.emplace_back(std::make_unique<SoftmaxNode>(std::move(attributes), context));
}
}  // namespace cudnn_frontend::graph