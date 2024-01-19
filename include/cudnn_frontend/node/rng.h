#pragma once

#include "../../cudnn_frontend_Rng.h"
#include "../../cudnn_frontend_Logging.h"

#include "../graph_helpers.h"
#include "../node_interface.h"

namespace cudnn_frontend::graph {

class RngNode : public INode {
    Rng_attributes attributes;

   public:
    RngNode(Rng_attributes&& attributes_, detail::Context const& context)
        : INode(context), attributes(std::move(attributes_)) {}

    Type
    getType() override final {
        return Type::RNG;
    }

    error_t
    pre_validate_node() const override final {
        getLogger() << "[cudnn_frontend] INFO: "
                    << "Validating RngNode " << attributes.name << "..." << std::endl;

        RETURN_CUDNN_FRONTEND_ERROR_IF(
            attributes.outputs.find(Rng_attributes::output_names::Y) == attributes.outputs.end(),
            error_code_t::ATTRIBUTE_NOT_SET,
            "rng output not set.");

        CHECK_CUDNN_FRONTEND_ERROR(attributes.validate_inputs());

        return {error_code_t::OK, ""};
    }

    error_t
    create_cudnn_tensors(int64_t& uid, std::unordered_map<int64_t, std::shared_ptr<cudnn_frontend::Tensor>>& tensors)
        const override final {
        getLogger() << "[cudnn_frontend] INFO: "
                    << "Building RngNode tensors " << attributes.name << "..." << std::endl;

        for (auto const& [name, tensor] : attributes.inputs) {
            (void)name;
            if (tensor) {
                CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(tensor, uid, tensors));
            }
        }
        for (auto const& [name, tensor] : attributes.outputs) {
            (void)name;
            if (tensor) {
                CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(tensor, uid, tensors));
            }
        }
        return {error_code_t::OK, ""};
    }

    error_t
    expand_and_infer_properties() override final {
        getLogger() << "[cudnn_frontend] INFO: Inferrencing properties for rng node " << attributes.name << "..."
                    << std::endl;

        auto y_tensor = attributes.outputs[Rng_attributes::output_names::Y];

        attributes.fill_from_context(context);

        // If user does not set shape and layout of the generated tensor,
        // Get it from node attributes
        // If layout is not set, generate the strides from layout

        if (y_tensor->get_dim().empty() && attributes.get_dim().size()) {
            y_tensor->set_dim(attributes.dim);
        }

        if (y_tensor->get_stride().empty()) {
            if (attributes.get_stride().size()) {
                y_tensor->set_stride(attributes.get_stride());
            } else {
                auto const& y_dim = y_tensor->get_dim();
                // Default to NHWC
                auto const& stride_order = detail::generate_NHWC_stride_order(y_dim.size());
                y_tensor->set_stride(detail::generate_stride(y_dim, stride_order));
            }
        }

        if (y_tensor->get_dim().empty() || y_tensor->get_stride().empty()) {
            return {error_code_t::SHAPE_DEDUCTION_FAILED, "RNG node output shape deduction failed"};
        }

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
    create_cudnn_operations(
        std::unordered_set<uid_t>& uids_involved_in_operations,
        std::vector<std::shared_ptr<cudnn_frontend::Operation>>& operations,
        std::unordered_map<int64_t, std::shared_ptr<cudnn_frontend::Tensor>>& tensors) const override final {
        getLogger() << "[cudnn_frontend] INFO: "
                    << "Building RngNode operations " << attributes.name << "..." << std::endl;

#ifndef NV_CUDNN_DISABLE_EXCEPTION
        try {
#endif

            RETURN_CUDNN_FRONTEND_ERROR_IF(attributes.get_distribution() != RngDistribution_t::BERNOULLI,
                                           error_code_t::ATTRIBUTE_NOT_SET,
                                           "no other distribution except bernoulli supported.");

            auto rng_descriptor = cudnn_frontend::RngDescBuilder()
                                      .setRngDistribution(attributes.get_distribution())
                                      .setBernoulliDistProbability(attributes.get_bernoulli_probability().value())
                                      .build();

            auto&& Rng_operation_builder = cudnn_frontend::OperationBuilder(DescriptorType_t::OPERATION_RNG_DESCRIPTOR);

            CUDNN_FE_VALIDATE_AND_ASSIGN_OUTPUT_TENSOR(Y, Rng_attributes::output_names::Y);
            Rng_operation_builder.setyDesc(*(tensors.at(Y->second->get_uid())));

            Rng_operation_builder.setRngDesc(rng_descriptor);

            if (attributes.seed.has_value()) {
                Rng_operation_builder.setSeed(attributes.get_seed().value());
            } else {
                CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(Seed, Rng_attributes::input_names::Seed);
                Rng_operation_builder.setSeedDesc(*(tensors.at(Seed->second->get_uid())));

                CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(Offset, Rng_attributes::input_names::Offset);
                Rng_operation_builder.setOffsetDesc(*(tensors.at(Offset->second->get_uid())));
            }

            auto operation = Rng_operation_builder.build();

            operations.push_back(std::make_shared<Operation_v8>(std::move(operation)));

#ifndef NV_CUDNN_DISABLE_EXCEPTION
        } catch (cudnn_frontend::cudnnException& e) {
            throw cudnnException(e.what(), e.getCudnnStatus());
        }
#endif

        auto const& non_virtual_uids = attributes.get_non_virtual_uids();
        uids_involved_in_operations.insert(non_virtual_uids.begin(), non_virtual_uids.end());
        return {error_code_t::OK, ""};
    }

    virtual void
    serialize(json& j) const override final {
        j = attributes;
    }
};

inline void INode::rng(std::shared_ptr<Tensor_attributes> seed,
        std::shared_ptr<Tensor_attributes> offset,
        Rng_attributes attributes,
        std::shared_ptr<Tensor_attributes> y) {
  attributes.inputs[Rng_attributes::input_names::Seed]   = seed;
  attributes.inputs[Rng_attributes::input_names::Offset] = offset;
  attributes.outputs[Rng_attributes::output_names::Y]    = y;
  sub_nodes.emplace_back(std::make_unique<RngNode>(std::move(attributes), context));
}

inline std::shared_ptr<Tensor_attributes>
INode::rng(std::shared_ptr<Tensor_attributes> seed,
           std::shared_ptr<Tensor_attributes> offset,
           Rng_attributes attributes) {
    attributes.inputs[Rng_attributes::input_names::Seed]   = seed;
    attributes.inputs[Rng_attributes::input_names::Offset] = offset;
    auto Y = attributes.outputs[Rng_attributes::output_names::Y] = output_tensor(attributes.name + "::Y");

    sub_nodes.emplace_back(std::make_unique<RngNode>(std::move(attributes), context));
    return Y;
}

}  // namespace cudnn_frontend::graph
