#pragma once

#include "../../cudnn_frontend_Rng.h"
#include "../../cudnn_frontend_Logging.h"

#include "../graph_helpers.h"
#include "../node_interface.h"

namespace cudnn_frontend::graph {

class RngNode : public NodeCRTP<RngNode> {
   public:
    Rng_attributes attributes;

    RngNode(Rng_attributes&& attributes_, detail::Context const& context)
        : NodeCRTP(context), attributes(std::move(attributes_)) {}

    Type
    getType() override final {
        return Type::RNG;
    }

    error_t
    infer_properties_node() override final {
        CUDNN_FE_LOG_LABEL_ENDL("INFO: Inferrencing properties for rng node " << attributes.name << "...");

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
    create_cudnn_operations(
        std::unordered_set<Tensor_attributes::uid_t>& uids_involved_in_operations,
        std::vector<std::shared_ptr<cudnn_frontend::Operation>>& operations,
        managed_backend_descriptor_t& raw_operations,
        std::unordered_map<int64_t, std::shared_ptr<cudnn_frontend::Tensor>>& tensors) const override final {
        CUDNN_FRONTEND_UNUSED(raw_operations);
        CUDNN_FE_LOG_LABEL_ENDL("INFO: Building RngNode operations " << attributes.name << "...");

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

#ifdef NV_CUDNN_DISABLE_EXCEPTION
        // disable exception macro is defined. Calling build will not throw.
        // Check status of desc and return error.
        auto operation = Rng_operation_builder.build();
        RETURN_CUDNN_FRONTEND_ERROR_IF(operation.get_status() != CUDNN_STATUS_SUCCESS,
                                       error_code_t::CUDNN_BACKEND_API_FAILED,
                                       operation.get_error());
        operations.push_back(std::make_shared<Operation_v8>(std::move(operation)));
#else
        // build() can throw
        // wrap in try catch
        try {
            auto operation = Rng_operation_builder.build();
            operations.push_back(std::make_shared<Operation_v8>(std::move(operation)));
        } catch (cudnn_frontend::cudnnException& e) {
            RETURN_CUDNN_FRONTEND_ERROR_IF(
                e.getCudnnStatus() != CUDNN_STATUS_SUCCESS, error_code_t::CUDNN_BACKEND_API_FAILED, e.what());
        }
#endif
        auto const& non_virtual_uids = attributes.get_non_virtual_uids();
        uids_involved_in_operations.insert(non_virtual_uids.begin(), non_virtual_uids.end());
        return {error_code_t::OK, ""};
    }

#ifndef CUDNN_FRONTEND_SKIP_JSON_LIB
    virtual void
    serialize(json& j) const override final {
        j = attributes;
        j.update(R"( {"tag": "RNG"})"_json);
    }
#endif
};

inline void
INode::rng(std::shared_ptr<Tensor_attributes> seed,
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