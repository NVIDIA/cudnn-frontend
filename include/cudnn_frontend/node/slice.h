#pragma once

namespace cudnn_frontend::graph {

class SliceNode : public NodeCRTP<SliceNode> {
   public:
    Slice_attributes attributes;

    SliceNode(Slice_attributes&& attributes_, detail::Context const& context)
        : NodeCRTP(context), attributes(std::move(attributes_)) {}

    Type
    getType() override final {
        return Type::SLICE;
    }

    error_t
    infer_properties_node() override final {
        getLogger() << "[cudnn_frontend] INFO: Inferrencing properties for slice node " << attributes.name << "..."
                    << std::endl;

        attributes.fill_from_context(context);

        auto output     = attributes.outputs.at(Slice_attributes::output_names::Y);
        auto output_dim = output->get_dim();

        if (output_dim.empty()) {
            for (size_t i = 0; i < attributes.slices.size(); ++i) {
                output_dim.push_back(attributes.slices[i].second - attributes.slices[i].first);
            }
            output->set_dim(output_dim);
        }

        auto const input            = attributes.inputs.at(Slice_attributes::input_names::X);
        auto const input_data_type  = input->get_data_type();
        auto const output_data_type = output->get_data_type();
        if (output_data_type == DataType_t::NOT_SET) {
            output->set_data_type(input_data_type);
        } else {
            RETURN_CUDNN_FRONTEND_ERROR_IF(output_data_type != input_data_type,
                                           error_code_t::INVALID_VALUE,
                                           "output and input tensor data types should match for slice operation.");
        }

        auto const input_stride = input->get_stride();
        if (output->get_stride().empty()) {
            // For simple slicing without changing the step, the stride remains the same
            // std::vector<int64_t> stride_order =
            //     detail::generate_stride_order_preserving_format(input_stride, output_dim.size());
            output->set_stride(input_stride);
        }

        return {error_code_t::OK, ""};
    }

    error_t
    create_cudnn_tensors_node(std::unordered_map<int64_t, std::shared_ptr<cudnn_frontend::Tensor>>& tensors,
                              int64_t& potential_uid,
                              std::unordered_set<int64_t> const& used_uids) const override final {
        // Do not make input tensor for backend.
        // But assign it a uid
        auto const input = attributes.inputs.at(Slice_attributes::input_names::X);
        if (input->has_uid() == false) {
            detail::assign_uid(input.get(), potential_uid, used_uids);
        }

        auto const output = attributes.outputs.at(Slice_attributes::output_names::Y);
        output->set_is_virtual(false);
        CHECK_CUDNN_FRONTEND_ERROR(detail::create_cudnn_tensor(output, tensors, potential_uid, used_uids));

        return {error_code_t::OK, ""};
    }

    error_t
    create_cudnn_operations(
        std::unordered_set<Tensor_attributes::uid_t>& uids_involved_in_operations,
        std::vector<std::shared_ptr<cudnn_frontend::Operation>>&,
        managed_backend_descriptor_t& raw_operations,
        std::unordered_map<int64_t, std::shared_ptr<cudnn_frontend::Tensor>>&) const override final {
        CUDNN_FRONTEND_UNUSED(raw_operations);
        // No corresponding backend operation

        auto const virutal_output = attributes.outputs.at(Slice_attributes::output_names::Y);
        if (virutal_output && virutal_output->get_is_virtual() == false) {
            uids_involved_in_operations.insert(virutal_output->get_uid());
            if (auto ragged_offset = virutal_output->get_ragged_offset()) {
                uids_involved_in_operations.insert(ragged_offset->get_uid());
            }
        }

        return {error_code_t::OK, ""};
    }

    error_t
    collect_variant_pack_replacements_node(
        std::unordered_map<Tensor_attributes::uid_t, std::pair<Tensor_attributes::uid_t, int64_t>>&
            variant_pack_replacements) const override final {
        auto const input  = attributes.inputs.at(Slice_attributes::input_names::X);
        auto const output = attributes.outputs.at(Slice_attributes::output_names::Y);

        variant_pack_replacements[input->get_uid()] = {output->get_uid(), attributes.get_offset()};

        return {error_code_t::OK, ""};
    };

#ifndef CUDNN_FRONTEND_SKIP_JSON_LIB
    virtual void
    serialize(json& j) const override final {
        j = attributes;
        j.update(R"( {"tag": "SLICE"})"_json);
    }
#endif
};

}  // namespace cudnn_frontend::graph