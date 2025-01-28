#pragma once

#include "../../cudnn_frontend_Logging.h"
#include "../../cudnn_frontend_shim.h"

#include "../graph_helpers.h"
#include "../node_interface.h"

namespace cudnn_frontend {

namespace graph {

class BlockScaleDequantizeNode : public NodeCRTP<BlockScaleDequantizeNode> {
   public:
    Block_scale_dequantize_attributes attributes;

    BlockScaleDequantizeNode(Block_scale_dequantize_attributes&& attributes_, detail::Context const& context)
        : NodeCRTP(context), attributes(std::move(attributes_)) {}

    Type
    getType() override final {
        return Type::BLOCK_SCALE_DEQUANTIZE;
    }

    error_t
    pre_validate_node() const override final {
        getLogger() << "[cudnn_frontend] INFO: " << "Validating BlockScaleDequantizeNode " << attributes.name << "..."
                    << std::endl;

        RETURN_CUDNN_FRONTEND_ERROR_IF(
            !attributes.block_size.has_value(), error_code_t::ATTRIBUTE_NOT_SET, "Block size not set\n");

        auto X     = attributes.inputs.at(Block_scale_dequantize_attributes::input_names::X);
        auto scale = attributes.inputs.at(Block_scale_dequantize_attributes::input_names::scale);
        auto Y     = attributes.outputs.at(Block_scale_dequantize_attributes::output_names::Y);

        RETURN_CUDNN_FRONTEND_ERROR_IF(
            !(X->get_data_type() == DataType_t::FP4_E2M1 || X->get_data_type() == DataType_t::FP8_E4M3 ||
              X->get_data_type() == DataType_t::FP8_E5M2),
            error_code_t::INVALID_VALUE,
            "Input datatype to dequantize node should be among E2M1, E4M3, E5M2\n");

        RETURN_CUDNN_FRONTEND_ERROR_IF(
            !(scale->get_data_type() == DataType_t::FP8_E4M3 || scale->get_data_type() == DataType_t::FP8_E8M0),
            error_code_t::INVALID_VALUE,
            "Scale datatype to dequantize node should be either E4M3 or E8M0\n");

        RETURN_CUDNN_FRONTEND_ERROR_IF(
            !(attributes.block_size.value() == 16 || attributes.block_size.value() == 32),
            error_code_t::INVALID_VALUE,
            "Block size for dequantize node should be 16 for nvfp4 - e2m1 or 32 for mxfp8 - e4m3, e5m2\n");

        RETURN_CUDNN_FRONTEND_ERROR_IF(
            (X->get_data_type() == DataType_t::FP4_E2M1 && attributes.block_size.value() != 16),
            error_code_t::INVALID_VALUE,
            "Block size for dequantize node should be 16 for e2m1 datatype\n");

        RETURN_CUDNN_FRONTEND_ERROR_IF(
            ((X->get_data_type() == DataType_t::FP8_E4M3 || X->get_data_type() == DataType_t::FP8_E5M2) &&
             attributes.block_size.value() != 32),
            error_code_t::INVALID_VALUE,
            "Block size for dequantize node should be 32 for e3m4 or e5m2 datatypes\n");

        RETURN_CUDNN_FRONTEND_ERROR_IF(
            (X->get_data_type() == DataType_t::FP4_E2M1 && scale->get_data_type() != DataType_t::FP8_E4M3),
            error_code_t::INVALID_VALUE,
            "Scale datatype should be E4M3 for input datatype E2M1\n");

        RETURN_CUDNN_FRONTEND_ERROR_IF(
            ((X->get_data_type() == DataType_t::FP8_E4M3 || X->get_data_type() == DataType_t::FP8_E5M2) &&
             scale->get_data_type() != DataType_t::FP8_E8M0),
            error_code_t::INVALID_VALUE,
            "Scale datatype should be E8M0 for input datatype E4M3 or E5M2\n");

        RETURN_CUDNN_FRONTEND_ERROR_IF(!(Y->get_is_virtual()),
                                       error_code_t::INVALID_VALUE,
                                       "Output tensor of dequantize node should be virtual\n");

        return {error_code_t::OK, ""};
    }

    error_t
    infer_properties_node() override final {
        getLogger() << "[cudnn_frontend] INFO: Inferencing properties for BlockScaleDequantizeNode " << attributes.name
                    << "..." << std::endl;

        attributes.fill_from_context(context);

        auto X     = attributes.inputs[Block_scale_dequantize_attributes::input_names::X];
        auto scale = attributes.inputs[Block_scale_dequantize_attributes::input_names::scale];
        auto Y     = attributes.outputs[Block_scale_dequantize_attributes::output_names::Y];

        // Only infer dims and strides if user did not set them
        if (Y->get_dim().empty()) {
            Y->set_dim(X->get_dim());
        }

        if (Y->get_stride().empty()) {
            Y->set_stride(X->get_stride());
        }

        return {error_code_t::OK, ""};
    }

    error_t
    create_cudnn_operations(
        std::unordered_set<Tensor_attributes::uid_t>& uids_involved_in_operations,
        std::vector<std::shared_ptr<cudnn_frontend::Operation>>& operations,
        managed_backend_descriptor_t& raw_operations,
        std::unordered_map<int64_t, std::shared_ptr<cudnn_frontend::Tensor>>& tensors) const override final {
        getLogger() << "[cudnn_frontend] INFO: " << "Building BlockScaleDequantizeNode operations " << attributes.name
                    << "..." << std::endl;
        auto cudnn_ver_error =
            error_t{error_code_t::GRAPH_NOT_SUPPORTED, "Block scale dequantize requires cuDNN v9.7.0"};

#if (CUDNN_VERSION >= 90700)  // TODO: v9.99 is new feature branch; switch to release branch when ready
        NV_CUDNN_FE_DYNAMIC_CHECK_CUDNN_BACKEND_VERSION(90700, cudnn_ver_error);
        CUDNN_FRONTEND_UNUSED(operations);
        auto block_scale_dequantize_operation = make_shared_backend_pointer(
            (cudnnBackendDescriptorType_t)CUDNN_BACKEND_OPERATION_BLOCK_SCALE_DEQUANTIZE_DESCRIPTOR);

        auto X         = attributes.inputs.find(Block_scale_dequantize_attributes::input_names::X)->second;
        auto backend_x = tensors[X->get_uid()]->get_desc()->get_backend_descriptor();
        CHECK_CUDNN_ERROR(detail::set_attribute(block_scale_dequantize_operation->get_backend_descriptor(),
                                                CUDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_XDESC,
                                                CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                1,
                                                &backend_x));

        auto scale         = attributes.inputs.find(Block_scale_dequantize_attributes::input_names::scale)->second;
        auto backend_scale = tensors[scale->get_uid()]->get_desc()->get_backend_descriptor();
        CHECK_CUDNN_ERROR(detail::set_attribute(block_scale_dequantize_operation->get_backend_descriptor(),
                                                CUDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_SCALE_DESC,
                                                CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                1,
                                                &backend_scale));

        auto Y         = attributes.outputs.find(Block_scale_dequantize_attributes::output_names::Y)->second;
        auto backend_y = tensors[Y->get_uid()]->get_desc()->get_backend_descriptor();
        CHECK_CUDNN_ERROR(detail::set_attribute(block_scale_dequantize_operation->get_backend_descriptor(),
                                                CUDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_YDESC,
                                                CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                1,
                                                &backend_y));

        cudnnDataType_t cudnn_data_type;
        CHECK_CUDNN_ERROR(detail::convert_to_cudnn_type(attributes.compute_data_type, cudnn_data_type));
        CHECK_CUDNN_ERROR(detail::set_attribute(block_scale_dequantize_operation->get_backend_descriptor(),
                                                CUDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_MATH_PREC,
                                                CUDNN_TYPE_DATA_TYPE,
                                                1,
                                                &cudnn_data_type));

        int32_t block_size = attributes.block_size.value();
        CHECK_CUDNN_ERROR(detail::set_attribute(block_scale_dequantize_operation->get_backend_descriptor(),
                                                CUDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_BLOCK_SIZE,
                                                CUDNN_TYPE_INT32,
                                                1,
                                                &block_size));

        CHECK_CUDNN_ERROR(detail::finalize(block_scale_dequantize_operation->get_backend_descriptor()));

        raw_operations.push_back(block_scale_dequantize_operation);

        auto const& non_virtual_uids = attributes.get_non_virtual_uids();
        uids_involved_in_operations.insert(non_virtual_uids.begin(), non_virtual_uids.end());
        return {error_code_t::OK, ""};
#else
        CUDNN_FRONTEND_UNUSED(uids_involved_in_operations);
        CUDNN_FRONTEND_UNUSED(operations);
        CUDNN_FRONTEND_UNUSED(raw_operations);
        CUDNN_FRONTEND_UNUSED(tensors);
        return cudnn_ver_error;
#endif
    }

#ifndef CUDNN_FRONTEND_SKIP_JSON_LIB
    virtual void
    serialize(json& j) const override final {
        j = attributes;
        j.update(R"( {"tag": "BLOCK_SCALE_DEQUANTIZE"})"_json);
    }
#endif
};

inline void
INode::block_scale_dequantize(std::shared_ptr<Tensor_attributes> x,
                              std::shared_ptr<Tensor_attributes> scale,
                              Block_scale_dequantize_attributes attributes,
                              std::shared_ptr<Tensor_attributes> y) {
    attributes.inputs[Block_scale_dequantize_attributes::input_names::X]     = x;
    attributes.inputs[Block_scale_dequantize_attributes::input_names::scale] = scale;
    attributes.outputs[Block_scale_dequantize_attributes::output_names::Y]   = y;
    sub_nodes.emplace_back(std::make_unique<BlockScaleDequantizeNode>(std::move(attributes), context));
}

}  // namespace graph

}  // namespace cudnn_frontend