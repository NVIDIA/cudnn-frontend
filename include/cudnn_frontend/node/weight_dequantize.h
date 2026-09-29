/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../node_interface.h"

namespace cudnn_frontend::graph {

class WeightDequantizeNode : public NodeCRTP<WeightDequantizeNode> {
   public:
    Weight_dequantize_attributes attributes;

    WeightDequantizeNode(Weight_dequantize_attributes&& value, detail::Context const& context)
        : NodeCRTP(context), attributes(std::move(value)) {}

    Type
    getType() override final {
        return Type::WEIGHT_DEQUANTIZE;
    }

    error_t
    pre_validate_node() const override final {
        auto const& p = attributes.program;
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            p.source.empty() || p.source.size() >= 1024 * 1024 || p.source.find('\0') != std::string::npos,
            error_code_t::INVALID_VALUE,
            "Weight dequantization source must contain 1..1048575 bytes without embedded NULs");
        auto letter      = [](char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_'; };
        bool valid_entry = !p.entry.empty() && p.entry.size() <= 127 && letter(p.entry.front());
        for (char c : p.entry) valid_entry &= letter(c) || (c >= '0' && c <= '9');
        RETURN_CUDNN_FRONTEND_ERROR_IF(!valid_entry,
                                       error_code_t::INVALID_VALUE,
                                       "Weight dequantization entry must be an unqualified ASCII identifier");
        RETURN_CUDNN_FRONTEND_ERROR_IF((p.abi_version != 1 && p.abi_version != 2) || p.tile_shape.size() != 2 ||
                                           p.tile_shape[0] <= 0 || p.tile_shape[1] <= 0,
                                       error_code_t::INVALID_VALUE,
                                       "Invalid weight dequantization ABI or tile shape");
        RETURN_CUDNN_FRONTEND_ERROR_IF(p.cta_smem_bytes < 0 || p.cta_smem_bytes > 1024 * 1024 ||
                                           p.stage_smem_bytes < 0 || p.stage_smem_bytes > 1024 * 1024 ||
                                           p.input_alignment <= 0 || p.input_alignment > 256 ||
                                           (p.input_alignment & (p.input_alignment - 1)) || p.constants.size() > 64,
                                       error_code_t::INVALID_VALUE,
                                       "Invalid weight dequantization resources or constants");
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            p.load_mode < 0 || p.load_mode > 2 ||
                (p.abi_version == 1 && (p.load_mode || p.storage_bits || p.row_stride_bytes)) ||
                (p.abi_version == 2 &&
                 (p.load_mode == 0 || (p.storage_bits != 2 && p.storage_bits != 4 && p.storage_bits != 8) ||
                  p.row_stride_bytes <= 0 || p.row_stride_bytes > INT32_MAX || p.row_stride_bytes % 32 ||
                  p.input_alignment < 32)),
            error_code_t::INVALID_VALUE,
            "Managed loads require ABI 2, mode TMA_BULK or VECTOR_256, 2/4/8 bits, and 32-byte aligned rows/inputs");
        RETURN_CUDNN_FRONTEND_ERROR_IF(attributes.auxiliary_count < 0 || attributes.auxiliary_count > 8,
                                       error_code_t::INVALID_VALUE,
                                       "Weight dequantization accepts zero to eight auxiliaries");
        RETURN_CUDNN_FRONTEND_ERROR_IF(attributes.inputs.size() != size_t(1 + attributes.auxiliary_count),
                                       error_code_t::INVALID_VALUE,
                                       "Weight dequantization input count does not match its ports");
        for (int64_t i = 0; i <= attributes.auxiliary_count; ++i) {
            auto it = attributes.inputs.find(static_cast<Weight_dequantize_attributes::input_names>(i));
            RETURN_CUDNN_FRONTEND_ERROR_IF(it == attributes.inputs.end() || !it->second,
                                           error_code_t::ATTRIBUTE_NOT_SET,
                                           "Missing weight dequantization input");
            auto const& t = it->second;
            RETURN_CUDNN_FRONTEND_ERROR_IF(
                t->get_is_virtual() || t->get_is_pass_by_value() || t->get_has_compile_time_constant(),
                error_code_t::INVALID_VALUE,
                "Weight storage and scales must be physical runtime tensors");
        }
        auto y = attributes.outputs.at(Weight_dequantize_attributes::output_names::Y);
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            !y || !y->get_is_virtual(), error_code_t::INVALID_VALUE, "Weight dequantization output must be virtual");
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            y->get_dim().size() != 3 || y->get_dim()[0] != 1 || y->get_dim()[1] <= 0 || y->get_dim()[2] <= 0 ||
                y->get_dim()[1] > INT32_MAX || y->get_dim()[2] > INT32_MAX,
            error_code_t::INVALID_VALUE,
            "Set the logical weight shape [1,K,N] explicitly; it cannot be inferred from bytes");
        if (p.load_mode != 0) {
            auto const& w             = attributes.inputs.at(Weight_dequantize_attributes::input_names::WEIGHTS);
            auto const& dim           = w->get_dim();
            auto const& stride        = w->get_stride();
            const int64_t minimum_row = ((y->get_dim()[2] + 63) / 64 * (64 * p.storage_bits / 8) + 31) / 32 * 32;
            RETURN_CUDNN_FRONTEND_ERROR_IF(
                dim.size() != 3 || dim[0] != 1 || dim[1] != 1 || dim[2] <= 0 || stride.size() != 3 || stride[2] != 1 ||
                    stride[1] < dim[2] || stride[0] < stride[1] ||
                    (w->get_data_type() != DataType_t::UINT8 && w->get_data_type() != DataType_t::INT8) ||
                    p.row_stride_bytes < minimum_row || y->get_dim()[1] > dim[2] / p.row_stride_bytes,
                error_code_t::INVALID_VALUE,
                "Managed weight storage must be a byte vector containing every padded K row");
        }
        return {error_code_t::OK, ""};
    }

    error_t
    infer_properties_node() override final {
        attributes.fill_from_context(context);
        // input_alignment is an explicit caller promise for all physical inputs.
        // Carry it into backend tensor descriptors as well as the program; the
        // backend checks actual variant-pack addresses on every execution.
        for (auto const& input : attributes.inputs)
            input.second->set_alignment(std::max(input.second->get_alignment(), attributes.program.input_alignment));
        auto y          = attributes.outputs.at(Weight_dequantize_attributes::output_names::Y);
        auto const& dim = y->get_dim();
        if (y->get_stride().empty()) y->set_stride({dim[1] * dim[2], dim[2], 1});
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            y->get_data_type() != DataType_t::HALF && y->get_data_type() != DataType_t::BFLOAT16,
            error_code_t::INVALID_VALUE,
            "Dequantized weights must be FP16 or BF16");
        // The program defines its arithmetic. A graph-wide compute default is
        // not a request to change the customer's device code.
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            attributes.compute_data_type != DataType_t::NOT_SET && attributes.compute_data_type != DataType_t::FLOAT,
            error_code_t::INVALID_VALUE,
            "Custom weight dequantization uses a program-defined conversion and FP32 GEMM computation");
        return {error_code_t::OK, ""};
    }

    error_t
    create_cudnn_operations(
        std::unordered_set<Tensor_attributes::uid_t>& uids,
        std::vector<std::shared_ptr<cudnn_frontend::Operation>>& operations,
        managed_backend_descriptor_t& raw_operations,
        std::unordered_map<int64_t, std::shared_ptr<cudnn_frontend::Tensor>>& tensors) const override final {
        CUDNN_FRONTEND_UNUSED(operations);
        // The prototype has no released-version guarantee. Its header explicitly
        // advertises these enum definitions; never guess enum numbers or infer
        // availability solely from CUDNN_VERSION. Runtime descriptor creation
        // below also rejects an unpatched library with the same version number.
#if defined(CUDNN_WEIGHT_DECODE_ABI_VERSION) && CUDNN_WEIGHT_DECODE_ABI_VERSION >= 1
        NV_CUDNN_FE_DYNAMIC_CHECK_CUDNN_BACKEND_VERSION(
            92800,
            (error_t{error_code_t::GRAPH_NOT_SUPPORTED, "Weight dequantization requires the prototype backend"}));
#if CUDNN_WEIGHT_DECODE_ABI_VERSION < 2
        RETURN_CUDNN_FRONTEND_ERROR_IF(attributes.program.abi_version == 2,
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "Managed weight loads require CUDNN_WEIGHT_DECODE_ABI_VERSION >= 2 headers");
#endif
        auto program = make_shared_backend_pointer(CUDNN_BACKEND_WEIGHT_DECODE_DESCRIPTOR);
        RETURN_CUDNN_FRONTEND_ERROR_IF(!program->is_good(),
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "Backend does not provide weight-decode program ABI 1");
        auto pd       = program->get_backend_descriptor();
        auto const& p = attributes.program;
        auto set =
            [&](cudnnBackendAttributeName_t name, cudnnBackendAttributeType_t type, int64_t count, void const* value) {
                return detail::set_attribute(pd, name, type, count, value);
            };
        _CUDNN_CHECK_CUDNN_ERROR(
            set(CUDNN_ATTR_WEIGHT_DECODE_SOURCE, CUDNN_TYPE_CHAR, p.source.size() + 1, p.source.c_str()));
        _CUDNN_CHECK_CUDNN_ERROR(
            set(CUDNN_ATTR_WEIGHT_DECODE_ENTRY, CUDNN_TYPE_CHAR, p.entry.size() + 1, p.entry.c_str()));
        _CUDNN_CHECK_CUDNN_ERROR(set(CUDNN_ATTR_WEIGHT_DECODE_ABI_VERSION, CUDNN_TYPE_INT64, 1, &p.abi_version));
        _CUDNN_CHECK_CUDNN_ERROR(set(CUDNN_ATTR_WEIGHT_DECODE_TILE_SHAPE, CUDNN_TYPE_INT64, 2, p.tile_shape.data()));
        _CUDNN_CHECK_CUDNN_ERROR(set(CUDNN_ATTR_WEIGHT_DECODE_CTA_SMEM_BYTES, CUDNN_TYPE_INT64, 1, &p.cta_smem_bytes));
        _CUDNN_CHECK_CUDNN_ERROR(
            set(CUDNN_ATTR_WEIGHT_DECODE_STAGE_SMEM_BYTES, CUDNN_TYPE_INT64, 1, &p.stage_smem_bytes));
        _CUDNN_CHECK_CUDNN_ERROR(
            set(CUDNN_ATTR_WEIGHT_DECODE_INPUT_ALIGNMENT, CUDNN_TYPE_INT64, 1, &p.input_alignment));
#if CUDNN_WEIGHT_DECODE_ABI_VERSION >= 2
        // Do not send new attributes for ABI 1: old prototype libraries still
        // support that unchanged contract, even with newer frontend headers.
        if (p.abi_version == 2) {
            _CUDNN_CHECK_CUDNN_ERROR(set(CUDNN_ATTR_WEIGHT_DECODE_LOAD_MODE, CUDNN_TYPE_INT64, 1, &p.load_mode));
            _CUDNN_CHECK_CUDNN_ERROR(set(CUDNN_ATTR_WEIGHT_DECODE_STORAGE_BITS, CUDNN_TYPE_INT64, 1, &p.storage_bits));
            _CUDNN_CHECK_CUDNN_ERROR(
                set(CUDNN_ATTR_WEIGHT_DECODE_ROW_STRIDE_BYTES, CUDNN_TYPE_INT64, 1, &p.row_stride_bytes));
        }
#endif
        if (!p.constants.empty())
            _CUDNN_CHECK_CUDNN_ERROR(
                set(CUDNN_ATTR_WEIGHT_DECODE_CONSTANTS, CUDNN_TYPE_INT64, p.constants.size(), p.constants.data()));
        _CUDNN_CHECK_CUDNN_ERROR(detail::finalize(pd));
        auto operation = make_shared_backend_pointer(CUDNN_BACKEND_OPERATION_WEIGHT_DECODE_DESCRIPTOR);
        _CUDNN_CHECK_CUDNN_ERROR(operation->get_status());
        auto op = operation->get_backend_descriptor();
        auto w  = tensors.at(attributes.inputs.at(Weight_dequantize_attributes::input_names::WEIGHTS)->get_uid())
                     ->get_raw_desc();
        auto y =
            tensors.at(attributes.outputs.at(Weight_dequantize_attributes::output_names::Y)->get_uid())->get_raw_desc();
        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(
            op, CUDNN_ATTR_OPERATION_WEIGHT_DECODE_PROGRAM, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &pd));
        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(
            op, CUDNN_ATTR_OPERATION_WEIGHT_DECODE_PACKED_DESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &w));
        std::vector<cudnnBackendDescriptor_t> auxiliary;
        for (int64_t i = 0; i < attributes.auxiliary_count; ++i) {
            auto t = attributes.inputs.at(static_cast<Weight_dequantize_attributes::input_names>(i + 1));
            auxiliary.push_back(tensors.at(t->get_uid())->get_raw_desc());
        }
        if (!auxiliary.empty())
            _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(op,
                                                           CUDNN_ATTR_OPERATION_WEIGHT_DECODE_AUX_DESCS,
                                                           CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                           auxiliary.size(),
                                                           auxiliary.data()));
        _CUDNN_CHECK_CUDNN_ERROR(
            detail::set_attribute(op, CUDNN_ATTR_OPERATION_WEIGHT_DECODE_YDESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &y));
        _CUDNN_CHECK_CUDNN_ERROR(detail::finalize(op));
        // Finalization copies the program. Only the operation belongs in the
        // operation graph; runtime weights/scales are ordinary variant-pack UIDs.
        raw_operations.push_back(operation);
        auto ids = attributes.get_non_virtual_uids();
        uids.insert(ids.begin(), ids.end());
        return {error_code_t::OK, ""};
#else
        CUDNN_FRONTEND_UNUSED(uids);
        CUDNN_FRONTEND_UNUSED(raw_operations);
        CUDNN_FRONTEND_UNUSED(tensors);
        return {error_code_t::GRAPH_NOT_SUPPORTED,
                "Weight dequantization requires headers with CUDNN_WEIGHT_DECODE_ABI_VERSION and a matching backend"};
#endif
    }

#ifndef CUDNN_FRONTEND_SKIP_JSON_LIB
    void
    serialize(json& j) const override final {
        j        = attributes;
        j["tag"] = "WEIGHT_DEQUANTIZE";
    }
#endif
};

}  // namespace cudnn_frontend::graph
