/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "../graph_interface.h"

namespace cudnn_frontend::graph::reduction_modifiers {

// How argmax and argmin order NaN. Each rule gives the lowest index among equal values, and -0 equals +0.
//   TORCH:      NaN is above every number for argmax and below every number for argmin, so the first NaN wins
//               (torch.argmax, torch.argmin).
//   XLA:        the XLA comparator. A NaN wins, but it is not commutative for two NaN values, so when a row holds
//               more than one NaN the result is one of the NaN positions, not specified which.
//   IGNORE_NAN: a NaN loses to every number; a row of NaN only gives its first position.
enum class Arg_nan_rule { TORCH, XLA, IGNORE_NAN };

namespace detail {

template <PointwiseMode_t Mode>
inline Custom_reduction_attributes
make_single_stream_modifier(Custom_reduction_attributes attributes, char const* name, char const* semantic_key) {
    attributes.set_reducer(
        [name](Custom_reduction_attributes::Graph_t reducer_graph,
               std::vector<Custom_reduction_attributes::Tensor_t> lhs,
               std::vector<Custom_reduction_attributes::Tensor_t> rhs) {
            auto const compute_type = lhs.at(0)->get_data_type();
            auto result             = reducer_graph->pointwise(
                lhs.at(0),
                rhs.at(0),
                Pointwise_attributes().set_name(name).set_mode(Mode).set_compute_data_type(compute_type));
            result->set_data_type(compute_type);
            return std::vector<Custom_reduction_attributes::Tensor_t>{std::move(result)};
        },
        semantic_key);
    return attributes;
}

using Tensor_t = Custom_reduction_attributes::Tensor_t;

inline Tensor_t
body_op(Custom_reduction_attributes::Graph_t const& body,
        char const* name,
        PointwiseMode_t mode,
        DataType_t compute_type,
        DataType_t result_type,
        Tensor_t const& x,
        Tensor_t const& b = nullptr,
        Tensor_t const& t = nullptr) {
    auto attributes = Pointwise_attributes().set_name(name).set_mode(mode).set_compute_data_type(compute_type);
    auto result     = t != nullptr   ? body->pointwise(x, b, t, attributes)
                      : b != nullptr ? body->pointwise(x, b, attributes)
                                     : body->pointwise(x, attributes);
    result->set_data_type(result_type).set_dim({1}).set_stride({1});
    return result;
}

// Stream 0 is the value and stream 1 the index. A compare uses the type of its operands. The logical modes and the
// select of two predicates read one-bit operands, so their compute type has no effect; they use the index type.
inline Custom_reduction_attributes::Reducer_t
arg_reducer(bool is_max, Arg_nan_rule rule) {
    return [is_max, rule](
               Custom_reduction_attributes::Graph_t body, std::vector<Tensor_t> lhs, std::vector<Tensor_t> rhs) {
        auto const B               = DataType_t::BOOLEAN;
        auto const T               = lhs.at(0)->get_data_type();
        auto const I               = lhs.at(1)->get_data_type();
        auto const& vl             = lhs.at(0);
        auto const& il             = lhs.at(1);
        auto const& vr             = rhs.at(0);
        auto const& ir             = rhs.at(1);
        auto const better          = is_max ? PointwiseMode_t::CMP_GT : PointwiseMode_t::CMP_LT;
        auto const better_or_equal = is_max ? PointwiseMode_t::CMP_GE : PointwiseMode_t::CMP_LE;
        auto const worse_or_equal  = is_max ? PointwiseMode_t::CMP_LE : PointwiseMode_t::CMP_GE;

        if (rule == Arg_nan_rule::XLA) {
            auto better_l = body_op(body, "better", better, T, B, vl, vr);
            auto nan_l    = body_op(body, "nan_l", PointwiseMode_t::CMP_NEQ, T, B, vl, vl);
            auto keep_vl  = body_op(body, "keep_value_l", PointwiseMode_t::LOGICAL_OR, I, B, better_l, nan_l);
            auto value    = body_op(body, "value", PointwiseMode_t::BINARY_SELECT, T, T, vl, vr, keep_vl);
            auto equal    = body_op(body, "equal", PointwiseMode_t::CMP_EQ, T, B, vl, vr);
            auto lower    = body_op(body, "lower", PointwiseMode_t::CMP_LT, I, B, il, ir);
            auto tie      = body_op(body, "tie_to_l", PointwiseMode_t::LOGICAL_AND, I, B, equal, lower);
            auto keep_il  = body_op(body, "keep_index_l", PointwiseMode_t::LOGICAL_OR, I, B, keep_vl, tie);
            auto index    = body_op(body, "index", PointwiseMode_t::BINARY_SELECT, I, I, il, ir, keep_il);
            return std::vector<Tensor_t>{value, index};
        }

        // TORCH and IGNORE_NAN are total orders on (value, index), so the body is commutative and associative.
        // The lhs wins on win_if_lower when its index is the lower one, else on win_if_higher.
        auto lower = body_op(body, "lower", PointwiseMode_t::CMP_LT, I, B, il, ir);
        Tensor_t win_if_lower;
        Tensor_t win_if_higher;
        if (rule == Arg_nan_rule::TORCH) {
            auto nan_l   = body_op(body, "nan_l", PointwiseMode_t::CMP_NEQ, T, B, vl, vl);
            auto at_l    = body_op(body, "better_or_equal", better_or_equal, T, B, vl, vr);
            win_if_lower = body_op(body, "win_if_lower", PointwiseMode_t::LOGICAL_OR, I, B, nan_l, at_l);
            // No compare mode is unordered, so "better or unordered" is NOT(worse or equal).
            auto not_better = body_op(body, "worse_or_equal", worse_or_equal, T, B, vl, vr);
            auto better_u   = body_op(body, "better_or_unordered", PointwiseMode_t::LOGICAL_NOT, I, B, not_better);
            auto number_r   = body_op(body, "number_r", PointwiseMode_t::CMP_EQ, T, B, vr, vr);
            win_if_higher   = body_op(body, "win_if_higher", PointwiseMode_t::LOGICAL_AND, I, B, better_u, number_r);
        } else {
            auto at_l       = body_op(body, "better_or_equal", better_or_equal, T, B, vl, vr);
            auto nan_r      = body_op(body, "nan_r", PointwiseMode_t::CMP_NEQ, T, B, vr, vr);
            win_if_lower    = body_op(body, "win_if_lower", PointwiseMode_t::LOGICAL_OR, I, B, at_l, nan_r);
            auto better_l   = body_op(body, "better", better, T, B, vl, vr);
            auto number_l   = body_op(body, "number_l", PointwiseMode_t::CMP_EQ, T, B, vl, vl);
            auto only_r_nan = body_op(body, "only_r_nan", PointwiseMode_t::LOGICAL_AND, I, B, nan_r, number_l);
            win_if_higher   = body_op(body, "win_if_higher", PointwiseMode_t::LOGICAL_OR, I, B, better_l, only_r_nan);
        }
        auto keep_l = body_op(body, "keep_l", PointwiseMode_t::BINARY_SELECT, I, B, win_if_lower, win_if_higher, lower);
        auto index  = body_op(body, "index", PointwiseMode_t::BINARY_SELECT, I, I, il, ir, keep_l);
        auto value  = body_op(body, "value", PointwiseMode_t::BINARY_SELECT, T, T, vl, vr, keep_l);
        return std::vector<Tensor_t>{value, index};
    };
}

template <typename T>
inline Custom_reduction_identity
identity_from_bits(typename serialization_detail::unsigned_scalar_bits<sizeof(T)>::type bits) {
    return Custom_reduction_identity(serialization_detail::scalar_from_raw_bits<T>(bits));
}

enum class Float_special { NEGATIVE_INFINITY, POSITIVE_INFINITY, QUIET_NAN };

// Any other type gets a FLOAT identity, so post_validate_node refuses a value stream that is not floating point.
inline Custom_reduction_identity
float_special_identity(DataType_t type, Float_special special) {
    auto const pick = [special](auto negative_infinity, auto positive_infinity, auto quiet_nan) {
        return special == Float_special::NEGATIVE_INFINITY   ? negative_infinity
               : special == Float_special::POSITIVE_INFINITY ? positive_infinity
                                                             : quiet_nan;
    };
    switch (type) {
        case DataType_t::HALF:
            return identity_from_bits<half>(pick(uint16_t{0xFC00}, uint16_t{0x7C00}, uint16_t{0x7E00}));
        case DataType_t::BFLOAT16:
            return identity_from_bits<nv_bfloat16>(pick(uint16_t{0xFF80}, uint16_t{0x7F80}, uint16_t{0x7FC0}));
        case DataType_t::DOUBLE:
            return identity_from_bits<double>(
                pick(uint64_t{0xFFF0000000000000}, uint64_t{0x7FF0000000000000}, uint64_t{0x7FF8000000000000}));
        default:
            return identity_from_bits<float>(pick(uint32_t{0xFF800000}, uint32_t{0x7F800000}, uint32_t{0x7FC00000}));
    }
}

// Any type other than INT64 gets an INT32 identity, so post_validate_node refuses any other index stream type.
inline Custom_reduction_identity
largest_index_identity(DataType_t type) {
    if (type == DataType_t::INT64) {
        return Custom_reduction_identity(std::numeric_limits<int64_t>::max());
    }
    return Custom_reduction_identity(std::numeric_limits<int32_t>::max());
}

inline char const*
arg_nan_rule_name(Arg_nan_rule rule) {
    switch (rule) {
        case Arg_nan_rule::TORCH:
            return "torch";
        case Arg_nan_rule::XLA:
            return "xla";
        case Arg_nan_rule::IGNORE_NAN:
            return "ignore_nan";
    }
    return "unknown";
}

inline Custom_reduction_attributes
make_arg_modifier(Custom_reduction_attributes attributes,
                  bool is_max,
                  DataType_t value_type,
                  DataType_t index_type,
                  Arg_nan_rule rule) {
    std::string const kind = is_max ? "argmax" : "argmin";
    // The largest index loses every tie, so the identity is two-sided for any index values. Under IGNORE_NAN a NaN
    // loses to every number, so the infinity identity would win against a NaN element; (NaN, largest index) does not.
    auto const value_special = rule == Arg_nan_rule::IGNORE_NAN
                                   ? Float_special::QUIET_NAN
                                   : (is_max ? Float_special::NEGATIVE_INFINITY : Float_special::POSITIVE_INFINITY);
    attributes.set_reducer(arg_reducer(is_max, rule),
                           "cudnn_frontend.reduction_modifiers." + kind + "." + arg_nan_rule_name(rule) + ".v1");
    attributes.set_identity_values(
        {float_special_identity(value_type, value_special), largest_index_identity(index_type)});
    return attributes;
}

}  // namespace detail

[[nodiscard]] inline Custom_reduction_attributes
sum(Custom_reduction_attributes attributes) {
    return detail::make_single_stream_modifier<PointwiseMode_t::ADD>(
        std::move(attributes), "reduction_modifiers::sum", "cudnn_frontend.reduction_modifiers.sum.v1");
}

[[nodiscard]] inline Custom_reduction_attributes
max(Custom_reduction_attributes attributes) {
    return detail::make_single_stream_modifier<PointwiseMode_t::MAX>(
        std::move(attributes), "reduction_modifiers::max", "cudnn_frontend.reduction_modifiers.max.v1");
}

[[nodiscard]] inline Custom_reduction_attributes
min(Custom_reduction_attributes attributes) {
    return detail::make_single_stream_modifier<PointwiseMode_t::MIN>(
        std::move(attributes), "reduction_modifiers::min", "cudnn_frontend.reduction_modifiers.min.v1");
}

// Two streams, (value, index); the index stream holds the position of each element along the reduced dimensions.
// Sets the reducer, the semantic key and the identities: (-inf, largest index) for argmax and (+inf, largest index)
// for argmin, or (NaN, largest index) for IGNORE_NAN, in value_type and index_type.
[[nodiscard]] inline Custom_reduction_attributes
argmax(Custom_reduction_attributes attributes,
       DataType_t value_type,
       DataType_t index_type = DataType_t::INT32,
       Arg_nan_rule rule     = Arg_nan_rule::TORCH) {
    return detail::make_arg_modifier(std::move(attributes), /*is_max=*/true, value_type, index_type, rule);
}

[[nodiscard]] inline Custom_reduction_attributes
argmin(Custom_reduction_attributes attributes,
       DataType_t value_type,
       DataType_t index_type = DataType_t::INT32,
       Arg_nan_rule rule     = Arg_nan_rule::TORCH) {
    return detail::make_arg_modifier(std::move(attributes), /*is_max=*/false, value_type, index_type, rule);
}

}  // namespace cudnn_frontend::graph::reduction_modifiers
