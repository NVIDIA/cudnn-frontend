/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#ifndef CUDNN_FRONTEND_SKIP_JSON_LIB

#include <climits>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "serialize.h"
#include <algorithm>

namespace cudnn_frontend::graph::plan_serialization_detail {

using uid_t        = Tensor_attributes::uid_t;
using Workspace    = std::unordered_map<uid_t, std::tuple<int64_t, int64_t, std::vector<float>>>;
using Replacements = std::unordered_map<uid_t, std::pair<uid_t, int64_t>>;

inline void
require(bool condition, char const *message) {
    if (!condition) {
        throw std::invalid_argument(message);
    }
}

inline int64_t
integer(json const &value) {
    return ::cudnn_frontend::serialization_detail::signed_integer(value);
}

inline uid_t
uid(std::string const &key) {
    return ::cudnn_frontend::serialization_detail::uid_key(key);
}

inline bool
same_scalar(Tensor_attributes::pass_by_values_t const &a, Tensor_attributes::pass_by_values_t const &b) {
    if (a.index() != b.index()) {
        return false;
    }
    return std::visit(
        [&](auto const &value) {
            using T           = std::decay_t<decltype(value)>;
            auto const &other = std::get<T>(b);
            return std::memcmp(&value, &other, sizeof(T)) == 0;
        },
        a);
}

inline void
validate_shape(Tensor_attributes const &tensor) {
    auto const &dim    = tensor.get_dim();
    auto const &stride = tensor.get_stride();
    require(!dim.empty() && dim.size() <= CUDNN_DIM_MAX && stride.size() == dim.size(),
            "Serialized tensor dimensions and strides have invalid ranks");
    for (size_t i = 0; i < dim.size(); ++i) {
        require(dim[i] >= 0 && stride[i] >= 0, "Serialized tensor dimensions or strides are invalid");
    }
}

inline int64_t
tensor_bytes(Tensor_attributes const &tensor) {
    validate_shape(tensor);
    auto const &dim    = tensor.get_dim();
    auto const &stride = tensor.get_stride();
    if (std::find(dim.begin(), dim.end(), 0) != dim.end()) {
        return 0;
    }
    int64_t span = 1;
    for (size_t i = 0; i < dim.size(); ++i) {
        if (stride[i] == 0) {
            continue;
        }
        require(dim[i] - 1 <= (std::numeric_limits<int64_t>::max() - span) / stride[i],
                "Serialized tensor address span overflows");
        span += (dim[i] - 1) * stride[i];
    }
    auto const bytes = detail::get_data_type_size(tensor.get_data_type());
    require(bytes > 0 && span <= std::numeric_limits<int64_t>::max() / static_cast<int64_t>(bytes),
            "Serialized tensor byte span is invalid");
    return span * static_cast<int64_t>(bytes);
}

// Tensor dumping copies the logical element count, independently of strides.
inline int64_t
dump_bytes(Tensor_attributes const &tensor) {
    validate_shape(tensor);
    if (std::find(tensor.get_dim().begin(), tensor.get_dim().end(), 0) != tensor.get_dim().end()) {
        return 0;
    }
    int64_t count = 1;
    for (auto extent : tensor.get_dim()) {
        require(count <= std::numeric_limits<int64_t>::max() / extent,
                "Serialized tensor dump element count overflows");
        count *= extent;
    }
    auto const width = static_cast<int64_t>(detail::get_data_type_size(tensor.get_data_type()));
    require(width > 0 && count <= std::numeric_limits<int64_t>::max() / width,
            "Serialized tensor dump byte count is invalid");
    return count * width;
}

// PBV argument packing reads one dtype-sized scalar, independently of shape.
// Unsupported sub-byte/Boolean scalar types fail backend packing before a read;
// a zero here is not a known allocation capacity for those types.
inline int64_t
scalar_read_bytes(Tensor_attributes const &tensor) {
    return static_cast<int64_t>(detail::get_data_type_size(tensor.get_data_type()));
}

inline int64_t
binding_read_bytes(Tensor_attributes const &tensor) {
    return tensor.get_is_pass_by_value() ? scalar_read_bytes(tensor) : tensor_bytes(tensor);
}

struct PlanMetadata {
    std::unordered_set<uid_t> uids;
    std::unordered_map<uid_t, Tensor_attributes::pass_by_values_t> scalars;
    Workspace workspace;
    std::unordered_map<uid_t, std::pair<int64_t, int64_t>> aligned;
    Replacements replacements;
    std::unordered_map<uid_t, std::shared_ptr<Tensor_attributes>> tensors;
    std::vector<std::pair<std::shared_ptr<Tensor_attributes>, char>> dumps;
    int64_t workspace_size = 0;
    uint64_t gid           = 0;
    bool has_structure     = false;
};

inline PlanMetadata
parse(json const &j) {
    PlanMetadata result;
    require(j.is_object(), "Serialized execution plan must be an object");
    require(j.at("cudnn_backend_data").is_string(), "Serialized backend plan must be a string");
    result.workspace_size = integer(j.at("fe_workspace_size"));
    require(result.workspace_size >= 0, "Serialized frontend workspace size must be nonnegative");

    auto const &uids = j.at("variant_pack_uids");
    require(uids.is_array() && uids.size() <= static_cast<size_t>(INT_MAX),
            "Serialized variant-pack UIDs must be an integer array");
    for (auto const &value : uids) {
        require(result.uids.insert(integer(value)).second, "Serialized variant-pack UID is repeated");
    }

    if (j.contains("pass_by_values")) {
        auto const &values = j.at("pass_by_values");
        require(values.is_object(), "Serialized pass_by_values must be an object");
        for (auto it = values.begin(); it != values.end(); ++it) {
            result.scalars.emplace(uid(it.key()), it.value().get<Tensor_attributes::pass_by_values_t>());
        }
    }

    if (j.contains("workspace_modifications")) {
        auto const &values = j.at("workspace_modifications");
        require(values.is_object(), "Serialized workspace_modifications must be an object");
        for (auto it = values.begin(); it != values.end(); ++it) {
            auto const key    = uid(it.key());
            auto const &entry = it.value();
            require(entry.is_array() && entry.size() == 3 && entry[2].is_array(),
                    "Serialized workspace modification must be [operation, offset, "
                    "words]");
            auto const kind   = integer(entry[0]);
            auto const offset = integer(entry[1]);
            auto const words  = entry[2].get<std::vector<float>>();
            require(kind >= 0 && kind <= 2, "Serialized workspace operation is unknown");
            require(offset >= 0 && offset <= result.workspace_size, "Serialized workspace offset is out of bounds");
            auto const available = result.workspace_size - offset;
            if (kind == 0) {
                require(words.size() <= static_cast<uint64_t>(available) / sizeof(float),
                        "Serialized workspace copy exceeds frontend workspace");
            } else if (kind == 1) {
                require(words.size() == 1 && std::isfinite(words[0]) && words[0] >= 0.0F &&
                            words[0] < std::ldexp(1.0F, 63) && std::trunc(words[0]) == words[0],
                        "Serialized workspace memset requires one nonnegative integral "
                        "byte count");
                require(static_cast<int64_t>(words[0]) <= available,
                        "Serialized workspace memset exceeds frontend workspace");
            } else {
                require(words.empty(), "Serialized pointer-only workspace operation must have no words");
            }
            require(result.scalars.count(key) == 0, "Serialized UID has conflicting scalar and workspace bindings");
            result.workspace.emplace(key, std::make_tuple(kind, offset, words));
        }
    }

    if (j.contains("aligned_operands")) {
        auto const &values = j.at("aligned_operands");
        require(values.is_object(), "Serialized aligned_operands must be an object");
        for (auto it = values.begin(); it != values.end(); ++it) {
            auto const &pair = it.value();
            require(pair.is_array() && pair.size() == 2, "Serialized aligned operand must have two integers");
            auto const alignment = integer(pair[0]);
            auto const width     = integer(pair[1]);
            require(alignment > 0 && (alignment & (alignment - 1)) == 0 &&
                        (width == 1 || width == 2 || width == 4 || width == 8 || width == 16 || width == 32),
                    "Serialized alignment or element width is invalid");
            result.aligned.emplace(uid(it.key()), std::make_pair(alignment, width));
        }
    }

    auto const &replacements = j.at("variant_pack_replacements");
    require(replacements.is_array(), "Serialized variant-pack replacements must be an array");
    std::unordered_set<uid_t> destinations;
    for (auto const &entry : replacements) {
        require(entry.is_array() && entry.size() == 2 && entry[1].is_array() && entry[1].size() == 2,
                "Serialized replacement must be [source, [destination, byte_offset]]");
        auto const source      = integer(entry[0]);
        auto const destination = integer(entry[1][0]);
        auto const offset      = integer(entry[1][1]);
        require(offset >= 0, "Serialized replacement offset must be nonnegative");
        require(destinations.insert(destination).second &&
                    result.replacements.emplace(source, std::make_pair(destination, offset)).second,
                "Serialized replacement has a duplicate source or destination");
        require(result.scalars.count(destination) == 0 && result.workspace.count(destination) == 0,
                "Serialized replacement destination has another binding owner");
    }

    // Validate all legacy note rows, but reconstruct the selected plan's actual notes from the backend.
    auto const &notes = j.at("behavior_notes");
    require(notes.is_array() && !notes.empty(), "Serialized behavior notes must contain note arrays");
    for (auto const &row : notes) {
        require(row.is_array(), "Serialized behavior-note row must be an array");
        for (auto const &note : row) {
            require(note.is_string() && json(note.get<BehaviorNote_t>()) == note,
                    "Serialized behavior note is unknown");
        }
    }

    result.has_structure = j.contains("nodes") || j.contains("tensors");
    if (result.has_structure) {
        auto const &gid = j.at("gid");
        require(gid.is_number_integer() && (gid.is_number_unsigned() || gid.get<int64_t>() >= 0),
                "Serialized graph ID must be an unsigned integer");
        result.gid          = gid.get<uint64_t>();
        auto const &tensors = j.at("tensors");
        require(tensors.is_array(), "Serialized graph tensors must be an array");
        if (j.contains("nodes")) {
            require(j.at("nodes").is_array(), "Serialized graph nodes must be an array");
        }
        for (auto const &info : tensors) {
            require(info.is_object() && info.at("dim").is_array() && info.at("stride").is_array() &&
                        info.at("is_virtual").is_boolean() && info.at("is_pass_by_value").is_boolean(),
                    "Serialized tensor record has invalid field types");
            auto const key = integer(info.at("uid"));
            for (auto const &value : info.at("dim")) {
                (void)integer(value);
            }
            for (auto const &value : info.at("stride")) {
                (void)integer(value);
            }
            for (auto field : {"ragged_offset_uid", "ragged_offset_multiplier", "alignment"}) {
                if (info.contains(field)) {
                    (void)integer(info.at(field));
                }
            }
            require(info.at("is_pass_by_value").get<bool>() || !info.contains("pass_by_value") ||
                        info.at("pass_by_value").is_null(),
                    "Serialized runtime scalar value requires its pass-by-value flag");
            auto tensor = std::make_shared<Tensor_attributes>(info.get<Tensor_attributes>());
            require(json(tensor->get_data_type()) == info.at("data_type"), "Serialized tensor data type is unknown");
            require(json(tensor->get_reordering_type()) == info.at("reordering_type"),
                    "Serialized tensor reordering type is unknown");
            require(tensor->validate().is_good(), "Serialized tensor has inconsistent attributes");
            auto const alignment = tensor->get_alignment();
            require(alignment > 0 && (alignment & (alignment - 1)) == 0,
                    "Serialized tensor alignment must be a positive power of two");
            validate_shape(*tensor);
            if (tensor->get_is_pass_by_value()) {
                require(std::count_if(tensor->get_dim().begin(),
                                      tensor->get_dim().end(),
                                      [](int64_t extent) { return extent > 1; }) <= 1,
                        "Serialized pass-by-value tensor has multiple nonsingleton "
                        "dimensions");
            }
            // Do not require byte-addressable storage for unconsumed virtual BOOLEAN or packed tensors.
            if (tensor->get_pass_by_value().has_value() && result.scalars.count(key) == 0) {
                require(!j.contains("pass_by_values"), "Serialized tensor scalar is absent from pass_by_values");
                require(result.workspace.count(key) == 0,
                        "Serialized UID has conflicting scalar and workspace bindings");
                result.scalars.emplace(key, tensor->get_pass_by_value().value());
            }
            auto scalar = result.scalars.find(key);
            if (scalar != result.scalars.end()) {
                require(tensor->get_is_pass_by_value() && tensor->get_pass_by_value().has_value() &&
                            same_scalar(scalar->second, tensor->get_pass_by_value().value()),
                        "Serialized scalar representations disagree");
                auto const capacity = std::visit([](auto const &value) { return sizeof(value); }, scalar->second);
                require(static_cast<uint64_t>(scalar_read_bytes(*tensor)) <= capacity,
                        "Serialized scalar tensor exceeds owned storage");
            }
            auto const work = result.workspace.find(key);
            if (work != result.workspace.end()) {
                require(!tensor->get_is_pass_by_value(), "Serialized workspace binding cannot own a host scalar");
                require(tensor_bytes(*tensor) <= result.workspace_size - std::get<1>(work->second),
                        "Serialized workspace tensor exceeds frontend workspace");
            }
            auto const aligned = result.aligned.find(key);
            if (aligned != result.aligned.end()) {
                require(
                    aligned->second.second == static_cast<int64_t>(detail::get_data_type_size(tensor->get_data_type())),
                    "Serialized aligned element width disagrees with tensor storage");
            }
            require(result.tensors.emplace(key, std::move(tensor)).second, "Serialized tensor UID is repeated");
        }
    }

    // Legacy tensor records may have reconstructed scalar owners after the top-level maps were read.
    for (auto const &[key, value] : result.scalars) {
        (void)value;
        require(result.workspace.count(key) == 0 && destinations.count(key) == 0,
                "Serialized scalar UID has another binding owner");
    }

    if (j.contains("tensors_to_dump")) {
        require(j.at("tensors_to_dump").is_array(), "Serialized tensor dumps must be an array");
        for (auto const &pair : j.at("tensors_to_dump")) {
            require(pair.is_array() && pair.size() == 2, "Serialized tensor dump must be [uid, format]");
            auto const key    = integer(pair[0]);
            auto const format = integer(pair[1]);
            require(format >= std::numeric_limits<char>::min() && format <= std::numeric_limits<char>::max(),
                    "Serialized tensor dump format is out of range");
            auto const tensor = result.tensors.find(key);
            // Plan-only blobs retain dump UIDs but deliberately omit their optional tensor metadata.
            if (tensor != result.tensors.end()) {
                result.dumps.emplace_back(tensor->second, static_cast<char>(format));
            }
        }
    }

    auto const maximum = static_cast<size_t>(INT_MAX);
    require(
        result.scalars.size() <= maximum && result.workspace.size() <= maximum && result.replacements.size() <= maximum,
        "Serialized variant-pack metadata has too many slots");
    require(result.uids.size() <= maximum - result.scalars.size() &&
                result.uids.size() + result.scalars.size() <= maximum - result.workspace.size() &&
                result.uids.size() + result.scalars.size() + result.workspace.size() <=
                    maximum - result.replacements.size(),
            "Serialized variant-pack metadata has too many slots");
    return result;
}

// Validate aliases in dependency order, carrying remaining allocation capacity through each chain.
// Caller-provided pointers have no allocation-size contract; retain only their declared tensor bounds.
inline void
validate_owned_spans(PlanMetadata const &metadata,
                     std::vector<std::pair<uid_t, std::pair<uid_t, int64_t>>> const &ordered) {
    std::unordered_map<uid_t, int64_t> capacities;
    std::unordered_set<uid_t> scalar_aliases;
    std::unordered_set<uid_t> workspace_aliases;
    for (auto const &[key, scalar] : metadata.scalars) {
        capacities.emplace(key,
                           std::visit([](auto const &value) { return static_cast<int64_t>(sizeof(value)); }, scalar));
        scalar_aliases.insert(key);
    }
    for (auto const &[key, work] : metadata.workspace) {
        capacities.emplace(key, metadata.workspace_size - std::get<1>(work));
        workspace_aliases.insert(key);
    }
    for (auto const &[source, edge] : ordered) {
        auto available  = capacities.find(source);
        auto const from = metadata.tensors.find(source);
        auto const to   = metadata.tensors.find(edge.first);
        if (workspace_aliases.count(source)) {
            require(to == metadata.tensors.end() || !to->second->get_is_pass_by_value(),
                    "Serialized workspace alias cannot bind a host scalar");
            workspace_aliases.insert(edge.first);
        }
        if (available == capacities.end() && from != metadata.tensors.end() &&
            detail::get_data_type_size(from->second->get_data_type()) > 0) {
            auto const span = tensor_bytes(*from->second);
            auto const read = from->second->get_is_pass_by_value() ? scalar_read_bytes(*from->second) : 0;
            available       = capacities.emplace(source, std::max(span, read)).first;
        }
        if (available != capacities.end()) {
            require(edge.second <= available->second, "Serialized replacement offset exceeds source capacity");
            auto remaining = available->second - edge.second;
            if (to != metadata.tensors.end()) {
                auto const target_bytes = binding_read_bytes(*to->second);
                require(target_bytes <= remaining, "Serialized replacement exceeds its source tensor span");
                if (from != metadata.tensors.end() && detail::get_data_type_size(from->second->get_data_type()) > 0) {
                    auto const span     = tensor_bytes(*from->second);
                    auto const read     = from->second->get_is_pass_by_value() ? scalar_read_bytes(*from->second) : 0;
                    auto const declared = std::max(span, read);
                    require(edge.second <= declared && target_bytes <= declared - edge.second,
                            "Serialized replacement exceeds its declared source tensor span");
                }
            } else if (scalar_aliases.count(source)) {
                // Legacy plan-only singleton PBV Slice writers legitimately emit zero-offset aliases.
                require(edge.second == 0,
                        "Serialized scalar replacement offset must "
                        "be zero without tensor metadata");
            }
            capacities[edge.first] = remaining;
        }
        if (scalar_aliases.count(source)) {
            scalar_aliases.insert(edge.first);
        }
    }
    for (auto const &[tensor, format] : metadata.dumps) {
        (void)format;
        auto const bytes = dump_bytes(*tensor);
        require(bytes <= tensor_bytes(*tensor), "Serialized tensor dump exceeds its declared storage span");
        auto const available = capacities.find(tensor->get_uid());
        require(available == capacities.end() || bytes <= available->second,
                "Serialized tensor dump exceeds its binding capacity");
    }
}

}  // namespace cudnn_frontend::graph::plan_serialization_detail

#endif
