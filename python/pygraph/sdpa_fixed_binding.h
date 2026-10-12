// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "variant_pack.h"
#include <algorithm>
#include <limits>
#include <string>
#include <utility>
#include <pybind11/stl.h>

namespace cudnn_frontend {
namespace python_bindings {

// Fixed declarations shared by half backward and SM80 forward. These retain no
// runtime storage observations, pointers, streams, or mutable argument frames.
class FixedSdpaOperands {
   protected:
    using Geometry = std::vector<std::pair<int64_t, int64_t>>;
    struct Operand {
        bool enabled  = false;
        int64_t bytes = 0, alignment = 1;
        std::string name, dtype;
        uint8_t code = 0, bits = 0;
        Geometry geometry;
    };
    std::vector<Operand> operands_;
    std::string name_;
    int64_t device_;
    [[noreturn]] void
    invalid(const std::string &message) const {
        throw pybind11::value_error(name_ + ": " + message);
    }

    int64_t
    multiply(int64_t a, int64_t b) const {
        if (a < 0 || b < 0 || (b && a > std::numeric_limits<int64_t>::max() / b))
            invalid("operand span must be nonnegative and fit in int64");
        return a * b;
    }

    int64_t
    end(int64_t pointer, int64_t bytes) const {
        if (pointer <= 0 || bytes < 0 || pointer > std::numeric_limits<int64_t>::max() - bytes)
            invalid("storage address range must fit in int64");
        return pointer + bytes;
    }

    Geometry
    geometry(std::span<const int64_t> shape, std::span<const int64_t> strides) const {
        if (!strides.empty() && strides.size() != shape.size()) invalid("shape and stride ranks differ");
        Geometry result;
        int64_t compact = 1;
        for (size_t j = shape.size(); j > 0; --j) {
            const auto i      = j - 1;
            const auto stride = strides.empty() ? compact : strides[i];
            if (shape[i] != 1) result.emplace_back(shape[i], stride);
            if (strides.empty()) compact = multiply(compact, shape[i]);
        }
        std::reverse(result.begin(), result.end());
        return result;
    }

    FixedSdpaOperands(std::string name, int64_t device) : name_(std::move(name)), device_(device) {
        if (device_ < 0) invalid("invalid fixed operand device");
    }

    void
    add_operand(const std::string &name, const pybind11::handle &value, const pybind11::handle &declared) {
        Operand op;
        op.name = name;
        if (!value.is_none()) {
            op.enabled = true;
            op.dtype   = value.attr("dtype").cast<std::string>();
            if (op.dtype == "bfloat16") {
                op.code = kDLBfloat;
                op.bits = 16;
            } else if (op.dtype == "float16" || op.dtype == "float32") {
                op.code = kDLFloat;
                op.bits = op.dtype == "float16" ? 16 : 32;
            } else if (op.dtype == "int32" || op.dtype == "int64") {
                // int64: bound ragged offsets (graph THD ports read on device).
                op.code = kDLInt;
                op.bits = op.dtype == "int32" ? 32 : 64;
            } else {
                invalid("native fixed binding does not support " + op.dtype);
            }
            op.bytes     = multiply(value.attr("span").cast<int64_t>(), op.bits / 8);
            op.alignment = value.attr("alignment").cast<int64_t>();
            if (op.alignment <= 0) invalid("invalid operand alignment");
        }
        if (!declared.is_none()) {
            auto pair    = declared.cast<pybind11::tuple>();
            auto shape   = pair[0].cast<std::vector<int64_t>>();
            auto strides = pair[1].cast<std::vector<int64_t>>();
            op.geometry  = geometry(shape, strides);
        }
        operands_.push_back(std::move(op));
    }

    bool
    validate(const Operand &op, const NativeOperandView &f, const std::string &device_label) const {
        if (!op.enabled) {
            if (f.filled) invalid(op.name + " was not compiled into this specialization");
            return false;
        }
        if (!f.filled) invalid(op.name + " is required by this specialization");
        if (!((f.device_type == kDLCUDA && f.device_id == device_) || (f.device_type == -1 && f.device_id == -1)))
            invalid(device_label + " must be on CUDA device " + std::to_string(device_));
        if (f.dtype.bits && (f.dtype.code != op.code || f.dtype.bits != op.bits || f.dtype.lanes != 1))
            invalid(op.name + " must be " + op.dtype);
        if (f.pointer <= 0 || f.pointer % op.alignment)
            invalid(op.name + " base address must be " + std::to_string(op.alignment) + "-byte aligned");
        if (f.observed_bytes >= 0 && f.observed_bytes < op.bytes)
            invalid(op.name + " backing storage is too small for the declared strides");
        end(f.pointer, op.bytes);
        return true;
    }
};
}  // namespace python_bindings
}  // namespace cudnn_frontend
