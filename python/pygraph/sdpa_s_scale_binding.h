// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "variant_pack.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <optional>
#include <string>
#include <vector>

#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace cudnn_frontend {
namespace python_bindings {

// Common dense/THD S binding. Only address-space declarations and host slots
// survive plan build. Device values are read by the kernel, including replay;
// host pass-by-value values enter the launch frame without a GPU copy or sync.
class SdpaSScaleBinding {
   public:
    explicit SdpaSScaleBinding(const nanobind::object &spec) {
        const auto quant = spec.attr("quant");
        if (nanobind::len(nanobind::object(quant.attr("sf_sizes"))))
            invalid("public S scales are per-tensor FP8 operands");
        host_   = nanobind::cast<std::array<bool, 2>>(quant.attr("s_scales"));
        device_ = nanobind::cast<int64_t>(spec.attr("device_index"));
        if (nanobind::hasattr(quant, "s_defaults"))
            defaults_ = nanobind::cast<std::array<std::optional<float>, 2>>(quant.attr("s_defaults"));
        const auto order = nanobind::cast<std::vector<std::string>>(spec.attr("order"));
        for (size_t i = 0; i < 2; ++i) {
            if (defaults_[i] && !host_[i]) invalid("embedded S scales require host pass-by-value mode");
            const auto arg   = std::string(name(i)) + (host_[i] ? "_value" : "_ptr");
            const auto found = std::find(order.begin(), order.end(), arg);
            if (found == order.end()) invalid("public S host is missing " + arg);
            slots_[i] = static_cast<size_t>(found - order.begin());
        }
    }

    static const char *
    name(size_t index) {
        return index == 0 ? "scale_s" : "descale_s";
    }

    void
    bind(const std::vector<NativeOperandView> &facts, size_t first, nanobind::list *frame) const {
        for (size_t i = 0; i < 2; ++i) {
            const auto &f = facts.at(first + i);
            if (!f.filled && defaults_[i]) {
                if (frame) (*frame)[slots_[i]] = nanobind::float_(*defaults_[i]);
                continue;
            }
            if (!f.filled) invalid(std::string(name(i)) + " is required by this specialization");
            if (f.dtype.code != kDLFloat || f.dtype.bits != 32 || f.dtype.lanes != 1 || !f.pointer || f.pointer % 4 ||
                (f.observed_bytes >= 0 && f.observed_bytes < 4) ||
                std::any_of(f.shape.begin(), f.shape.end(), [](int64_t n) { return n != 1; }))
                invalid(std::string(name(i)) + " requires one aligned float32 element with sufficient storage");
            if (host_[i]) {
                // Never dereference an unobserved raw address on the CPU.
                if (f.device_type != kDLCPU) invalid(std::string(name(i)) + " requires an observed CPU scalar");
                float value;
                std::memcpy(&value, reinterpret_cast<const void *>(f.pointer), sizeof(value));
                if (frame) (*frame)[slots_[i]] = nanobind::float_(value);
            } else {
                if (f.device_type >= 0 && (f.device_type != kDLCUDA || f.device_id != device_))
                    invalid(std::string(name(i)) + " must be on CUDA device " + std::to_string(device_));
                if (frame) (*frame)[slots_[i]] = nanobind::int_(f.pointer);
            }
        }
    }

   private:
    [[noreturn]] static void
    invalid(const std::string &message) {
        throw nanobind::value_error(("cudnn.sdpa: " + message).c_str());
    }
    std::array<bool, 2> host_{};
    std::array<size_t, 2> slots_{};
    std::array<std::optional<float>, 2> defaults_{};
    int64_t device_ = -1;
};

}  // namespace python_bindings
}  // namespace cudnn_frontend
