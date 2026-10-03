// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "variant_pack.h"

#include <algorithm>
#include <array>
#include <limits>
#include <string>
#include <utility>
#include <vector>
#include <pybind11/stl.h>

namespace cudnn_frontend {
namespace python_bindings {

// Shared dense/THD validation for opaque F8_128x4 scale storage. Only immutable
// tile sizes/host slots are retained; addresses, capacities and devices are current.
class SdpaMxScaleBinding {
   public:
    explicit SdpaMxScaleBinding(const pybind11::object &spec) {
        sizes_           = spec.attr("quant").attr("sf_sizes").cast<std::array<int64_t, 3>>();
        const auto order = spec.attr("order").cast<std::vector<std::string>>();
        for (size_t i = 0; i < names_.size(); ++i) {
            const auto found = std::find(order.begin(), order.end(), names_[i]);
            if (found == order.end()) invalid("MXFP8 host is missing " + std::string(names_[i]));
            indices_[i] = static_cast<size_t>(found - order.begin());
        }
        for (auto size : sizes_)
            if (size <= 0) invalid("MXFP8 tile size must be positive");
        device_ = spec.attr("device_index").cast<int64_t>();
        heads_  = {spec.attr("qh").cast<int64_t>(), spec.attr("kh").cast<int64_t>(), spec.attr("kh").cast<int64_t>()};
    }

    void
    bind(const std::vector<NativeOperandView> &facts,
         size_t first_sf,
         pybind11::tuple *frame,
         bool packed,
         bool paged,
         int64_t batch,
         int64_t sq,
         int64_t sk,
         int64_t page_size) const {
        std::array<int64_t, 3> tiles;
        for (size_t i = 0; i < 3; ++i) {
            const auto &f = facts[first_sf + i];
            if (!f.filled) invalid("MXFP8 requires " + std::string(names_[i]));
            if (f.device_type != -1 && (f.device_type != kDLCUDA || f.device_id != device_))
                invalid("MXFP8 scale storage must be on this plan's CUDA device");
            const auto bytes = byte_count(f);
            if (f.observed_bytes >= 0 && f.observed_bytes < bytes) invalid("MXFP8 scale observed storage is too small");
            if (bytes && (!f.pointer || f.pointer % 16)) invalid("MXFP8 scale storage must be 16-byte aligned");
            const auto row = multiply(heads_[i], sizes_[i]);
            int64_t count;
            if (packed) {
                if (bytes % row) invalid("MXFP8 scales must hold whole packed tile rows");
                count               = bytes / row;
                const auto &operand = facts[i];
                bool empty          = operand.observed_bytes == 0 ||
                             std::find(operand.shape.begin(), operand.shape.end(), 0) != operand.shape.end();
                if (!count && !empty) invalid("empty MXFP8 scales require a zero-capacity operand");
            } else if (paged && i > 0) {
                count = page_size / 128;
                if (facts[1].shape.empty()) invalid("MXFP8 paged K must expose its page count");
                if (bytes != multiply(multiply(facts[1].shape[0], row), count))
                    invalid("MXFP8 scale size does not match the page pool");
            } else {
                count = add(i == 0 ? sq : sk, 127) / 128;
                if (bytes != multiply(multiply(batch, row), count))
                    invalid("MXFP8 scale size does not match compiled dense geometry");
            }
            if (count > std::numeric_limits<int32_t>::max()) invalid("MXFP8 tile count exceeds Int32");
            tiles[i] = std::max<int64_t>(1, count);
            if (frame) (*frame)[indices_[i]] = pybind11::int_(bytes ? f.pointer : 0);
        }
        if (tiles[1] != tiles[2]) invalid("MXFP8 K/V scales must have the same packed tile count");
        if (frame) (*frame)[indices_[3]] = pybind11::make_tuple(tiles[0], tiles[1], tiles[2]);
    }

   private:
    [[noreturn]] static void
    invalid(const std::string &message) {
        throw pybind11::value_error("cudnn.sdpa: " + message);
    }
    static int64_t
    multiply(int64_t a, int64_t b) {
        if (a < 0 || b < 0 || (b && a > std::numeric_limits<int64_t>::max() / b))
            invalid("MXFP8 geometry must fit in int64");
        return a * b;
    }
    static int64_t
    add(int64_t a, int64_t b) {
        if (a < 0 || b < 0 || a > std::numeric_limits<int64_t>::max() - b) invalid("MXFP8 geometry must fit in int64");
        return a + b;
    }
    static int64_t
    byte_count(const NativeOperandView &f) {
        const auto code = f.dtype.code, bits = f.dtype.bits;
        const bool supported = (code == kDLFloat && (bits == 16 || bits == 32 || bits == 64)) ||
                               (code == kDLInt && (bits == 8 || bits == 32 || bits == 64)) ||
                               (code == kDLBfloat && bits == 16) ||
                               (bits == 8 && (code == kDLUInt || code == kDLBool || code == kDLFloat8_e4m3fn ||
                                              code == kDLFloat8_e5m2 || code == kDLFloat8_e8m0fnu));
        if (!supported || f.dtype.lanes != 1) invalid("unsupported MXFP8 scale storage dtype");
        if (!f.stride.empty() && f.stride.size() != f.shape.size()) invalid("MXFP8 scale shape/stride rank mismatch");
        std::vector<std::pair<int64_t, int64_t>> dimensions;
        int64_t compact = 1;
        bool empty      = false;
        for (size_t i = f.shape.size(); i-- > 0;) {
            const auto n = f.shape[i], stride = f.stride.empty() ? compact : f.stride[i];
            if (n < 0 || stride < 0) invalid("MXFP8 scale geometry must be nonnegative");
            empty |= n == 0;
            if (n > 1) dimensions.emplace_back(stride, n);
            compact = multiply(compact, n);
        }
        if (empty) return 0;
        std::sort(dimensions.begin(), dimensions.end());
        int64_t extent = 1;
        for (const auto &dim : dimensions) {
            if (dim.first != extent) invalid("MXFP8 scales require dense non-overlapping physical storage");
            extent = multiply(extent, dim.second);
        }
        return multiply(extent, bits / 8);
    }
    const std::array<const char *, 4> names_ = {"sf_q_ptr", "sf_k_ptr", "sf_v_ptr", "sf_tiles"};
    std::array<size_t, 4> indices_;
    std::array<int64_t, 3> sizes_, heads_;
    int64_t device_;
};

}  // namespace python_bindings
}  // namespace cudnn_frontend
