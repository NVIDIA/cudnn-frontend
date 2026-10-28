/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// XLA's variadic argmax over (value, s32 index) with only the index read, as a custom reduction. Every shape below
// is an argmax operand shape taken from XLA HLO. Each case is checked element for element against a host reference.

#include <catch2/catch_test_macros.hpp>
#include "../utils/helpers.h"

#include <cudnn_frontend.h>

#include <cuda_bf16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

namespace fe = cudnn_frontend;

using Attributes = fe::graph::Custom_reduction_attributes;

struct Xla_shape {
    std::vector<int64_t> dims;
    int64_t axis;
};

// (f32 value, s32 index) argmax operand shapes, reduced axis innermost.
std::vector<Xla_shape> const argmax_f32_shapes = {
    {{1, 1, 32000}, 2},   {{600, 1, 151936}, 2}, {{400, 1, 151936}, 2}, {{1, 1, 256128}, 2},   {{1, 1, 128256}, 2},
    {{128, 1, 32000}, 2}, {{16, 1, 151936}, 2},  {{1, 1, 151936}, 2},   {{400, 1, 262144}, 2}, {{1, 1, 262144}, 2},
    {{256, 1, 32000}, 2}, {{400, 1, 256128}, 2}, {{200, 1, 256128}, 2}, {{800, 1, 201088}, 2}, {{1, 1, 201088}, 2},
    {{32, 1, 32000}, 2},  {{16, 1, 32000}, 2},   {{400, 1, 128256}, 2}, {{600, 1, 128256}, 2}, {{256, 1, 202048}, 2},
    {{1, 1, 202048}, 2},  {{600, 1, 32000}, 2},  {{256, 1, 32768}, 2},  {{1, 1, 32768}, 2},    {{512, 1000}, 1},
    {{16, 1, 256128}, 2}, {{8, 1, 32000}, 2},    {{16, 1, 128256}, 2},  {{1, 256000}, 1},      {{32, 128, 32128}, 2},
};

// (bf16 value, s32 index) argmax operand shapes.
std::vector<Xla_shape> const argmax_bf16_shapes = {
    {{1, 262144}, 1},
};

std::string
shape_string(std::vector<int64_t> const& dims, int64_t axis) {
    std::string s = "[";
    for (size_t i = 0; i < dims.size(); ++i) {
        s += (i ? "," : "") + std::to_string(dims[i]);
    }
    return s + "]/axis" + std::to_string(axis);
}

std::vector<int64_t>
packed_stride(std::vector<int64_t> const& dims) {
    std::vector<int64_t> stride(dims.size(), 1);
    for (size_t i = dims.size(); i > 1; --i) {
        stride[i - 2] = stride[i - 1] * dims[i - 1];
    }
    return stride;
}

std::vector<int64_t>
keep_dims(std::vector<int64_t> dims, int64_t axis) {
    dims[static_cast<size_t>(axis)] = 1;
    return dims;
}

// The tensor seen as [outer, reduced, inner].
struct Layout {
    int64_t outer   = 1;
    int64_t reduced = 1;
    int64_t inner   = 1;

    Layout(std::vector<int64_t> const& dims, int64_t axis) {
        for (int64_t d = 0; d < static_cast<int64_t>(dims.size()); ++d) {
            auto const extent = dims[static_cast<size_t>(d)];
            if (d < axis) {
                outer *= extent;
            } else if (d == axis) {
                reduced = extent;
            } else {
                inner *= extent;
            }
        }
    }
    int64_t
    rows() const {
        return outer * inner;
    }
    int64_t
    elements() const {
        return outer * reduced * inner;
    }
    int64_t
    offset(int64_t row, int64_t r) const {
        return (row / inner) * reduced * inner + r * inner + (row % inner);
    }
};

class Device_buffer {
   public:
    explicit Device_buffer(size_t bytes) {
        if (bytes != 0) {
            CUDA_CHECK(cudaMalloc(&ptr_, bytes));
        }
    }
    ~Device_buffer() {
        if (ptr_ != nullptr) {
            cudaFree(ptr_);
        }
    }
    Device_buffer(Device_buffer const&) = delete;
    Device_buffer&
    operator=(Device_buffer const&) = delete;
    void*
    get() const {
        return ptr_;
    }

   private:
    void* ptr_ = nullptr;
};

struct Case_result {
    bool built          = false;
    std::string message = "";
    int64_t mismatches  = -1;
    int64_t checked     = 0;
};

void
check_case(std::string const& name, Case_result const& result) {
    INFO(name << (result.message.empty() ? "" : ": " + result.message));
    CHECK(result.built);
    CHECK(result.mismatches == 0);
}

fe::error_t
build(fe::graph::Graph& graph, cudnnHandle_t handle) {
    auto status = graph.validate();
    if (status.is_good()) {
        status = graph.build_operation_graph(handle);
    }
    if (status.is_good()) {
        status = graph.create_execution_plans({fe::HeurMode_t::A});
    }
    if (status.is_good()) {
        status = graph.check_support();
    }
    if (status.is_good()) {
        status = graph.build_plans(fe::BuildPlanPolicy_t::HEURISTICS_CHOICE);
    }
    return status;
}

using Arg_nan_rule = fe::graph::reduction_modifiers::Arg_nan_rule;

Attributes
xla_argmax(std::string const& name, int64_t axis, fe::DataType_t value_type, Arg_nan_rule rule = Arg_nan_rule::XLA) {
    return fe::graph::reduction_modifiers::argmax(
        Attributes().set_name(name).set_reduction_dims({axis}), value_type, fe::DataType_t::INT32, rule);
}

// Ties: the row maximum is at two positions. OneNaN, TwoNaN: one or two NaN per row. SignedZeroTies: +0 and -0 are
// the row maximum. AllNegInf: every value is -inf, so every element ties with the identity (-inf, largest index).
enum class Fill { Ties, OneNaN, TwoNaN, SignedZeroTies, AllNegInf };

char const*
fill_name(Fill fill) {
    switch (fill) {
        case Fill::Ties:
            return "ties";
        case Fill::OneNaN:
            return "one_nan";
        case Fill::TwoNaN:
            return "two_nan";
        case Fill::SignedZeroTies:
            return "signed_zero_ties";
        case Fill::AllNegInf:
            return "all_neg_inf";
    }
    return "";
}

// The torch-rule argmax as a left fold that keeps the accumulator on the left: the first NaN wins, otherwise the
// largest value, and among equal values (+0 equals -0) the smallest index value.
int32_t
reference_argmax(std::vector<float> const& values,
                 std::vector<int32_t> const& indices,
                 Layout const& layout,
                 int64_t row) {
    float best_value   = -std::numeric_limits<float>::infinity();
    int32_t best_index = std::numeric_limits<int32_t>::max();
    for (int64_t r = 0; r < layout.reduced; ++r) {
        auto const at    = layout.offset(row, r);
        auto const value = values[at];
        if (std::isnan(value)) {
            return indices[at];
        }
        if (value > best_value || (value == best_value && indices[at] < best_index)) {
            best_value = value;
            best_index = indices[at];
        }
    }
    return best_index;
}

// The exact set of correct device answers. XLA's comparator keeps lhs when both values are NaN, so it is not
// commutative there, and the reduction order picks among the NaN positions: each NaN index is correct. A row without
// NaN has one answer. The torch rule always has one answer: the first NaN, else reference_argmax.
std::vector<int32_t>
accepted_argmax(std::vector<float> const& values,
                std::vector<int32_t> const& indices,
                Layout const& layout,
                int64_t row,
                Arg_nan_rule rule) {
    if (rule == Arg_nan_rule::TORCH) {
        return {reference_argmax(values, indices, layout, row)};
    }
    std::vector<int32_t> nan_indices;
    for (int64_t r = 0; r < layout.reduced; ++r) {
        auto const at = layout.offset(row, r);
        if (std::isnan(values[at])) {
            nan_indices.push_back(indices[at]);
        }
    }
    if (!nan_indices.empty()) {
        return nan_indices;
    }
    return {reference_argmax(values, indices, layout, row)};
}

// Builds and runs one argmax. values/indices are host copies of what the device reads (values already rounded to
// the stream type). Returns the index output.
Case_result
run_argmax(cudnnHandle_t handle,
           std::vector<int64_t> const& dims,
           int64_t axis,
           fe::DataType_t value_type,
           std::vector<float> const& values,
           std::vector<int32_t> const& indices,
           Arg_nan_rule rule = Arg_nan_rule::XLA) {
    Case_result result;
    Layout const layout(dims, axis);

    fe::graph::Graph graph;
    graph.set_intermediate_data_type(fe::DataType_t::FLOAT);
    auto const stride = packed_stride(dims);
    auto X_value      = graph.tensor(
        fe::graph::Tensor_attributes().set_name("value").set_dim(dims).set_stride(stride).set_data_type(value_type));
    auto X_index =
        graph.tensor(fe::graph::Tensor_attributes().set_name("index").set_dim(dims).set_stride(stride).set_data_type(
            fe::DataType_t::INT32));
    auto outputs        = graph.custom_reduction({X_value, X_index}, xla_argmax("xla_argmax", axis, value_type, rule));
    auto const out_dims = keep_dims(dims, axis);
    // Only the index is read, as in XLA; the value result stays virtual.
    outputs.at(1)
        ->set_output(true)
        .set_data_type(fe::DataType_t::INT32)
        .set_dim(out_dims)
        .set_stride(packed_stride(out_dims));

    auto status = build(graph, handle);
    if (!status.is_good()) {
        result.message = status.get_message();
        return result;
    }
    result.built = true;

    size_t const value_bytes = value_type == fe::DataType_t::BFLOAT16 ? sizeof(nv_bfloat16) : sizeof(float);
    Device_buffer d_value(static_cast<size_t>(layout.elements()) * value_bytes);
    Device_buffer d_index(static_cast<size_t>(layout.elements()) * sizeof(int32_t));
    Device_buffer d_out(static_cast<size_t>(layout.rows()) * sizeof(int32_t));
    if (value_type == fe::DataType_t::BFLOAT16) {
        std::vector<nv_bfloat16> staged(values.size());
        for (size_t i = 0; i < values.size(); ++i) {
            staged[i] = __float2bfloat16(values[i]);
        }
        CUDA_CHECK(cudaMemcpy(d_value.get(), staged.data(), staged.size() * value_bytes, cudaMemcpyHostToDevice));
    } else {
        CUDA_CHECK(cudaMemcpy(d_value.get(), values.data(), values.size() * value_bytes, cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemcpy(d_index.get(), indices.data(), indices.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_out.get(), 0xFF, static_cast<size_t>(layout.rows()) * sizeof(int32_t)));

    int64_t workspace_size = 0;
    REQUIRE(graph.get_workspace_size(workspace_size).is_good());
    Device_buffer workspace(static_cast<size_t>(workspace_size));
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> pack = {
        {X_value, d_value.get()}, {X_index, d_index.get()}, {outputs.at(1), d_out.get()}};
    auto exec = graph.execute(handle, pack, workspace.get());
    if (!exec.is_good()) {
        result.built   = false;
        result.message = "execute: " + exec.get_message();
        return result;
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<int32_t> got(static_cast<size_t>(layout.rows()));
    CUDA_CHECK(cudaMemcpy(got.data(), d_out.get(), got.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
    result.mismatches = 0;
    for (int64_t row = 0; row < layout.rows(); ++row) {
        auto const accepted = accepted_argmax(values, indices, layout, row, rule);
        auto const value    = got[static_cast<size_t>(row)];
        if (std::find(accepted.begin(), accepted.end(), value) == accepted.end()) {
            if (result.mismatches == 0) {
                result.message = "first mismatch at row " + std::to_string(row) + ": got " + std::to_string(value) +
                                 ", want " + std::to_string(accepted.front()) +
                                 (accepted.size() > 1 ? " or another NaN index" : "");
            }
            ++result.mismatches;
        }
        ++result.checked;
    }
    return result;
}

float
round_to(fe::DataType_t type, float value) {
    return type == fe::DataType_t::BFLOAT16 ? __bfloat162float(__float2bfloat16(value)) : value;
}

// Fills the values of one argmax case; see Fill.
void
fill_argmax_values(std::vector<float>& values, Layout const& layout, fe::DataType_t type, Fill fill, uint32_t seed) {
    values.resize(static_cast<size_t>(layout.elements()));
    if (fill == Fill::AllNegInf) {
        std::fill(values.begin(), values.end(), -std::numeric_limits<float>::infinity());
        return;
    }
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> uniform(-1000.0f, 1000.0f);
    for (auto& v : values) {
        v = round_to(type, uniform(rng));
    }
    for (int64_t row = 0; row < layout.rows(); ++row) {
        std::uniform_int_distribution<int64_t> position(0, layout.reduced - 1);
        auto const p = position(rng);
        auto q       = position(rng);
        if (layout.reduced > 1) {
            while (q == p) {
                q = position(rng);
            }
        }
        auto const at_p = layout.offset(row, p);
        auto const at_q = layout.offset(row, q);
        switch (fill) {
            case Fill::Ties:
                values[at_p] = 2048.0f;
                values[at_q] = 2048.0f;
                break;
            case Fill::OneNaN:
                values[at_p] = std::numeric_limits<float>::quiet_NaN();
                break;
            case Fill::TwoNaN:
                values[at_p] = std::numeric_limits<float>::quiet_NaN();
                values[at_q] = std::numeric_limits<float>::quiet_NaN();
                break;
            case Fill::SignedZeroTies:
                for (int64_t r = 0; r < layout.reduced; ++r) {
                    auto& v = values[layout.offset(row, r)];
                    v       = round_to(type, -std::fabs(v) - 1.0f);
                }
                // Alternate which of the two positions holds -0.
                values[at_p] = row % 2 == 0 ? 0.0f : -0.0f;
                values[at_q] = row % 2 == 0 ? -0.0f : 0.0f;
                break;
            case Fill::AllNegInf:
                break;
        }
    }
}

std::vector<int32_t>
iota_along_axis(Layout const& layout) {
    std::vector<int32_t> indices(static_cast<size_t>(layout.elements()));
    for (int64_t row = 0; row < layout.rows(); ++row) {
        for (int64_t r = 0; r < layout.reduced; ++r) {
            indices[static_cast<size_t>(layout.offset(row, r))] = static_cast<int32_t>(r);
        }
    }
    return indices;
}

void
run_argmax_family(char const* family,
                  std::vector<Xla_shape> const& shapes,
                  fe::DataType_t value_type,
                  Arg_nan_rule rule = Arg_nan_rule::XLA) {
    auto handle_ptr = create_cudnn_handle();
    auto handle     = *handle_ptr;
    uint32_t seed   = 1;
    for (auto const& shape : shapes) {
        Layout const layout(shape.dims, shape.axis);
        auto const indices = iota_along_axis(layout);
        for (auto const fill : {Fill::Ties, Fill::OneNaN, Fill::TwoNaN, Fill::SignedZeroTies, Fill::AllNegInf}) {
            std::vector<float> values;
            fill_argmax_values(values, layout, value_type, fill, seed++);
            check_case(std::string(family) + " " + shape_string(shape.dims, shape.axis) + "/" + fill_name(fill),
                       run_argmax(handle, shape.dims, shape.axis, value_type, values, indices, rule));
        }
    }
}

char const*
custom_reduction_runtime_skip_reason() {
#if CUDART_VERSION < 13010
    return "Test requires cuda toolkit 13.1 or above";
#else
    if (CUDNN_VERSION < 92900 || fe::detail::get_backend_version() < 92900) {
        return "Custom reduction requires cuDNN v9.29.0";
    }
    if (!is_blackwell_computing_arch() || !is_arch_supported_by_cudnn()) {
        return "Custom reduction requires the TensorIR MemBound engine on data-center Blackwell";
    }
    return nullptr;
#endif
}

}  // namespace

TEST_CASE("Custom reduction XLA argmax f32 value, s32 index", "[custom_reduction][xla][argmax]") {
    if (auto const* reason = custom_reduction_runtime_skip_reason(); reason != nullptr) {
        SKIP(reason);
    }
    run_argmax_family("argmax_f32_s32", argmax_f32_shapes, fe::DataType_t::FLOAT);
}

TEST_CASE("Custom reduction XLA argmax bf16 value, s32 index", "[custom_reduction][xla][argmax]") {
    if (auto const* reason = custom_reduction_runtime_skip_reason(); reason != nullptr) {
        SKIP(reason);
    }
    run_argmax_family("argmax_bf16_s32", argmax_bf16_shapes, fe::DataType_t::BFLOAT16);
}

// The same shapes and data with the torch rule (the helper's default): the first NaN wins, so a row with two NaN has
// one answer.
TEST_CASE("Custom reduction XLA argmax shapes with the torch rule, f32 value, s32 index",
          "[custom_reduction][xla][argmax][torch]") {
    if (auto const* reason = custom_reduction_runtime_skip_reason(); reason != nullptr) {
        SKIP(reason);
    }
    run_argmax_family("argmax_torch_f32_s32", argmax_f32_shapes, fe::DataType_t::FLOAT, Arg_nan_rule::TORCH);
}

TEST_CASE("Custom reduction XLA argmax shapes with the torch rule, bf16 value, s32 index",
          "[custom_reduction][xla][argmax][torch]") {
    if (auto const* reason = custom_reduction_runtime_skip_reason(); reason != nullptr) {
        SKIP(reason);
    }
    run_argmax_family("argmax_torch_bf16_s32", argmax_bf16_shapes, fe::DataType_t::BFLOAT16, Arg_nan_rule::TORCH);
}

// XLA splits an argmax in two. The second stage reduces the (value, index) partials to the index alone, which
// executes: its indices are global positions, not an iota, so the tie-break compares index values. Its result is
// checked against the argmax of the whole unsplit row.
TEST_CASE("Custom reduction XLA split argmax second stage", "[custom_reduction][xla][argmax][split]") {
    if (auto const* reason = custom_reduction_runtime_skip_reason(); reason != nullptr) {
        SKIP(reason);
    }
    struct Split {
        int64_t batch;
        int64_t vocab;
        int64_t parts;
    };
    // Sourced (batch, vocab) pairs, each split into a partial count of the size XLA's split reduction produces.
    std::vector<Split> const splits = {
        {1, 32000, 500}, {600, 151936, 512}, {400, 262144, 512}, {800, 201088, 128}, {256, 32768, 128}};
    auto handle_ptr = create_cudnn_handle();
    auto handle     = *handle_ptr;
    uint32_t seed   = 1000;
    for (auto const& split : splits) {
        for (auto const fill : {Fill::Ties, Fill::OneNaN}) {
            int64_t const chunk = (split.vocab + split.parts - 1) / split.parts;
            Layout const full({split.batch, split.vocab}, 1);
            std::vector<float> row_values;
            fill_argmax_values(row_values, full, fe::DataType_t::FLOAT, fill, seed++);
            // Stage 1 on the host: per part, the (value, global index) winner.
            std::vector<float> values(static_cast<size_t>(split.batch * split.parts));
            std::vector<int32_t> indices(values.size());
            std::vector<int32_t> whole(static_cast<size_t>(split.batch));
            auto const full_iota = iota_along_axis(full);
            for (int64_t b = 0; b < split.batch; ++b) {
                whole[static_cast<size_t>(b)] = reference_argmax(row_values, full_iota, full, b);
                for (int64_t p = 0; p < split.parts; ++p) {
                    // The part as its own row: same comparator, global positions as the index.
                    int64_t const first = p * chunk;
                    int64_t const last  = std::min(split.vocab, first + chunk);
                    std::vector<float> part_values(row_values.begin() + b * split.vocab + first,
                                                   row_values.begin() + b * split.vocab + last);
                    std::vector<int32_t> part_indices(part_values.size());
                    for (size_t j = 0; j < part_indices.size(); ++j) {
                        part_indices[j] = static_cast<int32_t>(first + static_cast<int64_t>(j));
                    }
                    Layout const part({1, last - first}, 1);
                    auto const winner = reference_argmax(part_values, part_indices, part, 0);
                    auto const at     = static_cast<size_t>(b * split.parts + p);
                    values[at]        = part_values[static_cast<size_t>(winner - first)];
                    indices[at]       = winner;
                }
            }
            auto result = run_argmax(handle, {split.batch, split.parts}, 1, fe::DataType_t::FLOAT, values, indices);
            if (result.built) {
                // run_argmax compared against the argmax of the partials; also hold it to the unsplit row.
                Layout const partial({split.batch, split.parts}, 1);
                for (int64_t b = 0; b < split.batch; ++b) {
                    if (reference_argmax(values, indices, partial, b) != whole[static_cast<size_t>(b)]) {
                        ++result.mismatches;
                    }
                }
            }
            check_case("argmax_split_stage2 B" + std::to_string(split.batch) + "_V" + std::to_string(split.vocab) +
                           "_k" + std::to_string(split.parts) + "/" + fill_name(fill),
                       result);
        }
    }
}
