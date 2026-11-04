/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// argmax and argmin with reduction_modifiers, for each NaN rule, over FLOAT, BFLOAT16 and HALF values and an INT32
// index. Every row is checked against a host reference of the rule. The rows hold NaN (one, two, all, any payload and
// sign), ties, signed zeros, infinities, subnormals and random bit patterns.

#include <catch2/catch_test_macros.hpp>
#include "../utils/helpers.h"

#include <cudnn_frontend.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

namespace fe = cudnn_frontend;

using Attributes   = fe::graph::Custom_reduction_attributes;
using Arg_nan_rule = fe::graph::reduction_modifiers::Arg_nan_rule;

struct Rows_by_extent {
    int64_t rows;
    int64_t extent;
};

// Logits of a 129,280-token vocabulary, at decode batch sizes.
std::vector<Rows_by_extent> const vocabulary_shapes = {{1, 129280}, {8, 129280}, {64, 129280}};

// XLA argmax operand shapes, each reduced over its innermost axis, as [rows, extent].
std::vector<Rows_by_extent> const xla_shapes = {
    {1, 32000},    {600, 151936}, {400, 151936}, {1, 256128},   {1, 128256},   {128, 32000},  {16, 151936}, {1, 151936},
    {400, 262144}, {1, 262144},   {256, 32000},  {400, 256128}, {200, 256128}, {800, 201088}, {1, 201088},  {32, 32000},
    {16, 32000},   {400, 128256}, {600, 128256}, {256, 202048}, {1, 202048},   {600, 32000},  {256, 32768}, {1, 32768},
    {512, 1000},   {16, 256128},  {8, 32000},    {16, 128256},  {1, 256000},   {4096, 32128},
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

// The storage bits of one element type, and the exact float of a storage word.
struct Value_type {
    fe::DataType_t type;
    int bits;  // 16 or 32
    uint32_t positive_infinity;
    uint32_t quiet_nan;
    uint32_t mantissa_mask;

    uint32_t
    sign() const {
        return 1u << (bits - 1);
    }

    float
    to_float(uint32_t word) const {
        float value;
        if (type == fe::DataType_t::FLOAT) {
            std::memcpy(&value, &word, sizeof(value));
        } else if (type == fe::DataType_t::BFLOAT16) {
            uint32_t const widened = word << 16;
            std::memcpy(&value, &widened, sizeof(value));
        } else {
            __half_raw raw{};
            raw.x = static_cast<uint16_t>(word);
            value = cpu_half2float(half(raw));
        }
        return value;
    }

    uint32_t
    from_float(float value) const {
        uint32_t word;
        std::memcpy(&word, &value, sizeof(word));
        if (type == fe::DataType_t::FLOAT) {
            return word;
        }
        if (type == fe::DataType_t::BFLOAT16) {
            if (std::isnan(value)) {
                return (word >> 16) | 0x0040u;
            }
            return (word + 0x7FFFu + ((word >> 16) & 1u)) >> 16;
        }
        return static_cast<__half_raw>(cpu_float2half_rn(value)).x;
    }
};

Value_type const float_type    = {fe::DataType_t::FLOAT, 32, 0x7F800000u, 0x7FC00000u, 0x007FFFFFu};
Value_type const bfloat16_type = {fe::DataType_t::BFLOAT16, 16, 0x7F80u, 0x7FC0u, 0x007Fu};
Value_type const half_type     = {fe::DataType_t::HALF, 16, 0x7C00u, 0x7E00u, 0x03FFu};

uint64_t
splitmix64(uint64_t& state) {
    uint64_t z = (state += 0x9E3779B97F4A7C15ull);
    z          = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z          = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

// Row classes; pass p gives row r the class (r + p) % row_class_count, so a shape with one row sees each class too.
int64_t constexpr row_class_count = 17;

void
fill_row(Value_type const& vt, int row_class, uint32_t* words, int64_t extent, uint64_t& rng) {
    auto const random_position = [&]() {
        return static_cast<int64_t>(splitmix64(rng) % static_cast<uint64_t>(extent));
    };
    auto const random_value = [&]() {
        auto const unit = static_cast<float>(splitmix64(rng) >> 40) / static_cast<float>(1ull << 24);
        return vt.from_float(unit * 2000.0f - 1000.0f);
    };
    auto const any_nan = [&]() {
        auto const payload = static_cast<uint32_t>(splitmix64(rng)) & vt.mantissa_mask;
        auto const sign    = (splitmix64(rng) & 1u) ? vt.sign() : 0u;
        return sign | vt.positive_infinity | (payload == 0 ? 1u : payload);
    };
    uint32_t const negative_infinity = vt.sign() | vt.positive_infinity;
    auto const fill_all              = [&](uint32_t word) {
        for (int64_t i = 0; i < extent; ++i) {
            words[i] = word;
        }
    };
    auto const fill_random = [&]() {
        for (int64_t i = 0; i < extent; ++i) {
            words[i] = random_value();
        }
    };
    int64_t const p    = random_position();
    int64_t q          = random_position();
    int64_t const last = extent - 1;
    if (extent > 1) {
        while (q == p) {
            q = random_position();
        }
    }
    int64_t const first  = p < q ? p : q;
    int64_t const second = p < q ? q : p;
    switch (row_class) {
        case 0:  // random values
            fill_random();
            break;
        case 1:  // the best value at two positions, both orders of best and worst
            fill_random();
            words[p]                = vt.from_float(2048.0f);
            words[q]                = vt.from_float(2048.0f);
            words[(p + 1) % extent] = vt.from_float(-2048.0f);
            words[(q + 1) % extent] = vt.from_float(-2048.0f);
            break;
        case 2:  // one NaN
            fill_random();
            words[p] = any_nan();
            break;
        case 3:  // two adjacent NaN
            fill_random();
            words[p]                = any_nan();
            words[(p + 1) % extent] = any_nan();
            break;
        case 4:  // NaN first and last
            fill_random();
            words[0]    = any_nan();
            words[last] = any_nan();
            break;
        case 5:  // only NaN, mixed payloads and signs
            for (int64_t i = 0; i < extent; ++i) {
                words[i] = any_nan();
            }
            break;
        case 6:  // +0 and -0 tie as the largest value, in both orders, and as the smallest value, in both orders
        case 7:
        case 8:
        case 9: {
            uint32_t const background_sign = row_class < 8 ? vt.sign() : 0u;
            for (int64_t i = 0; i < extent; ++i) {
                words[i] = background_sign | vt.from_float(1.0f + static_cast<float>(splitmix64(rng) % 512u));
            }
            bool const positive_first = row_class % 2 == 0;
            words[first]              = positive_first ? 0u : vt.sign();
            words[second]             = positive_first ? vt.sign() : 0u;
            break;
        }
        case 10:  // only -inf
            fill_all(negative_infinity);
            break;
        case 11:  // only +inf
            fill_all(vt.positive_infinity);
            break;
        case 12:  // two +inf and a -inf among random values
            fill_random();
            words[first]                = vt.positive_infinity;
            words[second]               = vt.positive_infinity;
            words[(first + 1) % extent] = negative_infinity;
            break;
        case 13:  // +inf, then a NaN after it
            fill_random();
            words[first]  = vt.positive_infinity;
            words[second] = any_nan();
            break;
        case 14:  // the two smallest positive subnormal values on -inf, the larger one later, and a negative one
            fill_all(negative_infinity);
            words[first]                = 1u;
            words[second]               = 2u;
            words[(first + 1) % extent] = vt.sign() | 2u;
            break;
        case 15:  // a NaN at position 0, -inf after it
            fill_all(negative_infinity);
            words[0] = vt.quiet_nan;
            break;
        default:  // random storage words, NaN and infinity included
            for (int64_t i = 0; i < extent; ++i) {
                words[i] = static_cast<uint32_t>(splitmix64(rng)) & (vt.bits == 32 ? 0xFFFFFFFFu : 0xFFFFu);
            }
            break;
    }
}

// The rule's answer for one row, from the exact floats: the first NaN, the first best number (equal values tie, -0
// equals +0), whether the row holds a number.
struct Row_reference {
    int64_t first_nan  = -1;
    int64_t first_best = 0;
    bool has_number    = false;
};

Row_reference
reference_row(float const* values, int64_t extent, bool is_max) {
    Row_reference reference;
    float best = 0.0f;
    for (int64_t i = 0; i < extent; ++i) {
        float const v = values[i];
        if (std::isnan(v)) {
            if (reference.first_nan < 0) {
                reference.first_nan = i;
            }
            continue;
        }
        if (!reference.has_number || (is_max ? v > best : v < best)) {
            best                 = v;
            reference.first_best = i;
            reference.has_number = true;
        }
    }
    return reference;
}

bool
accepted(Row_reference const& reference, float const* values, int64_t extent, int32_t got, Arg_nan_rule rule) {
    if (got < 0 || got >= extent) {
        return false;
    }
    switch (rule) {
        case Arg_nan_rule::TORCH:
            return got == (reference.first_nan >= 0 ? reference.first_nan : reference.first_best);
        case Arg_nan_rule::XLA:
            return reference.first_nan >= 0 ? std::isnan(values[got]) : got == reference.first_best;
        case Arg_nan_rule::IGNORE_NAN:
            return got == (reference.has_number ? reference.first_best : 0);
    }
    return false;
}

char const*
rule_name(Arg_nan_rule rule) {
    return rule == Arg_nan_rule::TORCH ? "TORCH" : rule == Arg_nan_rule::XLA ? "XLA" : "IGNORE_NAN";
}

struct Arg_graph {
    bool is_max;
    Arg_nan_rule rule;
    std::shared_ptr<fe::graph::Graph> graph;
    std::shared_ptr<fe::graph::Tensor_attributes> value;
    std::shared_ptr<fe::graph::Tensor_attributes> index;
    std::shared_ptr<fe::graph::Tensor_attributes> out;
    int64_t mismatches = 0;
};

Arg_graph
build_arg_graph(cudnnHandle_t handle, Value_type const& vt, Rows_by_extent shape, bool is_max, Arg_nan_rule rule) {
    Arg_graph g{is_max, rule, std::make_shared<fe::graph::Graph>(), nullptr, nullptr, nullptr, 0};
    std::vector<int64_t> const dims   = {shape.rows, shape.extent};
    std::vector<int64_t> const stride = {shape.extent, 1};
    g.value                           = g.graph->tensor(
        fe::graph::Tensor_attributes().set_name("value").set_dim(dims).set_stride(stride).set_data_type(vt.type));
    g.index =
        g.graph->tensor(fe::graph::Tensor_attributes().set_name("index").set_dim(dims).set_stride(stride).set_data_type(
            fe::DataType_t::INT32));
    auto attributes = Attributes().set_name(is_max ? "argmax" : "argmin").set_reduction_dims({1});
    attributes      = is_max ? fe::graph::reduction_modifiers::argmax(attributes, vt.type, fe::DataType_t::INT32, rule)
                             : fe::graph::reduction_modifiers::argmin(attributes, vt.type, fe::DataType_t::INT32, rule);
    g.out           = g.graph->custom_reduction({g.value, g.index}, attributes).at(1);
    g.out->set_output(true);

    REQUIRE(g.graph->validate().is_good());
    REQUIRE(g.graph->build_operation_graph(handle).is_good());
    REQUIRE(g.graph->create_execution_plans({fe::HeurMode_t::A}).is_good());
    REQUIRE(g.graph->check_support().is_good());
    REQUIRE(g.graph->build_plans(fe::BuildPlanPolicy_t::HEURISTICS_CHOICE).is_good());
    return g;
}

// For each shape: six graphs (argmax, argmin, each rule) on one value tensor; row_class_count passes of fresh rows.
void
run_parity(Value_type const& vt, std::vector<Rows_by_extent> const& shapes) {
    auto handle_ptr = create_cudnn_handle();
    auto handle     = *handle_ptr;
    uint64_t rng    = 0x5EED0000ull + static_cast<uint64_t>(vt.type);
    for (auto const& shape : shapes) {
        INFO("value type " << static_cast<int>(vt.type) << " rows " << shape.rows << " extent " << shape.extent);
        std::vector<Arg_graph> graphs;
        for (bool const is_max : {true, false}) {
            for (auto const rule : {Arg_nan_rule::TORCH, Arg_nan_rule::XLA, Arg_nan_rule::IGNORE_NAN}) {
                graphs.push_back(build_arg_graph(handle, vt, shape, is_max, rule));
            }
        }
        int64_t workspace_size = 0;
        for (auto const& g : graphs) {
            int64_t size = 0;
            REQUIRE(g.graph->get_workspace_size(size).is_good());
            workspace_size = std::max(workspace_size, size);
        }

        auto const elements    = static_cast<size_t>(shape.rows * shape.extent);
        size_t const word_size = static_cast<size_t>(vt.bits / 8);
        std::vector<uint32_t> words(elements);
        std::vector<float> values(elements);
        std::vector<uint16_t> narrow(vt.bits == 16 ? elements : 0);
        std::vector<int32_t> iota(elements);
        for (size_t i = 0; i < elements; ++i) {
            iota[i] = static_cast<int32_t>(static_cast<int64_t>(i) % shape.extent);
        }
        Device_buffer d_value(elements * word_size);
        Device_buffer d_index(elements * sizeof(int32_t));
        Device_buffer d_out(static_cast<size_t>(shape.rows) * sizeof(int32_t));
        Device_buffer workspace(static_cast<size_t>(workspace_size));
        CUDA_CHECK(cudaMemcpy(d_index.get(), iota.data(), elements * sizeof(int32_t), cudaMemcpyHostToDevice));
        std::vector<int32_t> got(static_cast<size_t>(shape.rows));
        std::vector<Row_reference> max_reference(static_cast<size_t>(shape.rows));
        std::vector<Row_reference> min_reference(static_cast<size_t>(shape.rows));

        for (int64_t pass = 0; pass < row_class_count; ++pass) {
            for (int64_t row = 0; row < shape.rows; ++row) {
                auto* row_words = words.data() + row * shape.extent;
                fill_row(vt, static_cast<int>((row + pass) % row_class_count), row_words, shape.extent, rng);
                for (int64_t i = 0; i < shape.extent; ++i) {
                    values[static_cast<size_t>(row * shape.extent + i)] = vt.to_float(row_words[i]);
                }
                auto const* row_values                  = values.data() + row * shape.extent;
                max_reference[static_cast<size_t>(row)] = reference_row(row_values, shape.extent, true);
                min_reference[static_cast<size_t>(row)] = reference_row(row_values, shape.extent, false);
            }
            if (vt.bits == 16) {
                for (size_t i = 0; i < elements; ++i) {
                    narrow[i] = static_cast<uint16_t>(words[i]);
                }
                CUDA_CHECK(cudaMemcpy(d_value.get(), narrow.data(), elements * word_size, cudaMemcpyHostToDevice));
            } else {
                CUDA_CHECK(cudaMemcpy(d_value.get(), words.data(), elements * word_size, cudaMemcpyHostToDevice));
            }

            for (auto& g : graphs) {
                CUDA_CHECK(cudaMemset(d_out.get(), 0xFF, got.size() * sizeof(int32_t)));
                std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> pack = {
                    {g.value, d_value.get()}, {g.index, d_index.get()}, {g.out, d_out.get()}};
                REQUIRE(g.graph->execute(handle, pack, workspace.get()).is_good());
                CUDA_CHECK(cudaDeviceSynchronize());
                CUDA_CHECK(cudaMemcpy(got.data(), d_out.get(), got.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
                auto const& reference = g.is_max ? max_reference : min_reference;
                for (int64_t row = 0; row < shape.rows; ++row) {
                    auto const r = static_cast<size_t>(row);
                    if (!accepted(reference[r], values.data() + row * shape.extent, shape.extent, got[r], g.rule)) {
                        ++g.mismatches;
                    }
                }
            }
        }
        for (auto const& g : graphs) {
            INFO((g.is_max ? "argmax " : "argmin ") << rule_name(g.rule));
            CHECK(g.mismatches == 0);
        }
    }
}

// Every 16-bit pattern w, one per row, on a background that loses to every number: w alone; w after its next-lower
// neighbour (w must win); after its next-higher neighbour (the neighbour must win); and twice (the first must win).
void
run_every_pattern(Value_type const& vt) {
    int64_t constexpr extent = 16;
    int64_t constexpr rows   = 1 << 16;
    auto const ordered_key   = [&](uint32_t w) {
        return (w & vt.sign()) ? -static_cast<int32_t>(w & ~vt.sign()) : static_cast<int32_t>(w);
    };
    auto const from_key = [&](int32_t k) {
        return k >= 0 ? static_cast<uint32_t>(k) : (vt.sign() | static_cast<uint32_t>(-k));
    };
    auto const is_nan    = [&](uint32_t w) { return (w & ~vt.sign()) > vt.positive_infinity; };
    auto const neighbour = [&](uint32_t w, int step) {
        if (is_nan(w)) {
            return step < 0 ? vt.positive_infinity : (vt.sign() | vt.positive_infinity | 1u);
        }
        auto const n = from_key(ordered_key(w) + step);
        return is_nan(n) ? w : n;
    };
    std::vector<Rows_by_extent> const shape = {{rows, extent}};
    auto handle_ptr                         = create_cudnn_handle();
    auto handle                             = *handle_ptr;
    for (bool const is_max : {true, false}) {
        std::vector<Arg_graph> graphs;
        for (auto const rule : {Arg_nan_rule::TORCH, Arg_nan_rule::XLA, Arg_nan_rule::IGNORE_NAN}) {
            graphs.push_back(build_arg_graph(handle, vt, shape[0], is_max, rule));
        }
        int64_t workspace_size = 0;
        for (auto const& g : graphs) {
            int64_t size = 0;
            REQUIRE(g.graph->get_workspace_size(size).is_good());
            workspace_size = std::max(workspace_size, size);
        }
        uint32_t const background = is_max ? (vt.sign() | vt.positive_infinity) : vt.positive_infinity;
        size_t const elements     = static_cast<size_t>(rows * extent);
        std::vector<uint16_t> words(elements);
        std::vector<float> values(elements);
        std::vector<int32_t> iota(elements);
        for (size_t i = 0; i < elements; ++i) {
            iota[i] = static_cast<int32_t>(static_cast<int64_t>(i) % extent);
        }
        Device_buffer d_value(elements * sizeof(uint16_t));
        Device_buffer d_index(elements * sizeof(int32_t));
        Device_buffer d_out(static_cast<size_t>(rows) * sizeof(int32_t));
        Device_buffer workspace(static_cast<size_t>(workspace_size));
        CUDA_CHECK(cudaMemcpy(d_index.get(), iota.data(), elements * sizeof(int32_t), cudaMemcpyHostToDevice));
        std::vector<int32_t> got(static_cast<size_t>(rows));
        for (int probe = 0; probe < 4; ++probe) {
            for (int64_t row = 0; row < rows; ++row) {
                auto const w   = static_cast<uint32_t>(row);
                auto* row_word = words.data() + row * extent;
                for (int64_t i = 0; i < extent; ++i) {
                    row_word[i] = static_cast<uint16_t>(background);
                }
                row_word[11] = static_cast<uint16_t>(w);
                if (probe == 1) {
                    row_word[3] = static_cast<uint16_t>(neighbour(w, is_max ? -1 : 1));
                } else if (probe == 2) {
                    row_word[3] = static_cast<uint16_t>(neighbour(w, is_max ? 1 : -1));
                } else if (probe == 3) {
                    row_word[3] = static_cast<uint16_t>(w);
                }
                for (int64_t i = 0; i < extent; ++i) {
                    values[static_cast<size_t>(row * extent + i)] = vt.to_float(row_word[i]);
                }
            }
            CUDA_CHECK(cudaMemcpy(d_value.get(), words.data(), elements * sizeof(uint16_t), cudaMemcpyHostToDevice));
            for (auto& g : graphs) {
                CUDA_CHECK(cudaMemset(d_out.get(), 0xFF, got.size() * sizeof(int32_t)));
                std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> pack = {
                    {g.value, d_value.get()}, {g.index, d_index.get()}, {g.out, d_out.get()}};
                REQUIRE(g.graph->execute(handle, pack, workspace.get()).is_good());
                CUDA_CHECK(cudaDeviceSynchronize());
                CUDA_CHECK(cudaMemcpy(got.data(), d_out.get(), got.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
                for (int64_t row = 0; row < rows; ++row) {
                    auto const* row_values = values.data() + row * extent;
                    if (!accepted(reference_row(row_values, extent, is_max),
                                  row_values,
                                  extent,
                                  got[static_cast<size_t>(row)],
                                  g.rule)) {
                        ++g.mismatches;
                    }
                }
            }
        }
        for (auto const& g : graphs) {
            INFO((is_max ? "argmax " : "argmin ") << rule_name(g.rule));
            CHECK(g.mismatches == 0);
        }
    }
}

char const*
skip_reason() {
#if CUDART_VERSION < 13010
    return "Test requires cuda toolkit 13.1 or above";
#elif CUDNN_VERSION < 92900
    return "Custom reduction requires cuDNN v9.29.0";
#else
    if (fe::detail::get_backend_version() < 92900) {
        return "Custom reduction requires cuDNN v9.29.0";
    }
    if (!is_blackwell_computing_arch() || !is_arch_supported_by_cudnn()) {
        return "Custom reduction requires the TensorIR MemBound engine on data-center Blackwell";
    }
    return nullptr;
#endif
}

}  // namespace

TEST_CASE("Custom reduction argmax and argmin rules match their reference", "[custom_reduction][argmax]") {
    if (auto const* reason = skip_reason(); reason != nullptr) {
        SKIP(reason);
    }
    std::vector<Rows_by_extent> shapes = vocabulary_shapes;
    shapes.push_back({16, 151936});
    shapes.push_back({512, 1000});
    for (auto const* vt : {&float_type, &bfloat16_type, &half_type}) {
        run_parity(*vt, shapes);
    }
}

TEST_CASE("Custom reduction argmax and argmin rules hold for every 16-bit pattern", "[custom_reduction][argmax]") {
    if (auto const* reason = skip_reason(); reason != nullptr) {
        SKIP(reason);
    }
    run_every_pattern(bfloat16_type);
    run_every_pattern(half_type);
}

// Not run by default: every XLA argmax shape and the vocabulary shape up to 2,048 rows, for each value type.
TEST_CASE("Custom reduction argmax and argmin rules on every XLA and vocabulary shape",
          "[.][custom_reduction_argmax_all_shapes]") {
    if (auto const* reason = skip_reason(); reason != nullptr) {
        SKIP(reason);
    }
    std::vector<Rows_by_extent> shapes = xla_shapes;
    for (int64_t const rows : {1, 8, 64, 512, 2048}) {
        shapes.push_back({rows, 129280});
    }
    for (auto const* vt : {&float_type, &bfloat16_type, &half_type}) {
        run_parity(*vt, shapes);
    }
}
