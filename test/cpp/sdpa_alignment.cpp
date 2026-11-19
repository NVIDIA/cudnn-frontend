/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <catch2/catch_test_macros.hpp>
#include <cudnn_frontend.h>

TEST_CASE("SDPA alignment ignores only a ragged operand's batch stride", "[sdpa][validate][alignment][ragged]") {
    namespace fe = cudnn_frontend;
    using Tensor = fe::graph::Tensor_attributes;
    for (auto dtype : {fe::DataType_t::HALF,
                       fe::DataType_t::BFLOAT16,
                       fe::DataType_t::FP8_E4M3,
                       fe::DataType_t::FP8_E5M2,
                       fe::DataType_t::FLOAT}) {
        for (bool ragged : {false, true}) {
            auto tensor = std::make_shared<Tensor>(Tensor().set_dim({3, 8, 97, 128}).set_data_type(dtype));
            if (ragged) {
                tensor->set_ragged_offset(std::make_shared<Tensor>(
                    Tensor().set_dim({4, 1, 1, 1}).set_stride({1, 1, 1, 1}).set_data_type(fe::DataType_t::INT32)));
            }
            for (auto stride : {std::vector<int64_t>{2147483647, 128, 3072, 1},
                                std::vector<int64_t>{2147483632, 128, 3072, 1},
                                std::vector<int64_t>{0, 128, 3072, 1},
                                std::vector<int64_t>{1, 128, 3072, 1}}) {
                tensor->set_stride(stride);
                INFO("dtype=" << static_cast<int>(dtype) << " ragged=" << ragged << " batch stride=" << stride[0]);
                CHECK(fe::graph::validate_sdpa_operand_stride_alignment(tensor, "Q").is_good() ==
                      (ragged || stride[0] != 2147483647));
                // The same unused batch pitch cannot hide a real head or token pitch error.
                for (auto axis : {1, 2}) {
                    auto bad = stride;
                    ++bad[axis];
                    tensor->set_stride(bad);
                    CHECK_FALSE(fe::graph::validate_sdpa_operand_stride_alignment(tensor, "Q").is_good());
                }
            }
            // An unstepped singleton batch remains legal even without ragged offsets.
            tensor->set_dim({1, 8, 97, 128}).set_stride({2147483647, 128, 3072, 1});
            CHECK(fe::graph::validate_sdpa_operand_stride_alignment(tensor, "Q").is_good());
        }
    }
}

TEST_CASE("Ragged SDPA alignment survives plan metadata restoration", "[sdpa][serialize][alignment][ragged]") {
    namespace fe = cudnn_frontend;
    namespace md = fe::graph::plan_serialization_detail;
    using Tensor = fe::graph::Tensor_attributes;
    auto offsets = std::make_shared<Tensor>(
        Tensor().set_uid(2).set_dim({4, 1, 1, 1}).set_stride({1, 1, 1, 1}).set_data_type(fe::DataType_t::INT32));
    auto tensor = Tensor()
                      .set_uid(1)
                      .set_dim({3, 8, 97, 128})
                      .set_stride({2147483647, 128, 3072, 1})
                      .set_data_type(fe::DataType_t::BFLOAT16)
                      .set_ragged_offset(offsets);
    json state{{"json_version", "2.0"},
               {"cudnn_backend_data", "opaque"},
               {"variant_pack_uids", json::array({1, 2})},
               {"fe_workspace_size", 0},
               {"variant_pack_replacements", json::array()},
               {"behavior_notes", json::array({json::array()})},
               {"gid", 1},
               {"nodes", json::array()},
               {"tensors", json::array({json(tensor), json(*offsets)})},
               {"aligned_operands", {{"1", json::array({16, 2})}}}};
    // Existing two-integer alignment metadata remains readable. Addressing mode
    // comes from the resolved operand's metadata, not another tensor's offsets.
    auto parsed   = md::parse(state);
    auto restored = parsed.tensors.at(1);
    REQUIRE(restored->get_ragged_offset() == parsed.tensors.at(2));
    CHECK(fe::graph::validate_sdpa_operand_stride_alignment(restored, "Q").is_good());
    restored->set_stride({2147483647, 129, 3072, 1});
    CHECK_FALSE(fe::graph::validate_sdpa_operand_stride_alignment(restored, "Q").is_good());
    restored->set_stride({2147483647, 128, 3072, 1}).set_ragged_offset(nullptr);
    CHECK_FALSE(fe::graph::validate_sdpa_operand_stride_alignment(restored, "Q").is_good());

    auto explicit_state                       = state;
    explicit_state["ragged_aligned_operands"] = json::array({1});
    REQUIRE_NOTHROW(md::parse(explicit_state));
    auto opaque = explicit_state;
    opaque.erase("nodes");
    opaque.erase("tensors");
    opaque.erase("gid");
    REQUIRE_NOTHROW(md::parse(opaque));
    // The optional plan-only field must not waive alignment for an unrelated UID
    // or contradict a dense tensor. Keep legacy two-integer entries unchanged.
    for (auto const& value : std::vector<json>{nullptr,
                                               true,
                                               1,
                                               "1",
                                               json::array({true}),
                                               json::array({1.5}),
                                               json::array({1, 1}),
                                               json::array({2}),
                                               json::array({-17})}) {
        auto bad                       = opaque;
        bad["ragged_aligned_operands"] = value;
        REQUIRE_THROWS_AS(md::parse(bad), std::exception);
    }
    auto omitted                       = explicit_state;
    omitted["ragged_aligned_operands"] = json::array();
    REQUIRE_THROWS_AS(md::parse(omitted), std::exception);
    auto dense = explicit_state;
    tensor.set_ragged_offset(nullptr);
    dense["tensors"] = json::array({json(tensor), json(*offsets)});
    REQUIRE_THROWS_AS(md::parse(dense), std::exception);
}

namespace {
struct AlignmentTestAllocation {
    void* pointer = nullptr;
    explicit AlignmentTestAllocation(size_t bytes) {
        REQUIRE(cudaMalloc(&pointer, std::max<size_t>(bytes, 1)) == cudaSuccess);
    }
    ~AlignmentTestAllocation() { cudaFree(pointer); }
    AlignmentTestAllocation(AlignmentTestAllocation const&) = delete;
    AlignmentTestAllocation&
    operator=(AlignmentTestAllocation const&) = delete;
};
struct AlignmentTestHandle {
    cudnnHandle_t value = nullptr;
    AlignmentTestHandle() { REQUIRE(cudnnCreate(&value) == CUDNN_STATUS_SUCCESS); }
    ~AlignmentTestHandle() { cudnnDestroy(value); }
};
}  // namespace

TEST_CASE("Ragged SDPA overrides use operand addressing before and after plan reload",
          "[sdpa][alignment][ragged][native]") {
    namespace fe = cudnn_frontend;
    using Tensor = fe::graph::Tensor_attributes;
    if (cudnnGetVersion() < 92300) {
        SKIP("Runtime shape overrides require cuDNN 9.23 or newer");
    }
    int device = 0;
    cudaDeviceProp properties{};
    REQUIRE(cudaGetDevice(&device) == cudaSuccess);
    REQUIRE(cudaGetDeviceProperties(&properties, device) == cudaSuccess);
    if (properties.major < 9) {
        SKIP("This native ragged override regression requires SM90 or newer");
    }
    constexpr int64_t heads = 2, width = 64, tokens = 192;
    std::vector<int64_t> const shape{2, heads, 128, width};
    std::vector<int64_t> const aligned_strides{128 * heads * width, width, heads * width, 1};
    std::vector<int64_t> const ignored_batch_strides{2147483647, width, heads * width, 1};
    size_t const bytes = tokens * heads * width * sizeof(uint16_t);
    AlignmentTestHandle handle;
    for (auto dtype : {fe::DataType_t::HALF, fe::DataType_t::BFLOAT16}) {
        INFO("dtype=" << static_cast<int>(dtype));
        fe::graph::Graph graph;
        graph.set_io_data_type(dtype)
            .set_intermediate_data_type(fe::DataType_t::FLOAT)
            .set_compute_data_type(fe::DataType_t::FLOAT)
            .set_override_shape_enabled(true);
        auto offsets =
            graph.tensor(Tensor().set_dim({3, 1, 1, 1}).set_stride({1, 1, 1, 1}).set_data_type(fe::DataType_t::INT32));
        auto lengths =
            graph.tensor(Tensor().set_dim({2, 1, 1, 1}).set_stride({1, 1, 1, 1}).set_data_type(fe::DataType_t::INT32));
        auto operand = [&]() {
            return graph.tensor(
                Tensor().set_dim(shape).set_stride(aligned_strides).set_data_type(dtype).set_ragged_offset(offsets));
        };
        auto q = operand(), k = operand(), v = operand();
        auto output = graph.sdpa(q,
                                 k,
                                 v,
                                 fe::graph::SDPA_attributes()
                                     .set_generate_stats(false)
                                     .set_attn_scale(0.125F)
                                     .set_padding_mask(true)
                                     .set_seq_len_q(lengths)
                                     .set_seq_len_kv(lengths))[0];
        output->set_output(true)
            .set_dim(shape)
            .set_stride(aligned_strides)
            .set_data_type(dtype)
            .set_ragged_offset(offsets);
        auto status = graph.build(handle.value, {fe::HeurMode_t::A});
        INFO(status.get_message());
        REQUIRE(status.is_good());
        AlignmentTestAllocation q_data(bytes), k_data(bytes), v_data(bytes), o_data(bytes);
        AlignmentTestAllocation offsets_data(3 * sizeof(int32_t)), lengths_data(2 * sizeof(int32_t));
        AlignmentTestAllocation workspace(graph.get_workspace_size());
        REQUIRE(cudaMemset(q_data.pointer, 0, bytes) == cudaSuccess);
        REQUIRE(cudaMemset(k_data.pointer, 0, bytes) == cudaSuccess);
        int32_t const offsets_host[]{0, 64 * heads * width, tokens * heads * width};
        int32_t const lengths_host[]{64, 128};
        REQUIRE(cudaMemcpy(offsets_data.pointer, offsets_host, sizeof(offsets_host), cudaMemcpyHostToDevice) ==
                cudaSuccess);
        REQUIRE(cudaMemcpy(lengths_data.pointer, lengths_host, sizeof(lengths_host), cudaMemcpyHostToDevice) ==
                cudaSuccess);
        uint16_t const one = dtype == fe::DataType_t::HALF ? 0x3c00 : 0x3f80;
        uint16_t const two = 0x4000;
        std::vector<uint16_t> values(tokens * heads * width, two);
        std::fill(values.begin(), values.begin() + 64 * heads * width, one);
        REQUIRE(cudaMemcpy(v_data.pointer, values.data(), bytes, cudaMemcpyHostToDevice) == cudaSuccess);
        std::unordered_map<int64_t, void*> bindings{{q->get_uid(), q_data.pointer},
                                                    {k->get_uid(), k_data.pointer},
                                                    {v->get_uid(), v_data.pointer},
                                                    {output->get_uid(), o_data.pointer},
                                                    {offsets->get_uid(), offsets_data.pointer},
                                                    {lengths->get_uid(), lengths_data.pointer}};
        std::vector<int64_t> uids{q->get_uid(), k->get_uid(), v->get_uid(), output->get_uid()};
        std::vector<std::vector<int64_t>> shapes(4, shape), strides(4, ignored_batch_strides);
        std::vector<uint8_t> blob;
        REQUIRE(graph.serialize(blob).is_good());
        fe::graph::Graph restored;
        REQUIRE(restored.deserialize(handle.value, blob, false, false).is_good());
        REQUIRE(graph.serialize(blob, false).is_good());
        fe::graph::Graph plan_only;
        REQUIRE(plan_only.deserialize(handle.value, blob, false, false).is_good());
        REQUIRE(plan_only.serialize(blob, false).is_good());
        fe::graph::Graph reserialized;
        REQUIRE(reserialized.deserialize(handle.value, blob, false, false).is_good());
        for (auto current : {&graph, &restored, &plan_only, &reserialized}) {
            // Zero Q/K gives a uniform mean within each document. Different V
            // constants make using another document's batch origin observable.
            REQUIRE(cudaMemset(o_data.pointer, 0xff, bytes) == cudaSuccess);
            status = current->execute(handle.value, bindings, workspace.pointer, uids, shapes, strides);
            INFO(status.get_message());
            REQUIRE(status.is_good());
            std::vector<uint16_t> actual(values.size());
            REQUIRE(cudaMemcpy(actual.data(), o_data.pointer, bytes, cudaMemcpyDeviceToHost) == cudaSuccess);
            REQUIRE(actual == values);

            for (auto axis : {1, 2}) {
                auto bad = strides;
                ++bad[0][axis];
                REQUIRE(cudaMemset(o_data.pointer, 0xff, bytes) == cudaSuccess);
                status = current->execute(handle.value, bindings, workspace.pointer, uids, shapes, bad);
                REQUIRE(status.get_code() == fe::error_code_t::INVALID_VARIANT_PACK);
                REQUIRE(status.get_message().find("override stride") != std::string::npos);
                REQUIRE(cudaMemcpy(actual.data(), o_data.pointer, bytes, cudaMemcpyDeviceToHost) == cudaSuccess);
                REQUIRE(std::all_of(actual.begin(), actual.end(), [](uint16_t value) { return value == 0xffff; }));
            }
            // Ignoring an unused stride does not relax the bound pointer check.
            auto bad          = bindings;
            bad[q->get_uid()] = static_cast<char*>(q_data.pointer) + 2;
            status            = current->execute(handle.value, bad, workspace.pointer, uids, shapes, strides);
            REQUIRE(status.get_code() == fe::error_code_t::INVALID_VARIANT_PACK);
            REQUIRE(status.get_message().find("device pointer is not aligned") != std::string::npos);
        }
    }
}
