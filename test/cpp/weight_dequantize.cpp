/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <cudnn_frontend.h>

namespace {
namespace fe = cudnn_frontend;
using Tensor = std::shared_ptr<fe::graph::Tensor_attributes>;

struct DecodeGraph {
    fe::graph::Graph graph;
    Tensor weights, decoded;
    std::vector<Tensor> auxiliary;

    explicit DecodeGraph(fe::graph::Weight_dequantize_program const& program, int count = 2) {
        graph.set_io_data_type(fe::DataType_t::HALF)
            .set_intermediate_data_type(fe::DataType_t::HALF)
            .set_compute_data_type(fe::DataType_t::FLOAT);
        weights = graph.tensor(fe::graph::Tensor_attributes()
                                   .set_name("weights")
                                   .set_uid(2)
                                   .set_dim({1, 1, 4096})
                                   .set_stride({4096, 4096, 1})
                                   .set_data_type(fe::DataType_t::UINT8));
        for (int i = 0; i < count; ++i)
            auxiliary.push_back(graph.tensor(fe::graph::Tensor_attributes()
                                                 .set_name("scale" + std::to_string(i))
                                                 .set_uid(10 + i)
                                                 .set_dim({1, 1, 16})
                                                 .set_stride({16, 16, 1})
                                                 .set_data_type(fe::DataType_t::FLOAT)));
        decoded = graph.weight_dequantize(
            weights, auxiliary, fe::graph::Weight_dequantize_attributes().set_name("dequant").set_program(program));
        decoded->set_uid(3).set_dim({1, 64, 64});
        auto a = graph.tensor(fe::graph::Tensor_attributes().set_uid(1).set_dim({1, 32, 64}).set_stride({2048, 64, 1}));
        graph.matmul(a, decoded, fe::graph::Matmul_attributes().set_name("gemm"))->set_uid(4).set_output(true);
    }
};

auto
program() {
    return fe::graph::Weight_dequantize_program()
        .set_source("// Customer CUDA C++\n")
        .set_entry("decode")
        .set_constants({4, 24, 20});
}
}  // namespace

TEST_CASE("Custom weight dequantization validates its public contract", "[weight_dequantize][validate]") {
    auto p = program();
    SECTION("empty source") { p.source.clear(); }
    SECTION("embedded NUL") { p.source = std::string("a\0b", 3); }
    SECTION("oversized source") { p.source.assign(1024 * 1024, 'x'); }
    SECTION("entry is not an identifier") { p.entry = "decode;"; }
    SECTION("entry starts with digit") { p.entry = "1decode"; }
    SECTION("empty entry") { p.entry.clear(); }
    SECTION("oversized entry") { p.entry.assign(128, 'x'); }
    SECTION("unknown ABI") { p.abi_version = 2; }
    SECTION("missing tile dimension") { p.tile_shape = {32}; }
    SECTION("nonpositive tile") { p.tile_shape = {32, 0}; }
    SECTION("negative CTA scratch") { p.cta_smem_bytes = -1; }
    SECTION("excessive stage scratch") { p.stage_smem_bytes = 1024 * 1024 + 1; }
    SECTION("alignment not a power of two") { p.input_alignment = 3; }
    SECTION("zero alignment") { p.input_alignment = 0; }
    SECTION("excessive alignment") { p.input_alignment = 512; }
    SECTION("too many constants") { p.constants.resize(65); }
    REQUIRE(DecodeGraph(p).graph.validate().get_code() == fe::error_code_t::INVALID_VALUE);
}

TEST_CASE("Custom weight dequantization requires logical dimensions and physical inputs",
          "[weight_dequantize][validate]") {
    DecodeGraph g(program());
    SECTION("logical shape is not storage shape") { g.decoded->set_dim({}); }
    SECTION("batched weights") { g.decoded->set_dim({2, 64, 64}); }
    SECTION("overflowing dimensions") { g.decoded->set_dim({1, int64_t(INT32_MAX) + 1, 64}); }
    SECTION("float decoded output") { g.decoded->set_data_type(fe::DataType_t::FLOAT); }
    SECTION("materialized decoded output") { g.decoded->set_output(true); }
    SECTION("virtual storage") { g.weights->set_is_virtual(true); }
    SECTION("virtual scale") { g.auxiliary[0]->set_is_virtual(true); }
    SECTION("pass by value scale") { g.auxiliary[0]->set_is_pass_by_value(true); }
    REQUIRE_FALSE(g.graph.validate().is_good());
}

TEST_CASE("Custom weight dequantization owns program and ordered auxiliaries", "[weight_dequantize][serialize]") {
    auto count = GENERATE(0, 2, 8);
    auto p     = program();
    DecodeGraph g(p, count);
    p.source = "changed after constructing graph";
    p.constants.clear();
    REQUIRE(g.graph.validate().is_good());
    REQUIRE(g.decoded->get_stride() == std::vector<int64_t>{4096, 64, 1});
    json saved = g.graph;
    REQUIRE(saved["nodes"][0]["program"]["source"] == program().source);
    REQUIRE(saved["nodes"][0]["program"]["constants"] == program().constants);
    for (int i = 0; i < count; ++i)
        REQUIRE(saved["nodes"][0]["inputs"]["AUX_" + std::to_string(i)] == g.auxiliary[i]->get_uid());
    fe::graph::Graph restored;
    REQUIRE(restored.deserialize(saved).is_good());
    REQUIRE(restored.validate().is_good());
    REQUIRE(json(restored) == saved);
    REQUIRE(restored.key() == g.graph.key());
}

TEST_CASE("Custom weight dequantization rejects excessive auxiliaries", "[weight_dequantize][validate]") {
    REQUIRE(DecodeGraph(program(), 9).graph.validate().get_code() == fe::error_code_t::INVALID_VALUE);
}

TEST_CASE("Custom weight dequantization program affects graph identity", "[weight_dequantize][serialize]") {
    DecodeGraph original(program());
    REQUIRE(original.graph.validate().is_good());
    auto p = program();
    SECTION("source") { p.source += "// different conversion\n"; }
    SECTION("entry") { p.entry = "another_decode"; }
    SECTION("constants") { p.constants[0] = 2; }
    SECTION("CTA scratch") { p.cta_smem_bytes = 256; }
    SECTION("stage scratch") { p.stage_smem_bytes = 256; }
    SECTION("alignment") { p.input_alignment = 32; }
    SECTION("tile") { p.tile_shape = {64, 64}; }
    DecodeGraph different(p);
    REQUIRE(different.graph.validate().is_good());
    REQUIRE(original.graph.key() != different.graph.key());
}

TEST_CASE("Custom weight dequantization reports unavailable headers", "[weight_dequantize][compatibility]") {
#if !defined(CUDNN_WEIGHT_DECODE_ABI_VERSION) || CUDNN_WEIGHT_DECODE_ABI_VERSION < 1
    fe::graph::WeightDequantizeNode node{fe::graph::Weight_dequantize_attributes(), fe::detail::Context()};
    std::unordered_set<int64_t> uids;
    std::vector<std::shared_ptr<fe::Operation>> operations;
    fe::graph::managed_backend_descriptor_t raw;
    std::unordered_map<int64_t, std::shared_ptr<fe::Tensor>> tensors;
    REQUIRE(node.create_cudnn_operations(uids, operations, raw, tensors).get_code() ==
            fe::error_code_t::GRAPH_NOT_SUPPORTED);
#else
    SUCCEED("Experimental header capability is available");
#endif
}
