/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <catch2/catch_test_macros.hpp>

#include <cudnn_frontend.h>

#include <cmath>
#include <thread>
#include <vector>

TEST_CASE("Tensor attributes", "[tensor][serialize]") {
    namespace fe = cudnn_frontend;

    auto tensor_attributes = fe::graph::Tensor_attributes()
                                 .set_name("image")
                                 .set_dim({4, 32, 16, 16})
                                 .set_stride({32 * 16 * 16, 1, 32 * 16, 32})
                                 .set_is_virtual(true)
                                 .set_is_pass_by_value(true)
                                 .set_uid(12312)
                                 .set_reordering_type(fe::TensorReordering_t::F16x16)
                                 .set_data_type(fe::DataType_t::HALF);

    json j                              = tensor_attributes;
    auto tensor_attributes_deserialized = j;

    REQUIRE(tensor_attributes_deserialized == tensor_attributes);
}

TEST_CASE("Tensor attributes alignment", "[tensor][serialize]") {
    namespace fe = cudnn_frontend;

    auto make_tensor = []() -> fe::graph::Tensor_attributes {
        return fe::graph::Tensor_attributes()
            .set_name("image")
            .set_dim({4, 32, 16, 16})
            .set_stride({32 * 16 * 16, 1, 32 * 16, 32})
            .set_uid(12312)
            .set_data_type(fe::DataType_t::HALF);
    };

    SECTION("a non-default alignment survives the round trip") {
        auto tensor_attributes = make_tensor().set_alignment(4);

        json j = tensor_attributes;
        REQUIRE(j.contains("alignment"));
        REQUIRE(j["alignment"].get<int64_t>() == 4);

        auto deserialized = j.get<fe::graph::Tensor_attributes>();
        REQUIRE(deserialized.get_alignment() == 4);
    }

    SECTION("the default alignment is not emitted") {
        auto tensor_attributes = make_tensor();
        REQUIRE(tensor_attributes.get_alignment() == fe::graph::Tensor_attributes::default_alignment);

        json j = tensor_attributes;
        REQUIRE_FALSE(j.contains("alignment"));

        auto deserialized = j.get<fe::graph::Tensor_attributes>();
        REQUIRE(deserialized.get_alignment() == fe::graph::Tensor_attributes::default_alignment);
    }

    SECTION("a payload with no alignment key deserializes to the default") {
        json j = make_tensor().set_alignment(4);
        REQUIRE(j.erase("alignment") == 1);

        fe::graph::Tensor_attributes deserialized;
        REQUIRE_NOTHROW(deserialized = j.get<fe::graph::Tensor_attributes>());
        REQUIRE(deserialized.get_alignment() == fe::graph::Tensor_attributes::default_alignment);
    }
}

TEST_CASE("conv graph serialization preserves alignment", "[graph][serialize]") {
    namespace fe = cudnn_frontend;

    fe::graph::Graph graph;

    auto x = graph.tensor(fe::graph::Tensor_attributes());
    x->set_name("image")
        .set_dim({4, 32, 16, 16})
        .set_stride({32 * 16 * 16, 1, 32 * 16, 32})
        .set_data_type(fe::DataType_t::HALF)
        .set_alignment(4);

    auto w = graph.tensor(fe::graph::Tensor_attributes());
    w->set_name("weight")
        .set_dim({64, 32, 3, 3})
        .set_stride({32 * 3 * 3, 1, 32 * 3, 32})
        .set_data_type(fe::DataType_t::HALF);

    auto conv_fprop_attributes = fe::graph::Conv_fprop_attributes()
                                     .set_name("conv_fprop")
                                     .set_padding({1, 1})
                                     .set_stride({1, 1})
                                     .set_dilation({1, 1})
                                     .set_compute_data_type(fe::DataType_t::FLOAT);

    auto y = graph.conv_fprop(x, w, conv_fprop_attributes);
    y->set_name("output").set_output(true).set_data_type(fe::DataType_t::HALF);

    auto alignment_of = [](json const& serialized, std::string const& name) -> int64_t {
        for (auto const& tensor : serialized["tensors"]) {
            if (tensor.contains("name") && tensor["name"].get<std::string>() == name) {
                return tensor.value("alignment", fe::graph::Tensor_attributes::default_alignment);
            }
        }
        return -1;
    };

    json j = graph;

    REQUIRE(alignment_of(j, "image") == 4);
    REQUIRE(alignment_of(j, "weight") == fe::graph::Tensor_attributes::default_alignment);

    fe::graph::Graph graph_deserialized;
    REQUIRE(graph_deserialized.deserialize(j).is_good());

    json j2 = graph_deserialized;

    REQUIRE(alignment_of(j2, "image") == 4);
    REQUIRE(j == j2);
}

TEST_CASE("Context serialization", "[context][serialize]") {
    namespace fe = cudnn_frontend;

    fe::graph::Graph graph;

    graph.set_io_data_type(fe::DataType_t::HALF)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT)
        .set_sm_count(24);

    json j = graph;

    std::cout << j << std::endl;

    fe::graph::Graph graph_deserialized;

    REQUIRE(graph_deserialized.deserialize(j).is_good());

    json j2 = graph_deserialized;

    REQUIRE(j == j2);

    auto status = fe::graph::Graph().deserialize(j, true);
    REQUIRE(status.is_bad());
    REQUIRE(status.get_message().find("enforce_precompiled") != std::string::npos);

    REQUIRE(graph.validate().is_good());
}

TEST_CASE("Conv fprop attributes", "[conv_fprop][serialize]") {
    namespace fe = cudnn_frontend;

    auto x = std::make_shared<fe::graph::Tensor_attributes>();
    x->set_name("image")
        .set_dim({4, 32, 16, 16})
        .set_stride({32 * 16 * 16, 1, 32 * 16, 32})
        .set_is_virtual(true)
        .set_is_pass_by_value(true)
        .set_uid(12312)
        .set_reordering_type(fe::TensorReordering_t::F16x16)
        .set_data_type(fe::DataType_t::HALF);

    auto conv_fprop_attributes = fe::graph::Conv_fprop_attributes()
                                     .set_name("conv_fprop")
                                     .set_padding({1, 1})
                                     .set_stride({1, 1})
                                     .set_dilation({1, 1})
                                     .set_compute_data_type(fe::DataType_t::FLOAT);

    json j                                  = conv_fprop_attributes;
    auto conv_fprop_attributes_deserialized = j;

    REQUIRE(conv_fprop_attributes_deserialized == conv_fprop_attributes);
}

TEST_CASE("Graph key", "[graph][key]") {
    namespace fe = cudnn_frontend;

    fe::graph::Graph graph;
    graph.set_io_data_type(fe::DataType_t::HALF)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    auto X = graph.tensor(
        fe::graph::Tensor_attributes().set_name("image").set_dim({4, 16, 64}).set_stride({16 * 64, 1, 16}));
    auto Y = graph.tensor(
        fe::graph::Tensor_attributes().set_name("filter").set_dim({4, 64, 32}).set_stride({32 * 64, 1, 64}));

    fe::graph::Matmul_attributes matmul;
    auto Z = graph.matmul(X, Y, matmul);

    auto scale_options = fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::MUL);
    auto S             = graph.tensor(
        fe::graph::Tensor_attributes().set_name("scale").set_dim({4, 16, 32}).set_stride({16 * 32, 32, 1}));
    auto scale_output = graph.pointwise(Z, S, scale_options);

    auto bias_options = fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::ADD);
    auto B =
        graph.tensor(fe::graph::Tensor_attributes().set_name("bias").set_dim({4, 16, 32}).set_stride({16 * 32, 32, 1}));
    auto bias_output = graph.pointwise(scale_output, B, bias_options);

    auto relu_options = fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::RELU_FWD);
    auto O            = graph.pointwise(bias_output, relu_options);
    O->set_output(true);

    cudnnHandle_t handle;
    cudnnCreate(&handle);

    REQUIRE(graph.validate().is_good());

    REQUIRE(graph.build_operation_graph(handle).is_good());
    auto key = graph.key();

    REQUIRE(graph.create_execution_plans({fe::HeurMode_t::A}).is_good());
    REQUIRE(key == graph.key());

    REQUIRE(graph.check_support().is_good());
    REQUIRE(key == graph.key());

    REQUIRE(graph.build_plans().is_good());
    REQUIRE(key == graph.key());

    cudnnDestroy(handle);
}

TEST_CASE("Graph key dynamic shape", "[graph][key][dynamic_shape]") {
    namespace fe = cudnn_frontend;
    if (cudnnGetCudartVersion() < 12000) {
        SKIP("Test requires cuda toolkit 12.0 or above");
    }
    // clang-format off
    struct {
        int64_t b,    m,    n,    k;
    } shapes[] = {
        {       4,   16,   32,   64},
        {       8,   32,   64,  128},
    };
    // clang-format on

    constexpr int shapes_count = sizeof(shapes) / sizeof(shapes[0]);
    size_t key                 = 0;  // Save key between runs to verify that dim and stride information is deleted

    for (int idx_shape = 0; idx_shape < shapes_count; idx_shape++) {
        auto b = shapes[idx_shape].b;
        auto m = shapes[idx_shape].m;
        auto n = shapes[idx_shape].n;
        auto k = shapes[idx_shape].k;

        fe::graph::Graph graph;
        graph.set_io_data_type(fe::DataType_t::HALF)
            .set_intermediate_data_type(fe::DataType_t::FLOAT)
            .set_compute_data_type(fe::DataType_t::FLOAT)
            .set_dynamic_shape_enabled(true);

        auto X =
            graph.tensor(fe::graph::Tensor_attributes().set_name("image").set_dim({b, m, k}).set_stride({m * k, 1, m}));
        auto Y = graph.tensor(
            fe::graph::Tensor_attributes().set_name("filter").set_dim({b, k, n}).set_stride({n * k, 1, k}));

        fe::graph::Matmul_attributes matmul;
        auto Z = graph.matmul(X, Y, matmul);

        auto scale_options = fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::MUL);
        auto S =
            graph.tensor(fe::graph::Tensor_attributes().set_name("scale").set_dim({b, m, n}).set_stride({m * n, n, 1}));
        auto scale_output = graph.pointwise(Z, S, scale_options);

        auto bias_options = fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::ADD);
        auto B =
            graph.tensor(fe::graph::Tensor_attributes().set_name("bias").set_dim({b, m, n}).set_stride({m * n, n, 1}));
        auto bias_output = graph.pointwise(scale_output, B, bias_options);

        auto relu_options = fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::RELU_FWD);
        auto O            = graph.pointwise(bias_output, relu_options);
        O->set_output(true);

        cudnnHandle_t handle;
        cudnnCreate(&handle);

        auto status = graph.validate();
        if (cudnnGetVersion() >= 90400) {
            REQUIRE(status.is_good());
        } else {
            REQUIRE(status.is_bad());
            SKIP("Dynamic shapes not supported pre 9.4");
        }

        REQUIRE(graph.build_operation_graph(handle).is_good());

        if (!key) {
            key = graph.key();
        }

        REQUIRE(graph.create_execution_plans({fe::HeurMode_t::A}).is_good());
        REQUIRE(key == graph.key());

        REQUIRE(graph.check_support().is_good());
        REQUIRE(key == graph.key());

        REQUIRE(graph.build_plans().is_good());
        REQUIRE(key == graph.key());

        cudnnDestroy(handle);
    }
}

TEST_CASE("Matmul fp8 fusion", "[graph][serialize]") {
    namespace fe = cudnn_frontend;
    // matmul problem size
    int64_t const b = 16;
    int64_t const m = 32;
    int64_t const n = 64;
    int64_t const k = 128;

    fe::graph::Graph graph{};

    // Create the two non-virtual input tensors A and B.
    // There are read from global memory.
    auto A_attributes = fe::graph::Tensor_attributes()
                            .set_name("A")
                            .set_dim({b, m, k})
                            .set_stride({m * k, k, 1})
                            .set_data_type(fe::DataType_t::FP8_E4M3);
    auto A = graph.tensor(A_attributes);

    auto B_attributes = fe::graph::Tensor_attributes()
                            .set_name("B")
                            .set_dim({b, k, n})
                            .set_stride({k * n, 1, k})
                            .set_data_type(fe::DataType_t::FP8_E4M3);
    auto B = graph.tensor(B_attributes);

    auto A_descale_attributes = fe::graph::Tensor_attributes()
                                    .set_name("descale0")
                                    .set_dim({1, 1, 1})
                                    .set_stride({1, 1, 1})
                                    .set_data_type(fe::DataType_t::FLOAT);
    auto B_descale_attributes = fe::graph::Tensor_attributes()
                                    .set_name("descale1")
                                    .set_dim({1, 1, 1})
                                    .set_stride({1, 1, 1})
                                    .set_data_type(fe::DataType_t::FLOAT);

    auto A_descale = graph.tensor(A_descale_attributes);
    auto B_descale = graph.tensor(B_descale_attributes);

    auto matmul_attributes =
        // fe::graph::Matmul_attributes().set_name("GEMM").set_compute_data_type(fe::DataType_t::FLOAT);
        fe::graph::Matmul_attributes().set_name("GEMM").set_compute_data_type(fe::DataType_t::FLOAT);
    auto C = graph.matmul(A, B, matmul_attributes);
    C->set_data_type(fe::DataType_t::FLOAT);

    // Add scale_A operation
    auto pw_0_attributes = fe::graph::Pointwise_attributes()
                               //    .set_name("pw0_Mul")
                               .set_mode(fe::PointwiseMode_t::MUL)
                               .set_compute_data_type(fe::DataType_t::FLOAT);
    auto C_after_pw_0 = graph.pointwise(C, A_descale, pw_0_attributes);
    C_after_pw_0->set_data_type(fe::DataType_t::FLOAT);

    // Add descale_B operation
    auto pw_1_attributes = fe::graph::Pointwise_attributes()
                               //    .set_name("pw1_Mul")
                               .set_mode(fe::PointwiseMode_t::MUL)
                               .set_compute_data_type(fe::DataType_t::FLOAT);
    auto C_after_pw_1 = graph.pointwise(C_after_pw_0, B_descale, pw_1_attributes);
    C_after_pw_1->set_output(true).set_data_type(fe::DataType_t::BFLOAT16);

    json j = graph;

    std::cout << j << std::endl;

    fe::graph::Graph graph_deserialized;

    REQUIRE(graph_deserialized.deserialize(j).is_good());

    json j2 = graph_deserialized;

    REQUIRE(j == j2);

    REQUIRE(graph.validate().is_good());

    std::cout << "Validating deserialized graph" << std::endl;

    cudnnHandle_t handle;  // Handle to use during deserialize and execute

    cudnnCreate(&handle);

    REQUIRE(graph_deserialized.validate().is_good());

    REQUIRE(graph_deserialized.build_operation_graph(handle).is_good());

    cudnnDestroy(handle);
}

TEST_CASE("conv graph serialization", "[graph][serialize]") {
    namespace fe = cudnn_frontend;

    fe::graph::Graph graph;

    auto x = graph.tensor(fe::graph::Tensor_attributes());
    x->set_name("image")
        .set_dim({4, 32, 16, 16})
        .set_stride({32 * 16 * 16, 1, 32 * 16, 32})
        .set_is_virtual(false)
        .set_is_pass_by_value(false)
        .set_reordering_type(fe::TensorReordering_t::NONE)
        .set_data_type(fe::DataType_t::HALF);

    auto w = graph.tensor(fe::graph::Tensor_attributes());
    w->set_name("weight")
        .set_dim({64, 32, 3, 3})
        .set_stride({32 * 3 * 3, 1, 32 * 3, 32})
        .set_is_virtual(false)
        .set_is_pass_by_value(false)
        .set_reordering_type(fe::TensorReordering_t::NONE)
        .set_data_type(fe::DataType_t::HALF);

    auto conv_fprop_attributes = fe::graph::Conv_fprop_attributes()
                                     .set_name("conv_fprop")
                                     .set_padding({1, 1})
                                     .set_stride({1, 1})
                                     .set_dilation({1, 1})
                                     .set_compute_data_type(fe::DataType_t::FLOAT);

    auto y = graph.conv_fprop(x, w, conv_fprop_attributes);

    auto b = graph.tensor(fe::graph::Tensor_attributes());
    b->set_name("bias")
        .set_dim({1, 64, 1, 1})
        .set_stride({64, 1, 64, 64})
        .set_is_virtual(false)
        .set_is_pass_by_value(false)
        .set_reordering_type(fe::TensorReordering_t::NONE)
        .set_data_type(fe::DataType_t::HALF);

    auto pointwise_attributes = fe::graph::Pointwise_attributes()
                                    .set_name("bias")
                                    .set_mode(fe::PointwiseMode_t::ADD)
                                    .set_compute_data_type(fe::DataType_t::FLOAT);

    auto o = graph.pointwise(y, b, pointwise_attributes);

    auto reduction_attributes = fe::graph::Reduction_attributes()
                                    .set_name("reduction")
                                    .set_mode(fe::ReductionMode_t::ADD)
                                    .set_compute_data_type(fe::DataType_t::FLOAT);
    auto r = graph.reduction(o, reduction_attributes);

    r->set_output(true).set_data_type(fe::DataType_t::HALF);

    json j = graph;

    fe::graph::Graph graph_deserialized;

    REQUIRE(graph_deserialized.deserialize(j).is_good());

    json j2 = graph_deserialized;

    REQUIRE(j == j2);

    REQUIRE(graph_deserialized.validate().is_good());
}

TEST_CASE("sdpa graph serialization", "[graph][serialize]") {
    namespace fe = cudnn_frontend;

    fe::graph::Graph graph;
    graph.set_io_data_type(fe::DataType_t::HALF)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    int64_t b    = 3;     // batch size
    int64_t h    = 4;     // head dim
    int64_t s_q  = 1024;  // q tensor is padded to this seq length
    int64_t s_kv = 1024;  // k and v tensor is padded to this seq length
    int64_t d    = 128;   // hidden dim

    auto Q = graph.tensor(fe::graph::Tensor_attributes()
                              .set_name("Q")
                              .set_dim({b, h, s_q, d})
                              .set_stride({3 * h * d, 3 * d, 3 * b * h * d, 1}));
    auto K = graph.tensor(fe::graph::Tensor_attributes()
                              .set_name("K")
                              .set_dim({b, h, s_kv, d})
                              .set_stride({3 * h * d, 3 * d, 3 * b * h * d, 1}));
    auto V = graph.tensor(fe::graph::Tensor_attributes()
                              .set_name("V")
                              .set_dim({b, h, s_kv, d})
                              .set_stride({3 * h * d, 3 * d, 3 * b * h * d, 1}));

    auto attn_scale = graph.tensor(fe::graph::Tensor_attributes()
                                       .set_name("attn_scale")
                                       .set_dim({1, 1, 1, 1})
                                       .set_stride({1, 1, 1, 1})
                                       .set_is_pass_by_value(true)
                                       .set_data_type(fe::DataType_t::FLOAT));

    auto seed   = graph.tensor(fe::graph::Tensor_attributes()
                                 .set_name("Seed")
                                 .set_dim({1, 1, 1, 1})
                                 .set_stride({1, 1, 1, 1})
                                 .set_data_type(fe::DataType_t::INT32));
    auto offset = graph.tensor(fe::graph::Tensor_attributes()
                                   .set_name("Offset")
                                   .set_dim({1, 1, 1, 1})
                                   .set_stride({1, 1, 1, 1})
                                   .set_data_type(fe::DataType_t::INT32));

    auto bias = graph.tensor(fe::graph::Tensor_attributes()
                                 .set_name("bias")
                                 .set_dim({b, 1, s_q, s_kv})
                                 .set_stride({s_q * s_kv, s_q * s_kv, s_kv, 1}));

    auto seq_q  = graph.tensor(fe::graph::Tensor_attributes()
                                  .set_name("seq_q")
                                  .set_dim({b, 1, 1, 1})
                                  .set_stride({1, 1, 1, 1})
                                  .set_data_type(fe::DataType_t::INT32));
    auto seq_kv = graph.tensor(fe::graph::Tensor_attributes()
                                   .set_name("seq_kv")
                                   .set_dim({b, 1, 1, 1})
                                   .set_stride({1, 1, 1, 1})
                                   .set_data_type(fe::DataType_t::INT32));

    auto sdpa_options = fe::graph::SDPA_attributes()
                            .set_name("flash_attention")
                            .set_generate_stats(true)
                            .set_attn_scale(attn_scale)
                            .set_alibi_mask(true)
                            .set_diagonal_band_right_bound(0)
                            .set_dropout(0.1f, seed, offset)
                            .set_bias(bias)
                            .set_padding_mask(true)
                            .set_seq_len_q(seq_q)
                            .set_seq_len_kv(seq_kv);

    auto [O, stats] = graph.sdpa(Q, K, V, sdpa_options);

    O->set_output(true).set_dim({b, h, s_q, d}).set_stride({h * d, d, b * h * d, 1});
    stats->set_output(true).set_data_type(fe::DataType_t::FLOAT);

    json j = graph;

    fe::graph::Graph graph_deserialized;
    REQUIRE(graph_deserialized.deserialize(j).is_good());
    json j2 = graph_deserialized;

    REQUIRE(j == j2);
}

TEST_CASE("sdpa backward graph serialization", "[graph][serialize]") {
    namespace fe = cudnn_frontend;

    int64_t b    = 3;     // batch size
    int64_t h    = 4;     // head dim
    int64_t s_q  = 1024;  // q tensor is padded to this seq length
    int64_t s_kv = 1024;  // k and v tensor is padded to this seq length
    int64_t d    = 128;   // hidden dim

    fe::graph::Graph graph;
    graph.set_io_data_type(fe::DataType_t::HALF)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    std::shared_ptr<fe::graph::Tensor_attributes> bias, dropout_seed, dropout_offset;

    auto q = graph.tensor(
        fe::graph::Tensor_attributes().set_name("Q").set_dim({b, h, s_q, d}).set_stride({h * s_q * d, s_q * d, d, 1}));
    auto k = graph.tensor(fe::graph::Tensor_attributes()
                              .set_name("K")
                              .set_dim({b, h, s_kv, d})
                              .set_stride({h * s_kv * d, s_kv * d, d, 1}));
    auto v = graph.tensor(fe::graph::Tensor_attributes()
                              .set_name("V")
                              .set_dim({b, h, s_kv, d})
                              .set_stride({h * s_kv * d, s_kv * d, d, 1}));
    auto o = graph.tensor(
        fe::graph::Tensor_attributes().set_name("O").set_dim({b, h, s_q, d}).set_stride({h * s_q * d, s_q * d, d, 1}));
    auto dO = graph.tensor(
        fe::graph::Tensor_attributes().set_name("dO").set_dim({b, h, s_q, d}).set_stride({h * s_q * d, s_q * d, d, 1}));
    auto stats = graph.tensor(fe::graph::Tensor_attributes()
                                  .set_name("stats")
                                  .set_dim({b, h, s_q, 1})
                                  .set_stride({h * s_q, s_q, 1, 1})
                                  .set_data_type(fe::DataType_t::FLOAT));

    auto attn_scale = graph.tensor(fe::graph::Tensor_attributes()
                                       .set_name("attn_scale")
                                       .set_dim({1, 1, 1, 1})
                                       .set_stride({1, 1, 1, 1})
                                       .set_is_pass_by_value(true)
                                       .set_data_type(fe::DataType_t::FLOAT));

    bias = graph.tensor(fe::graph::Tensor_attributes()
                            .set_name("bias")
                            .set_dim({b, 1, s_q, s_kv})
                            .set_stride({s_q * s_kv, s_q * s_kv, s_kv, 1}));

    dropout_seed   = graph.tensor(fe::graph::Tensor_attributes()
                                    .set_name("Seed")
                                    .set_dim({1, 1, 1, 1})
                                    .set_stride({1, 1, 1, 1})
                                    .set_data_type(fe::DataType_t::INT32));
    dropout_offset = graph.tensor(fe::graph::Tensor_attributes()
                                      .set_name("Offset")
                                      .set_dim({1, 1, 1, 1})
                                      .set_stride({1, 1, 1, 1})
                                      .set_data_type(fe::DataType_t::INT32));

    auto sdpa_backward_options = fe::graph::SDPA_backward_attributes()
                                     .set_name("flash_attention_backward")
                                     .set_diagonal_alignment(cudnn_frontend::DiagonalAlignment_t::TOP_LEFT)
                                     .set_diagonal_band_right_bound(0)
                                     .set_attn_scale(attn_scale)
                                     .set_bias(bias)
                                     .set_dropout(0.1f, dropout_seed, dropout_offset);

    auto [dQ, dK, dV] = graph.sdpa_backward(q, k, v, o, dO, stats, sdpa_backward_options);

    dQ->set_output(true).set_dim({b, h, s_q, d}).set_stride({h * s_q * d, s_q * d, d, 1});
    dK->set_output(true).set_dim({b, h, s_kv, d}).set_stride({h * s_kv * d, s_kv * d, d, 1});
    dV->set_output(true).set_dim({b, h, s_kv, d}).set_stride({h * s_kv * d, s_kv * d, d, 1});

    json j = graph;
    fe::graph::Graph graph_deserialized;
    REQUIRE(graph_deserialized.deserialize(j).is_good());
    json j2 = graph_deserialized;

    REQUIRE(j == j2);

    REQUIRE(graph_deserialized.validate().is_good());
}

// Round-trips a plan through binary serialize/deserialize and confirms the variant
// pack template is populated on the deserialized graph. The check verifies that
// deserialize(handle, ...) eagerly prepares the template (instead of waiting for
// first execute).
TEST_CASE("Plan deserialize prepares variant pack template", "[graph][serialize][deserialize]") {
    namespace fe = cudnn_frontend;

    constexpr int64_t a_uid = 1, b_uid = 2, c_uid = 3;

    fe::graph::Graph graph;
    graph.set_io_data_type(fe::DataType_t::HALF)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    auto A = graph.tensor(
        fe::graph::Tensor_attributes().set_name("A").set_dim({4, 16, 64}).set_stride({16 * 64, 64, 1}).set_uid(a_uid));
    auto B = graph.tensor(
        fe::graph::Tensor_attributes().set_name("B").set_dim({4, 64, 32}).set_stride({64 * 32, 32, 1}).set_uid(b_uid));

    auto C = graph.matmul(A, B, fe::graph::Matmul_attributes().set_name("matmul"));
    C->set_output(true).set_uid(c_uid);

    cudnnHandle_t handle;
    cudnnCreate(&handle);

    REQUIRE(graph.build(handle, {fe::HeurMode_t::A}).is_good());

    std::vector<uint8_t> serialized_data;
    REQUIRE(graph.serialize(serialized_data).is_good());

    fe::graph::Graph graph_deserialized;
    REQUIRE(graph_deserialized.deserialize(handle, serialized_data, true).is_good());

    // Variant pack template should already be populated; no execute needed.
    auto const user_uids = graph_deserialized.get_variant_pack_uids_sorted();
    REQUIRE(user_uids == std::vector<int64_t>{a_uid, b_uid, c_uid});

    cudnnDestroy(handle);
}

// Exercises the run_warmup=false fast path of deserialize(handle, ...). Skipping
// the throwaway warmup capture must not change the resulting plan: the variant
// pack template is built by prepare_variant_pack_template(), which is independent
// of warmup, so the deserialized graph is still fully usable.
//
// Note the additive argument order: deserialize(handle, data, enforce_precompiled, run_warmup).
// Both bools are spelled explicitly below so the positional meaning is unambiguous.
TEST_CASE("Plan deserialize with run_warmup=false still prepares template", "[graph][serialize][deserialize]") {
    namespace fe = cudnn_frontend;

    constexpr int64_t a_uid = 1, b_uid = 2, c_uid = 3;

    fe::graph::Graph graph;
    graph.set_io_data_type(fe::DataType_t::HALF)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    auto A = graph.tensor(
        fe::graph::Tensor_attributes().set_name("A").set_dim({4, 16, 64}).set_stride({16 * 64, 64, 1}).set_uid(a_uid));
    auto B = graph.tensor(
        fe::graph::Tensor_attributes().set_name("B").set_dim({4, 64, 32}).set_stride({64 * 32, 32, 1}).set_uid(b_uid));

    auto C = graph.matmul(A, B, fe::graph::Matmul_attributes().set_name("matmul"));
    C->set_output(true).set_uid(c_uid);

    cudnnHandle_t handle;
    cudnnCreate(&handle);

    REQUIRE(graph.build(handle, {fe::HeurMode_t::A}).is_good());

    // serialize the graph
    std::vector<uint8_t> serialized_data;
    REQUIRE(graph.serialize(serialized_data).is_good());

    // expected uids
    std::vector<int64_t> const expected_uids{a_uid, b_uid, c_uid};

    // test the blob overload, run_warmup=false.
    // Check for success and variant pack uids are the same as the expected uids
    SECTION("blob overload, run_warmup=false") {
        fe::graph::Graph graph_deserialized;
        REQUIRE(
            graph_deserialized.deserialize(handle, serialized_data, /*enforce_precompiled=*/false, /*run_warmup=*/false)
                .is_good());
        // check the variant pack uids are the same as the expected uids
        REQUIRE(graph_deserialized.get_variant_pack_uids_sorted() == expected_uids);
        // tensor metadata must still resolve with warmup skipped (deserialized_tensor_properties)
        fe::graph::Tensor_attributes queried;
        REQUIRE(graph_deserialized.query_tensor_attributes_of_uid(a_uid, queried).is_good());
    }

    // test the json overload, run_warmup=false.
    // same assertion via the pre-parsed-json overload, covering the path that avoids a second from_ubjson.
    SECTION("json overload, run_warmup=false") {
        json const j = json::from_ubjson(serialized_data);
        fe::graph::Graph graph_deserialized;
        auto const status =
            graph_deserialized.deserialize(handle, j, /*enforce_precompiled=*/false, /*run_warmup=*/false);
        REQUIRE(status.is_good());
        REQUIRE(graph_deserialized.get_variant_pack_uids_sorted() == expected_uids);
    }

    // test the run_warmup=false matches the default warmup=true template
    // Confirms warmup is purely a priming step with no effect on the executable plan.
    SECTION("run_warmup=false matches the default warmup=true template") {
        fe::graph::Graph warmed;
        REQUIRE(
            warmed.deserialize(handle, serialized_data, /*enforce_precompiled=*/false, /*run_warmup=*/true).is_good());

        fe::graph::Graph skipped;
        auto const skipped_status =
            skipped.deserialize(handle, serialized_data, /*enforce_precompiled=*/false, /*run_warmup=*/false);
        REQUIRE(skipped_status.is_good());

        // The plan-side state the fast execute path relies on is identical either way.
        REQUIRE(warmed.get_variant_pack_uids_sorted() == skipped.get_variant_pack_uids_sorted());
        REQUIRE(skipped.get_variant_pack_uids_sorted() == expected_uids);
    }

    cudnnDestroy(handle);
}

// serialize(data, serialize_structure=false) omits the graph structure
// (nodes/tensors) while keeping the plan reloadable via deserialize(handle, ...).
// The default (true) still emits the structure, so existing callers are unaffected.
TEST_CASE("serialize_structure flag controls structural payload", "[graph][serialize][deserialize]") {
    namespace fe = cudnn_frontend;

    fe::graph::Graph graph;
    graph.set_io_data_type(fe::DataType_t::HALF)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    auto A = graph.tensor(
        fe::graph::Tensor_attributes().set_name("A").set_dim({4, 16, 64}).set_stride({16 * 64, 64, 1}).set_uid(1));
    auto B = graph.tensor(
        fe::graph::Tensor_attributes().set_name("B").set_dim({4, 64, 32}).set_stride({64 * 32, 32, 1}).set_uid(2));
    auto C = graph.matmul(A, B, fe::graph::Matmul_attributes().set_name("matmul"));
    C->set_output(true).set_uid(3);

    cudnnHandle_t handle;
    cudnnCreate(&handle);

    REQUIRE(graph.build(handle, {fe::HeurMode_t::A}).is_good());

    std::vector<uint8_t> with_structure, without_structure;
    REQUIRE(graph.serialize(with_structure).is_good());  // default: serialize_structure=true
    REQUIRE(graph.serialize(without_structure, /*serialize_structure=*/false).is_good());

    json const j_with    = json::from_ubjson(with_structure);
    json const j_without = json::from_ubjson(without_structure);

    // Default emits the structure; opting out drops it and shrinks the blob.
    REQUIRE(j_with.contains("nodes"));
    REQUIRE_FALSE(j_without.contains("nodes"));
    REQUIRE(without_structure.size() < with_structure.size());

    // Both remain reloadable through the plan path with identical variant packs.
    auto const expected_uids = graph.get_variant_pack_uids_sorted();
    for (auto const& blob : {with_structure, without_structure}) {
        fe::graph::Graph reloaded;
        REQUIRE(reloaded.deserialize(handle, blob, /*enforce_precompiled=*/false, /*run_warmup=*/false).is_good());
        REQUIRE(reloaded.get_variant_pack_uids_sorted() == expected_uids);
    }

    cudnnDestroy(handle);
}

// A graph loaded via deserialize(handle, ...) might have no node subtree,
// so serialize() must source its pass-by-value and workspace modifications from the
// cached members instead of walking the (now empty) subtree.
TEST_CASE("Plan re-serialize preserves pass-by-value and workspace modifications", "[graph][serialize][deserialize]") {
    namespace fe = cudnn_frontend;

    fe::graph::Graph graph;
    graph.set_io_data_type(fe::DataType_t::HALF)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    constexpr int64_t N = 4;
    auto X              = graph.tensor(fe::graph::Tensor_attributes()
                              .set_name("X")
                              .set_dim({N, N, N})
                              .set_stride({N * N, N, 1})
                              .set_data_type(fe::DataType_t::HALF)
                              .set_uid(1));
    auto scalar         = graph.tensor(5.0f);
    scalar->set_name("scalar").set_uid(2);
    auto Y = graph.pointwise(X,
                             scalar,
                             fe::graph::Pointwise_attributes()
                                 .set_name("add")
                                 .set_mode(fe::PointwiseMode_t::ADD)
                                 .set_compute_data_type(fe::DataType_t::FLOAT));
    Y->set_output(true).set_data_type(fe::DataType_t::HALF).set_uid(3);

    cudnnHandle_t handle;
    cudnnCreate(&handle);

    REQUIRE(graph.build(handle, {fe::HeurMode_t::A}).is_good());

    std::vector<uint8_t> blob;
    REQUIRE(graph.serialize(blob, /*serialize_structure=*/false).is_good());

    // The fresh graph must actually carry pass-by-value data, else the test is moot.
    json const j_fresh = json::from_ubjson(blob);
    REQUIRE(j_fresh.contains("pass_by_values"));
    REQUIRE_FALSE(j_fresh["pass_by_values"].empty());

    auto const expected_uids = graph.get_variant_pack_uids_sorted();

    fe::graph::Graph reloaded;
    REQUIRE(reloaded.deserialize(handle, blob, /*enforce_precompiled=*/false, /*run_warmup=*/false).is_good());
    REQUIRE(reloaded.get_variant_pack_uids_sorted() == expected_uids);

    // Re-serialize the plan-only graph; the cached maps must survive intact.
    std::vector<uint8_t> next;
    REQUIRE(reloaded.serialize(next, /*serialize_structure=*/false).is_good());

    json const j_next = json::from_ubjson(next);
    REQUIRE(j_next["pass_by_values"] == j_fresh["pass_by_values"]);
    REQUIRE(j_next["workspace_modifications"] == j_fresh["workspace_modifications"]);

    cudnnDestroy(handle);
}

// ============================================================
// Helpers shared by handle-less deserialize tests below.
// ============================================================

namespace {
namespace fe = cudnn_frontend;

// Build a small NHWC FP32 conv graph and return (graph, X-uid, W-uid, Y-uid).
// N=1, C=4, H=8, W=8, K=4, R=1, S=1, padding=0, stride=1, dilation=1.
// This shape is simple but exercises the full stack.
struct ConvGraph {
    static constexpr int64_t N = 1, C = 4, H = 8, W = 8, K = 4, R = 1, S = 1;
    std::shared_ptr<fe::graph::Graph> graph;
    std::shared_ptr<fe::graph::Tensor_attributes> X, W_t, Y;

    ConvGraph() {
        graph = std::make_shared<fe::graph::Graph>();
        graph->set_io_data_type(fe::DataType_t::FLOAT)
            .set_intermediate_data_type(fe::DataType_t::FLOAT)
            .set_compute_data_type(fe::DataType_t::FLOAT);
        // NHWC layout
        X = graph->tensor(
            fe::graph::Tensor_attributes().set_name("X").set_dim({N, C, H, W}).set_stride({C * H * W, 1, C * W, C}));
        W_t = graph->tensor(
            fe::graph::Tensor_attributes().set_name("W").set_dim({K, C, R, S}).set_stride({C * R * S, 1, C * S, C}));
        auto opts = fe::graph::Conv_fprop_attributes().set_padding({0, 0}).set_stride({1, 1}).set_dilation({1, 1});
        Y         = graph->conv_fprop(X, W_t, opts);
        Y->set_output(true);
    }
};

// Build a conv+relu fusion graph for the RTC test.
struct FusionGraph {
    static constexpr int64_t N = 16, C = 128, H = 64, W = 64, K = 256, R = 3, S = 3;
    std::shared_ptr<fe::graph::Graph> graph;
    std::shared_ptr<fe::graph::Tensor_attributes> X, W_t, Y;

    FusionGraph() {
        graph = std::make_shared<fe::graph::Graph>();
        graph->set_io_data_type(fe::DataType_t::HALF)
            .set_intermediate_data_type(fe::DataType_t::FLOAT)
            .set_compute_data_type(fe::DataType_t::FLOAT);
        // NHWC layout
        X = graph->tensor(
            fe::graph::Tensor_attributes().set_name("X").set_dim({N, C, H, W}).set_stride({C * H * W, 1, C * W, C}));
        W_t = graph->tensor(
            fe::graph::Tensor_attributes().set_name("W").set_dim({K, C, R, S}).set_stride({C * R * S, 1, C * S, C}));
        auto conv_opts = fe::graph::Conv_fprop_attributes().set_padding({1, 1}).set_stride({1, 1}).set_dilation({1, 1});
        auto conv      = graph->conv_fprop(X, W_t, conv_opts);
        Y = graph->pointwise(conv, fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::RELU_FWD));
        Y->set_output(true).set_data_type(fe::DataType_t::HALF);
    }
};

// Return the deviceVer encoded in a serialized DeviceProperties byte buffer.
// The wire format is a JSON string, e.g. {"deviceVer":890,...}.
// Returns -1 if the key is not found.
static int
parse_device_ver(std::vector<uint8_t> const& buf) {
    std::string s(reinterpret_cast<const char*>(buf.data()), buf.size());
    auto pos = s.find("\"deviceVer\":");
    if (pos == std::string::npos) return -1;
    pos += std::string("\"deviceVer\":").size();
    return std::stoi(s.substr(pos));
}

// Produce a DeviceProperties whose deviceVer has been changed to a different
// SM than the current device. Returns nullptr if the edit fails.
static std::shared_ptr<fe::DeviceProperties>
make_wrong_arch_devprop() {
    auto dp_real = std::make_shared<fe::DeviceProperties>();
    if (dp_real->set_device_id(0).build().is_bad()) return nullptr;

    std::vector<uint8_t> buf;
    if (dp_real->serialize(buf).is_bad()) return nullptr;

    int real_ver = parse_device_ver(buf);
    if (real_ver < 0) return nullptr;

    // Pick a different SM: add 100 (one major step up) if possible, else subtract 100.
    int wrong_ver = (real_ver < 900) ? (real_ver + 100) : (real_ver - 100);

    std::string s(reinterpret_cast<const char*>(buf.data()), buf.size());
    std::string needle = "\"deviceVer\":" + std::to_string(real_ver);
    std::string repl   = "\"deviceVer\":" + std::to_string(wrong_ver);
    auto pos           = s.find(needle);
    if (pos == std::string::npos) return nullptr;
    s.replace(pos, needle.size(), repl);

    std::vector<uint8_t> bad(s.begin(), s.end());
    auto dp_wrong = std::make_shared<fe::DeviceProperties>();
    if (dp_wrong->deserialize(bad).is_bad()) return nullptr;
    return dp_wrong;
}

}  // namespace

// ============================================================
// Compile-only overload resolution (no GPU, no cuDNN handle).
// Verifies that std::vector<uint8_t> selects the new handle-less
// deserialize overload and not the structural-graph json overload.
// ============================================================

TEST_CASE("Handle-less deserialize overload resolution", "[serialize][graph]") {
    namespace fe = cudnn_frontend;
    // This test is purely compile-time: if the wrong overload is selected the
    // assertion message distinguishes plan vs. structural deserialize.
    // We call it on an empty graph that will fail at runtime (no devprop),
    // but the call must COMPILE to the blob overload, not convert to json.
    fe::graph::Graph g;
    std::vector<uint8_t> blob{0, 1, 2};  // intentionally invalid
    auto err = g.deserialize(blob);
    // No devprop set → ATTRIBUTE_NOT_SET, not a json parse error.
    REQUIRE(err.is_bad());
    REQUIRE(err.get_code() == fe::error_code_t::ATTRIBUTE_NOT_SET);
}

// ============================================================
// Negative: no device properties set → clean ATTRIBUTE_NOT_SET error.
// ============================================================

TEST_CASE("Handle-less deserialize with no devprop returns ATTRIBUTE_NOT_SET", "[serialize][graph]") {
#if (CUDNN_VERSION < 90800)
    SKIP("Handle-less plan deserialize requires cuDNN >= 9.8 headers");
#else
    namespace fe = cudnn_frontend;

    fe::graph::Graph g;
    std::vector<uint8_t> dummy(64, 0);
    auto status = g.deserialize(dummy);
    REQUIRE(status.is_bad());
    REQUIRE(status.get_code() == fe::error_code_t::ATTRIBUTE_NOT_SET);
    REQUIRE(status.get_message().find("set_device_properties") != std::string::npos);
#endif
}

// ============================================================
// Main runtime tests — gated at 9.11 (matching the deviceless sample).
// ============================================================

TEST_CASE("Handle-less plan deserialize", "[serialize][graph]") {
#if (CUDNN_VERSION < 90800)
    SKIP("Handle-less plan deserialize requires cuDNN >= 9.8 headers");
#else
    if (cudnn_frontend::detail::get_backend_version() < 91100) {
        SKIP("Handle-less plan deserialize runtime tests require cuDNN >= 9.11");
    }

    // ── build a device properties descriptor ──────────────────────────────
    auto dp = std::make_shared<cudnn_frontend::DeviceProperties>();
    REQUIRE(dp->set_device_id(0).build().is_good());

    namespace fe = cudnn_frontend;

    SECTION("wrong-arch devprop causes arch mismatch on deserialize") {
        // This is the primary discriminator: if devprop were ignored,
        // deserialize would succeed regardless of which SM is claimed.
        ConvGraph cg;
        cg.graph->set_device_properties(dp);
        REQUIRE(cg.graph->build({fe::HeurMode_t::A, fe::HeurMode_t::FALLBACK}).is_good());
        std::vector<uint8_t> blob;
        REQUIRE(cg.graph->serialize(blob).is_good());

        auto dp_wrong = make_wrong_arch_devprop();
        REQUIRE(dp_wrong != nullptr);

        fe::graph::Graph deser;
        deser.set_device_properties(dp_wrong);
        auto status = deser.deserialize(blob);
        REQUIRE(status.is_bad());
        // Backend returns NOT_SUPPORTED for arch mismatch during plan finalize.
        auto const code  = status.get_code();
        bool is_plan_err = (code == fe::error_code_t::GRAPH_EXECUTION_PLAN_CREATION_FAILED);
        bool is_be_err   = (code == fe::error_code_t::CUDNN_BACKEND_API_FAILED);
        REQUIRE((is_plan_err || is_be_err));
    }

    SECTION("RTC fusion plan carries RUNTIME_COMPILATION behavior note") {
        FusionGraph fg;
        fg.graph->set_device_properties(dp);
        REQUIRE(fg.graph->build_operation_graph().is_good());
        REQUIRE(fg.graph->create_execution_plans({fe::HeurMode_t::A, fe::HeurMode_t::FALLBACK}).is_good());
        fg.graph->select_behavior_notes({fe::BehaviorNote_t::RUNTIME_COMPILATION});
        if (fg.graph->check_support().is_bad()) {
            SKIP("No RUNTIME_COMPILATION engine available for this graph on this GPU/cuDNN");
        }
        REQUIRE(fg.graph->build_plans(fe::BuildPlanPolicy_t::HEURISTICS_CHOICE).is_good());

        std::vector<fe::BehaviorNote_t> notes;
        REQUIRE(fg.graph->get_behavior_notes(notes).is_good());
        bool has_rtc = std::find(notes.begin(), notes.end(), fe::BehaviorNote_t::RUNTIME_COMPILATION) != notes.end();
        REQUIRE(has_rtc);

        std::vector<uint8_t> blob;
        REQUIRE(fg.graph->serialize(blob).is_good());

        fe::graph::Graph deser;
        deser.set_device_properties(dp);
        REQUIRE(deser.deserialize(blob).is_good());
    }

    SECTION("plan-only blob (serialize_structure=false) + handle-less deserialize") {
        ConvGraph cg;
        cg.graph->set_device_properties(dp);
        REQUIRE(cg.graph->build({fe::HeurMode_t::A, fe::HeurMode_t::FALLBACK}).is_good());

        std::vector<uint8_t> blob;
        REQUIRE(cg.graph->serialize(blob, /*serialize_structure=*/false).is_good());

        fe::graph::Graph deser;
        deser.set_device_properties(dp);
        auto status = deser.deserialize(blob);
        REQUIRE(status.is_good());

        int64_t ws = 0;
        REQUIRE(deser.get_workspace_size(ws).is_good());
    }

    SECTION("enforce_precompiled=true rejects blob with no plan data") {
        // enforce_precompiled checks that the blob contains serialized plan bytes
        // (cudnn_backend_data field). Manufacture a blob without that field.
        ConvGraph cg;
        cg.graph->set_device_properties(dp);
        REQUIRE(cg.graph->build({fe::HeurMode_t::A, fe::HeurMode_t::FALLBACK}).is_good());

        std::vector<uint8_t> full_blob;
        REQUIRE(cg.graph->serialize(full_blob).is_good());

        auto j = json::from_ubjson(full_blob);
        j.erase("cudnn_backend_data");
        auto no_plan_blob = json::to_ubjson(j);

        fe::graph::Graph deser;
        deser.set_device_properties(dp);
        auto status = deser.deserialize(no_plan_blob, /*enforce_precompiled=*/true);
        REQUIRE(status.is_bad());
        auto const code  = status.get_code();
        bool is_plan_err = (code == fe::error_code_t::GRAPH_EXECUTION_PLAN_CREATION_FAILED);
        REQUIRE(is_plan_err);
    }

    SECTION("handle overload still uses handle even when devprop is set on graph") {
        // Decision #7: the existing deserialize(handle, blob) overloads forward an
        // explicit device_prop=nullptr so the graph's device_properties is NOT consulted.
        ConvGraph cg;
        cg.graph->set_device_properties(dp);
        REQUIRE(cg.graph->build({fe::HeurMode_t::A, fe::HeurMode_t::FALLBACK}).is_good());
        std::vector<uint8_t> blob;
        REQUIRE(cg.graph->serialize(blob).is_good());

        auto dp_wrong = make_wrong_arch_devprop();
        REQUIRE(dp_wrong != nullptr);

        cudnnHandle_t handle;
        REQUIRE(cudnnCreate(&handle) == CUDNN_STATUS_SUCCESS);

        // Graph has incompatible devprop, but we deserialize via handle overload.
        // Should succeed because handle overload ignores graph's device_properties.
        fe::graph::Graph deser;
        deser.set_device_properties(dp_wrong);  // deliberately incompatible
        auto status = deser.deserialize(handle, blob, false, false);
        REQUIRE(status.is_good());  // handle path used, not devprop

        cudnnDestroy(handle);
    }

    SECTION("concurrency: N threads share one devprop, each deserializes its own blob") {
        // Concurrent rehydration is the primary motivation for this feature.
        // One shared read-only DeviceProperties descriptor is used by all threads;
        // each thread deserializes into its own Graph.
        constexpr int kThreads = 4;

        ConvGraph cg;
        cg.graph->set_device_properties(dp);
        REQUIRE(cg.graph->build({fe::HeurMode_t::A, fe::HeurMode_t::FALLBACK}).is_good());
        std::vector<uint8_t> blob;
        REQUIRE(cg.graph->serialize(blob).is_good());

        std::vector<int> results(kThreads, 0);
        std::vector<std::thread> threads;
        threads.reserve(kThreads);

        for (int t = 0; t < kThreads; ++t) {
            threads.emplace_back([&dp, &blob, &results, t]() {
                fe::graph::Graph g;
                g.set_device_properties(dp);  // shared descriptor
                results[t] = g.deserialize(blob).is_good() ? 1 : 0;
            });
        }
        for (auto& th : threads) th.join();

        for (int t = 0; t < kThreads; ++t) {
            REQUIRE(results[t] == 1);
        }
    }
#endif  // CUDNN_VERSION >= 90800
}
