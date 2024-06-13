/*
 * Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <catch2/catch_test_macros.hpp>

#include <cudnn_frontend.h>

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

    REQUIRE(graph.check_support(handle).is_good());
    REQUIRE(key == graph.key());

    REQUIRE(graph.build_plans(handle).is_good());
    REQUIRE(key == graph.key());
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
        .set_dim({1, 32, 1, 1})
        .set_stride({32, 1, 32, 32})
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
                            .set_is_inference(false)
                            .set_attn_scale(attn_scale)
                            .set_alibi_mask(true)
                            .set_causal_mask(false)
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

    REQUIRE(graph_deserialized.validate().is_good());
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
                                     .set_causal_mask(true)
                                     .set_attn_scale(attn_scale)
                                     .set_bias(bias)
                                     .set_dropout(0.1f, dropout_seed, dropout_offset)
                                     .set_deterministic_algorithm(true);

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