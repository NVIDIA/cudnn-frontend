/*
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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
#include "../utils/helpers.h"

#include <cudnn_frontend.h>

TEST_CASE("CSBR Graph with serialization", "[conv][graph][serialization]") {
    enum UIDs {
        x_tensor,
        w_tensor,
        y_tensor,
        scale_tensor,
        bias_tensor,
    };

#if (CUDNN_VERSION < 8905)
    SKIP("Serialization tests is not supported in cudnn versions prior to 8.9.5");
#endif

    int64_t n = 8, c = 32, h = 16, w = 16, k = 64, r = 3, s = 3;

    cudnnHandle_t handle;  // Handle to use during deserialize and execute

    checkCudnnErr(cudnnCreate(&handle));

    auto build_and_validate_graph_helper =
        [](int64_t n, int64_t c, int64_t h, int64_t w, int64_t k, int64_t r, int64_t s)
        -> std::shared_ptr<cudnn_frontend::graph::Graph> {
        auto graph = std::make_shared<cudnn_frontend::graph::Graph>();
        graph->set_io_data_type(cudnn_frontend::DataType_t::HALF)
            .set_intermediate_data_type(cudnn_frontend::DataType_t::FLOAT)
            .set_compute_data_type(cudnn_frontend::DataType_t::FLOAT);

        auto X = graph->tensor(cudnn_frontend::graph::Tensor_attributes()
                                   .set_name("image")
                                   .set_uid(x_tensor)
                                   .set_dim({n, c, h, w})
                                   .set_stride({c * h * w, 1, c * w, c}));

        auto W = graph->tensor(cudnn_frontend::graph::Tensor_attributes()
                                   .set_uid(w_tensor)
                                   .set_name("filter")
                                   .set_dim({k, c, r, s})
                                   .set_stride({c * r * s, 1, c * s, c}));

        auto conv_options =
            cudnn_frontend::graph::Conv_fprop_attributes().set_padding({1, 1}).set_stride({1, 1}).set_dilation({1, 1});
        auto conv_output = graph->conv_fprop(X, W, conv_options);

        auto S = graph->tensor(cudnn_frontend::graph::Tensor_attributes()
                                   .set_uid(scale_tensor)
                                   .set_name("scale")
                                   .set_dim({1, k, 1, 1})
                                   .set_stride({k, 1, k, k}));
        auto scale_options =
            cudnn_frontend::graph::Pointwise_attributes().set_mode(cudnn_frontend::PointwiseMode_t::MUL);
        auto scale_output = graph->pointwise(conv_output, S, scale_options);

        auto B = graph->tensor(cudnn_frontend::graph::Tensor_attributes()
                                   .set_name("bias")
                                   .set_uid(bias_tensor)
                                   .set_dim({1, k, 1, 1})
                                   .set_stride({k, 1, k, k}));
        auto bias_options =
            cudnn_frontend::graph::Pointwise_attributes().set_mode(cudnn_frontend::PointwiseMode_t::ADD);
        auto bias_output = graph->pointwise(scale_output, B, bias_options);

        auto relu_options =
            cudnn_frontend::graph::Pointwise_attributes().set_mode(cudnn_frontend::PointwiseMode_t::RELU_FWD);
        auto Y = graph->pointwise(bias_output, relu_options);
        Y->set_output(true).set_uid(y_tensor);

        REQUIRE(graph->validate().is_good());

        return graph;
    };

    // Check support

    auto check_support = [build_and_validate_graph_helper](
                             int64_t n, int64_t c, int64_t h, int64_t w, int64_t k, int64_t r, int64_t s) -> bool {
        cudnnHandle_t handle;

        checkCudnnErr(cudnnCreate(&handle));

        auto graph = build_and_validate_graph_helper(n, c, h, w, k, r, s);

        REQUIRE(graph->build_operation_graph(handle).is_good());

        REQUIRE(graph->create_execution_plans({cudnn_frontend::HeurMode_t::A}).is_good());

        REQUIRE(graph->check_support(handle).is_good());

        cudnnDestroy(handle);

        return true;
    };

    // Serialization Phase

    auto serialize =
        [build_and_validate_graph_helper](
            int64_t n, int64_t c, int64_t h, int64_t w, int64_t k, int64_t r, int64_t s) -> std::vector<uint8_t> {
        cudnnHandle_t handle;

        std::vector<uint8_t> serialized_data;

        checkCudnnErr(cudnnCreate(&handle));

        auto graph = build_and_validate_graph_helper(n, c, h, w, k, r, s);

        REQUIRE(graph->build_operation_graph(handle).is_good());

        REQUIRE(graph->create_execution_plans({cudnn_frontend::HeurMode_t::A}).is_good());

        REQUIRE(graph->check_support(handle).is_good());

        REQUIRE(graph->build_plans(handle).is_good());

        // Insert auto-tuning logic here

        REQUIRE(graph->serialize(serialized_data).is_good());

        cudnnDestroy(handle);

        return serialized_data;
    };

    auto deserialize = [](cudnnHandle_t handle,
                          std::vector<uint8_t> const& data) -> std::shared_ptr<cudnn_frontend::graph::Graph> {
        auto graph = std::make_shared<cudnn_frontend::graph::Graph>();

        REQUIRE(graph->deserialize(handle, data).is_good());

        return graph;
    };

    // Check if the graph is supported
    REQUIRE(check_support(n, c, h, w, k, r, s));

    // Serialize the graph.
    auto serialize_data = serialize(n, c, h, w, k, r, s);

    // Deserialize the graph and execute
    auto graph = deserialize(handle, serialize_data);

    Surface<half> x_device_memory(n * c * h * w, false);
    Surface<half> w_device_memory(k * c * r * s, false);
    Surface<half> s_device_memory(k, false);
    Surface<half> b_device_memory(k, false);
    Surface<half> y_device_memory(n * k * h * w, false);  // Should be p, q.

    Surface<int8_t> workspace(graph->get_workspace_size(), false);

    std::unordered_map<int64_t, void*> variant_pack = {{x_tensor, x_device_memory.devPtr},
                                                       {w_tensor, w_device_memory.devPtr},
                                                       {scale_tensor, s_device_memory.devPtr},
                                                       {bias_tensor, b_device_memory.devPtr},
                                                       {y_tensor, y_device_memory.devPtr}};

    REQUIRE(graph->execute(handle, variant_pack, workspace.devPtr).is_good());

    cudnnDestroy(handle);
}

TEST_CASE("SDPA Graph with serialization", "[sdpa][graph][serialization]") {
    int64_t b    = 12;    // batch size
    int64_t h    = 6;     // head dim
    int64_t s_q  = 1024;  // q tensor is padded to this seq length
    int64_t s_kv = 1024;  // k and v tensor is padded to this seq length
    int64_t d    = 128;   // hidden dim

#if (CUDNN_VERSION < 8905)
    SKIP("Serialization tests is not supported in cudnn versions prior to 8.9.5");
#endif

    // Mode of sdpa operation
    bool is_inference = true;

    // attention scale
    bool is_attn_scale   = true;
    float attn_scale_cpu = 0.5f;

    // Dropout configutation
    bool use_dropout_with_rng = true;
    float dropout_probability = 0.1f;

    enum UIDs { uid_Q, uid_K, uid_V, uid_ATTN_SCALE, uid_SEED, uid_OFFSET, uid_O, uid_STATS };

    auto build_and_validate_graph_helper =
        [](int64_t b,
           int64_t h,
           int64_t s_q,
           int64_t s_kv,
           int64_t d,
           bool is_attn_scale,
           bool is_inference,
           bool use_dropout_with_rng,
           float dropout_probability) -> std::shared_ptr<cudnn_frontend::graph::Graph> {
        namespace fe = cudnn_frontend;

        auto graph = std::make_shared<fe::graph::Graph>();

        graph->set_io_data_type(fe::DataType_t::HALF)
            .set_intermediate_data_type(fe::DataType_t::FLOAT)
            .set_compute_data_type(fe::DataType_t::FLOAT);

        auto Q = graph->tensor(fe::graph::Tensor_attributes()
                                   .set_name("Q")
                                   .set_dim({b, h, s_q, d})
                                   .set_uid(uid_Q)
                                   .set_stride({3 * h * d, 3 * d, 3 * b * h * d, 1}));
        auto K = graph->tensor(fe::graph::Tensor_attributes()
                                   .set_name("K")
                                   .set_uid(uid_K)
                                   .set_dim({b, h, s_kv, d})
                                   .set_stride({3 * h * d, 3 * d, 3 * b * h * d, 1}));
        auto V = graph->tensor(fe::graph::Tensor_attributes()
                                   .set_name("V")
                                   .set_uid(uid_V)
                                   .set_dim({b, h, s_kv, d})
                                   .set_stride({3 * h * d, 3 * d, 3 * b * h * d, 1}));

        auto attn_scale = is_attn_scale ? graph->tensor(fe::graph::Tensor_attributes()
                                                            .set_name("attn_scale")
                                                            .set_dim({1, 1, 1, 1})
                                                            .set_uid(uid_ATTN_SCALE)
                                                            .set_stride({1, 1, 1, 1})
                                                            .set_is_pass_by_value(true)
                                                            .set_data_type(fe::DataType_t::FLOAT))
                                        : nullptr;

        auto sdpa_options = fe::graph::SDPA_attributes().set_name("flash_attention").set_is_inference(is_inference);

        sdpa_options.set_causal_mask(true);
        sdpa_options.set_alibi_mask(true);

        if (is_attn_scale) {
            sdpa_options.set_attn_scale(attn_scale);
        };

        auto seed = use_dropout_with_rng ? graph->tensor(fe::graph::Tensor_attributes()
                                                             .set_name("Seed")
                                                             .set_uid(uid_SEED)
                                                             .set_dim({1, 1, 1, 1})
                                                             .set_stride({1, 1, 1, 1})
                                                             .set_data_type(fe::DataType_t::INT32))
                                         : nullptr;

        auto offset = use_dropout_with_rng ? graph->tensor(fe::graph::Tensor_attributes()
                                                               .set_uid(uid_OFFSET)
                                                               .set_name("Offset")
                                                               .set_dim({1, 1, 1, 1})
                                                               .set_stride({1, 1, 1, 1})
                                                               .set_data_type(fe::DataType_t::INT32))
                                           : nullptr;

        if (use_dropout_with_rng) {
            sdpa_options.set_dropout(dropout_probability, seed, offset);
        }

        auto [O, stats] = graph->sdpa(Q, K, V, sdpa_options);

        O->set_output(true).set_dim({b, h, s_q, d}).set_uid(uid_O).set_stride({h * d, d, b * h * d, 1});

        // Check that Stats tensor is real, which is only when its training step
        if (is_inference) {
            REQUIRE(stats == nullptr);
        } else {
            stats->set_output(true).set_uid(uid_STATS).set_data_type(fe::DataType_t::FLOAT);
        }

        REQUIRE(graph->validate().is_good());

        return graph;
    };

    auto check_support = [build_and_validate_graph_helper](int64_t b,
                                                           int64_t h,
                                                           int64_t s_q,
                                                           int64_t s_kv,
                                                           int64_t d,
                                                           bool is_attn_scale,
                                                           bool is_inference,
                                                           bool use_dropout_with_rng,
                                                           float dropout_probability) -> bool {
        cudnnHandle_t handle;

        checkCudnnErr(cudnnCreate(&handle));

        auto graph = build_and_validate_graph_helper(
            b, h, s_q, s_kv, d, is_attn_scale, is_inference, use_dropout_with_rng, dropout_probability);

        REQUIRE(graph->build_operation_graph(handle).is_good());

        REQUIRE(graph->create_execution_plans({cudnn_frontend::HeurMode_t::A}).is_good());

        REQUIRE(graph->check_support(handle).is_good());

        cudnnDestroy(handle);

        return true;
    };

    auto serialize = [build_and_validate_graph_helper](int64_t b,
                                                       int64_t h,
                                                       int64_t s_q,
                                                       int64_t s_kv,
                                                       int64_t d,
                                                       bool is_attn_scale,
                                                       bool is_inference,
                                                       bool use_dropout_with_rng,
                                                       float dropout_probability) -> std::vector<uint8_t> {
        cudnnHandle_t handle;

        std::vector<uint8_t> serialized_data;

        checkCudnnErr(cudnnCreate(&handle));

        auto graph = build_and_validate_graph_helper(
            b, h, s_q, s_kv, d, is_attn_scale, is_inference, use_dropout_with_rng, dropout_probability);

        REQUIRE(graph->build_operation_graph(handle).is_good());

        REQUIRE(graph->create_execution_plans({cudnn_frontend::HeurMode_t::A}).is_good());

        REQUIRE(graph->check_support(handle).is_good());

        REQUIRE(graph->build_plans(handle).is_good());

        // Insert auto-tuning logic here

        REQUIRE(graph->serialize(serialized_data).is_good());

        cudnnDestroy(handle);

        return serialized_data;
    };

    auto deserialize = [](cudnnHandle_t handle,
                          std::vector<uint8_t> const& data) -> std::shared_ptr<cudnn_frontend::graph::Graph> {
        auto graph = std::make_shared<cudnn_frontend::graph::Graph>();

        REQUIRE(graph->deserialize(handle, data).is_good());

        return graph;
    };

    // Check support
    REQUIRE(check_support(b, h, s_q, s_kv, d, is_attn_scale, is_inference, use_dropout_with_rng, dropout_probability));

    // Serialize the graph.
    auto serialize_data =
        serialize(b, h, s_q, s_kv, d, is_attn_scale, is_inference, use_dropout_with_rng, dropout_probability);

    cudnnHandle_t handle;
    checkCudnnErr(cudnnCreate(&handle));

    auto graph = deserialize(handle, serialize_data);

    //// Build variant pack
    Surface<half> qkvTensor(b * s_q * 3 * h * d, false);
    Surface<half> oTensor(b * s_q * h * d, false);
    void* devPtrQ = qkvTensor.devPtr;
    void* devPtrK = (qkvTensor.devPtr + d);
    void* devPtrV = (qkvTensor.devPtr + 2 * d);
    void* devPtrO = oTensor.devPtr;

    int32_t scaleSize  = 1;
    int32_t seed_value = 123456;
    Surface<int32_t> dropoutSeed(scaleSize, false, seed_value);
    Surface<int32_t> dropoutOffset(scaleSize, false, (int32_t)1);

    Surface<int8_t> workspace(graph->get_workspace_size(), false);

    std::cout << "Graph requires workspace " << graph->get_workspace_size() << std::endl;

    std::unordered_map<int64_t, void*> variant_pack = {{uid_Q, devPtrQ},
                                                       {uid_K, devPtrK},
                                                       {uid_V, devPtrV},
                                                       {uid_ATTN_SCALE, &attn_scale_cpu},
                                                       {uid_SEED, dropoutSeed.devPtr},
                                                       {uid_OFFSET, dropoutOffset.devPtr},
                                                       {uid_O, devPtrO}};

    REQUIRE(graph->execute(handle, variant_pack, workspace.devPtr).is_good());

    checkCudnnErr(cudnnDestroy(handle));
}