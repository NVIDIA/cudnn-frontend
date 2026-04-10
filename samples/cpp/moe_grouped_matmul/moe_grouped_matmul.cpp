/*
 * Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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

#include <random>

#include "../utils/helpers.h"

#include <cudnn_frontend.h>
#include <cublasLt.h>

TEST_CASE("WoQ MoeGroupedMatmul", "[MoeGroupedMatmul][graph]") {
#if (CUDNN_VERSION < 91800)
    SKIP("MoE is not supported in cudnn versions prior to 9.18.0");
#endif

    if (is_arch_supported_by_cudnn() == false) {
        SKIP("Architecture is not supported by current cudnn version");
    }
    namespace fe = cudnn_frontend;

    // problem size
    int64_t const batch_size  = 2;
    int64_t const num_experts = 3;
    int64_t const top_k       = 2;
    int64_t const token_num   = 512;
    int64_t const weight_size = 256;
    int64_t const hidden_size = 512;
    int64_t const block_size  = 128;

    // Initialize input tensors
    Surface<int8_t> token_gpu(
        div_up(batch_size * token_num * top_k * hidden_size *
                   cudnn_frontend::detail::get_element_size_in_bits(cudnn_frontend::DataType_t::HALF),
               8));
    Surface<int8_t> weight_gpu(
        div_up(num_experts * hidden_size * weight_size *
                   cudnn_frontend::detail::get_element_size_in_bits(cudnn_frontend::DataType_t::INT4),
               8));
    Surface<int8_t> block_scale_gpu(
        div_up(num_experts * div_up(hidden_size, block_size) * weight_size *
                   cudnn_frontend::detail::get_element_size_in_bits(cudnn_frontend::DataType_t::HALF),
               8));
    Surface<int8_t> first_token_offset_gpu(div_up(
        batch_size * num_experts * cudnn_frontend::detail::get_element_size_in_bits(cudnn_frontend::DataType_t::INT32),
        8));
    Surface<int8_t> moe_grouped_matmul_gpu(
        div_up(batch_size * token_num * top_k * weight_size *
                   cudnn_frontend::detail::get_element_size_in_bits(cudnn_frontend::DataType_t::HALF),
               8));

    std::vector<int32_t> first_token_offset_cpu({0, 128, 512, 768, 1152, 1536});
    CUDA_CHECK(cudaMemcpy(first_token_offset_gpu.devPtr,
                          first_token_offset_cpu.data(),
                          first_token_offset_cpu.size() * sizeof(int32_t),
                          cudaMemcpyHostToDevice));

    // Make cudnn graph
    fe::graph::Graph graph{};

    graph.set_intermediate_data_type(fe::DataType_t::HALF);
    graph.set_compute_data_type(fe::DataType_t::HALF);

    auto tensor_token = graph.tensor(fe::graph::Tensor_attributes()
                                         .set_name("token")
                                         .set_dim({1, batch_size * token_num * top_k, hidden_size})
                                         .set_stride({batch_size * token_num * top_k * hidden_size, hidden_size, 1})
                                         .set_data_type(fe::DataType_t::HALF));

    auto tensor_weight = graph.tensor(fe::graph::Tensor_attributes()
                                          .set_name("weight")
                                          .set_dim({num_experts, hidden_size, weight_size})
                                          .set_stride({hidden_size * weight_size, 1, hidden_size})
                                          .set_data_type(fe::DataType_t::INT4));

    auto tensor_block_scale = graph.tensor(
        fe::graph::Tensor_attributes()
            .set_name("block_scale")
            .set_dim({num_experts, div_up(hidden_size, block_size), weight_size})
            .set_stride({div_up(hidden_size, block_size) * weight_size, 1, div_up(hidden_size, block_size)})
            .set_data_type(fe::DataType_t::HALF));

    auto tensor_first_token_offset = graph.tensor(fe::graph::Tensor_attributes()
                                                      .set_name("first_token_offset")
                                                      .set_dim({batch_size * num_experts, 1, 1})
                                                      .set_stride({1, 1, 1})
                                                      .set_data_type(fe::DataType_t::INT32));

    auto dequantize_weight_attr = fe::graph::Block_scale_dequantize_attributes()
                                      .set_block_size({block_size, 1})
                                      .set_compute_data_type(fe::DataType_t::HALF);

    auto tensor_dequantized_weight =
        graph.block_scale_dequantize(tensor_weight, tensor_block_scale, dequantize_weight_attr);
    tensor_dequantized_weight->set_data_type(fe::DataType_t::HALF);

    auto moe_grouped_matmul_attr = fe::graph::Moe_grouped_matmul_attributes()
                                       .set_name("moe_grouped_matmul")
                                       .set_mode(fe::MoeGroupedMatmulMode_t::NONE)
                                       .set_compute_data_type(fe::DataType_t::HALF)
                                       .set_top_k(top_k);

    auto tensor_moe_grouped_matmul = graph.moe_grouped_matmul(
        tensor_token, tensor_dequantized_weight, tensor_first_token_offset, nullptr, nullptr, moe_grouped_matmul_attr);

    tensor_moe_grouped_matmul->set_data_type(fe::DataType_t::HALF);
    tensor_moe_grouped_matmul->set_output(true);

    std::cout << graph << std::endl;
    REQUIRE(graph.validate().is_good());

    // Create a unique_ptr for the cuDNN handle
    auto handle_ptr = create_cudnn_handle();
    auto handle     = *handle_ptr;

    REQUIRE(graph.build_operation_graph(handle).is_good());
    REQUIRE(graph.create_execution_plans({fe::HeurMode_t::A}).is_good());
    REQUIRE(graph.check_support().is_good());

    REQUIRE(graph.build_plans(fe::BuildPlanPolicy_t::ALL).is_good());

    // Run cudnn graph
    int64_t workspace_size = 0;
    REQUIRE(graph.get_workspace_size(workspace_size).is_good());
    Surface<int8_t> workspace(workspace_size);

    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
        {tensor_token, token_gpu.devPtr},
        {tensor_weight, weight_gpu.devPtr},
        {tensor_block_scale, block_scale_gpu.devPtr},
        {tensor_first_token_offset, first_token_offset_gpu.devPtr},
        {tensor_moe_grouped_matmul, moe_grouped_matmul_gpu.devPtr}};
    REQUIRE(graph.execute(handle, variant_pack, workspace.devPtr).is_good());
}

TEST_CASE("BF16 MoeGroupedMatmulBwd", "[MoeGroupedMatmulBwd][graph]") {
#if (CUDNN_VERSION < 92200) || (CUDNN_VERSION >= 99900)
    SKIP("MoE is not supported in cudnn versions prior to 9.22.0");
#endif

    if (cublasLtGetVersion() < 130500) {
        SKIP("Not supported by cublasLt version");
    }

    if (is_arch_supported_by_cudnn() == false) {
        SKIP("Architecture is not supported by current cudnn version");
    }
    namespace fe = cudnn_frontend;

    // problem size
    int64_t const num_experts = 36;
    int64_t const token_num   = 2000;
    int64_t const weight_size = 248;
    int64_t const hidden_size = 520;
    std::vector<int32_t> first_token_offset_cpu({0,   1,   2,   3,    4,    5,    6,    7,    8,    9,    10,   11,
                                                 12,  13,  14,  15,   16,   17,   18,   127,  255,  383,  483,  515,
                                                 643, 718, 924, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900});

    // Initialize input tensors
    Surface<int8_t> doutput_gpu(
        div_up(token_num * weight_size *
                   cudnn_frontend::detail::get_element_size_in_bits(cudnn_frontend::DataType_t::BFLOAT16),
               8));
    Surface<int8_t> token_gpu(
        div_up(token_num * hidden_size *
                   cudnn_frontend::detail::get_element_size_in_bits(cudnn_frontend::DataType_t::BFLOAT16),
               8));
    Surface<int8_t> first_token_offset_gpu(
        div_up(num_experts * cudnn_frontend::detail::get_element_size_in_bits(cudnn_frontend::DataType_t::INT32), 8));
    Surface<int8_t> dweight_gpu(
        div_up(num_experts * hidden_size * weight_size *
                   cudnn_frontend::detail::get_element_size_in_bits(cudnn_frontend::DataType_t::BFLOAT16),
               8));

    CUDA_CHECK(cudaMemcpy(first_token_offset_gpu.devPtr,
                          first_token_offset_cpu.data(),
                          first_token_offset_cpu.size() * sizeof(int32_t),
                          cudaMemcpyHostToDevice));

    // Make cudnn graph
    fe::graph::Graph graph{};

    graph.set_intermediate_data_type(fe::DataType_t::FLOAT);
    graph.set_compute_data_type(fe::DataType_t::FLOAT);

    auto tensor_doutput = graph.tensor(fe::graph::Tensor_attributes()
                                           .set_name("doutput")
                                           .set_dim({1, token_num, weight_size})
                                           .set_stride({token_num * weight_size, weight_size, 1})
                                           .set_data_type(fe::DataType_t::BFLOAT16));

    auto tensor_token = graph.tensor(fe::graph::Tensor_attributes()
                                         .set_name("token")
                                         .set_dim({1, token_num, hidden_size})
                                         .set_stride({token_num * hidden_size, hidden_size, 1})
                                         .set_data_type(fe::DataType_t::BFLOAT16));

    auto tensor_first_token_offset = graph.tensor(fe::graph::Tensor_attributes()
                                                      .set_name("first_token_offset")
                                                      .set_dim({num_experts, 1, 1})
                                                      .set_stride({1, 1, 1})
                                                      .set_data_type(fe::DataType_t::INT32));

    auto moe_grouped_matmul_bwd_attr = fe::graph::Moe_grouped_matmul_bwd_attributes()
                                           .set_name("moe_grouped_matmul_bwd")
                                           .set_compute_data_type(fe::DataType_t::FLOAT);

    auto tensor_dweight = graph.moe_grouped_matmul_bwd(
        tensor_doutput, tensor_token, tensor_first_token_offset, moe_grouped_matmul_bwd_attr);

    tensor_dweight->set_data_type(fe::DataType_t::BFLOAT16);
    tensor_dweight->set_output(true);

    std::cout << graph << std::endl;
    REQUIRE(graph.validate().is_good());

    // Create a unique_ptr for the cuDNN handle
    auto handle_ptr = create_cudnn_handle();
    auto handle     = *handle_ptr;

    REQUIRE(graph.build_operation_graph(handle).is_good());
    REQUIRE(graph.create_execution_plans({fe::HeurMode_t::A}).is_good());
    REQUIRE(graph.check_support().is_good());

    REQUIRE(graph.build_plans(fe::BuildPlanPolicy_t::ALL).is_good());

    // Run cudnn graph
    int64_t workspace_size = 0;
    REQUIRE(graph.get_workspace_size(workspace_size).is_good());
    Surface<int8_t> workspace(workspace_size);

    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
        {tensor_doutput, doutput_gpu.devPtr},
        {tensor_token, token_gpu.devPtr},
        {tensor_first_token_offset, first_token_offset_gpu.devPtr},
        {tensor_dweight, dweight_gpu.devPtr}};
    REQUIRE(graph.execute(handle, variant_pack, workspace.devPtr).is_good());
}
