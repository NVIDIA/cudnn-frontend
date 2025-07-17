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

#include <catch2/generators/catch_generators.hpp>

#include "../utils/helpers.h"

#include <cudnn_frontend.h>
#include <cudnn_frontend_utils.h>

namespace GeneralBlockScaleMatmul {

struct TestParams {
    int64_t b = -1;
    int64_t m = -1;
    int64_t n = -1;
    int64_t k = -1;

    int32_t block_size_a_m = -1;
    int32_t block_size_a_k = -1;
    int32_t block_size_b_n = -1;
    int32_t block_size_b_k = -1;

    cudnn_frontend::DataType_t datatype_a = cudnn_frontend::DataType_t::NOT_SET;
    cudnn_frontend::DataType_t datatype_b = cudnn_frontend::DataType_t::NOT_SET;

    cudnn_frontend::DataType_t datatype_block_scale_a = cudnn_frontend::DataType_t::NOT_SET;
    cudnn_frontend::DataType_t datatype_block_scale_b = cudnn_frontend::DataType_t::NOT_SET;

    cudnn_frontend::DataType_t after_dequant_datatype_a = cudnn_frontend::DataType_t::NOT_SET;
    cudnn_frontend::DataType_t after_dequant_datatype_b = cudnn_frontend::DataType_t::NOT_SET;

    cudnn_frontend::DataType_t datatype_d = cudnn_frontend::DataType_t::NOT_SET;

    cudnn_frontend::DataType_t compute_math_precision = cudnn_frontend::DataType_t::NOT_SET;

    TestParams(int64_t b_,
               int64_t m_,
               int64_t n_,
               int64_t k_,
               int32_t block_size_a_m_,
               int32_t block_size_a_k_,
               int32_t block_size_b_k_,
               int32_t block_size_b_n_,
               cudnn_frontend::DataType_t datatype_a_,
               cudnn_frontend::DataType_t datatype_b_,
               cudnn_frontend::DataType_t datatype_block_scale_a_,
               cudnn_frontend::DataType_t datatype_block_scale_b_,
               cudnn_frontend::DataType_t after_dequant_datatype_a_,
               cudnn_frontend::DataType_t after_dequant_datatype_b_,
               cudnn_frontend::DataType_t datatype_d_,
               cudnn_frontend::DataType_t compute_math_precision_) {
        b = b_;
        m = m_;
        n = n_;
        k = k_;

        block_size_a_m = block_size_a_m_;
        block_size_a_k = block_size_a_k_;
        block_size_b_k = block_size_b_k_;
        block_size_b_n = block_size_b_n_;

        datatype_a               = datatype_a_;
        datatype_b               = datatype_b_;
        datatype_block_scale_a   = datatype_block_scale_a_;
        datatype_block_scale_b   = datatype_block_scale_b_;
        after_dequant_datatype_a = after_dequant_datatype_a_;
        after_dequant_datatype_b = after_dequant_datatype_b_;
        datatype_d               = datatype_d_;
        compute_math_precision   = compute_math_precision_;
    }
};

TEST_CASE("General Block Scale Matmul", "[matmul][graph]") {
    namespace fe = cudnn_frontend;

#if (CUDNN_VERSION < 91100)
    SKIP("General matmul with block scaling is not supported in cudnn versions prior to 9.11.0");
#endif

    if (cudnnGetCudartVersion() < 12000) {
        SKIP("Test requires cuda toolkit 12.0 or above");
    }

    auto test_params = GENERATE(
        // General fp16 block scale matmul with 1x128 & 128x128 block size
        TestParams(2,
                   512,
                   512,
                   512,
                   1,
                   128,
                   128,
                   128,
                   cudnn_frontend::DataType_t::HALF,
                   cudnn_frontend::DataType_t::HALF,
                   cudnn_frontend::DataType_t::HALF,
                   cudnn_frontend::DataType_t::HALF,
                   cudnn_frontend::DataType_t::HALF,
                   cudnn_frontend::DataType_t::HALF,
                   cudnn_frontend::DataType_t::HALF,
                   cudnn_frontend::DataType_t::HALF),
        // Int4 WoQ Matmul with 1x128 & 128x1 block size
        TestParams(2,
                   512,
                   512,
                   512,
                   1,
                   128,
                   128,
                   1,
                   cudnn_frontend::DataType_t::INT4,
                   cudnn_frontend::DataType_t::HALF,
                   cudnn_frontend::DataType_t::FLOAT,
                   cudnn_frontend::DataType_t::FLOAT,
                   cudnn_frontend::DataType_t::HALF,
                   cudnn_frontend::DataType_t::HALF,
                   cudnn_frontend::DataType_t::HALF,
                   cudnn_frontend::DataType_t::HALF),
        // fp8 block scale matmul with non-classical block size
        TestParams(2,
                   512,
                   512,
                   512,
                   47,
                   32,
                   32,
                   17,
                   cudnn_frontend::DataType_t::BFLOAT16,
                   cudnn_frontend::DataType_t::BFLOAT16,
                   cudnn_frontend::DataType_t::FLOAT,
                   cudnn_frontend::DataType_t::FLOAT,
                   cudnn_frontend::DataType_t::BFLOAT16,
                   cudnn_frontend::DataType_t::BFLOAT16,
                   cudnn_frontend::DataType_t::FLOAT,
                   cudnn_frontend::DataType_t::FLOAT));

    auto b = test_params.b;
    auto m = test_params.m;
    auto n = test_params.n;
    auto k = test_params.k;

    auto block_size_a_m = test_params.block_size_a_m;
    auto block_size_a_k = test_params.block_size_a_k;
    auto block_size_b_k = test_params.block_size_b_k;
    auto block_size_b_n = test_params.block_size_b_n;

    auto datatype_a = test_params.datatype_a;
    auto datatype_b = test_params.datatype_b;

    auto datatype_block_scale_a = test_params.datatype_block_scale_a;
    auto datatype_block_scale_b = test_params.datatype_block_scale_b;

    auto after_dequant_datatype_a = test_params.after_dequant_datatype_a;
    auto after_dequant_datatype_b = test_params.after_dequant_datatype_b;

    auto datatype_d = test_params.datatype_d;

    auto compute_math_precision = test_params.compute_math_precision;

    Surface<int8_t> tensor_a_gpu(div_up(b * m * k * cudnn_frontend::detail::get_element_size_in_bits(datatype_a), 8),
                                 false);
    Surface<int8_t> tensor_b_gpu(div_up(b * k * n * cudnn_frontend::detail::get_element_size_in_bits(datatype_b), 8),
                                 false);

    int64_t block_scale_dim_a_m = div_up(m, block_size_a_m);
    int64_t block_scale_dim_a_k = div_up(k, block_size_a_k);
    int64_t block_scale_dim_b_k = div_up(k, block_size_b_k);
    int64_t block_scale_dim_b_n = div_up(n, block_size_b_n);

    Surface<int8_t> block_descale_a_gpu(
        div_up(b * block_scale_dim_a_m * block_scale_dim_a_k *
                   cudnn_frontend::detail::get_element_size_in_bits(datatype_block_scale_a),
               8),
        false);
    Surface<int8_t> block_descale_b_gpu(
        div_up(b * block_scale_dim_b_k * block_scale_dim_b_n *
                   cudnn_frontend::detail::get_element_size_in_bits(datatype_block_scale_b),
               8),
        false);

    Surface<int8_t> tensor_d_gpu(div_up(b * m * n * cudnn_frontend::detail::get_element_size_in_bits(datatype_d), 8),
                                 false);

    fe::graph::Graph graph{};

    graph.set_intermediate_data_type(fe::DataType_t::FLOAT);
    graph.set_compute_data_type(fe::DataType_t::FLOAT);

    auto tensor_a = graph.tensor(fe::graph::Tensor_attributes()
                                     .set_name("tensor_a")
                                     .set_data_type(datatype_a)
                                     .set_dim({b, m, k})
                                     .set_stride({m * k, k, 1}));

    auto tensor_b = graph.tensor(fe::graph::Tensor_attributes()
                                     .set_name("tensor_b")
                                     .set_data_type(datatype_b)
                                     .set_dim({b, k, n})
                                     .set_stride({k * n, 1, k}));

    auto block_descale_a =
        graph.tensor(fe::graph::Tensor_attributes()
                         .set_name("block_descale_a")
                         .set_data_type(datatype_block_scale_a)
                         .set_dim({b, block_scale_dim_a_m, block_scale_dim_a_k})
                         .set_stride({block_scale_dim_a_m * block_scale_dim_a_k, block_scale_dim_a_k, 1}));

    auto block_descale_b =
        graph.tensor(fe::graph::Tensor_attributes()
                         .set_name("block_descale_b")
                         .set_data_type(datatype_block_scale_b)
                         .set_dim({b, block_scale_dim_b_k, block_scale_dim_b_n})
                         .set_stride({block_scale_dim_b_k * block_scale_dim_b_n, 1, block_scale_dim_b_k}));

    auto dequantize_attr_a = fe::graph::Block_scale_dequantize_attributes()
                                 .set_block_size({block_size_a_m, block_size_a_k})
                                 .set_compute_data_type(compute_math_precision);

    auto dequant_tensor_a = graph.block_scale_dequantize(tensor_a, block_descale_a, dequantize_attr_a);

    auto dequantize_attr_b = fe::graph::Block_scale_dequantize_attributes()
                                 .set_block_size({block_size_b_k, block_size_b_n})
                                 .set_compute_data_type(compute_math_precision);

    auto dequant_tensor_b = graph.block_scale_dequantize(tensor_b, block_descale_b, dequantize_attr_b);

    // This explicit data type setting is necessary,
    // otherwise float tensor core instructions will be utilized by default, causing unoptimized performance
    // just explicitly set to cudnn_frontend::DataType_t::HALF if no idea what data type should use
    dequant_tensor_a->set_data_type(after_dequant_datatype_a);
    dequant_tensor_b->set_data_type(after_dequant_datatype_b);

    auto matmul_attributes =
        fe::graph::Matmul_attributes().set_name("GEMM").set_compute_data_type(compute_math_precision);

    auto tensor_d = graph.matmul(dequant_tensor_a, dequant_tensor_b, matmul_attributes);

    tensor_d->set_data_type(datatype_d);
    tensor_d->set_is_virtual(false);

    std::cout << graph << std::endl;

    REQUIRE(graph.validate().is_good());

    // Create a unique_ptr for the cuDNN handle
    auto handle_ptr = create_cudnn_handle();
    auto handle     = *handle_ptr;

    REQUIRE(graph.build_operation_graph(handle).is_good());
    REQUIRE(graph.create_execution_plans({fe::HeurMode_t::A}).is_good());

    REQUIRE(graph.check_support().is_good());

    REQUIRE(graph.build_plans(fe::BuildPlanPolicy_t::HEURISTICS_CHOICE).is_good());

    int64_t workspace_size = 0;
    REQUIRE(graph.get_workspace_size(workspace_size).is_good());
    Surface<int8_t> workspace(workspace_size, false);

    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
        {tensor_a, tensor_a_gpu.devPtr},
        {tensor_b, tensor_b_gpu.devPtr},
        {block_descale_a, block_descale_a_gpu.devPtr},
        {block_descale_b, block_descale_b_gpu.devPtr},
        {tensor_d, tensor_d_gpu.devPtr}};

    REQUIRE(graph.execute(handle, variant_pack, workspace.devPtr).is_good());
}

};  // namespace GeneralBlockScaleMatmul