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

#include <random>

#include "../utils/helpers.h"

#include <cudnn_frontend.h>

int64_t
round_up_to_multiple(int64_t x, int64_t multiple) {
    return ((x + multiple - 1) / multiple) * multiple;
}

size_t
bytes_for_n_elems(size_t n_elems, cudnn_frontend::DataType_t datatype) {
    switch (datatype) {
        case cudnn_frontend::DataType_t::FLOAT:
            return n_elems * 4;
            break;
        case cudnn_frontend::DataType_t::HALF:
        case cudnn_frontend::DataType_t::BFLOAT16:
            return n_elems * 2;
            break;
        case cudnn_frontend::DataType_t::FP8_E4M3:
        case cudnn_frontend::DataType_t::FP8_E5M2:
        case cudnn_frontend::DataType_t::FP8_E8M0:
            return n_elems;
        case cudnn_frontend::DataType_t::FP4_E2M1:
            return round_up_to_multiple(n_elems, 2) / 2;
        default:
            return 0;
            break;
    }
}

struct TestParams {
    int64_t b = -1;
    int64_t m = -1;
    int64_t n = -1;
    int64_t k = -1;

    int32_t block_size = -1;

    cudnn_frontend::DataType_t datatype_a = cudnn_frontend::DataType_t::NOT_SET;
    cudnn_frontend::DataType_t datatype_b = cudnn_frontend::DataType_t::NOT_SET;

    cudnn_frontend::DataType_t datatype_block_scale = cudnn_frontend::DataType_t::NOT_SET;

    cudnn_frontend::DataType_t datatype_d = cudnn_frontend::DataType_t::NOT_SET;

    TestParams(int64_t b_,
               int64_t m_,
               int64_t n_,
               int64_t k_,
               int32_t block_size_,
               cudnn_frontend::DataType_t datatype_a_,
               cudnn_frontend::DataType_t datatype_b_,
               cudnn_frontend::DataType_t datatype_block_scale_,
               cudnn_frontend::DataType_t datatype_d_) {
        b = b_;
        m = m_;
        n = n_;
        k = k_;

        block_size = block_size_;

        datatype_a           = datatype_a_;
        datatype_b           = datatype_b_;
        datatype_block_scale = datatype_block_scale_;
        datatype_d           = datatype_d_;
    }
};

TEST_CASE("Matmul block scale", "[matmul][graph][FP4]") {
    namespace fe = cudnn_frontend;

#if (CUDNN_VERSION < 90700)
    SKIP("Matmul with block scaling is not supported in cudnn versions prior to 9.7.0");
#endif
    if (check_device_arch_newer_than("blackwell") == false) {
        SKIP("block scale requires Blackwell and up");
    }

    // auto test_params = GENERATE(TestParams(1,
    //                                        128,
    //                                        128,
    //                                        64,
    //                                        16,
    //                                        cudnn_frontend::DataType_t::FP4_E2M1,
    //                                        cudnn_frontend::DataType_t::FP4_E2M1,
    //                                        cudnn_frontend::DataType_t::FP8_E4M3,
    //                                        cudnn_frontend::DataType_t::FLOAT));

    auto test_params = GENERATE(TestParams(1,
                                           128,
                                           128,
                                           64,
                                           16,
                                           cudnn_frontend::DataType_t::FP4_E2M1,
                                           cudnn_frontend::DataType_t::FP4_E2M1,
                                           cudnn_frontend::DataType_t::FP8_E4M3,
                                           cudnn_frontend::DataType_t::FLOAT),
                                TestParams(1,
                                           128,
                                           128,
                                           64,
                                           16,
                                           cudnn_frontend::DataType_t::FP4_E2M1,
                                           cudnn_frontend::DataType_t::FP4_E2M1,
                                           cudnn_frontend::DataType_t::FP8_E4M3,
                                           cudnn_frontend::DataType_t::HALF),
                                TestParams(1,
                                           128,
                                           128,
                                           64,
                                           16,
                                           cudnn_frontend::DataType_t::FP4_E2M1,
                                           cudnn_frontend::DataType_t::FP4_E2M1,
                                           cudnn_frontend::DataType_t::FP8_E4M3,
                                           cudnn_frontend::DataType_t::BFLOAT16),
                                TestParams(3,
                                           137,
                                           268,
                                           160,
                                           16,
                                           cudnn_frontend::DataType_t::FP4_E2M1,
                                           cudnn_frontend::DataType_t::FP4_E2M1,
                                           cudnn_frontend::DataType_t::FP8_E4M3,
                                           cudnn_frontend::DataType_t::FLOAT),
                                TestParams(3,
                                           137,
                                           272,
                                           160,
                                           16,
                                           cudnn_frontend::DataType_t::FP4_E2M1,
                                           cudnn_frontend::DataType_t::FP4_E2M1,
                                           cudnn_frontend::DataType_t::FP8_E4M3,
                                           cudnn_frontend::DataType_t::HALF),
                                TestParams(3,
                                           137,
                                           272,
                                           160,
                                           16,
                                           cudnn_frontend::DataType_t::FP4_E2M1,
                                           cudnn_frontend::DataType_t::FP4_E2M1,
                                           cudnn_frontend::DataType_t::FP8_E4M3,
                                           cudnn_frontend::DataType_t::BFLOAT16),
                                TestParams(1,
                                           128,
                                           128,
                                           128,
                                           32,
                                           cudnn_frontend::DataType_t::FP8_E4M3,
                                           cudnn_frontend::DataType_t::FP8_E4M3,
                                           cudnn_frontend::DataType_t::FP8_E8M0,
                                           cudnn_frontend::DataType_t::FLOAT),
                                TestParams(1,
                                           128,
                                           128,
                                           128,
                                           32,
                                           cudnn_frontend::DataType_t::FP8_E4M3,
                                           cudnn_frontend::DataType_t::FP8_E4M3,
                                           cudnn_frontend::DataType_t::FP8_E8M0,
                                           cudnn_frontend::DataType_t::HALF),
                                TestParams(1,
                                           128,
                                           128,
                                           128,
                                           32,
                                           cudnn_frontend::DataType_t::FP8_E4M3,
                                           cudnn_frontend::DataType_t::FP8_E4M3,
                                           cudnn_frontend::DataType_t::FP8_E8M0,
                                           cudnn_frontend::DataType_t::BFLOAT16),
                                TestParams(1,
                                           128,
                                           128,
                                           128,
                                           32,
                                           cudnn_frontend::DataType_t::FP8_E4M3,
                                           cudnn_frontend::DataType_t::FP8_E5M2,
                                           cudnn_frontend::DataType_t::FP8_E8M0,
                                           cudnn_frontend::DataType_t::FLOAT),
                                TestParams(1,
                                           128,
                                           128,
                                           128,
                                           32,
                                           cudnn_frontend::DataType_t::FP8_E4M3,
                                           cudnn_frontend::DataType_t::FP8_E5M2,
                                           cudnn_frontend::DataType_t::FP8_E8M0,
                                           cudnn_frontend::DataType_t::HALF),
                                TestParams(1,
                                           128,
                                           128,
                                           128,
                                           32,
                                           cudnn_frontend::DataType_t::FP8_E4M3,
                                           cudnn_frontend::DataType_t::FP8_E5M2,
                                           cudnn_frontend::DataType_t::FP8_E8M0,
                                           cudnn_frontend::DataType_t::BFLOAT16),
                                TestParams(1,
                                           128,
                                           128,
                                           128,
                                           32,
                                           cudnn_frontend::DataType_t::FP8_E5M2,
                                           cudnn_frontend::DataType_t::FP8_E4M3,
                                           cudnn_frontend::DataType_t::FP8_E8M0,
                                           cudnn_frontend::DataType_t::FLOAT),
                                TestParams(1,
                                           128,
                                           128,
                                           128,
                                           32,
                                           cudnn_frontend::DataType_t::FP8_E5M2,
                                           cudnn_frontend::DataType_t::FP8_E4M3,
                                           cudnn_frontend::DataType_t::FP8_E8M0,
                                           cudnn_frontend::DataType_t::HALF),
                                TestParams(1,
                                           128,
                                           128,
                                           128,
                                           32,
                                           cudnn_frontend::DataType_t::FP8_E5M2,
                                           cudnn_frontend::DataType_t::FP8_E4M3,
                                           cudnn_frontend::DataType_t::FP8_E8M0,
                                           cudnn_frontend::DataType_t::BFLOAT16),
                                TestParams(3,
                                           137,
                                           268,
                                           160,
                                           32,
                                           cudnn_frontend::DataType_t::FP8_E4M3,
                                           cudnn_frontend::DataType_t::FP8_E4M3,
                                           cudnn_frontend::DataType_t::FP8_E8M0,
                                           cudnn_frontend::DataType_t::FLOAT),
                                TestParams(3,
                                           137,
                                           272,
                                           160,
                                           32,
                                           cudnn_frontend::DataType_t::FP8_E4M3,
                                           cudnn_frontend::DataType_t::FP8_E4M3,
                                           cudnn_frontend::DataType_t::FP8_E8M0,
                                           cudnn_frontend::DataType_t::HALF),
                                TestParams(3,
                                           137,
                                           272,
                                           160,
                                           32,
                                           cudnn_frontend::DataType_t::FP8_E4M3,
                                           cudnn_frontend::DataType_t::FP8_E4M3,
                                           cudnn_frontend::DataType_t::FP8_E8M0,
                                           cudnn_frontend::DataType_t::BFLOAT16),
                                TestParams(3,
                                           137,
                                           268,
                                           160,
                                           32,
                                           cudnn_frontend::DataType_t::FP8_E4M3,
                                           cudnn_frontend::DataType_t::FP8_E5M2,
                                           cudnn_frontend::DataType_t::FP8_E8M0,
                                           cudnn_frontend::DataType_t::FLOAT),
                                TestParams(3,
                                           137,
                                           272,
                                           160,
                                           32,
                                           cudnn_frontend::DataType_t::FP8_E4M3,
                                           cudnn_frontend::DataType_t::FP8_E5M2,
                                           cudnn_frontend::DataType_t::FP8_E8M0,
                                           cudnn_frontend::DataType_t::HALF),
                                TestParams(3,
                                           137,
                                           272,
                                           160,
                                           32,
                                           cudnn_frontend::DataType_t::FP8_E4M3,
                                           cudnn_frontend::DataType_t::FP8_E5M2,
                                           cudnn_frontend::DataType_t::FP8_E8M0,
                                           cudnn_frontend::DataType_t::BFLOAT16),
                                TestParams(3,
                                           137,
                                           268,
                                           160,
                                           32,
                                           cudnn_frontend::DataType_t::FP8_E5M2,
                                           cudnn_frontend::DataType_t::FP8_E4M3,
                                           cudnn_frontend::DataType_t::FP8_E8M0,
                                           cudnn_frontend::DataType_t::FLOAT),
                                TestParams(3,
                                           137,
                                           272,
                                           160,
                                           32,
                                           cudnn_frontend::DataType_t::FP8_E5M2,
                                           cudnn_frontend::DataType_t::FP8_E4M3,
                                           cudnn_frontend::DataType_t::FP8_E8M0,
                                           cudnn_frontend::DataType_t::HALF),
                                TestParams(3,
                                           137,
                                           272,
                                           160,
                                           32,
                                           cudnn_frontend::DataType_t::FP8_E5M2,
                                           cudnn_frontend::DataType_t::FP8_E4M3,
                                           cudnn_frontend::DataType_t::FP8_E8M0,
                                           cudnn_frontend::DataType_t::BFLOAT16));

    int64_t const b = test_params.b;
    int64_t const m = test_params.m;
    int64_t const n = test_params.n;
    int64_t const k = test_params.k;

    auto datatype_a = test_params.datatype_a;
    auto datatype_b = test_params.datatype_b;

    auto datatype_block_scale = test_params.datatype_block_scale;

    auto block_size = test_params.block_size;

    auto datatype_d = test_params.datatype_d;

    Surface<int8_t> tensor_a_gpu(bytes_for_n_elems(b * m * k, datatype_a), false);
    Surface<int8_t> tensor_b_gpu(bytes_for_n_elems(b * k * n, datatype_b), false);

    static constexpr int indestructible_128x4_block_128 = 128;
    static constexpr int indestructible_128x4_block_4   = 4;

    int64_t rounded_m             = round_up_to_multiple(m, indestructible_128x4_block_128);
    int64_t rounded_n             = round_up_to_multiple(n, indestructible_128x4_block_128);
    int64_t rounded_block_scale_k = round_up_to_multiple(k / block_size, indestructible_128x4_block_4);

    Surface<int8_t> block_descale_a_gpu(bytes_for_n_elems(b * rounded_m * rounded_block_scale_k, datatype_block_scale),
                                        false);
    Surface<int8_t> block_descale_b_gpu(bytes_for_n_elems(b * rounded_block_scale_k * rounded_n, datatype_block_scale),
                                        false);

    Surface<int8_t> tensor_d_gpu(bytes_for_n_elems(b * m * n, datatype_d), false);

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

    auto block_descale_a = graph.tensor(fe::graph::Tensor_attributes()
                                            .set_name("block_descale_a")
                                            .set_data_type(datatype_block_scale)
                                            .set_dim({b, rounded_m, rounded_block_scale_k})
                                            .set_stride({rounded_m * rounded_block_scale_k, rounded_block_scale_k, 1})
                                            .set_reordering_type(cudnn_frontend::TensorReordering_t::F8_128x4));

    auto block_descale_b = graph.tensor(fe::graph::Tensor_attributes()
                                            .set_name("block_descale_b")
                                            .set_data_type(datatype_block_scale)
                                            .set_dim({b, rounded_block_scale_k, rounded_n})
                                            .set_stride({rounded_n * rounded_block_scale_k, 1, rounded_block_scale_k})
                                            .set_reordering_type(cudnn_frontend::TensorReordering_t::F8_128x4));

    auto nvfp4_dequantize_attr = fe::graph::Block_scale_dequantize_attributes().set_block_size(block_size);

    auto dequan_tensor_a = graph.block_scale_dequantize(tensor_a, block_descale_a, nvfp4_dequantize_attr);

    auto dequan_tensor_b = graph.block_scale_dequantize(tensor_b, block_descale_b, nvfp4_dequantize_attr);

    auto matmul_attributes =
        fe::graph::Matmul_attributes().set_name("GEMM").set_compute_data_type(fe::DataType_t::FLOAT);

    auto tensor_d = graph.matmul(dequan_tensor_a, dequan_tensor_b, matmul_attributes);

    tensor_d->set_data_type(datatype_d);
    tensor_d->set_is_virtual(false);

    std::cout << graph << std::endl;

    REQUIRE(graph.validate().is_good());

    // Create a unique_ptr for the cuDNN handle
    auto handle_ptr = create_cudnn_handle();
    auto handle     = *handle_ptr;

    REQUIRE(graph.build_operation_graph(handle).is_good());
    REQUIRE(graph.create_execution_plans({fe::HeurMode_t::A}).is_good());

    REQUIRE(graph.check_support(handle).is_good());

    REQUIRE(graph.build_plans(handle, fe::BuildPlanPolicy_t::HEURISTICS_CHOICE).is_good());

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
