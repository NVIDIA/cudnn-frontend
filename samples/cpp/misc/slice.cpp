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
#include "../utils/helpers.h"

#include <cudnn_frontend.h>

TEST_CASE("Slice gemm", "[slice][gemm][graph][fusion]") {
    namespace fe = cudnn_frontend;

    constexpr int B_start  = 1;
    constexpr int B        = 8;
    constexpr int B_end    = 2;
    constexpr int B_actual = B_start + B + B_end;

    constexpr int M_start  = 3;
    constexpr int M        = 16;
    constexpr int M_end    = 4;
    constexpr int M_actual = M_start + M + M_end;

    constexpr int N = 32;

    constexpr int K = 64;

    constexpr int a_uid = 1, b_uid = 2, c_uid = 3;

    fe::graph::Graph graph{};

    graph.set_io_data_type(fe::DataType_t::HALF).set_compute_data_type(fe::DataType_t::FLOAT);

    auto A = graph.tensor(fe::graph::Tensor_attributes()
                              .set_dim({B_actual, M_actual, K})
                              .set_stride({M_actual * K, K, 1})
                              .set_uid(a_uid));

    auto slice_params = fe::graph::Slice_attributes().set_name("slice").set_slices(
        {{B_start, B_start + B}, {M_start, M_start + M}, {0, K}});
    auto A_slice = graph.slice(A, slice_params);
    A_slice->set_data_type(fe::DataType_t::HALF);

    auto B0 = graph.tensor(fe::graph::Tensor_attributes().set_dim({B, K, N}).set_stride({K * N, N, 1}).set_uid(b_uid));

    auto C0 = graph.matmul(A_slice, B0, fe::graph::Matmul_attributes().set_name("matmul"));
    C0->set_data_type(fe::DataType_t::FLOAT);

    auto C =
        graph.pointwise(C0, fe::graph::Pointwise_attributes().set_name("relu").set_mode(fe::PointwiseMode_t::RELU_FWD));
    C->set_output(true).set_uid(c_uid);

    // Create a unique_ptr for the cuDNN handle
    auto handle_ptr = create_cudnn_handle();
    auto handle     = *handle_ptr;

    REQUIRE(graph.build(handle, {fe::HeurMode_t::A}).is_good());

    std::vector<uint8_t> serialized_data;
    REQUIRE(graph.serialize(serialized_data).is_good());

    Surface<half> A_gpu(B_actual * M_actual * K, false);
    Surface<half> B_gpu(B * K * N, false);
    Surface<half> C_gpu(B * M * N, false);
    std::unordered_map<int64_t, void *> variant_pack = {
        {a_uid, A_gpu.devPtr}, {b_uid, B_gpu.devPtr}, {c_uid, C_gpu.devPtr}};
    int64_t workspace_size = 0;
    REQUIRE(graph.get_workspace_size(workspace_size).is_good());
    Surface<int8_t> workspace(workspace_size, false);

    fe::graph::Graph graph2;
    REQUIRE(graph2.deserialize(handle, serialized_data).is_good());
    auto result = graph2.execute(handle, variant_pack, workspace.devPtr);
    if (!result.is_good()) {
        std::cerr << result.get_message();
        REQUIRE(false);
    }
}
