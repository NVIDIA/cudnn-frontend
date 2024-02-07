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
#include "../utils/helpers.h"

#include <cudnn_frontend.h>

TEST_CASE("Reduction", "[reduction]") {
    namespace fe    = cudnn_frontend;
    constexpr int n = 64;
    if (cudnnGetVersion() < 8600) {
        SKIP("TEST REQUIRES minimum cudnn version 8.6.0");
    }
    Surface<float> A_gpu(n * n * n * n, false);
    fe::graph::Graph graph{};
    auto A = graph.tensor(fe::graph::Tensor_attributes()
                              .set_dim({n, n, n, n})
                              .set_stride({n * n * n, 1, n * n, n})
                              .set_data_type(fe::DataType_t::FLOAT));
    auto C = graph.reduction(A,
                             fe::graph::Reduction_attributes()
                                 .set_mode(fe::ReductionMode_t::MAX)
                                 .set_compute_data_type(fe::DataType_t::FLOAT));
    C->set_output(true).set_data_type(fe::DataType_t::FLOAT).set_dim({1, 1, 1, 1});
    REQUIRE(graph.validate().is_good());
    cudnnHandle_t handle;
    checkCudnnErr(cudnnCreate(&handle));
    REQUIRE(graph.build_operation_graph(handle).is_good());
    REQUIRE(graph.create_execution_plans({fe::HeurMode_t::A}).is_good());
    REQUIRE(graph.build_plans(handle, fe::BuildPlanPolicy_t::HEURISTICS_CHOICE).is_good());
    Surface<float> C_gpu(n * n * n * n, false);
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {{A, A_gpu.devPtr},
                                                                                             {C, C_gpu.devPtr}};
    Surface<int8_t> workspace(graph.get_workspace_size(), false);
    REQUIRE(graph.execute(handle, variant_pack, workspace.devPtr).is_good());
    checkCudnnErr(cudnnDestroy(handle));
}
