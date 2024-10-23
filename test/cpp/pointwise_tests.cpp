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

TEST_CASE("Pointwise shape deduction", "[pointwise_shape_deduction]") {
    namespace fe = cudnn_frontend;

    cudnnHandle_t handle;
    cudnnCreate(&handle);

    fe::graph::Graph graph;
    graph.set_io_data_type(fe::DataType_t::HALF)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    auto in0 = graph.tensor(
        fe::graph::Tensor_attributes().set_name("in0").set_dim({8, 128, 16000, 1}).set_stride({2048000, 1, 128, 128}));

    auto in1 = graph.tensor(
        fe::graph::Tensor_attributes().set_name("in1").set_dim({1, 128, 1, 1}).set_stride({128, 1, 128, 128}));

    auto add_options = fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::ADD);

    auto out_0 = graph.pointwise(in0, in1, add_options);

    out_0->set_output(true);

    REQUIRE(graph.validate().is_good());

    REQUIRE(graph.build_operation_graph(handle).is_good());

    REQUIRE(out_0->get_dim() == in0->get_dim());
    REQUIRE(out_0->get_stride() == in0->get_stride());

    cudnnDestroy(handle);
}

TEST_CASE("Pointwise Add shape deduction", "[pointwise_shape_deduction]") {
    namespace fe = cudnn_frontend;

    cudnnHandle_t handle;
    cudnnCreate(&handle);

    fe::graph::Graph graph;
    graph.set_io_data_type(fe::DataType_t::HALF)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    auto in0 = graph.tensor(
        fe::graph::Tensor_attributes().set_name("in0").set_dim({1, 4194304, 1}).set_stride({1, 1, 4194304}));

    auto in1 =
        graph.tensor(fe::graph::Tensor_attributes().set_name("in1").set_dim({1, 4194304, 32}).set_stride({1, 32, 1}));

    auto add_options = fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::ADD);

    auto out_0 = graph.pointwise(in0, in1, add_options);
    out_0->set_output(true);

    REQUIRE(graph.validate().is_good());

    REQUIRE(graph.build_operation_graph(handle).is_good());

    REQUIRE(out_0->get_dim() == in1->get_dim());
    REQUIRE(out_0->get_stride() == in1->get_stride());

    cudnnDestroy(handle);
}