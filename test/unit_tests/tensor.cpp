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

TEST_CASE("tensor query checks", "[query_tensor_attributes_of_uid]") {
    namespace fe = cudnn_frontend;

    fe::graph::Graph graph;
    graph.set_io_data_type(fe::DataType_t::HALF)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    int64_t uid      = 1;
    std::string name = "image";

    auto X = graph.tensor(fe::graph::Tensor_attributes()
                              .set_name(name)
                              .set_dim({8, 32, 16, 16})
                              .set_stride({32 * 16 * 16, 1, 32 * 16, 32})
                              .set_uid(uid));

    fe::graph::Tensor_attributes t;

    REQUIRE(graph.query_tensor_attributes_of_uid(uid, t).is_good());

    REQUIRE(t.get_name() == name);
}