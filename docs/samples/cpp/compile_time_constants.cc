// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Construct an FMA graph whose scale and bias are compile-time constants.

#include <cudnn_frontend.h>

#include <iostream>

int
main() {
    namespace fe = cudnn_frontend;
    using fe::graph::ScalarType;

    fe::graph::Graph graph;
    auto x = graph.tensor(fe::graph::Tensor_attributes()
                              .set_name("X")
                              .set_dim({1, 1024})
                              .set_stride({1024, 1})
                              .set_data_type(fe::DataType_t::FLOAT));

    auto scale = graph.tensor(2.5f, ScalarType::COMPILE_TIME_CONST);
    scale->set_name("scale").set_dim({1, 1}).set_stride({1, 1});
    auto scaled = graph.pointwise(x,
                                  scale,
                                  fe::graph::Pointwise_attributes()
                                      .set_name("mul")
                                      .set_mode(fe::PointwiseMode_t::MUL)
                                      .set_compute_data_type(fe::DataType_t::FLOAT));

    auto bias = graph.tensor(1.0f, ScalarType::COMPILE_TIME_CONST);
    bias->set_name("bias").set_dim({1, 1}).set_stride({1, 1});
    auto y = graph.pointwise(scaled,
                             bias,
                             fe::graph::Pointwise_attributes()
                                 .set_name("add")
                                 .set_mode(fe::PointwiseMode_t::ADD)
                                 .set_compute_data_type(fe::DataType_t::FLOAT));
    y->set_name("Y").set_output(true).set_data_type(fe::DataType_t::FLOAT);

    auto status = graph.validate();
    if (status.is_bad()) {
        std::cerr << status.get_message() << '\n';
        return 1;
    }
    std::cout << "compile-time constant graph validated\n";
    return 0;
}
