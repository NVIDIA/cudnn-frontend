/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#include "my_scale_op.h"
#include <cudnn_frontend.h>
#include "helpers.h"

void
run_my_scale_op(int64_t* tensorDim,
                [[maybe_unused]] cudnnDataType_t dataType,
                void* devPtrX,
                void* devPtrScale,
                void* devPtrY) {
    namespace fe = cudnn_frontend;

    auto handle_ptr = create_cudnn_handle();
    auto handle     = *handle_ptr;

    // Build graph using the new Graph API (same as samples/cpp/)
    auto graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(fe::DataType_t::FLOAT).set_compute_data_type(fe::DataType_t::FLOAT);

    auto N = tensorDim[0];
    auto C = tensorDim[1];
    auto H = tensorDim[2];
    auto W = tensorDim[3];

    auto X = graph->tensor(
        fe::graph::Tensor_attributes().set_name("X").set_dim({N, C, H, W}).set_stride({C * H * W, 1, C * W, C}));

    auto Scale =
        graph->tensor(fe::graph::Tensor_attributes().set_name("Scale").set_dim({1, 1, 1, 1}).set_stride({1, 1, 1, 1}));

    auto scale_options = fe::graph::Pointwise_attributes()
                             .set_mode(fe::PointwiseMode_t::MUL)
                             .set_compute_data_type(fe::DataType_t::FLOAT);
    auto Y             = graph->pointwise(X, Scale, scale_options);
    Y->set_output(true).set_data_type(fe::DataType_t::FLOAT);

    REQUIRE(graph->validate().is_good());
    REQUIRE(graph->build_operation_graph(handle).is_good());
    REQUIRE(graph->create_execution_plans({fe::HeurMode_t::A}).is_good());
    REQUIRE(graph->check_support().is_good());
    REQUIRE(graph->build_plans().is_good());

    int64_t total_elems = N * C * H * W;
    Surface<float> x_tensor(total_elems, false);
    Surface<float> y_tensor(total_elems, false);

    // Read scale scalar and upload
    float h_scale;
    checkCudaErr(cudaMemcpy(&h_scale, devPtrScale, sizeof(float), cudaMemcpyDeviceToHost));
    Surface<float> scale_tensor(1, false, h_scale);

    std::unordered_map<int64_t, void*> variant_pack = {
        {X->get_uid(), devPtrX}, {Scale->get_uid(), scale_tensor.devPtr}, {Y->get_uid(), devPtrY}};

    int64_t workspace_size = 0;
    REQUIRE(graph->get_workspace_size(workspace_size).is_good());
    Surface<int8_t> workspace(workspace_size);

    std::cout << *graph << std::endl;
    REQUIRE(graph->execute(handle, variant_pack, workspace.devPtr).is_good());
}
