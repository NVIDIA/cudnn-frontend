// Complete standalone C++ broadcasting sample.
// API page: docs/api/cpp/add.md
// API: add
// Use case: Broadcasting
// Source: adapted from `samples/cpp/misc/pointwise.cpp`.

#include <cuda_runtime.h>
#include <cudnn_frontend.h>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <unordered_map>

namespace fe = cudnn_frontend;

void
check(fe::error_t status) {
    if (!status.is_good()) throw std::runtime_error(status.get_message());
}

template <typename T>
T*
managed(int64_t count) {
    T* pointer = nullptr;
    cudaMallocManaged(&pointer, count * sizeof(T));
    return pointer;
}

int
main() {
    float* a_data = managed<float>(24);
    float* b_data = managed<float>(6);
    float* y_data = managed<float>(24);
    std::fill_n(a_data, 24, 1.0f);
    b_data[0] = 10.0f;
    b_data[1] = 20.0f;
    b_data[2] = 30.0f;
    std::copy_n(b_data, 3, b_data + 3);

    fe::graph::Graph graph;
    graph.set_io_data_type(fe::DataType_t::FLOAT)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);
    auto a = graph.tensor(fe::graph::Tensor_attributes()
                              .set_name("A")
                              .set_dim({2, 3, 4})
                              .set_stride({12, 4, 1})
                              .set_data_type(fe::DataType_t::FLOAT));
    auto b = graph.tensor(fe::graph::Tensor_attributes()
                              .set_name("B")
                              .set_dim({2, 3, 1})
                              .set_stride({3, 1, 1})
                              .set_data_type(fe::DataType_t::FLOAT));
    auto y = graph.pointwise(a,
                             b,
                             fe::graph::Pointwise_attributes()
                                 .set_name("broadcast_add")
                                 .set_mode(fe::PointwiseMode_t::ADD)
                                 .set_compute_data_type(fe::DataType_t::FLOAT));
    y->set_output(true).set_name("Y").set_data_type(fe::DataType_t::FLOAT);

    cudnnHandle_t handle = nullptr;
    cudnnCreate(&handle);
    check(graph.build(handle, {fe::HeurMode_t::A, fe::HeurMode_t::FALLBACK}));
    void* workspace     = nullptr;
    auto workspace_size = graph.get_workspace_size();
    if (workspace_size > 0) cudaMalloc(&workspace, workspace_size);
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
        {a, a_data}, {b, b_data}, {y, y_data}};
    check(graph.execute(handle, variant_pack, workspace));
    if (cudaDeviceSynchronize() != cudaSuccess) throw std::runtime_error("execution failed");

    for (int64_t i = 0; i < 24; ++i) {
        auto channel = (i / 4) % 3;
        if (y_data[i] != 11.0f + 10.0f * channel)
            throw std::runtime_error("broadcast result mismatch");
    }
    std::cout << "broadcast add output: shape=[2, 3, 4], values=[" << y_data[0] << ", " << y_data[4]
              << ", " << y_data[8] << "]\n";

    cudaFree(a_data);
    cudaFree(b_data);
    cudaFree(y_data);
    cudaFree(workspace);
    cudnnDestroy(handle);
    return 0;
}
