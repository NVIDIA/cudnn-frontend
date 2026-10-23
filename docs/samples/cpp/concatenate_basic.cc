// Concatenate tensors and compare with an index-based reference.

#include <cuda_runtime.h>
#include <cudnn_frontend.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <unordered_map>

namespace fe = cudnn_frontend;

void
check(fe::error_t status) {
    if (!status.is_good()) throw std::runtime_error(status.get_message());
}

template <typename T>
T *
managed(int64_t count) {
    T *data = nullptr;
    cudaMallocManaged(&data, count * sizeof(T));
    return data;
}

using VariantPack = std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void *>;

void
build_and_execute(fe::graph::Graph &graph, VariantPack &variant_pack) {
    cudnnHandle_t handle = nullptr;
    cudnnCreate(&handle);
    check(graph.build(handle, {fe::HeurMode_t::A, fe::HeurMode_t::FALLBACK}));
    void *workspace     = nullptr;
    auto workspace_size = graph.get_workspace_size();
    if (workspace_size > 0) cudaMalloc(&workspace, workspace_size);
    check(graph.execute(handle, variant_pack, workspace));
    if (cudaDeviceSynchronize() != cudaSuccess) throw std::runtime_error("execution failed");
    cudaFree(workspace);
    cudnnDestroy(handle);
}

int
main() {
    float *x = managed<float>(24);
    float *z = managed<float>(24);
    float *y = managed<float>(48);
    std::mt19937 generator(1234);
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
    for (int64_t index = 0; index < 24; ++index) {
        x[index] = distribution(generator);
        z[index] = distribution(generator);
    }

    fe::graph::Graph graph;
    graph.set_io_data_type(fe::DataType_t::FLOAT)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);
    auto X = graph.tensor(
        fe::graph::Tensor_attributes().set_name("X").set_dim({2, 3, 4}).set_stride({12, 4, 1}));
    auto Z = graph.tensor(
        fe::graph::Tensor_attributes().set_name("Z").set_dim({2, 3, 4}).set_stride({12, 4, 1}));
    auto Y = graph.concatenate(
        {X, Z}, fe::graph::Concatenate_attributes().set_name("concatenate").set_axis(1));
    Y->set_output(true).set_name("Y").set_dim({2, 6, 4}).set_stride({24, 4, 1});

    VariantPack variant_pack = {{X, x}, {Z, z}, {Y, y}};
    build_and_execute(graph, variant_pack);
    for (int64_t batch = 0; batch < 2; ++batch) {
        for (int64_t row = 0; row < 6; ++row) {
            for (int64_t column = 0; column < 4; ++column) {
                auto y_offset      = batch * 24 + row * 4 + column;
                auto source_offset = batch * 12 + (row % 3) * 4 + column;
                auto expected      = row < 3 ? x[source_offset] : z[source_offset];
                auto actual        = y[y_offset];
                if (!std::isfinite(actual) || !std::isfinite(expected) ||
                    std::fabs(actual - expected) > 1.0e-6f) {
                    throw std::runtime_error("concatenate result mismatch");
                }
            }
        }
    }
    std::cout << "concatenate result matches reference\n";
    cudaFree(x);
    cudaFree(z);
    cudaFree(y);
    return 0;
}
