// Execute 2-D max pooling and compare with an index-based reference.

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cudnn_frontend.h>

#include <cmath>
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
    T* data = nullptr;
    cudaMallocManaged(&data, count * sizeof(T));
    return data;
}

int
main() {
    __half* x = managed<__half>(16);
    __half* y = managed<__half>(4);
    for (int64_t index = 0; index < 16; ++index)
        x[index] = __float2half(static_cast<float>(index + 1));

    fe::graph::Graph graph;
    graph.set_io_data_type(fe::DataType_t::HALF).set_compute_data_type(fe::DataType_t::FLOAT);
    auto X          = graph.tensor(fe::graph::Tensor_attributes()
                              .set_name("X")
                              .set_dim({1, 1, 4, 4})
                              .set_stride({16, 16, 4, 1})
                              .set_data_type(fe::DataType_t::HALF));
    auto [Y, Index] = graph.resample(X,
                                     fe::graph::Resample_attributes()
                                         .set_generate_index(false)
                                         .set_resampling_mode(fe::ResampleMode_t::MAXPOOL)
                                         .set_padding_mode(fe::PaddingMode_t::NEG_INF_PAD)
                                         .set_window({2, 2})
                                         .set_stride({2, 2})
                                         .set_pre_padding({0, 0})
                                         .set_post_padding({0, 0}));
    if (Index != nullptr)
        throw std::runtime_error("inference max pooling returned an index tensor");
    Y->set_output(true)
        .set_name("Y")
        .set_dim({1, 1, 2, 2})
        .set_stride({4, 4, 2, 1})
        .set_data_type(fe::DataType_t::HALF);

    cudnnHandle_t handle = nullptr;
    cudnnCreate(&handle);
    check(graph.build(handle, {fe::HeurMode_t::A, fe::HeurMode_t::FALLBACK}));
    void* workspace     = nullptr;
    auto workspace_size = graph.get_workspace_size();
    if (workspace_size > 0) cudaMalloc(&workspace, workspace_size);
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
        {X, x}, {Y, y}};
    check(graph.execute(handle, variant_pack, workspace));
    if (cudaDeviceSynchronize() != cudaSuccess) throw std::runtime_error("execution failed");

    float expected[] = {6.0f, 8.0f, 14.0f, 16.0f};
    for (int64_t index = 0; index < 4; ++index) {
        float actual = __half2float(y[index]);
        if (!std::isfinite(actual) || std::fabs(actual - expected[index]) > 1.0e-3f) {
            throw std::runtime_error("max-pooling result mismatch");
        }
    }
    std::cout << "max-pooling output: shape=[1, 1, 2, 2], values=[6, 8, 14, 16]\n";

    cudaFree(workspace);
    cudnnDestroy(handle);
    cudaFree(x);
    cudaFree(y);
    return 0;
}
