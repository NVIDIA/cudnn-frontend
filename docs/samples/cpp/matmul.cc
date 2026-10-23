#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <unordered_map>

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cudnn_frontend.h>

namespace fe = cudnn_frontend;

void
check(fe::error_t status) {
    if (!status.is_good()) throw std::runtime_error(status.get_message());
}

template <typename T>
T *
managed(std::size_t count) {
    T *pointer = nullptr;
    cudaMallocManaged(&pointer, count * sizeof(T));
    return pointer;
}

typedef std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void *> variant_pack_t;

int
main() {
    // 1. Fix the graph's tensor dimensions. B has a singleton batch dimension,
    //    so cuDNN broadcasts it across the two batches of A.
    constexpr int64_t batch = 2;
    constexpr int64_t m     = 4;
    constexpr int64_t k     = 8;
    constexpr int64_t n     = 6;

    // 2. Allocate unified buffers that both the host and GPU can access.
    auto *a_dev = managed<nv_bfloat16>(batch * m * k);
    auto *b_dev = managed<nv_bfloat16>(k * n);
    auto *c_dev = managed<float>(batch * m * n);

    // 3. Initialize the unified input buffers on the host. Fixed values make
    //    every expected output element deterministic: 8 * 1 * 2 = 16.
    std::fill_n(a_dev, batch * m * k, __float2bfloat16(1.0f));
    std::fill_n(b_dev, k * n, __float2bfloat16(2.0f));

    // 4. Create the handle that owns the cuDNN execution context.
    cudnnHandle_t handle = nullptr;
    cudnnCreate(&handle);

    // 5. Set graph-wide data-type policy. Inputs use BF16, while intermediate
    //    values and matmul accumulation use FP32 for greater numeric accuracy.
    fe::graph::Graph graph;
    graph.set_io_data_type(fe::DataType_t::BFLOAT16)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    // 6. Describe tensor metadata. Dimensions and strides tell cuDNN how to
    //    interpret each runtime pointer; they do not allocate or copy data.
    auto a = graph.tensor(fe::graph::Tensor_attributes()
                              .set_name("A")
                              .set_dim({batch, m, k})
                              .set_stride({m * k, k, 1})
                              .set_data_type(fe::DataType_t::BFLOAT16));
    auto b = graph.tensor(fe::graph::Tensor_attributes()
                              .set_name("B")
                              .set_dim({1, k, n})
                              .set_stride({k * n, n, 1})
                              .set_data_type(fe::DataType_t::BFLOAT16));

    // 7. Add the matmul operation and make C a materialized FP32 graph output.
    auto attr = fe::graph::Matmul_attributes().set_compute_data_type(fe::DataType_t::FLOAT);
    auto c    = graph.matmul(a, b, attr);
    c->set_name("C").set_output(true).set_data_type(fe::DataType_t::FLOAT);

    // 8. Validate the graph, query heuristic engine configurations, check
    //    support on this GPU, and build a reusable execution plan.
    check(graph.build(handle, {fe::HeurMode_t::A, fe::HeurMode_t::FALLBACK}));

    // 9. Allocate the temporary storage required by the selected plan.
    void *workspace     = nullptr;
    auto workspace_size = graph.get_workspace_size();
    if (workspace_size > 0) cudaMalloc(&workspace, workspace_size);

    // 10. Bind graph tensors to runtime device pointers. The same built graph
    //     can be executed again with other compatible buffers.
    variant_pack_t variant_pack = {{a, a_dev}, {b, b_dev}, {c, c_dev}};

    // 11. Enqueue the plan on the handle's CUDA stream.
    check(graph.execute(handle, variant_pack, workspace));

    // 12. Wait for GPU work before reading the unified output on the host, then
    //     verify every element directly from the managed allocation.
    if (cudaDeviceSynchronize() != cudaSuccess) throw std::runtime_error("execution failed");
    for (int64_t i = 0; i < batch * m * n; ++i) {
        float value = c_dev[i];
        if (std::abs(value - 16.0f) > 0.01f) {
            std::cerr << "unexpected result: " << value << "\n";
            return 1;
        }
    }
    std::cout << "matmul ok: [" << batch << ", " << m << ", " << n
              << "], workspace=" << workspace_size << " bytes\n";

    cudaFree(a_dev);
    cudaFree(b_dev);
    cudaFree(c_dev);
    cudaFree(workspace);
    cudnnDestroy(handle);
    return 0;
}
