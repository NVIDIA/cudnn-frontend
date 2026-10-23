// Execute matmul, bias, and ReLU in one cuDNN graph.

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
    constexpr int64_t batch = 2;
    constexpr int64_t m     = 4;
    constexpr int64_t k     = 8;
    constexpr int64_t n     = 6;

    auto *a_dev    = managed<nv_bfloat16>(batch * m * k);
    auto *b_dev    = managed<nv_bfloat16>(k * n);
    auto *bias_dev = managed<nv_bfloat16>(n);
    auto *y_dev    = managed<nv_bfloat16>(batch * m * n);

    for (int64_t index = 0; index < batch * m * k; ++index) {
        a_dev[index] = __float2bfloat16(static_cast<float>(index % 17 - 8) / 8.0f);
    }
    for (int64_t index = 0; index < k * n; ++index) {
        b_dev[index] = __float2bfloat16(static_cast<float>(index % 13 - 6) / 8.0f);
    }
    for (int64_t column = 0; column < n; ++column) {
        bias_dev[column] = __float2bfloat16(static_cast<float>(column - 3) / 4.0f);
    }

    cudnnHandle_t handle = nullptr;
    cudnnCreate(&handle);

    // Define the graph by recording tensor metadata and operations.
    fe::graph::Graph graph;
    graph.set_io_data_type(fe::DataType_t::BFLOAT16)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    auto a    = graph.tensor(fe::graph::Tensor_attributes()
                              .set_name("A")
                              .set_dim({batch, m, k})
                              .set_stride({m * k, k, 1})
                              .set_data_type(fe::DataType_t::BFLOAT16));
    auto b    = graph.tensor(fe::graph::Tensor_attributes()
                              .set_name("B")
                              .set_dim({1, k, n})
                              .set_stride({k * n, n, 1})
                              .set_data_type(fe::DataType_t::BFLOAT16));
    auto bias = graph.tensor(fe::graph::Tensor_attributes()
                                 .set_name("bias")
                                 .set_dim({1, 1, n})
                                 .set_stride({n, n, 1})
                                 .set_data_type(fe::DataType_t::BFLOAT16));

    auto product = graph.matmul(
        a, b, fe::graph::Matmul_attributes().set_compute_data_type(fe::DataType_t::FLOAT));
    auto shifted = graph.pointwise(product,
                                   bias,
                                   fe::graph::Pointwise_attributes()
                                       .set_name("bias_add")
                                       .set_mode(fe::PointwiseMode_t::ADD)
                                       .set_compute_data_type(fe::DataType_t::FLOAT));
    auto result  = graph.pointwise(shifted,
                                  fe::graph::Pointwise_attributes()
                                      .set_name("relu")
                                      .set_mode(fe::PointwiseMode_t::RELU_FWD)
                                      .set_compute_data_type(fe::DataType_t::FLOAT));
    result->set_name("Y").set_output(true).set_data_type(fe::DataType_t::BFLOAT16);

    // Validate the graph structure and infer unspecified tensor properties.
    check(graph.validate());

    // Create the cuDNN backend graph representation.
    check(graph.build_operation_graph(handle));

    // Query heuristic candidates for the graph.
    check(graph.create_execution_plans({fe::HeurMode_t::A, fe::HeurMode_t::FALLBACK}));

    // Check that the candidates support this graph on the current GPU.
    check(graph.check_support());

    // Select the preferred supported candidate and build its execution plan.
    check(graph.build_plans(fe::BuildPlanPolicy_t::HEURISTICS_CHOICE));

    // Allocate the temporary workspace required by the execution plan.
    void *workspace     = nullptr;
    auto workspace_size = graph.get_workspace_size();
    if (workspace_size > 0) cudaMalloc(&workspace, workspace_size);

    // Bind application buffers to graph tensors and execute the plan.
    variant_pack_t variant_pack = {{a, a_dev}, {b, b_dev}, {bias, bias_dev}, {result, y_dev}};
    check(graph.execute(handle, variant_pack, workspace));

    if (cudaDeviceSynchronize() != cudaSuccess) throw std::runtime_error("execution failed");
    for (int64_t index = 0; index < batch * m * n; ++index) {
        int64_t batch_index = index / (m * n);
        int64_t row         = (index / n) % m;
        int64_t column      = index % n;
        float expected      = __bfloat162float(bias_dev[column]);
        for (int64_t inner = 0; inner < k; ++inner) {
            expected += __bfloat162float(a_dev[(batch_index * m + row) * k + inner]) *
                        __bfloat162float(b_dev[inner * n + column]);
        }
        expected     = expected < 0.0f ? 0.0f : expected;
        float actual = __bfloat162float(y_dev[index]);
        if (!std::isfinite(actual) || !std::isfinite(expected) ||
            std::abs(actual - expected) > 0.125f) {
            std::cerr << "unexpected result: " << actual << "\n";
            return 1;
        }
    }
    std::cout << "fusion ok: [" << batch << ", " << m << ", " << n
              << "], workspace=" << workspace_size << " bytes\n";
    cudaFree(a_dev);
    cudaFree(b_dev);
    cudaFree(bias_dev);
    cudaFree(y_dev);
    cudaFree(workspace);
    cudnnDestroy(handle);
    return 0;
}
