// Build, filter, autotune, and inspect plans for a small BF16 matmul.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cudnn_frontend.h>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
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
    constexpr int64_t batch = 1, m = 4, k = 16, n = 8;
    auto* a_data = managed<nv_bfloat16>(m * k);
    auto* b_data = managed<nv_bfloat16>(k * n);
    auto* c_data = managed<float>(m * n);
    for (int64_t i = 0; i < m * k; ++i) a_data[i] = __float2bfloat16(1.0f);
    for (int64_t i = 0; i < k * n; ++i) b_data[i] = __float2bfloat16(1.0f);

    cudnnHandle_t handle = nullptr;
    cudnnCreate(&handle);

    fe::graph::Graph graph;
    graph.set_io_data_type(fe::DataType_t::BFLOAT16).set_compute_data_type(fe::DataType_t::FLOAT);
    auto a = graph.tensor(fe::graph::Tensor_attributes()
                              .set_name("A")
                              .set_dim({batch, m, k})
                              .set_stride({m * k, k, 1})
                              .set_data_type(fe::DataType_t::BFLOAT16));
    auto b = graph.tensor(fe::graph::Tensor_attributes()
                              .set_name("B")
                              .set_dim({batch, k, n})
                              .set_stride({k * n, n, 1})
                              .set_data_type(fe::DataType_t::BFLOAT16));
    auto c = graph.matmul(
        a, b, fe::graph::Matmul_attributes().set_compute_data_type(fe::DataType_t::FLOAT));
    c->set_output(true)
        .set_name("C")
        .set_dim({batch, m, n})
        .set_stride({m * n, n, 1})
        .set_data_type(fe::DataType_t::FLOAT);

    check(graph.validate());
    std::cout << "validated graph" << std::endl;
    check(graph.build_operation_graph(handle));
    std::cout << "built operation graph" << std::endl;

    // --8<-- [start:planning]
    check(graph.create_execution_plans({fe::HeurMode_t::A, fe::HeurMode_t::FALLBACK}));
    graph.deselect_shared_mem_greater_than(128 * 1024);
    check(graph.check_support());
    check(graph.build_plans(fe::BuildPlanPolicy_t::ALL));
    std::cout << "built supported plans" << std::endl;
    // --8<-- [end:planning]

    // get_autotune_workspace_size() currently dereferences failed-plan slots.
    // Query each built plan through the checked API instead.
    int64_t workspace_bytes = 0;
    for (int64_t index = 0; index < graph.get_execution_plan_count(); ++index) {
        int64_t plan_workspace_bytes = 0;
        if (graph.get_workspace_size_plan_at_index(index, plan_workspace_bytes).is_good()) {
            workspace_bytes = std::max(workspace_bytes, plan_workspace_bytes);
        }
    }
    void* workspace = nullptr;
    if (workspace_bytes > 0) cudaMalloc(&workspace, workspace_bytes);
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
        {a, a_data}, {b, b_data}, {c, c_data}};

    // --8<-- [start:autotune]
    std::cout << "starting autotune with " << workspace_bytes << " workspace bytes" << std::endl;
    check(graph.autotune(handle, variant_pack, workspace));
    std::cout << "autotune complete" << std::endl;
    // --8<-- [end:autotune]

    // --8<-- [start:inspection]
    std::string plan_name;
    check(graph.get_plan_name(plan_name));
    check(graph.prepare_variant_pack_template());
    auto const uids = graph.get_variant_pack_uids_sorted();
    graph.apply_oss_slot_indices_to_plans();
    std::cout << "plan=" << plan_name << ", uids=" << uids.size() << "\n";
    // --8<-- [end:inspection]

    if (workspace) cudaFree(workspace);
    cudnnDestroy(handle);
    cudaFree(a_data);
    cudaFree(b_data);
    cudaFree(c_data);
    std::cout << "plan inspection executed successfully\n";
    return 0;
}
