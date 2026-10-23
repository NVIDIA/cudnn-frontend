// Cache a built C++ frontend graph by the tensor metadata that defines its contract.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cudnn_frontend.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <vector>

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

struct TensorMetadata {
    std::vector<int64_t> dim;
    std::vector<int64_t> stride;
    fe::DataType_t data_type;

    bool
    operator==(TensorMetadata const& other) const {
        return dim == other.dim && stride == other.stride && data_type == other.data_type;
    }
};

struct GraphKey {
    int device;
    TensorMetadata a;
    TensorMetadata b;
    fe::DataType_t compute_data_type;
    fe::DataType_t output_data_type;

    bool
    operator==(GraphKey const& other) const {
        return device == other.device && a == other.a && b == other.b &&
               compute_data_type == other.compute_data_type &&
               output_data_type == other.output_data_type;
    }
};

void
hash_combine(std::size_t& seed, std::size_t value) {
    seed ^= value + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

struct GraphKeyHash {
    std::size_t
    operator()(GraphKey const& key) const {
        std::size_t seed = std::hash<int>{}(key.device);
        auto hash_tensor = [&seed](TensorMetadata const& tensor) {
            hash_combine(seed, std::hash<int>{}(static_cast<int>(tensor.data_type)));
            for (auto value : tensor.dim) hash_combine(seed, std::hash<int64_t>{}(value));
            for (auto value : tensor.stride) hash_combine(seed, std::hash<int64_t>{}(value));
        };
        hash_tensor(key.a);
        hash_tensor(key.b);
        hash_combine(seed, std::hash<int>{}(static_cast<int>(key.compute_data_type)));
        hash_combine(seed, std::hash<int>{}(static_cast<int>(key.output_data_type)));
        return seed;
    }
};

struct GraphEntry {
    fe::graph::Graph graph;
    std::shared_ptr<fe::graph::Tensor_attributes> a;
    std::shared_ptr<fe::graph::Tensor_attributes> b;
    std::shared_ptr<fe::graph::Tensor_attributes> c;
    cudnnHandle_t handle = nullptr;
    void* workspace      = nullptr;

    void
    execute(nv_bfloat16* a_data, nv_bfloat16* b_data, float* c_data) {
        std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
            {a, a_data}, {b, b_data}, {c, c_data}};
        check(graph.execute(handle, variant_pack, workspace));
        if (cudaDeviceSynchronize() != cudaSuccess) throw std::runtime_error("execution failed");
    }

    ~GraphEntry() {
        if (workspace) cudaFree(workspace);
        if (handle) cudnnDestroy(handle);
    }
};

class MatmulGraphCache {
   public:
    GraphEntry&
    get(GraphKey const& key) {
        auto iterator = entries.find(key);
        if (iterator != entries.end()) return *iterator->second;

        auto entry              = build(key);
        auto [inserted, unused] = entries.emplace(key, std::move(entry));
        (void)unused;
        builds++;
        return *inserted->second;
    }

    std::size_t
    build_count() const {
        return builds;
    }

   private:
    static std::unique_ptr<GraphEntry>
    build(GraphKey const& key) {
        auto entry = std::make_unique<GraphEntry>();
        cudnnCreate(&entry->handle);

        entry->graph.set_io_data_type(key.a.data_type)
            .set_intermediate_data_type(key.compute_data_type)
            .set_compute_data_type(key.compute_data_type);
        entry->a = entry->graph.tensor(fe::graph::Tensor_attributes()
                                           .set_name("A")
                                           .set_dim(key.a.dim)
                                           .set_stride(key.a.stride)
                                           .set_data_type(key.a.data_type));
        entry->b = entry->graph.tensor(fe::graph::Tensor_attributes()
                                           .set_name("B")
                                           .set_dim(key.b.dim)
                                           .set_stride(key.b.stride)
                                           .set_data_type(key.b.data_type));
        entry->c = entry->graph.matmul(
            entry->a,
            entry->b,
            fe::graph::Matmul_attributes().set_compute_data_type(key.compute_data_type));
        entry->c->set_name("C").set_output(true).set_data_type(key.output_data_type);

        check(entry->graph.build(entry->handle, {fe::HeurMode_t::A, fe::HeurMode_t::FALLBACK}));
        int64_t const workspace_size = entry->graph.get_workspace_size();
        if (workspace_size > 0) cudaMalloc(&entry->workspace, workspace_size);
        return entry;
    }

    std::unordered_map<GraphKey, std::unique_ptr<GraphEntry>, GraphKeyHash> entries;
    std::size_t builds = 0;
};

void
cached_matmul(MatmulGraphCache& cache,
              GraphKey const& key,
              nv_bfloat16* a_data,
              nv_bfloat16* b_data,
              float* c_data) {
    cache.get(key).execute(a_data, b_data, c_data);
}

void
check_output(float const* output, int64_t count, float expected) {
    for (int64_t index = 0; index < count; ++index) {
        if (std::fabs(output[index] - expected) > 0.01f) {
            throw std::runtime_error("cached matmul produced an unexpected result");
        }
    }
}

int
main() {
    constexpr int64_t batch = 2;
    constexpr int64_t m     = 4;
    constexpr int64_t k     = 8;
    constexpr int64_t n     = 6;

    int device = 0;
    cudaGetDevice(&device);
    GraphKey const key = {
        device,
        {{batch, m, k}, {m * k, k, 1}, fe::DataType_t::BFLOAT16},
        {{1, k, n}, {k * n, n, 1}, fe::DataType_t::BFLOAT16},
        fe::DataType_t::FLOAT,
        fe::DataType_t::FLOAT,
    };

    auto* a_first  = managed<nv_bfloat16>(batch * m * k);
    auto* b_first  = managed<nv_bfloat16>(k * n);
    auto* c_first  = managed<float>(batch * m * n);
    auto* a_second = managed<nv_bfloat16>(batch * m * k);
    auto* b_second = managed<nv_bfloat16>(k * n);
    auto* c_second = managed<float>(batch * m * n);
    std::fill_n(a_first, batch * m * k, __float2bfloat16(1.0f));
    std::fill_n(b_first, k * n, __float2bfloat16(2.0f));
    std::fill_n(a_second, batch * m * k, __float2bfloat16(3.0f));
    std::fill_n(b_second, k * n, __float2bfloat16(4.0f));

    MatmulGraphCache cache;
    cached_matmul(cache, key, a_first, b_first, c_first);
    cached_matmul(cache, key, a_second, b_second, c_second);
    check_output(c_first, batch * m * n, 16.0f);
    check_output(c_second, batch * m * n, 96.0f);
    if (cache.build_count() != 1)
        throw std::runtime_error("compatible executions did not reuse the graph");
    std::cout << "C++ graph cache ok: builds=1 executions=2\n";

    cudaFree(a_first);
    cudaFree(b_first);
    cudaFree(c_first);
    cudaFree(a_second);
    cudaFree(b_second);
    cudaFree(c_second);
    return 0;
}
