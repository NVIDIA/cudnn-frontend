/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <cudnn_frontend.h>
#include <cuda_bf16.h>
#include <nvrtc.h>
#include "../utils/helpers.h"
#include "weight_dequantize/ggml_decode.h"
#include "weight_dequantize/ggml_fixture.h"

namespace {
// Customer-supplied CUDA C++ defines both the narrow numerical type and its
// scaling scheme. The backend supplies the ABI types and conversion helper.
constexpr char source[] = R"CUDA(
__device__ void decode(const FortWeightDecodeTileV1& t, const void* storage,
    const void* const* auxiliary, void*, void*, FortWeightDecodeValue* output) {
    const int bits=int(t.constants[0]), scaling=int(t.constants[1]);
    const long long block_k=t.constants[2], block_n=t.constants[3];
    const long long blocks_n=(t.full_n+block_n-1)/block_n;
    const auto* block_scales=static_cast<const float*>(auxiliary[0]);
    const auto* global_scale=static_cast<const float*>(auxiliary[1]);
    for (int i=t.thread_id; i<t.tile_k*t.tile_n; i+=t.thread_count) {
        const int row=i/t.tile_n, col=i%t.tile_n;
        if (row>=t.valid_k || col>=t.valid_n) continue;
        const long long k=t.k_begin+row, n=t.n_begin+col;
        const long long bit_index=(k*t.full_n+n)*bits;
        const unsigned code=(static_cast<const unsigned char*>(storage)[bit_index/8]
            >> (bit_index%8)) & ((1u<<bits)-1);
        // Give the code its signed numerical meaning, then dequantize it.
        const int value=int(code)-((code & (1u<<(bits-1))) ? (1<<bits) : 0);
        float weight=float(value);
        if (scaling & 2) weight *= block_scales[(k/block_k)*blocks_n+n/block_n];
        if (scaling & 1) weight *= global_scale[0];
        // Both scale operations precede the single rounding to the MMA type.
        output[row*t.output_stride+col]=fort_weight_decode_from_float(weight);
    }
}
)CUDA";
}  // namespace

TEST_CASE("Custom weight dequantization matmul", "[matmul][weight_dequantize][graph]") {
    namespace fe = cudnn_frontend;
#if !defined(CUDNN_WEIGHT_DECODE_ABI_VERSION) || CUDNN_WEIGHT_DECODE_ABI_VERSION < 1
    SKIP("Build with the experimental weight-dequantization backend headers");
#else
    if (get_compute_capability() != 120) SKIP("Prototype supports SM120 only");
    int major = 0, minor = 0;
    REQUIRE(nvrtcVersion(&major, &minor) == NVRTC_SUCCESS);
    if (major * 1000 + minor * 10 < 12080) SKIP("Prototype requires NVRTC 12.8 or newer");
    cudnnBackendDescriptor_t probe = nullptr;
    if (cudnnBackendCreateDescriptor(CUDNN_BACKEND_WEIGHT_DECODE_DESCRIPTOR, &probe) != CUDNN_STATUS_SUCCESS)
        SKIP("Runtime library lacks experimental weight-dequantization descriptors");
    REQUIRE(cudnnBackendDestroyDescriptor(probe) == CUDNN_STATUS_SUCCESS);

    auto bits    = GENERATE(8, 4, 2);
    auto scaling = GENERATE(1, 2, 3);  // global, block, block followed by global
    auto bf16    = GENERATE(false, true);
    CAPTURE(bits, scaling, bf16);
    constexpr int64_t m = 37, k = 83, n = 75, lda = 88, block_k = 24, block_n = 20;
    constexpr int64_t blocks_n    = (n + block_n - 1) / block_n;
    constexpr int64_t scale_count = ((k + block_k - 1) / block_k) * blocks_n;
    int64_t byte_count            = (k * n * bits + 7) / 8;
    auto dtype                    = bf16 ? fe::DataType_t::BFLOAT16 : fe::DataType_t::HALF;
    auto encode                   = [bf16](float x) -> unsigned short {
        return bf16 ? __bfloat16_as_ushort(__float2bfloat16(x)) : __half_as_ushort(__float2half(x));
    };
    auto to_float = [bf16](unsigned short x) -> float {
        return bf16 ? __bfloat162float(__ushort_as_bfloat16(x)) : __half2float(__ushort_as_half(x));
    };

    fe::graph::Graph graph;
    graph.set_io_data_type(dtype).set_intermediate_data_type(dtype).set_compute_data_type(fe::DataType_t::FLOAT);
    auto storage      = graph.tensor(fe::graph::Tensor_attributes()
                                    .set_name("storage")
                                    .set_uid(2)
                                    .set_dim({1, 1, byte_count})
                                    .set_stride({byte_count, byte_count, 1})
                                    .set_data_type(fe::DataType_t::UINT8));
    auto block_scales = graph.tensor(fe::graph::Tensor_attributes()
                                         .set_name("block_scales")
                                         .set_uid(5)
                                         .set_dim({1, 1, scale_count})
                                         .set_stride({scale_count, scale_count, 1})
                                         .set_data_type(fe::DataType_t::FLOAT));
    auto global_scale = graph.tensor(fe::graph::Tensor_attributes()
                                         .set_name("global_scale")
                                         .set_uid(6)
                                         .set_dim({1, 1, 1})
                                         .set_stride({1, 1, 1})
                                         .set_data_type(fe::DataType_t::FLOAT));
    auto program      = fe::graph::Weight_dequantize_program().set_source(source).set_entry("decode").set_constants(
        {bits, scaling, block_k, block_n});
    // No decoder scratch is necessary for these conversions. The logical B is
    // virtual: the engine produces tiles directly in shared memory for MMA.
    auto b =
        graph.weight_dequantize(storage,
                                {block_scales, global_scale},
                                fe::graph::Weight_dequantize_attributes().set_name("dequant").set_program(program));
    b->set_dim({1, k, n}).set_data_type(dtype).set_uid(3);
    auto a = graph.tensor(
        fe::graph::Tensor_attributes().set_name("A").set_uid(1).set_dim({1, m, k}).set_stride({m * lda, lda, 1}));
    auto c = graph.matmul(a, b, fe::graph::Matmul_attributes().set_name("gemm"));
    c->set_output(true).set_data_type(fe::DataType_t::FLOAT).set_uid(4);
    REQUIRE(graph.validate().is_good());

    // Exercise structural serialization through the public C++ API. Source,
    // constants, and the order of the two auxiliary tensors survive this copy.
    fe::graph::Graph restored;
    REQUIRE(restored.deserialize(json(graph)).is_good());
    REQUIRE(restored.validate().is_good());
    REQUIRE(restored.key() == graph.key());
    auto handle_owner = create_cudnn_handle();
    auto handle       = *handle_owner;
    REQUIRE(restored.build_operation_graph(handle).is_good());
    // Pin the prototype FORT/native matmul engine instead of testing heuristic ranking.
    REQUIRE(restored.create_execution_plan(10, {}).is_good());
    REQUIRE(restored.check_support().is_good());
    REQUIRE(restored.build_plans().is_good());
    int64_t workspace_size = -1;
    REQUIRE(restored.get_workspace_size(workspace_size).is_good());
    REQUIRE(workspace_size >= 0);
    // Small GEMM grids may use split-K with FP32 partial outputs. Always query
    // the plan; decoder scratch being zero does not imply zero GEMM workspace.
    Surface<uint8_t> workspace(std::max<int64_t>(1, workspace_size), 0);

    std::vector<unsigned short> a_host(m * lda, 0);
    for (int64_t row = 0; row < m; ++row)
        for (int64_t col = 0; col < k; ++col)
            a_host[row * lda + col] = encode(float((row * 3 + col * 7) % 13 - 6) / 16);
    std::vector<int> original(k * n);
    std::vector<uint8_t> bytes(byte_count, 0);
    for (int64_t i = 0; i < k * n; ++i) {
        original[i]   = int((i * 13 + i / 7) % (1 << bits)) - (1 << (bits - 1));
        unsigned code = unsigned(original[i]) & ((1u << bits) - 1);
        bytes[i * bits / 8] |= code << ((i * bits) % 8);
    }
    Surface<unsigned short> a_gpu(a_host.size(), 0);
    Surface<uint8_t> w_gpu(bytes.size(), 0);
    Surface<float> block_gpu(scale_count, 0), global_gpu(1, 0), c_gpu(m * n, 0);
    CUDA_CHECK(cudaMemcpy(a_gpu.devPtr, a_host.data(), a_host.size() * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(w_gpu.devPtr, bytes.data(), bytes.size(), cudaMemcpyHostToDevice));
    std::unordered_map<int64_t, void*> variant_pack = {
        {1, a_gpu.devPtr}, {2, w_gpu.devPtr}, {4, c_gpu.devPtr}, {5, block_gpu.devPtr}, {6, global_gpu.devPtr}};
    std::vector<float> scales(scale_count), weights(k * n), actual(m * n);
    // Reuse the plan while changing the global scale, then the block scales.
    // The original signed values form an independent CPU reference: it does
    // not decode the byte stream with another copy of the CUDA algorithm.
    for (int pass = 0; pass < 3; ++pass) {
        for (int64_t i = 0; i < scale_count; ++i) scales[i] = float(i % 7 + 1 + (pass == 2 ? 2 : 0)) / 16;
        float global = pass == 0 ? 0.375f : 0.625f;
        CUDA_CHECK(cudaMemcpy(block_gpu.devPtr, scales.data(), scale_count * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(global_gpu.devPtr, &global, sizeof(float), cudaMemcpyHostToDevice));
        REQUIRE(restored.execute(handle, variant_pack, workspace_size ? workspace.devPtr : nullptr).is_good());
        CUDA_CHECK(cudaMemcpy(actual.data(), c_gpu.devPtr, actual.size() * sizeof(float), cudaMemcpyDeviceToHost));
        for (int64_t row = 0; row < k; ++row)
            for (int64_t col = 0; col < n; ++col) {
                float w = float(original[row * n + col]);
                if (scaling & 2) w *= scales[(row / block_k) * blocks_n + col / block_n];
                if (scaling & 1) w *= global;
                weights[row * n + col] = to_float(encode(w));
            }
        for (int64_t row = 0; row < m; ++row)
            for (int64_t col = 0; col < n; ++col) {
                double expected = 0;
                for (int64_t red = 0; red < k; ++red)
                    expected += double(to_float(a_host[row * lda + red])) * weights[red * n + col];
                auto value = actual[row * n + col];
                REQUIRE(std::isfinite(value));
                REQUIRE(std::abs(double(value) - expected) <= 0.005 + 0.005 * std::abs(expected));
            }
    }
#endif
}

// Original ggml blocks: decoder owns every load, including embedded scales and
// the IQ1_S grid. These examples do not repack or materialize a dense B tensor.
TEST_CASE("Custom weight dequantization ggml blocks", "[matmul][weight_dequantize][graph]") {
    namespace fe = cudnn_frontend;
#if !defined(CUDNN_WEIGHT_DECODE_ABI_VERSION) || CUDNN_WEIGHT_DECODE_ABI_VERSION < 1
    SKIP("Build with the experimental weight-dequantization backend headers");
#else
    if (get_compute_capability() != 120) SKIP("Prototype supports SM120 only");
    cudnnBackendDescriptor_t probe = nullptr;
    if (cudnnBackendCreateDescriptor(CUDNN_BACKEND_WEIGHT_DECODE_DESCRIPTOR, &probe) != CUDNN_STATUS_SUCCESS)
        SKIP("Runtime library lacks experimental weight-dequantization descriptors");
    REQUIRE(cudnnBackendDestroyDescriptor(probe) == CUDNN_STATUS_SUCCESS);
    auto format   = GENERATE(4, 6, 1);  // Q4_K, Q6_K, IQ1_S
    auto bf16     = GENERATE(false, true);
    auto shape    = GENERATE(0, 1, 2, 3);  // tails, identity, split-K, wide K128 group
    bool identity = shape == 1, split = shape == 2, wide = shape == 3;
    int device;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    const int64_t m   = wide ? 1 : (identity ? 256 : (split ? 16 : 37));
    const int64_t k   = wide ? 128 : (identity ? 256 : (split ? 544 : 515));
    const int64_t n   = wide ? 8LL * 64 * prop.multiProcessorCount : (identity ? 64 : (split ? 256 : 75));
    const int64_t lda = (k + 7) / 8 * 8;
    // Keep aligned vector loads covered on identity/wide shapes and exercise
    // the FLOAT-aligned scalar fallback on partial and split-K shapes.
    const int grid_offset = format == 1 && !identity && !wide ? (split ? 2 : 1) : 0;
    CAPTURE(format, bf16, shape, grid_offset);
    GgmlWeightFixture fixture(format, k, n);
    auto dtype = bf16 ? fe::DataType_t::BFLOAT16 : fe::DataType_t::HALF;
    auto round = [bf16](float x) -> float {
        return bf16 ? __bfloat162float(__float2bfloat16(x)) : __half2float(__float2half(x));
    };
    fe::graph::Graph graph;
    graph.set_io_data_type(dtype).set_intermediate_data_type(dtype).set_compute_data_type(fe::DataType_t::FLOAT);
    auto vector = [&](int64_t uid, int64_t count, fe::DataType_t type) {
        return graph.tensor(fe::graph::Tensor_attributes()
                                .set_uid(uid)
                                .set_dim({1, 1, count})
                                .set_stride({count, count, 1})
                                .set_data_type(type));
    };
    auto w = vector(2, fixture.packed.size(), fe::DataType_t::UINT8);
    w->set_alignment(1);
    std::vector<std::shared_ptr<fe::graph::Tensor_attributes> > aux;
    if (format == 1) {
        aux.push_back(vector(5, fixture.codebook.size(), fe::DataType_t::FLOAT));
        aux.back()->set_alignment(4);
    }
    // Bytewise field reads and the Q4 word-load fallback permit an unaligned base. The FLOAT grid
    // still requires natural alignment, which the engine checks separately.
    auto program = fe::graph::Weight_dequantize_program()
                       .set_source(ggml_decode_source)
                       .set_entry("decode")
                       .set_constants({format})
                       .set_input_alignment(1);
    auto b = graph.weight_dequantize(w, aux, fe::graph::Weight_dequantize_attributes().set_program(program));
    b->set_dim({1, k, n}).set_data_type(dtype);
    auto a = graph.tensor(fe::graph::Tensor_attributes().set_uid(1).set_dim({1, m, k}).set_stride({m * lda, lda, 1}));
    auto c = graph.matmul(a, b, fe::graph::Matmul_attributes());
    c->set_output(true).set_data_type(fe::DataType_t::FLOAT).set_uid(4);
    REQUIRE(graph.validate().is_good());
    auto owner = create_cudnn_handle();
    REQUIRE(graph.build_operation_graph(*owner).is_good());
    REQUIRE(graph.create_execution_plan(10, {}).is_good());
    REQUIRE(graph.check_support().is_good());
    REQUIRE(graph.build_plans().is_good());
    auto workspace_size = graph.get_workspace_size();
    REQUIRE(workspace_size >= 0);
    REQUIRE((workspace_size > 0) == split);
    Surface<uint8_t> workspace(std::max<int64_t>(1, workspace_size), 0);
    std::vector<unsigned short> ah(m * lda, 0);
    std::vector<float> af(m * k);
    for (int64_t r = 0; r < m; ++r)
        for (int64_t j = 0; j < k; ++j) {
            float value   = identity ? float(r == j) : float((r * 7 + j * 3) % 17 - 8) / 16;
            af[r * k + j] = round(value);
            ah[r * lda + j] =
                bf16 ? __bfloat16_as_ushort(__float2bfloat16(value)) : __half_as_ushort(__float2half(value));
        }
    for (auto& value : fixture.expected) value = round(value);
    Surface<unsigned short> ag(ah.size(), 0);
    Surface<uint8_t> wg(fixture.packed.size() + 1, 0);
    Surface<float> grid(fixture.codebook.size() + grid_offset, 0), cg(m * n, 0);
    CUDA_CHECK(cudaMemcpy(ag.devPtr, ah.data(), ah.size() * 2, cudaMemcpyHostToDevice));
    // Cover the aligned Q4 word path as well as the unaligned byte fallback.
    const int packed_offset = wide ? 0 : 1;
    CUDA_CHECK(
        cudaMemcpy(wg.devPtr + packed_offset, fixture.packed.data(), fixture.packed.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        grid.devPtr + grid_offset, fixture.codebook.data(), fixture.codebook.size() * 4, cudaMemcpyHostToDevice));
    std::unordered_map<int64_t, void*> bindings{{1, ag.devPtr}, {2, wg.devPtr + packed_offset}, {4, cg.devPtr}};
    if (format == 1) bindings[5] = grid.devPtr + grid_offset;
    REQUIRE(graph.execute(*owner, bindings, workspace_size ? workspace.devPtr : nullptr).is_good());
    std::vector<float> actual(m * n);
    CUDA_CHECK(cudaMemcpy(actual.data(), cg.devPtr, actual.size() * 4, cudaMemcpyDeviceToHost));
    for (int64_t r = 0; r < m; ++r)
        for (int64_t col = 0; col < n; ++col) {
            double expected = 0;
            for (int64_t j = 0; j < k; ++j) expected += double(af[r * k + j]) * fixture.expected[j * n + col];
            const float value = actual[r * n + col];
            REQUIRE(std::isfinite(value));
            if (identity)
                REQUIRE(value == expected);
            else
                REQUIRE(std::abs(double(value) - expected) <= 0.005 + 0.005 * std::abs(expected));
        }
#endif
}
