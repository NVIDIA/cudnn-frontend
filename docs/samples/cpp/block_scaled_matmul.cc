// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Complete standalone C++ sample.
// API pages: docs/api/cpp/block_scale_dequantize.md and block_scale_quantize.md
// APIs: block_scale_dequantize, matmul, block_scale_quantize
// Use case: MXFP8 block-scaled matmul
// Source: adapted from `samples/cpp/matmul/blackwell_nvfp4_mxfp8_block_scale_matmul.cpp`.

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

constexpr int64_t
div_up(int64_t value, int64_t divisor) {
    return (value + divisor - 1) / divisor;
}

constexpr int64_t
round_up(int64_t value, int64_t multiple) {
    return div_up(value, multiple) * multiple;
}

template <typename T>
T*
managed(int64_t count) {
    T* pointer = nullptr;
    cudaMallocManaged(&pointer, count * sizeof(T));
    return pointer;
}

using VariantPack = std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*>;

void
build_and_execute(fe::graph::Graph& graph, VariantPack& variant_pack) {
    cudnnHandle_t handle = nullptr;
    cudnnCreate(&handle);
    check(graph.build(handle, {fe::HeurMode_t::A, fe::HeurMode_t::FALLBACK}));

    void* workspace     = nullptr;
    auto workspace_size = graph.get_workspace_size();
    if (workspace_size > 0) cudaMalloc(&workspace, workspace_size);
    check(graph.execute(handle, variant_pack, workspace));
    if (cudaDeviceSynchronize() != cudaSuccess) throw std::runtime_error("execution failed");

    cudaFree(workspace);
    cudnnDestroy(handle);
}

int
main() {
#if CUDNN_VERSION < 91400
    std::cout << "SKIP: block-scale quantize requires cuDNN 9.14 or newer\n";
    return 0;
#else
    int64_t const batch                 = 1;
    int64_t const rows                  = 128;
    int64_t const columns               = 128;
    int64_t const inner                 = 128;
    int64_t const block_size            = 32;
    int64_t const packed_rows           = round_up(rows, 128);
    int64_t const packed_columns        = round_up(columns, 128);
    int64_t const packed_inner          = round_up(div_up(inner, block_size), 4);
    int64_t const packed_output_columns = round_up(div_up(columns, block_size), 4);

    uint8_t* a_data            = managed<uint8_t>(batch * rows * inner);
    uint8_t* b_data            = managed<uint8_t>(batch * inner * columns);
    uint8_t* descale_a_data    = managed<uint8_t>(batch * packed_rows * packed_inner);
    uint8_t* descale_b_data    = managed<uint8_t>(batch * packed_inner * packed_columns);
    uint8_t* quantized_data    = managed<uint8_t>(batch * rows * columns);
    uint8_t* output_scale_data = managed<uint8_t>(batch * packed_rows * packed_output_columns);

    std::fill_n(a_data, batch * rows * inner, uint8_t{0});
    std::fill_n(b_data, batch * inner * columns, uint8_t{0});
    // E8M0 uses exponent-only storage. 0x7f represents a scale of 1.
    std::fill_n(descale_a_data, batch * packed_rows * packed_inner, uint8_t{0x7f});
    std::fill_n(descale_b_data, batch * packed_inner * packed_columns, uint8_t{0x7f});

    fe::graph::Graph graph{};
    graph.set_io_data_type(fe::DataType_t::HALF)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);
    auto A         = graph.tensor(fe::graph::Tensor_attributes()
                              .set_name("A")
                              .set_dim({batch, rows, inner})
                              .set_stride({rows * inner, inner, 1})
                              .set_data_type(fe::DataType_t::FP8_E4M3));
    auto B         = graph.tensor(fe::graph::Tensor_attributes()
                              .set_name("B")
                              .set_dim({batch, inner, columns})
                              .set_stride({inner * columns, 1, inner})
                              .set_data_type(fe::DataType_t::FP8_E4M3));
    auto descale_a = graph.tensor(fe::graph::Tensor_attributes()
                                      .set_name("descale_a")
                                      .set_dim({batch, packed_rows, packed_inner})
                                      .set_stride({packed_rows * packed_inner, packed_inner, 1})
                                      .set_data_type(fe::DataType_t::FP8_E8M0)
                                      .set_reordering_type(fe::TensorReordering_t::F8_128x4));
    auto descale_b = graph.tensor(fe::graph::Tensor_attributes()
                                      .set_name("descale_b")
                                      .set_dim({batch, packed_inner, packed_columns})
                                      .set_stride({packed_inner * packed_columns, 1, packed_inner})
                                      .set_data_type(fe::DataType_t::FP8_E8M0)
                                      .set_reordering_type(fe::TensorReordering_t::F8_128x4));

    auto dequantized_a = graph.block_scale_dequantize(
        A,
        descale_a,
        fe::graph::Block_scale_dequantize_attributes().set_block_size({1, block_size}));
    auto dequantized_b = graph.block_scale_dequantize(
        B,
        descale_b,
        fe::graph::Block_scale_dequantize_attributes().set_block_size({block_size, 1}));
    auto product =
        graph.matmul(dequantized_a,
                     dequantized_b,
                     fe::graph::Matmul_attributes().set_name("matmul").set_compute_data_type(
                         fe::DataType_t::FLOAT));
    auto [quantized, output_scale] =
        graph.block_scale_quantize(product,
                                   fe::graph::Block_scale_quantize_attributes()
                                       .set_name("quantize_output")
                                       .set_block_size(block_size)
                                       .set_axis(2)
                                       .set_transpose(false));
    quantized->set_output(true).set_data_type(fe::DataType_t::FP8_E4M3);
    output_scale->set_output(true)
        .set_dim({batch, packed_rows, packed_output_columns})
        .set_stride({packed_rows * packed_output_columns, packed_output_columns, 1})
        .set_data_type(fe::DataType_t::FP8_E8M0)
        .set_reordering_type(fe::TensorReordering_t::F8_128x4);

    VariantPack variant_pack = {{A, a_data},
                                {B, b_data},
                                {descale_a, descale_a_data},
                                {descale_b, descale_b_data},
                                {quantized, quantized_data},
                                {output_scale, output_scale_data}};
    build_and_execute(graph, variant_pack);

    for (int64_t index = 0; index < batch * rows * columns; ++index) {
        if (quantized_data[index] != 0) {
            throw std::runtime_error("zero inputs produced a nonzero quantized output");
        }
    }
    std::cout << "MXFP8 block-scaled matmul ok: zero result and output scales produced\n";

    cudaFree(a_data);
    cudaFree(b_data);
    cudaFree(descale_a_data);
    cudaFree(descale_b_data);
    cudaFree(quantized_data);
    cudaFree(output_scale_data);
    return 0;
#endif
}
