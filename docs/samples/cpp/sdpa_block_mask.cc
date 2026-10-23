// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Complete standalone packed-UINT8 block-mask SDPA sample.
// API page: docs/api/cpp/sdpa.md
// API: sdpa
// Use case: Block mask
// Source: adapted from `samples/cpp/sdpa/fp16_fwd_with_block_mask.cpp`.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cudnn_frontend.h>

#include <cstdint>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace fe = cudnn_frontend;

template <typename T>
T*
managed(int64_t count) {
    T* ptr             = nullptr;
    cudaError_t status = cudaMallocManaged(&ptr, count * sizeof(T));
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("cudaMallocManaged: ") + cudaGetErrorString(status));
    }
    return ptr;
}

using VariantPack = std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*>;

void
build_and_execute(fe::graph::Graph& graph, VariantPack& variant_pack) {
    cudnnHandle_t handle       = nullptr;
    void* workspace            = nullptr;
    cudnnStatus_t cudnn_status = cudnnCreate(&handle);
    if (cudnn_status != CUDNN_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("cudnnCreate: ") + cudnnGetErrorString(cudnn_status));
    }

    try {
        auto status = graph.build(handle, {fe::HeurMode_t::A, fe::HeurMode_t::FALLBACK});
        if (!status.is_good()) {
            throw std::runtime_error(std::string("graph.build: ") + status.get_message());
        }

        int64_t workspace_size = graph.get_workspace_size();
        if (workspace_size > 0) {
            cudaError_t cuda_status = cudaMalloc(&workspace, workspace_size);
            if (cuda_status != cudaSuccess) {
                throw std::runtime_error(std::string("cudaMalloc(workspace): ") +
                                         cudaGetErrorString(cuda_status));
            }
        }

        status = graph.execute(handle, variant_pack, workspace);
        if (!status.is_good()) {
            throw std::runtime_error(std::string("graph.execute: ") + status.get_message());
        }

        cudaError_t cuda_status = cudaDeviceSynchronize();
        if (cuda_status != cudaSuccess) {
            throw std::runtime_error(std::string("cudaDeviceSynchronize: ") +
                                     cudaGetErrorString(cuda_status));
        }
    } catch (...) {
        if (workspace) cudaFree(workspace);
        cudnnDestroy(handle);
        throw;
    }

    if (workspace) {
        cudaError_t cuda_status = cudaFree(workspace);
        if (cuda_status != cudaSuccess) {
            throw std::runtime_error(std::string("cudaFree(workspace): ") +
                                     cudaGetErrorString(cuda_status));
        }
    }

    cudnn_status = cudnnDestroy(handle);
    if (cudnn_status != CUDNN_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("cudnnDestroy: ") + cudnnGetErrorString(cudnn_status));
    }
}

int
main() {
    try {
        int64_t const B          = 1;
        int64_t const H          = 1;
        int64_t const S_Q        = 128;
        int64_t const S_KV       = 1024;
        int64_t const D          = 64;
        int64_t const TILE_SIZE  = 128;
        int64_t const Q_TILES    = (S_Q + TILE_SIZE - 1) / TILE_SIZE;
        int64_t const KV_TILES   = (S_KV + TILE_SIZE - 1) / TILE_SIZE;
        int64_t const MASK_BYTES = (KV_TILES + 7) / 8;

        nv_bfloat16* Q_dev          = managed<nv_bfloat16>(B * H * S_Q * D);
        nv_bfloat16* K_dev          = managed<nv_bfloat16>(B * H * S_KV * D);
        nv_bfloat16* V_dev          = managed<nv_bfloat16>(B * H * S_KV * D);
        nv_bfloat16* O_dev          = managed<nv_bfloat16>(B * H * S_Q * D);
        uint8_t* mask_dev           = managed<uint8_t>(B * H * Q_TILES * MASK_BYTES);
        uint8_t const visible_tiles = (1U << 0) | (1U << 2);

        for (int64_t i = 0; i < B * H * S_Q * D; ++i) Q_dev[i] = __float2bfloat16(0.0f);
        for (int64_t i = 0; i < B * H * S_KV * D; ++i) K_dev[i] = __float2bfloat16(0.0f);
        for (int64_t token = 0; token < S_KV; ++token) {
            float const tile_value = static_cast<float>(token / TILE_SIZE + 1);
            for (int64_t d = 0; d < D; ++d) V_dev[token * D + d] = __float2bfloat16(tile_value);
        }
        mask_dev[0] = visible_tiles;

        fe::graph::Graph graph{};
        graph.set_io_data_type(fe::DataType_t::BFLOAT16)
            .set_intermediate_data_type(fe::DataType_t::FLOAT)
            .set_compute_data_type(fe::DataType_t::FLOAT);
        auto Q          = graph.tensor(fe::graph::Tensor_attributes()
                                  .set_name("Q")
                                  .set_dim({B, H, S_Q, D})
                                  .set_stride({H * S_Q * D, S_Q * D, D, 1})
                                  .set_data_type(fe::DataType_t::BFLOAT16));
        auto K          = graph.tensor(fe::graph::Tensor_attributes()
                                  .set_name("K")
                                  .set_dim({B, H, S_KV, D})
                                  .set_stride({H * S_KV * D, S_KV * D, D, 1})
                                  .set_data_type(fe::DataType_t::BFLOAT16));
        auto V          = graph.tensor(fe::graph::Tensor_attributes()
                                  .set_name("V")
                                  .set_dim({B, H, S_KV, D})
                                  .set_stride({H * S_KV * D, S_KV * D, D, 1})
                                  .set_data_type(fe::DataType_t::BFLOAT16));
        auto block_mask = graph.tensor(
            fe::graph::Tensor_attributes()
                .set_name("block_mask")
                .set_dim({B, H, Q_TILES, MASK_BYTES})
                .set_stride({H * Q_TILES * MASK_BYTES, Q_TILES * MASK_BYTES, MASK_BYTES, 1})
                .set_data_type(fe::DataType_t::UINT8));
        auto attrs = fe::graph::SDPA_attributes()
                         .set_name("sdpa")
                         .set_generate_stats(false)
                         .set_attn_scale(1.0f / std::sqrt(static_cast<float>(D)))
                         .set_block_mask(block_mask)
                         .set_implementation(fe::AttentionImplementation_t::UNIFIED)
                         .set_compute_data_type(fe::DataType_t::FLOAT);
        auto outputs = graph.sdpa(Q, K, V, attrs);
        outputs[0]
            ->set_output(true)
            .set_name("O")
            .set_dim({B, H, S_Q, D})
            .set_stride({H * S_Q * D, S_Q * D, D, 1})
            .set_data_type(fe::DataType_t::BFLOAT16);
        if (outputs[1] != nullptr)
            throw std::runtime_error("inference unexpectedly produced statistics");

        VariantPack variant_pack = {
            {Q, Q_dev}, {K, K_dev}, {V, V_dev}, {block_mask, mask_dev}, {outputs[0], O_dev}};
        build_and_execute(graph, variant_pack);

        float const expected = 2.0f;
        for (int64_t i = 0; i < B * H * S_Q * D; ++i) {
            float const actual = __bfloat162float(O_dev[i]);
            if (std::abs(actual - expected) > 0.01f) {
                throw std::runtime_error(
                    "block mask output does not match the visible-tile reference");
            }
        }

        std::cout << "packed UINT8 block-mask SDPA passed\n";
        cudaFree(Q_dev);
        cudaFree(K_dev);
        cudaFree(V_dev);
        cudaFree(O_dev);
        cudaFree(mask_dev);
    } catch (std::exception const& e) {
        std::cerr << "block-mask sample failed: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
