/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <catch2/catch_test_macros.hpp>
#include "../utils/helpers.h"

#include <cudnn_frontend.h>

#include <cuda_bf16.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

namespace fe = cudnn_frontend;

char const*
custom_reduction_runtime_skip_reason() {
#if CUDART_VERSION < 13010
    return "Test requires cuda toolkit 13.1 or above";
#elif CUDNN_VERSION < 92900
    return "Custom reduction requires cuDNN v9.29.0";
#else
    if (fe::detail::get_backend_version() < 92900) {
        return "Custom reduction requires cuDNN v9.29.0";
    }
    if (!is_blackwell_computing_arch()) {
        return "Custom reduction requires the TensorIR MemBound engine on data-center Blackwell";
    }
    if (!is_arch_supported_by_cudnn()) {
        return "Architecture is not supported by the current cuDNN version";
    }
    return nullptr;
#endif
}

void
build_supported_graph(fe::graph::Graph& graph, cudnnHandle_t handle) {
    REQUIRE(graph.validate().is_good());
    REQUIRE(graph.build_operation_graph(handle).is_good());
    REQUIRE(graph.create_execution_plans({fe::HeurMode_t::A}).is_good());
    REQUIRE(graph.check_support().is_good());
    REQUIRE(graph.build_plans(fe::BuildPlanPolicy_t::HEURISTICS_CHOICE).is_good());
}

template <typename T>
std::vector<T>
copy_from_device(T const* device, size_t count) {
    std::vector<T> host(count);
    CUDA_CHECK(cudaMemcpy(host.data(), device, count * sizeof(T), cudaMemcpyDeviceToHost));
    return host;
}

uint16_t
raw_bits(nv_bfloat16 value) {
    return static_cast<__nv_bfloat16_raw>(value).x;
}

}  // namespace

TEST_CASE("Custom reduction F32 sum helper matches built-in reduction and replay",
          "[membound][custom_reduction][graph][serialization]") {
    namespace fe = cudnn_frontend;

#if CUDART_VERSION < 13010
    SKIP("Test requires cuda toolkit 13.1 or above");
    return;
#endif

#if (CUDNN_VERSION < 92900)
    SKIP("Custom reduction requires cuDNN v9.29.0");
#else
    if (fe::detail::get_backend_version() < 92900) {
        SKIP("Custom reduction requires cuDNN v9.29.0");
    }
#endif
    if (!is_blackwell_computing_arch()) {
        SKIP("Custom reduction requires the TensorIR MemBound engine on data-center Blackwell");
    }
    if (!is_arch_supported_by_cudnn()) {
        SKIP("Architecture is not supported by the current cuDNN version");
    }

    constexpr int64_t batch        = 3;
    constexpr int64_t reduced_size = 8;
    constexpr int64_t inner        = 5;
    constexpr int64_t input_size   = batch * reduced_size * inner;
    constexpr int64_t output_size  = batch * inner;

    fe::graph::Graph graph;
    graph.set_io_data_type(fe::DataType_t::FLOAT).set_compute_data_type(fe::DataType_t::FLOAT);

    auto X = graph.tensor(fe::graph::Tensor_attributes()
                              .set_name("X")
                              .set_dim({batch, reduced_size, inner})
                              .set_stride({reduced_size * inner, inner, 1})
                              .set_data_type(fe::DataType_t::FLOAT));

    auto outputs = graph.custom_reduction(
        {X},
        fe::graph::reduction_modifiers::sum(fe::graph::Custom_reduction_attributes()
                                                .set_name("custom_sum")
                                                .set_reduction_dims({1})
                                                .set_identity_values({fe::graph::Custom_reduction_identity(0.0f)})
                                                .set_compute_data_types({fe::DataType_t::FLOAT})));
    auto Y = outputs.at(0);
    Y->set_name("Y")
        .set_dim({batch, 1, inner})
        .set_stride({inner, inner, 1})
        .set_output(true)
        .set_data_type(fe::DataType_t::FLOAT);

    fe::graph::Graph built_in_graph;
    built_in_graph.set_io_data_type(fe::DataType_t::FLOAT).set_compute_data_type(fe::DataType_t::FLOAT);
    auto built_in_X = built_in_graph.tensor(fe::graph::Tensor_attributes()
                                                .set_name("built_in_X")
                                                .set_dim({batch, reduced_size, inner})
                                                .set_stride({reduced_size * inner, inner, 1})
                                                .set_data_type(fe::DataType_t::FLOAT));
    auto built_in_Y = built_in_graph.reduction(built_in_X,
                                               fe::graph::Reduction_attributes()
                                                   .set_name("built_in_sum")
                                                   .set_mode(fe::ReductionMode_t::ADD)
                                                   .set_compute_data_type(fe::DataType_t::FLOAT));
    built_in_Y->set_name("built_in_Y")
        .set_dim({batch, 1, inner})
        .set_stride({inner, inner, 1})
        .set_output(true)
        .set_data_type(fe::DataType_t::FLOAT);

    REQUIRE(graph.validate().is_good());
    REQUIRE(built_in_graph.validate().is_good());
    auto const graph_key_before_expansion = graph.key();

    auto handle_ptr = create_cudnn_handle();
    auto handle     = *handle_ptr;

    REQUIRE(graph.build_operation_graph(handle).is_good());
    REQUIRE(graph.key() == graph_key_before_expansion);
    REQUIRE(graph.create_execution_plans({fe::HeurMode_t::A}).is_good());
    REQUIRE(graph.check_support().is_good());
    REQUIRE(graph.build_plans(fe::BuildPlanPolicy_t::HEURISTICS_CHOICE).is_good());
    REQUIRE(built_in_graph.build_operation_graph(handle).is_good());
    REQUIRE(built_in_graph.create_execution_plans({fe::HeurMode_t::A}).is_good());
    REQUIRE(built_in_graph.check_support().is_good());
    REQUIRE(built_in_graph.build_plans(fe::BuildPlanPolicy_t::HEURISTICS_CHOICE).is_good());

    std::vector<float> input(input_size);
    for (int64_t i = 0; i < input_size; ++i) {
        input[static_cast<size_t>(i)] = static_cast<float>(i % 11 - 5);
    }

    std::vector<float> expected(output_size, 0.0f);
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t r = 0; r < reduced_size; ++r) {
            for (int64_t k = 0; k < inner; ++k) {
                expected[static_cast<size_t>(b * inner + k)] +=
                    input[static_cast<size_t>((b * reduced_size + r) * inner + k)];
            }
        }
    }

    Surface<float> X_gpu(input_size, 0.0f);
    Surface<float> Y_gpu(output_size, 0.0f);
    Surface<float> built_in_Y_gpu(output_size, 0.0f);
    CUDA_CHECK(cudaMemcpy(X_gpu.devPtr, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice));

    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {{X, X_gpu.devPtr},
                                                                                             {Y, Y_gpu.devPtr}};
    int64_t workspace_size                                                                = 0;
    REQUIRE(graph.get_workspace_size(workspace_size).is_good());
    Surface<int8_t> workspace(static_cast<size_t>(workspace_size));
    REQUIRE(graph.execute(handle, variant_pack, workspace.devPtr).is_good());

    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> built_in_variant_pack = {
        {built_in_X, X_gpu.devPtr}, {built_in_Y, built_in_Y_gpu.devPtr}};
    int64_t built_in_workspace_size = 0;
    REQUIRE(built_in_graph.get_workspace_size(built_in_workspace_size).is_good());
    Surface<int8_t> built_in_workspace(static_cast<size_t>(built_in_workspace_size));
    REQUIRE(built_in_graph.execute(handle, built_in_variant_pack, built_in_workspace.devPtr).is_good());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> original_output(output_size);
    std::vector<float> built_in_output(output_size);
    CUDA_CHECK(cudaMemcpy(
        original_output.data(), Y_gpu.devPtr, original_output.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        built_in_output.data(), built_in_Y_gpu.devPtr, built_in_output.size() * sizeof(float), cudaMemcpyDeviceToHost));
    REQUIRE(original_output == expected);
    REQUIRE(built_in_output == expected);
    REQUIRE(original_output == built_in_output);

    std::vector<uint8_t> serialized_plan;
    // Keep the default serialize_structure=true path: reducer callbacks are not
    // reconstructible, but the materialized execution plan is replayable.
    REQUIRE(graph.serialize(serialized_plan).is_good());
    REQUIRE_FALSE(serialized_plan.empty());

    // Mutate a real custom-reduction plan, but never execute or warm up rejected metadata.
    // Each case must fail before auxiliary copies, scalar pointer binding, or plan publication.
    auto const valid_metadata = json::from_ubjson(serialized_plan);
    std::vector<json> malformed_metadata;
    for (auto entry : {json::array({1, 0, json::array()}),
                       json::array({0, -1, json::array()}),
                       json::array({1, 0, json::array({"7FC00000"})}),
                       json::array({1, 0, json::array({"5F000000"})}),
                       json::array({0, 0, json::array({"00000000"})})}) {
        auto malformed                       = valid_metadata;
        malformed["fe_workspace_size"]       = 0;
        malformed["workspace_modifications"] = {{"777", entry}};
        malformed_metadata.push_back(std::move(malformed));
    }
    for (auto key : {"777suffix", "+777", "0777", "777 "}) {
        auto malformed                   = valid_metadata;
        malformed["pass_by_values"][key] = json{{"index", 0}, {"value", 1}};
        malformed_metadata.push_back(std::move(malformed));
    }
    for (auto replacements :
         {json::array({json::array({777, json::array({777, 0})})}),
          json::array({json::array({777, json::array({778, 0})}), json::array({778, json::array({777, 0})})}),
          json::array({json::array({777, json::array({778, 0})}), json::array({777, json::array({779, 0})})})}) {
        auto malformed                         = valid_metadata;
        malformed["variant_pack_replacements"] = std::move(replacements);
        malformed_metadata.push_back(std::move(malformed));
    }
    {
        auto malformed                     = valid_metadata;
        malformed["pass_by_values"]["777"] = json{{"index", 6}, {"value", 0}};
        malformed_metadata.push_back(std::move(malformed));
    }
    for (auto const& malformed : malformed_metadata) {
        fe::graph::Graph restored;
        REQUIRE(restored.deserialize(handle, valid_metadata, true, false).is_good());
        auto const before_uids = restored.get_variant_pack_uids_sorted();
        fe::error_t status;
        REQUIRE_NOTHROW(status = restored.deserialize(handle, malformed, true, false));
        INFO(status.get_message());
        REQUIRE(status.get_code() == fe::error_code_t::INVALID_VALUE);
        REQUIRE(restored.get_variant_pack_uids_sorted() == before_uids);
        std::vector<uint8_t> after;
        REQUIRE(restored.serialize(after).is_good());
        REQUIRE(json::from_ubjson(after)["pass_by_values"] == valid_metadata["pass_by_values"]);
        REQUIRE(restored.deserialize(handle, serialized_plan, true, false).is_good());
    }
    REQUIRE(graph.key() == graph_key_before_expansion);

    // Preserve legacy plan-only pointer aliases even when the root UID is not consumed by the backend.
    // Serialize the chain in reverse dependency order and bind the real output into owned workspace.
    for (bool structure : {false, true}) {
        auto aliased = valid_metadata;
        if (!structure) {
            aliased.erase("nodes");
            aliased.erase("tensors");
        }
        aliased["fe_workspace_size"]         = 256;
        aliased["workspace_modifications"]   = {{"777", json::array({2, 0, json::array()})}};
        aliased["variant_pack_replacements"] = json::array(
            {json::array({778, json::array({Y->get_uid(), 0})}), json::array({777, json::array({778, 0})})});
        fe::graph::Graph alias_graph;
        REQUIRE(alias_graph.deserialize(handle, aliased, true, false).is_good());
        std::vector<uint8_t> alias_blob;
        REQUIRE(alias_graph.serialize(alias_blob, structure).is_good());
        fe::graph::Graph alias_reloaded;
        REQUIRE(alias_reloaded.deserialize(handle, alias_blob, true, false).is_good());
        if (structure) {
            fe::graph::Tensor_attributes queried;
            REQUIRE(alias_reloaded.query_tensor_attributes_of_uid(Y->get_uid(), queried).is_good());
            REQUIRE(queried.get_dim() == Y->get_dim());
        }
        int64_t alias_workspace_size = 0;
        REQUIRE(alias_reloaded.get_workspace_size(alias_workspace_size).is_good());
        Surface<int8_t> alias_workspace(static_cast<size_t>(alias_workspace_size));
        std::unordered_map<int64_t, void*> input_only = {{X->get_uid(), X_gpu.devPtr}};
        REQUIRE(alias_reloaded.execute(handle, input_only, alias_workspace.devPtr).is_good());
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<float> alias_output(output_size);
        CUDA_CHECK(cudaMemcpy(
            alias_output.data(), alias_workspace.devPtr, alias_output.size() * sizeof(float), cudaMemcpyDeviceToHost));
        REQUIRE(alias_output == expected);
    }

    auto restored_handle_ptr = create_cudnn_handle();
    auto restored_handle     = *restored_handle_ptr;

    fe::graph::Graph restored_graph;
    REQUIRE(restored_graph
                .deserialize(restored_handle,
                             serialized_plan,
                             /*enforce_precompiled=*/true,
                             /*run_warmup=*/false)
                .is_good());

    Surface<float> restored_Y_gpu(output_size, 0.0f);
    std::unordered_map<int64_t, void*> restored_variant_pack = {{X->get_uid(), X_gpu.devPtr},
                                                                {Y->get_uid(), restored_Y_gpu.devPtr}};
    int64_t restored_workspace_size                          = 0;
    REQUIRE(restored_graph.get_workspace_size(restored_workspace_size).is_good());
    Surface<int8_t> restored_workspace(static_cast<size_t>(restored_workspace_size));
    REQUIRE(restored_graph.execute(restored_handle, restored_variant_pack, restored_workspace.devPtr).is_good());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> restored_output(output_size);
    CUDA_CHECK(cudaMemcpy(
        restored_output.data(), restored_Y_gpu.devPtr, restored_output.size() * sizeof(float), cudaMemcpyDeviceToHost));
    REQUIRE(restored_output == expected);
    REQUIRE(restored_output == original_output);
    REQUIRE(restored_output == built_in_output);
}

TEST_CASE("Custom reduction FP16 sum uses its default compute type across two axes",
          "[membound][custom_reduction][graph][fp16][multi_axis]") {
    if (auto const* reason = custom_reduction_runtime_skip_reason(); reason != nullptr) {
        SKIP(reason);
    }

    constexpr int64_t dim0        = 2;
    constexpr int64_t retained    = 3;
    constexpr int64_t dim2        = 4;
    constexpr int64_t input_size  = dim0 * retained * dim2;
    constexpr int64_t output_size = retained;

    fe::graph::Graph graph;
    graph.set_io_data_type(fe::DataType_t::HALF).set_compute_data_type(fe::DataType_t::HALF);
    auto X       = graph.tensor(fe::graph::Tensor_attributes()
                              .set_name("X_fp16")
                              .set_dim({dim0, retained, dim2})
                              .set_stride({retained * dim2, dim2, 1})
                              .set_data_type(fe::DataType_t::HALF));
    auto outputs = graph.custom_reduction(
        {X},
        fe::graph::reduction_modifiers::sum(
            fe::graph::Custom_reduction_attributes()
                .set_name("custom_sum_fp16_default_compute")
                .set_reduction_dims({0, 2})
                .set_identity_values({fe::graph::Custom_reduction_identity(cpu_float2half_rn(0.0f))})));
    auto Y = outputs.at(0);
    Y->set_name("Y_fp16")
        .set_dim({1, retained, 1})
        .set_stride({retained, 1, 1})
        .set_output(true)
        .set_data_type(fe::DataType_t::HALF);

    // The built-in reference uses its conventional FP32 accumulator. Every staged value and partial sum is an
    // exactly representable integer, so it remains an exact numeric oracle for the custom path's default FP16 body.
    fe::graph::Graph built_in_graph;
    built_in_graph.set_io_data_type(fe::DataType_t::HALF).set_compute_data_type(fe::DataType_t::FLOAT);
    auto built_in_X = built_in_graph.tensor(fe::graph::Tensor_attributes()
                                                .set_name("built_in_X_fp16")
                                                .set_dim({dim0, retained, dim2})
                                                .set_stride({retained * dim2, dim2, 1})
                                                .set_data_type(fe::DataType_t::HALF));
    auto built_in_Y = built_in_graph.reduction(built_in_X,
                                               fe::graph::Reduction_attributes()
                                                   .set_name("built_in_sum_fp16")
                                                   .set_mode(fe::ReductionMode_t::ADD)
                                                   .set_compute_data_type(fe::DataType_t::FLOAT));
    built_in_Y->set_name("built_in_Y_fp16")
        .set_dim({1, retained, 1})
        .set_stride({retained, 1, 1})
        .set_output(true)
        .set_data_type(fe::DataType_t::HALF);

    auto handle_ptr = create_cudnn_handle();
    auto handle     = *handle_ptr;
    build_supported_graph(graph, handle);
    build_supported_graph(built_in_graph, handle);

    std::vector<half> input(input_size);
    std::vector<float> expected(output_size, 0.0f);
    for (int64_t i = 0; i < dim0; ++i) {
        for (int64_t r = 0; r < retained; ++r) {
            for (int64_t k = 0; k < dim2; ++k) {
                auto const value = static_cast<float>((i * 3 + r * 2 + k) % 5 - 2);
                auto const index = static_cast<size_t>((i * retained + r) * dim2 + k);
                input[index]     = cpu_float2half_rn(value);
                expected[static_cast<size_t>(r)] += value;
            }
        }
    }

    Surface<half> X_gpu(input_size, cpu_float2half_rn(0.0f));
    Surface<half> Y_gpu(output_size, cpu_float2half_rn(0.0f));
    Surface<half> built_in_Y_gpu(output_size, cpu_float2half_rn(0.0f));
    CUDA_CHECK(cudaMemcpy(X_gpu.devPtr, input.data(), input.size() * sizeof(half), cudaMemcpyHostToDevice));

    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {{X, X_gpu.devPtr},
                                                                                             {Y, Y_gpu.devPtr}};
    int64_t workspace_size                                                                = 0;
    REQUIRE(graph.get_workspace_size(workspace_size).is_good());
    Surface<int8_t> workspace(static_cast<size_t>(workspace_size));
    REQUIRE(graph.execute(handle, variant_pack, workspace.devPtr).is_good());

    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> built_in_variant_pack = {
        {built_in_X, X_gpu.devPtr}, {built_in_Y, built_in_Y_gpu.devPtr}};
    int64_t built_in_workspace_size = 0;
    REQUIRE(built_in_graph.get_workspace_size(built_in_workspace_size).is_good());
    Surface<int8_t> built_in_workspace(static_cast<size_t>(built_in_workspace_size));
    REQUIRE(built_in_graph.execute(handle, built_in_variant_pack, built_in_workspace.devPtr).is_good());
    CUDA_CHECK(cudaDeviceSynchronize());

    auto const custom_output   = copy_from_device(Y_gpu.devPtr, static_cast<size_t>(output_size));
    auto const built_in_output = copy_from_device(built_in_Y_gpu.devPtr, static_cast<size_t>(output_size));
    for (size_t i = 0; i < static_cast<size_t>(output_size); ++i) {
        REQUIRE(__half2float(custom_output[i]) == expected[i]);
        REQUIRE(__half2float(built_in_output[i]) == expected[i]);
        REQUIRE(__half2float(custom_output[i]) == __half2float(built_in_output[i]));
    }
}

TEST_CASE("Custom reduction BF16 sum forwards padded non-contiguous three-axis input with FP32 compute",
          "[membound][custom_reduction][graph][bf16][multi_axis][padded]") {
    if (auto const* reason = custom_reduction_runtime_skip_reason(); reason != nullptr) {
        SKIP(reason);
    }

    constexpr int64_t dim0                = 2;
    constexpr int64_t retained            = 3;
    constexpr int64_t dim2                = 4;
    constexpr int64_t dim3                = 5;
    constexpr int64_t physical_input_size = 216;
    constexpr int64_t logical_output_size = retained;
    constexpr int64_t input_stride0       = 120;
    constexpr int64_t input_stride1       = 35;
    constexpr int64_t input_stride2       = 7;
    constexpr int64_t input_stride3       = 1;

    fe::graph::Graph graph;
    graph.set_io_data_type(fe::DataType_t::BFLOAT16).set_compute_data_type(fe::DataType_t::FLOAT);
    auto X       = graph.tensor(fe::graph::Tensor_attributes()
                              .set_name("X_bf16_padded")
                              .set_dim({dim0, retained, dim2, dim3})
                              .set_stride({input_stride0, input_stride1, input_stride2, input_stride3})
                              .set_data_type(fe::DataType_t::BFLOAT16));
    auto outputs = graph.custom_reduction(
        {X},
        fe::graph::reduction_modifiers::sum(
            fe::graph::Custom_reduction_attributes()
                .set_name("custom_sum_bf16_padded")
                .set_reduction_dims({0, 2, 3})
                .set_identity_values({fe::graph::Custom_reduction_identity(__float2bfloat16(0.0f))})
                .set_compute_data_types({fe::DataType_t::FLOAT})));
    auto Y = outputs.at(0);
    Y->set_name("Y_bf16_padded")
        .set_dim({1, retained, 1, 1})
        .set_stride({retained, 1, 1, 1})
        .set_output(true)
        .set_data_type(fe::DataType_t::BFLOAT16);

    fe::graph::Graph built_in_graph;
    built_in_graph.set_io_data_type(fe::DataType_t::BFLOAT16).set_compute_data_type(fe::DataType_t::FLOAT);
    auto built_in_X =
        built_in_graph.tensor(fe::graph::Tensor_attributes()
                                  .set_name("built_in_X_bf16_padded")
                                  .set_dim({dim0, retained, dim2, dim3})
                                  .set_stride({input_stride0, input_stride1, input_stride2, input_stride3})
                                  .set_data_type(fe::DataType_t::BFLOAT16));
    auto built_in_Y = built_in_graph.reduction(built_in_X,
                                               fe::graph::Reduction_attributes()
                                                   .set_name("built_in_sum_bf16_padded")
                                                   .set_mode(fe::ReductionMode_t::ADD)
                                                   .set_compute_data_type(fe::DataType_t::FLOAT));
    built_in_Y->set_name("built_in_Y_bf16_padded")
        .set_dim({1, retained, 1, 1})
        .set_stride({retained, 1, 1, 1})
        .set_output(true)
        .set_data_type(fe::DataType_t::BFLOAT16);

    auto handle_ptr = create_cudnn_handle();
    auto handle     = *handle_ptr;
    build_supported_graph(graph, handle);
    build_supported_graph(built_in_graph, handle);

    std::vector<nv_bfloat16> input(static_cast<size_t>(physical_input_size), __float2bfloat16(64.0f));
    std::vector<float> expected(static_cast<size_t>(logical_output_size), 0.0f);
    for (int64_t i = 0; i < dim0; ++i) {
        for (int64_t r = 0; r < retained; ++r) {
            for (int64_t k = 0; k < dim2; ++k) {
                for (int64_t l = 0; l < dim3; ++l) {
                    auto const value  = static_cast<float>((i * 5 + r * 3 + k * 2 + l) % 7 - 3);
                    auto const offset = static_cast<size_t>(i * input_stride0 + r * input_stride1 + k * input_stride2 +
                                                            l * input_stride3);
                    input[offset]     = __float2bfloat16(value);
                    expected[static_cast<size_t>(r)] += value;
                }
            }
        }
    }

    Surface<nv_bfloat16> X_gpu(physical_input_size, __float2bfloat16(64.0f));
    Surface<nv_bfloat16> Y_gpu(logical_output_size, __float2bfloat16(0.0f));
    Surface<nv_bfloat16> built_in_Y_gpu(logical_output_size, __float2bfloat16(0.0f));
    CUDA_CHECK(cudaMemcpy(X_gpu.devPtr, input.data(), input.size() * sizeof(nv_bfloat16), cudaMemcpyHostToDevice));

    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {{X, X_gpu.devPtr},
                                                                                             {Y, Y_gpu.devPtr}};
    int64_t workspace_size                                                                = 0;
    REQUIRE(graph.get_workspace_size(workspace_size).is_good());
    Surface<int8_t> workspace(static_cast<size_t>(workspace_size));
    REQUIRE(graph.execute(handle, variant_pack, workspace.devPtr).is_good());

    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> built_in_variant_pack = {
        {built_in_X, X_gpu.devPtr}, {built_in_Y, built_in_Y_gpu.devPtr}};
    int64_t built_in_workspace_size = 0;
    REQUIRE(built_in_graph.get_workspace_size(built_in_workspace_size).is_good());
    Surface<int8_t> built_in_workspace(static_cast<size_t>(built_in_workspace_size));
    REQUIRE(built_in_graph.execute(handle, built_in_variant_pack, built_in_workspace.devPtr).is_good());
    CUDA_CHECK(cudaDeviceSynchronize());

    auto const custom_output   = copy_from_device(Y_gpu.devPtr, static_cast<size_t>(logical_output_size));
    auto const built_in_output = copy_from_device(built_in_Y_gpu.devPtr, static_cast<size_t>(logical_output_size));
    for (size_t i = 0; i < static_cast<size_t>(logical_output_size); ++i) {
        auto const expected_bf16 = __float2bfloat16(expected[i]);
        REQUIRE(raw_bits(custom_output[i]) == raw_bits(expected_bf16));
        REQUIRE(raw_bits(built_in_output[i]) == raw_bits(expected_bf16));
        REQUIRE(raw_bits(custom_output[i]) == raw_bits(built_in_output[i]));
    }
}

TEST_CASE("Custom reduction INT32 sum uses default compute across every input axis",
          "[membound][custom_reduction][graph][int32][multi_axis]") {
    if (auto const* reason = custom_reduction_runtime_skip_reason(); reason != nullptr) {
        SKIP(reason);
    }

    constexpr int64_t dim0        = 2;
    constexpr int64_t dim1        = 3;
    constexpr int64_t dim2        = 2;
    constexpr int64_t dim3        = 4;
    constexpr int64_t input_size  = dim0 * dim1 * dim2 * dim3;
    constexpr int64_t output_size = 1;

    fe::graph::Graph graph;
    graph.set_io_data_type(fe::DataType_t::INT32).set_compute_data_type(fe::DataType_t::INT32);
    auto X = graph.tensor(fe::graph::Tensor_attributes()
                              .set_name("X_s32")
                              .set_dim({dim0, dim1, dim2, dim3})
                              .set_stride({dim1 * dim2 * dim3, dim2 * dim3, dim3, 1})
                              .set_data_type(fe::DataType_t::INT32));
    auto outputs =
        graph.custom_reduction({X},
                               fe::graph::reduction_modifiers::sum(
                                   fe::graph::Custom_reduction_attributes()
                                       .set_name("custom_sum_s32_full_rank")
                                       .set_reduction_dims({0, 1, 2, 3})
                                       .set_identity_values({fe::graph::Custom_reduction_identity(int32_t{0})})));
    auto Y = outputs.at(0);
    Y->set_name("Y_s32")
        .set_dim({1, 1, 1, 1})
        .set_stride({1, 1, 1, 1})
        .set_output(true)
        .set_data_type(fe::DataType_t::INT32);

    auto handle_ptr = create_cudnn_handle();
    auto handle     = *handle_ptr;
    build_supported_graph(graph, handle);

    constexpr int32_t pattern[] = {int32_t{-7}, int32_t{3}, int32_t{2147483}, int32_t{-1000000}};
    std::vector<int32_t> input(static_cast<size_t>(input_size));
    int32_t expected = 0;
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = pattern[i % 4];
        expected += input[i];
    }

    Surface<int32_t> X_gpu(input_size, int32_t{0});
    Surface<int32_t> Y_gpu(output_size, int32_t{0});
    CUDA_CHECK(cudaMemcpy(X_gpu.devPtr, input.data(), input.size() * sizeof(int32_t), cudaMemcpyHostToDevice));

    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {{X, X_gpu.devPtr},
                                                                                             {Y, Y_gpu.devPtr}};
    int64_t workspace_size                                                                = 0;
    REQUIRE(graph.get_workspace_size(workspace_size).is_good());
    Surface<int8_t> workspace(static_cast<size_t>(workspace_size));
    REQUIRE(graph.execute(handle, variant_pack, workspace.devPtr).is_good());
    CUDA_CHECK(cudaDeviceSynchronize());

    auto const output = copy_from_device(Y_gpu.devPtr, static_cast<size_t>(output_size));
    REQUIRE(output == std::vector<int32_t>{expected});
}

TEST_CASE("Custom reduction N=2 results funnel through the parent graph to one output",
          "[membound][custom_reduction][graph][multi_stream][funnel][serialization]") {
    if (auto const* reason = custom_reduction_runtime_skip_reason(); reason != nullptr) {
        SKIP(reason);
    }

    constexpr int64_t outer       = 2;
    constexpr int64_t reduced     = 4;
    constexpr int64_t inner       = 3;
    constexpr int64_t input_size  = outer * reduced * inner;
    constexpr int64_t output_size = outer * inner;

    fe::graph::Graph graph;
    graph.set_io_data_type(fe::DataType_t::FLOAT).set_compute_data_type(fe::DataType_t::FLOAT);
    auto X0 = graph.tensor(fe::graph::Tensor_attributes()
                               .set_name("X0_funnel")
                               .set_dim({outer, reduced, inner})
                               .set_stride({reduced * inner, inner, 1})
                               .set_data_type(fe::DataType_t::FLOAT));
    auto X1 = graph.tensor(fe::graph::Tensor_attributes()
                               .set_name("X1_funnel")
                               .set_dim({outer, reduced, inner})
                               .set_stride({reduced * inner, inner, 1})
                               .set_data_type(fe::DataType_t::FLOAT));

    auto reducer = [](fe::graph::Custom_reduction_attributes::Graph_t reducer_graph,
                      std::vector<fe::graph::Custom_reduction_attributes::Tensor_t> lhs,
                      std::vector<fe::graph::Custom_reduction_attributes::Tensor_t> rhs) {
        std::vector<fe::graph::Custom_reduction_attributes::Tensor_t> reduced_values;
        reduced_values.reserve(lhs.size());
        for (size_t i = 0; i < lhs.size(); ++i) {
            auto value = reducer_graph->pointwise(lhs.at(i),
                                                  rhs.at(i),
                                                  fe::graph::Pointwise_attributes()
                                                      .set_name("funnel_stream_add" + std::to_string(i))
                                                      .set_mode(fe::PointwiseMode_t::ADD)
                                                      .set_compute_data_type(fe::DataType_t::FLOAT));
            value->set_data_type(fe::DataType_t::FLOAT).set_dim({1}).set_stride({1});
            reduced_values.emplace_back(std::move(value));
        }
        return reduced_values;
    };

    auto outputs =
        graph.custom_reduction({X0, X1},
                               fe::graph::Custom_reduction_attributes()
                                   .set_name("custom_sum_two_stream_funnel")
                                   .set_reducer(std::move(reducer), "samples.custom_reduction.two_stream_sum.v1")
                                   .set_reduction_dims({1})
                                   .set_identity_values({fe::graph::Custom_reduction_identity(0.0f),
                                                         fe::graph::Custom_reduction_identity(0.0f)})
                                   .set_compute_data_types({fe::DataType_t::FLOAT, fe::DataType_t::FLOAT}));
    auto Y0 = outputs.at(0);
    auto Y1 = outputs.at(1);
    Y0->set_name("Y0_funnel")
        .set_dim({outer, 1, inner})
        .set_stride({inner, inner, 1})
        .set_data_type(fe::DataType_t::FLOAT);
    Y1->set_name("Y1_funnel")
        .set_dim({outer, 1, inner})
        .set_stride({inner, inner, 1})
        .set_data_type(fe::DataType_t::FLOAT);
    REQUIRE(Y0->get_is_virtual());
    REQUIRE(Y1->get_is_virtual());

    auto Z = graph.pointwise(Y0,
                             Y1,
                             fe::graph::Pointwise_attributes()
                                 .set_name("combine_funnel_results")
                                 .set_mode(fe::PointwiseMode_t::ADD)
                                 .set_compute_data_type(fe::DataType_t::FLOAT));
    Z->set_name("Z_funnel")
        .set_dim({outer, 1, inner})
        .set_stride({inner, inner, 1})
        .set_output(true)
        .set_data_type(fe::DataType_t::FLOAT);
    REQUIRE_FALSE(Z->get_is_virtual());

    auto handle_ptr = create_cudnn_handle();
    auto handle     = *handle_ptr;
    build_supported_graph(graph, handle);

    std::vector<float> input0(static_cast<size_t>(input_size));
    std::vector<float> input1(static_cast<size_t>(input_size));
    std::vector<float> expected(static_cast<size_t>(output_size), 0.0f);
    for (int64_t i = 0; i < outer; ++i) {
        for (int64_t r = 0; r < reduced; ++r) {
            for (int64_t k = 0; k < inner; ++k) {
                auto const index = static_cast<size_t>((i * reduced + r) * inner + k);
                input0[index]    = static_cast<float>((i + 2 * r + k) % 7 - 3);
                input1[index]    = static_cast<float>((3 * i + r + 2 * k) % 5 - 2);
                expected[static_cast<size_t>(i * inner + k)] += input0[index] + input1[index];
            }
        }
    }

    Surface<float> X0_gpu(input_size, 0.0f);
    Surface<float> X1_gpu(input_size, 0.0f);
    Surface<float> Z_gpu(output_size, 0.0f);
    CUDA_CHECK(cudaMemcpy(X0_gpu.devPtr, input0.data(), input0.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(X1_gpu.devPtr, input1.data(), input1.size() * sizeof(float), cudaMemcpyHostToDevice));

    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
        {X0, X0_gpu.devPtr}, {X1, X1_gpu.devPtr}, {Z, Z_gpu.devPtr}};
    int64_t workspace_size = 0;
    REQUIRE(graph.get_workspace_size(workspace_size).is_good());
    Surface<int8_t> workspace(static_cast<size_t>(workspace_size));
    REQUIRE(graph.execute(handle, variant_pack, workspace.devPtr).is_good());
    CUDA_CHECK(cudaDeviceSynchronize());

    auto const output = copy_from_device(Z_gpu.devPtr, static_cast<size_t>(output_size));
    REQUIRE(output == expected);

    std::vector<uint8_t> serialized_plan;
    REQUIRE(graph.serialize(serialized_plan).is_good());
    REQUIRE_FALSE(serialized_plan.empty());

    auto restored_handle_ptr = create_cudnn_handle();
    auto restored_handle     = *restored_handle_ptr;
    fe::graph::Graph restored_graph;
    REQUIRE(restored_graph
                .deserialize(restored_handle,
                             serialized_plan,
                             /*enforce_precompiled=*/true,
                             /*run_warmup=*/false)
                .is_good());

    Surface<float> restored_Z_gpu(output_size, 0.0f);
    std::unordered_map<int64_t, void*> restored_variant_pack = {
        {X0->get_uid(), X0_gpu.devPtr}, {X1->get_uid(), X1_gpu.devPtr}, {Z->get_uid(), restored_Z_gpu.devPtr}};
    int64_t restored_workspace_size = 0;
    REQUIRE(restored_graph.get_workspace_size(restored_workspace_size).is_good());
    Surface<int8_t> restored_workspace(static_cast<size_t>(restored_workspace_size));
    REQUIRE(restored_graph.execute(restored_handle, restored_variant_pack, restored_workspace.devPtr).is_good());
    CUDA_CHECK(cudaDeviceSynchronize());

    auto const restored_output = copy_from_device(restored_Z_gpu.devPtr, static_cast<size_t>(output_size));
    REQUIRE(restored_output == output);
    REQUIRE(restored_output == expected);
}

TEST_CASE("Custom reduction executes a FLOAT RELU_FWD body with the default upper clip",
          "[membound][custom_reduction][graph][relu]") {
    if (auto const* reason = custom_reduction_runtime_skip_reason(); reason != nullptr) {
        SKIP(reason);
    }

    constexpr int64_t outer       = 4;
    constexpr int64_t reduced     = 64;
    constexpr int64_t inner       = 8;
    constexpr int64_t input_size  = outer * reduced * inner;
    constexpr int64_t output_size = outer * inner;

    fe::graph::Graph graph;
    graph.set_io_data_type(fe::DataType_t::FLOAT).set_compute_data_type(fe::DataType_t::FLOAT);
    auto X = graph.tensor(fe::graph::Tensor_attributes()
                              .set_name("X_relu_max")
                              .set_dim({outer, reduced, inner})
                              .set_stride({reduced * inner, inner, 1})
                              .set_data_type(fe::DataType_t::FLOAT));

    // max(lhs, rhs) written as rhs + relu(lhs - rhs). Without RELU_FWD the body returns lhs, so the result would
    // stay at the identity. -1000 is an identity on either side because the data stay in [-25, 25].
    auto reducer = [](fe::graph::Custom_reduction_attributes::Graph_t reducer_graph,
                      std::vector<fe::graph::Custom_reduction_attributes::Tensor_t> lhs,
                      std::vector<fe::graph::Custom_reduction_attributes::Tensor_t> rhs) {
        auto difference = reducer_graph->pointwise(lhs.at(0),
                                                   rhs.at(0),
                                                   fe::graph::Pointwise_attributes()
                                                       .set_name("relu_max_sub")
                                                       .set_mode(fe::PointwiseMode_t::SUB)
                                                       .set_compute_data_type(fe::DataType_t::FLOAT));
        difference->set_data_type(fe::DataType_t::FLOAT).set_dim({1}).set_stride({1});
        auto excess = reducer_graph->pointwise(difference,
                                               fe::graph::Pointwise_attributes()
                                                   .set_name("relu_max_relu")
                                                   .set_mode(fe::PointwiseMode_t::RELU_FWD)
                                                   .set_compute_data_type(fe::DataType_t::FLOAT));
        excess->set_data_type(fe::DataType_t::FLOAT).set_dim({1}).set_stride({1});
        auto result = reducer_graph->pointwise(rhs.at(0),
                                               excess,
                                               fe::graph::Pointwise_attributes()
                                                   .set_name("relu_max_add")
                                                   .set_mode(fe::PointwiseMode_t::ADD)
                                                   .set_compute_data_type(fe::DataType_t::FLOAT));
        result->set_data_type(fe::DataType_t::FLOAT).set_dim({1}).set_stride({1});
        return std::vector<fe::graph::Custom_reduction_attributes::Tensor_t>{std::move(result)};
    };

    auto outputs = graph.custom_reduction({X},
                                          fe::graph::Custom_reduction_attributes()
                                              .set_name("custom_relu_max")
                                              .set_reducer(std::move(reducer), "samples.custom_reduction.relu_max.v1")
                                              .set_reduction_dims({1})
                                              .set_identity_values({fe::graph::Custom_reduction_identity(-1000.0f)})
                                              .set_compute_data_types({fe::DataType_t::FLOAT}));
    auto Y       = outputs.at(0);
    Y->set_name("Y_relu_max")
        .set_dim({outer, 1, inner})
        .set_stride({inner, inner, 1})
        .set_output(true)
        .set_data_type(fe::DataType_t::FLOAT);

    auto handle_ptr = create_cudnn_handle();
    auto handle     = *handle_ptr;
    build_supported_graph(graph, handle);

    // Multiples of 0.5 in [-25, 25], so every difference and sum is exact. Column (0, 0) is all negative, so its
    // maximum is negative too.
    std::vector<float> input(static_cast<size_t>(input_size));
    std::vector<float> expected(static_cast<size_t>(output_size), -1000.0f);
    for (int64_t i = 0; i < outer; ++i) {
        for (int64_t r = 0; r < reduced; ++r) {
            for (int64_t k = 0; k < inner; ++k) {
                auto const index = static_cast<size_t>((i * reduced + r) * inner + k);
                auto value       = static_cast<float>((37 * i + 11 * r + 5 * k) % 101) * 0.5f - 25.0f;
                if (i == 0 && k == 0) {
                    value = -1.0f - static_cast<float>(r % 40) * 0.5f;
                }
                input[index]  = value;
                auto& maximum = expected[static_cast<size_t>(i * inner + k)];
                maximum       = std::max(maximum, value);
            }
        }
    }

    Surface<float> X_gpu(input_size, 0.0f);
    Surface<float> Y_gpu(output_size, 0.0f);
    CUDA_CHECK(cudaMemcpy(X_gpu.devPtr, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice));

    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {{X, X_gpu.devPtr},
                                                                                             {Y, Y_gpu.devPtr}};
    int64_t workspace_size                                                                = 0;
    REQUIRE(graph.get_workspace_size(workspace_size).is_good());
    Surface<int8_t> workspace(static_cast<size_t>(workspace_size));
    REQUIRE(graph.execute(handle, variant_pack, workspace.devPtr).is_good());
    CUDA_CHECK(cudaDeviceSynchronize());

    auto const output = copy_from_device(Y_gpu.devPtr, static_cast<size_t>(output_size));
    REQUIRE(output == expected);
}

TEST_CASE("Custom reduction executes a body with a compile-time constant",
          "[membound][custom_reduction][graph][constant]") {
    if (auto const* reason = custom_reduction_runtime_skip_reason(); reason != nullptr) {
        SKIP(reason);
    }

    constexpr int64_t outer       = 4;
    constexpr int64_t reduced     = 64;
    constexpr int64_t inner       = 8;
    constexpr int64_t input_size  = outer * reduced * inner;
    constexpr int64_t output_size = outer * inner;
    constexpr float lowest        = -std::numeric_limits<float>::infinity();

    fe::graph::Graph graph;
    graph.set_io_data_type(fe::DataType_t::FLOAT)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);
    auto X = graph.tensor(fe::graph::Tensor_attributes()
                              .set_name("X_constant_max")
                              .set_dim({outer, reduced, inner})
                              .set_stride({reduced * inner, inner, 1})
                              .set_data_type(fe::DataType_t::FLOAT));

    // max(max(lhs, rhs), c) with the constant c = -inf. -inf is the identity of max, so it is a two-sided identity of
    // this body for every input, as the reduction contract requires.
    auto reducer = [lowest](fe::graph::Custom_reduction_attributes::Graph_t reducer_graph,
                            std::vector<fe::graph::Custom_reduction_attributes::Tensor_t> lhs,
                            std::vector<fe::graph::Custom_reduction_attributes::Tensor_t> rhs) {
        auto larger   = reducer_graph->pointwise(lhs.at(0),
                                               rhs.at(0),
                                               fe::graph::Pointwise_attributes()
                                                   .set_name("constant_max_max")
                                                   .set_mode(fe::PointwiseMode_t::MAX)
                                                   .set_compute_data_type(fe::DataType_t::FLOAT));
        auto constant = reducer_graph->tensor(lowest, fe::graph::ScalarType::COMPILE_TIME_CONST);
        auto result   = reducer_graph->pointwise(larger,
                                               constant,
                                               fe::graph::Pointwise_attributes()
                                                   .set_name("constant_max_constant")
                                                   .set_mode(fe::PointwiseMode_t::MAX)
                                                   .set_compute_data_type(fe::DataType_t::FLOAT));
        return std::vector<fe::graph::Custom_reduction_attributes::Tensor_t>{std::move(result)};
    };

    auto outputs =
        graph.custom_reduction({X},
                               fe::graph::Custom_reduction_attributes()
                                   .set_name("custom_constant_max")
                                   .set_reducer(std::move(reducer), "samples.custom_reduction.constant_max.v1")
                                   .set_reduction_dims({1})
                                   .set_identity_values({fe::graph::Custom_reduction_identity(lowest)}));
    auto Y = outputs.at(0);
    Y->set_name("Y_constant_max").set_dim({outer, 1, inner}).set_stride({inner, inner, 1}).set_output(true);

    auto handle_ptr = create_cudnn_handle();
    auto handle     = *handle_ptr;
    build_supported_graph(graph, handle);

    // Multiples of 0.5 in [-25, 25]. Column (0, 0) stays at -11 or below, so a wrong constant above -11 changes its
    // result.
    std::vector<float> input(static_cast<size_t>(input_size));
    std::vector<float> expected(static_cast<size_t>(output_size), lowest);
    for (int64_t i = 0; i < outer; ++i) {
        for (int64_t r = 0; r < reduced; ++r) {
            for (int64_t k = 0; k < inner; ++k) {
                auto const index = static_cast<size_t>((i * reduced + r) * inner + k);
                auto value       = static_cast<float>((37 * i + 11 * r + 5 * k) % 101) * 0.5f - 25.0f;
                if (i == 0 && k == 0) {
                    value = -11.0f - static_cast<float>(r % 20) * 0.5f;
                }
                input[index]  = value;
                auto& maximum = expected[static_cast<size_t>(i * inner + k)];
                maximum       = std::max(maximum, value);
            }
        }
    }
    REQUIRE(expected.at(0) == -11.0f);

    Surface<float> X_gpu(input_size, 0.0f);
    Surface<float> Y_gpu(output_size, 0.0f);
    CUDA_CHECK(cudaMemcpy(X_gpu.devPtr, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice));

    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {{X, X_gpu.devPtr},
                                                                                             {Y, Y_gpu.devPtr}};
    int64_t workspace_size                                                                = 0;
    REQUIRE(graph.get_workspace_size(workspace_size).is_good());
    Surface<int8_t> workspace(static_cast<size_t>(workspace_size));
    REQUIRE(graph.execute(handle, variant_pack, workspace.devPtr).is_good());
    CUDA_CHECK(cudaDeviceSynchronize());

    auto const output = copy_from_device(Y_gpu.devPtr, static_cast<size_t>(output_size));
    REQUIRE(output == expected);
}
