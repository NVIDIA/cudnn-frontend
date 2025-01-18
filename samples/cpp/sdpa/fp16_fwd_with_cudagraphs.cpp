/*
 * Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF  MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include "../utils/helpers.h"
#include <catch2/catch_test_macros.hpp>

#include <cuda_runtime_api.h>

#include <cudnn_frontend.h>
namespace fe = cudnn_frontend;

/*
Run this example by using command:
bin/samples "Toy sdpa forward as CUDA graph"

This example shows how to construct a sdpa forward graph
as a CUDA graph, then instantiate and execute it in a simple way.

For an example showing how to construct the CUDA graph as a
child of a larger CUDA graph, see ../misc/cudagraphs.cpp.
*/

// Tensors in forward pass
#define Q_UID 1
#define K_UID 2
#define V_UID 3
#define O_UID 4
#define STATS_UID 5
#define BIAS_UID 6
#define SEQ_LEN_Q_UID 7
#define SEQ_LEN_KV_UID 8

// Declare the function from fp16_fwd.cpp
std::shared_ptr<fe::graph::Graph>
create_sdpa_forward_graph(int64_t const b,
                          int64_t const h_q,
                          int64_t const h_k,
                          int64_t const h_v,
                          int64_t const s_q,
                          int64_t const s_kv,
                          int64_t const d_qk,
                          int64_t const d_v,
                          float const attn_scale  = 1.0f,
                          bool const is_inference = false,
                          bool const causal_mask  = false,
                          bool const alibi_mask   = false,
                          bool const padding_mask = false,
                          bool has_attn_bias      = false);

// Convenience class to encapsulate SDPA test data for this example
class SdpaTestData {
   public:
    SdpaTestData(int64_t const b,
                 int64_t const h_q,
                 int64_t const h_k,
                 int64_t const h_v,
                 int64_t const s_q,
                 int64_t const s_kv,
                 int64_t const d_qk,
                 int64_t const d_v,
                 int64_t const workspace_size,
                 bool const is_inference,
                 bool const padding_mask,
                 bool const has_attn_bias,
                 float const qkv_fill_value)
        : q_tensor(b * h_q * s_q * d_qk, false, cpu_float2half_rn(qkv_fill_value)),
          k_tensor(b * h_k * d_qk * s_kv, false, cpu_float2half_rn(qkv_fill_value)),
          v_tensor(b * h_v * d_v * s_kv, false, cpu_float2half_rn(qkv_fill_value)),
          o_tensor(b * s_q * h_q * d_qk, false),
          bias_tensor(b * 1 * s_q * s_kv, false),
          devActualSeqlenQ(b, false, /*fillValue=*/20),
          devActualSeqlenKV(b, false, /*fillValue=*/20),
          statsTensor(b * h_q * s_q * 1, false),
          workspace(workspace_size, false),
          is_inference_(is_inference),
          padding_mask_(padding_mask),
          has_attn_bias_(has_attn_bias) {}

    std::unordered_map<fe::graph::Tensor_attributes::uid_t, void *>
    build_variant_pack() {
        std::unordered_map<fe::graph::Tensor_attributes::uid_t, void *> variant_pack;
        variant_pack[Q_UID] = q_tensor.devPtr;
        variant_pack[K_UID] = k_tensor.devPtr;
        variant_pack[V_UID] = v_tensor.devPtr;
        variant_pack[O_UID] = o_tensor.devPtr;
        if (has_attn_bias_) {
            variant_pack[BIAS_UID] = bias_tensor.devPtr;
        }
        if (padding_mask_) {
            variant_pack[SEQ_LEN_Q_UID]  = devActualSeqlenQ.devPtr;
            variant_pack[SEQ_LEN_KV_UID] = devActualSeqlenKV.devPtr;
        }
        if (is_inference_ == false) {
            variant_pack[STATS_UID] = statsTensor.devPtr;
        }
        return variant_pack;
    }

    void *
    get_workspace_ptr() {
        return workspace.devPtr;
    }

    void
    sync_outputs() {
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(
            o_tensor.hostPtr, o_tensor.devPtr, sizeof(o_tensor.hostPtr[0]) * o_tensor.n_elems, cudaMemcpyDeviceToHost));
        if (is_inference_ == false) {
            CUDA_CHECK(cudaMemcpy(statsTensor.hostPtr,
                                  statsTensor.devPtr,
                                  sizeof(statsTensor.hostPtr[0]) * statsTensor.n_elems,
                                  cudaMemcpyDeviceToHost));
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    template <typename T>
    bool
    equal_tensors(Surface<T> &a, Surface<T> &b) {
        REQUIRE(a.n_elems == b.n_elems);
        for (int i = 0; i < a.n_elems; i++) {
            if (a.hostPtr[i] != b.hostPtr[i]) {
                return false;
            }
        }
        return true;
    }

    bool
    equal_outputs(SdpaTestData &other) {
        REQUIRE(is_inference_ == other.is_inference_);
        sync_outputs();
        other.sync_outputs();
        if (!equal_tensors(o_tensor, other.o_tensor)) return false;
        if (!is_inference_ && !equal_tensors(statsTensor, other.statsTensor)) return false;
        return true;
    }

   private:
    Surface<half> q_tensor;
    Surface<half> k_tensor;
    Surface<half> v_tensor;
    Surface<half> o_tensor;
    Surface<half> bias_tensor;
    Surface<int32_t> devActualSeqlenQ;
    Surface<int32_t> devActualSeqlenKV;
    Surface<float> statsTensor;
    Surface<int8_t> workspace;
    bool is_inference_;
    bool padding_mask_;
    bool has_attn_bias_;
};

TEST_CASE("Toy sdpa forward as CUDA graph", "[graph][sdpa][flash][forward][cudagraph]") {
    // cuDNN only supports native CUDA graphs in CUDA 12.0 and above.
    // Because the below test depends on some CUDA graph APIs that changed
    // between CUDA 11.x and 12.0, it wouldn't even compile in <12.0 anyway,
    // so we just disable the whole test by #if in that case.
#if (CUDART_VERSION < 12000)
    SKIP("Test requires cuda toolkit 12.0 or above");
#else
    // Also check the CUDA version at runtime, for good measure.
    if (cudnnGetCudartVersion() < 12000) {
        SKIP("Test requires cuda toolkit 12.0 or above");
    }

    // cuDNN only supports native CUDA graphs for sdpa in 9.6 or above.
    if (cudnnGetVersion() < 90600) {
        SKIP("Test requires cudnn 9.6.0 or above");
        return;
    }

    int64_t b          = 3;     // batch size
    int64_t h_q        = 4;     // head dim
    int64_t h_k        = 4;     // head dim
    int64_t h_v        = 4;     // head dim
    int64_t s_q        = 1024;  // q tensor is padded to this seq length
    int64_t s_kv       = 1024;  // k and v tensor is padded to this seq length
    int64_t d_qk       = 128;   // hidden dim
    int64_t d_v        = 128;   // hidden dim
    bool is_inference  = false;
    float attn_scale   = 0.123f;
    bool causal_mask   = true;
    bool padding_mask  = (cudnnGetVersion() >= 8903);
    bool alibi_mask    = false;  // TODO: (cudnnGetVersion() >= 8904)
    bool has_attn_bias = (cudnnGetVersion() >= 8903);

    // Create a unique_ptr for the cuDNN handle
    auto handle_ptr = create_cudnn_handle();
    auto handle     = *handle_ptr;

    auto graph = create_sdpa_forward_graph(b,
                                           h_q,
                                           h_k,
                                           h_v,
                                           s_q,
                                           s_kv,
                                           d_qk,
                                           d_v,
                                           attn_scale,
                                           is_inference,
                                           causal_mask,
                                           alibi_mask,
                                           padding_mask,
                                           has_attn_bias);

    // Validate the graph and lower the FE graph to BE graph
    REQUIRE(graph->validate().is_good());
    REQUIRE(graph->build_operation_graph(handle).is_good());
    REQUIRE(graph->create_execution_plans({cudnn_frontend::HeurMode_t::A}).is_good());

    // Make sure the selected execution plan supports cuda graph
    graph->select_behavior_notes({cudnn_frontend::BehaviorNote_t::SUPPORTS_CUDA_GRAPH_NATIVE_API});
    REQUIRE(graph->check_support(handle).is_good());
    REQUIRE(graph->build_plans(handle).is_good());

    int64_t workspace_size = 0;
    REQUIRE(graph->get_workspace_size(workspace_size).is_good());

    //// Create a CUDA graph.
    // The responsibility to destroy is on the user.
    cudaGraph_t cudnn_cuda_graph;
    CUDA_CHECK(cudaGraphCreate(&cudnn_cuda_graph, 0));  // 0 is just what the API says to pass

    // Create the first variant pack.
    SdpaTestData test_data_1(b,
                             h_q,
                             h_k,
                             h_v,
                             s_q,
                             s_kv,
                             d_qk,
                             d_v,
                             workspace_size,
                             is_inference,
                             padding_mask,
                             has_attn_bias,
                             /*fillValue_qkv=*/1.1f);
    auto variant_pack_1 = test_data_1.build_variant_pack();

    // Populate and instantiate the graph, then launch it.
    REQUIRE(graph->populate_cuda_graph(handle, variant_pack_1, test_data_1.get_workspace_ptr(), cudnn_cuda_graph)
                .is_good());
    cudaGraphExec_t cuda_graph_exec;
    CUDA_CHECK(cudaGraphInstantiate(&cuda_graph_exec, cudnn_cuda_graph, NULL, NULL, 0));
    CUDA_CHECK(cudaGraphLaunch(cuda_graph_exec, 0));

    // Functional correctness:
    // Execute the SDPA directly and check that the results are the same as using a CUDA graph.
    SdpaTestData test_data_2(test_data_1);
    auto variant_pack_2 = test_data_2.build_variant_pack();
    REQUIRE(graph->execute(handle, variant_pack_2, test_data_2.get_workspace_ptr()).is_good());
    REQUIRE(test_data_1.equal_outputs(test_data_2));

    // Update the existing CUDA graph with different data.
    SdpaTestData test_data_3(b,
                             h_q,
                             h_k,
                             h_v,
                             s_q,
                             s_kv,
                             d_qk,
                             d_v,
                             workspace_size,
                             is_inference,
                             padding_mask,
                             has_attn_bias,
                             /*fillValue_qkv=*/1.3f);
    auto variant_pack_3 = test_data_3.build_variant_pack();
    REQUIRE(
        graph->update_cuda_graph(handle, variant_pack_3, test_data_3.get_workspace_ptr(), cudnn_cuda_graph).is_good());
    cudaGraphExecUpdateResultInfo result_info;
    CUDA_CHECK(cudaGraphExecUpdate(cuda_graph_exec, cudnn_cuda_graph, &result_info));
    CUDA_CHECK(cudaGraphLaunch(cuda_graph_exec, 0));

    // Functional correctness:
    // Execute the SDPA directly and check that the results are the same as using a CUDA graph.
    SdpaTestData test_data_4(test_data_3);
    auto variant_pack_4 = test_data_4.build_variant_pack();
    REQUIRE(graph->execute(handle, variant_pack_4, test_data_4.get_workspace_ptr()).is_good());
    REQUIRE(test_data_3.equal_outputs(test_data_4));

    // Because original and updated graph have different inputs, their outputs should *not* match
    REQUIRE(!test_data_1.equal_outputs(test_data_3));

    //// Cleanup
    CUDA_CHECK(cudaGraphExecDestroy(cuda_graph_exec));
    CUDA_CHECK(cudaGraphDestroy(cudnn_cuda_graph));
#endif  // CUDART_VERSION < 12000
}
