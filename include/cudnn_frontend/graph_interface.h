#pragma once

#include <unordered_map>
#include <string>

#include "../cudnn_frontend_version.h"
#include "node/batchnorm.h"
#include "node/batchnorm_inference.h"
#include "node/bn_finalize.h"
#include "node/conv_fprop.h"
#include "node/conv_dgrad.h"
#include "node/conv_wgrad.h"
#include "node/dbn.h"
#include "node/dln.h"
#include "node/dbn_weight.h"
#include "node/genstats.h"
#include "node/layernorm.h"
#include "node/instancenorm.h"
#include "node/rmsnorm.h"
#include "node/resample.h"
#include "node/reshape.h"
#include "node/slice.h"
// #include "node/scaled_dot_product_attention.h"
#include "node/scaled_dot_product_flash_attention.h"
#include "node/sdpa_fp8.h"
#include "node/sdpa_fp8_bwd.h"
#include "node/block_scale_quantize.h"
#include "node/block_scale_dequantize.h"
#include "node/concatenate.h"

#include "backend/backend_descriptor.h"
#include "plans.h"
#include "knobs.h"
#include "graph_helpers.h"
#include "backend/kernel_cache.h"

namespace cudnn_frontend::graph {

class Graph : public ICudnn, public INode {
   private:
    std::unordered_set<std::shared_ptr<Tensor_attributes>> full_graph_inputs;
    std::unordered_set<Tensor_attributes::uid_t> used_uids;
    int64_t fe_workspace_size = 0;

    std::unordered_set<std::shared_ptr<Tensor_attributes>> deserialized_tensor_properties;
    std::unordered_map<uid_t, pass_by_values_t> deserialized_pass_by_value;
    std::unordered_map<uid_t, std::tuple<int64_t, int64_t, std::vector<float>>> deserialized_workspace_modifications;

    error_t
    get_pre_assigned_uids(std::unordered_set<Tensor_attributes::uid_t> &used_uids) {
        for (auto const &input : full_graph_inputs) {
            if (input->has_uid()) {
                auto uid  = input->get_uid();
                auto iter = used_uids.find(uid);
                RETURN_CUDNN_FRONTEND_ERROR_IF(iter != used_uids.end(),
                                               error_code_t::INVALID_VALUE,
                                               "uid " + std::to_string(uid) + " for tensor named " + input->get_name() +
                                                   " has been already assigned to another tensor.");
                used_uids.insert(uid);
            }
        }
        for (auto const &output : full_graph_outputs) {
            if (output->has_uid()) {
                auto uid  = output->get_uid();
                auto iter = used_uids.find(uid);
                RETURN_CUDNN_FRONTEND_ERROR_IF(iter != used_uids.end(),
                                               error_code_t::INVALID_VALUE,
                                               "uid " + std::to_string(uid) + " for tensor named " +
                                                   output->get_name() +
                                                   " has been already assigned to another tensor.");
                used_uids.insert(uid);
            }
        }

        return {error_code_t::OK, ""};
    }

    error_t
    pre_validate_node() const override final {
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            (is_dynamic_shape_enabled || kernel_cache != nullptr) && detail::get_backend_version() < 90400,
            error_code_t::GRAPH_NOT_SUPPORTED,
            "Dynamic shapes or kernel caching enabled, but cuDNN version < 9.4!");
        RETURN_CUDNN_FRONTEND_ERROR_IF(((is_dynamic_shape_enabled == false) && (kernel_cache != nullptr)),
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "Kernel caching enabled but dynamic shapes is disabled");
        if (detail::get_backend_version() != detail::get_compiled_version()) {
            CUDNN_FE_LOG_LABEL_ENDL("INFO: The cuDNN version used at compilation ("
                                    << detail::get_compiled_version() << ") and the one used at runtime ("
                                    << detail::get_backend_version() << ") differ.");
        }
        return {error_code_t::OK, ""};
    }

    error_t
    infer_properties_node() override final {
        return {error_code_t::OK, ""};
    }

    error_t
    post_validate_node() const override final {
        return {error_code_t::OK, ""};
    }

    virtual error_t
    collect_pass_by_value_tensors_node(
        std::unordered_map<uid_t, pass_by_values_t> &pass_by_values) const override final {
        for (auto [uid, value] : deserialized_pass_by_value) {
            pass_by_values.emplace(uid, value);
        }
        return {error_code_t::OK, ""};
    }

    virtual error_t
    collect_tensors_in_workspace_node(
        std::unordered_map<Tensor_attributes::uid_t, std::tuple<int64_t, int64_t, std::vector<float>>>
            &worskspace_modifications,
        int64_t &) const override {
        for (auto [uid, value] : deserialized_workspace_modifications) {
            worskspace_modifications.emplace(uid, value);
        }
        return {error_code_t::OK, ""};
    }

    virtual error_t
    create_cudnn_tensors_node(std::unordered_map<int64_t, std::shared_ptr<cudnn_frontend::Tensor>> &,
                              int64_t &,
                              std::unordered_set<int64_t> const &) const override final {
        return {error_code_t::OK, ""};
    }

    error_t
    extend_tensor_map_with_workspace_tensors_(
        std::unordered_map<int64_t, void *> &tensor_to_pointer_map,
        void *workspace,
        std::unordered_map<uid_t, std::tuple<int64_t, int64_t, std::vector<float>>> const &worskspace_modifications)
        const {
        for (auto const &[uid, data] : worskspace_modifications) {
            tensor_to_pointer_map.emplace(uid, static_cast<char *>(workspace) + std::get<1>(data));
        }
        return {error_code_t::OK, ""};
    }

    error_t
    extend_tensor_map_with_pass_by_value_tensors_(
        std::unordered_map<int64_t, void *> &tensor_to_pointer_map,
        std::unordered_map<uid_t, pass_by_values_t> &tensor_to_pass_by_value) const {
        for (auto &[uid, value] : tensor_to_pass_by_value) {
            if (half *half_value_ptr = std::get_if<half>(&value)) {
                tensor_to_pointer_map.emplace(uid, half_value_ptr);
            } else if (nv_bfloat16 *nv_bfloat16_value_ptr = std::get_if<nv_bfloat16>(&value)) {
                tensor_to_pointer_map.emplace(uid, nv_bfloat16_value_ptr);
            } else if (int32_t *int32_t_value_ptr = std::get_if<int32_t>(&value)) {
                tensor_to_pointer_map.emplace(uid, int32_t_value_ptr);
            } else if (int64_t *int64_t_value_ptr = std::get_if<int64_t>(&value)) {
                tensor_to_pointer_map.emplace(uid, int64_t_value_ptr);
            } else if (float *float_value_ptr = std::get_if<float>(&value)) {
                tensor_to_pointer_map.emplace(uid, float_value_ptr);
            } else {
                RETURN_CUDNN_FRONTEND_ERROR_IF(
                    true, error_code_t::INVALID_VARIANT_PACK, "Unexpected type for pass by value tensor.");
            }
        }
        return {error_code_t::OK, ""};
    }

    error_t
    make_variant_pack_replacements(
        std::unordered_map<int64_t, void *> &tensor_to_pointer_map,
        std::unordered_map<Tensor_attributes::uid_t, std::pair<Tensor_attributes::uid_t, int64_t>> replacements) const {
        for (auto &[from_uid, value] : replacements) {
            const auto &[to_uid, start_offset] = value;

            // Check if from_uid exists in the map
            auto it = tensor_to_pointer_map.find(from_uid);
            RETURN_CUDNN_FRONTEND_ERROR_IF(it == tensor_to_pointer_map.end(),
                                           error_code_t::INVALID_VARIANT_PACK,
                                           "Variant pack expected uid " + std::to_string(from_uid) + " but not found.");

            // Perform pointer arithmetic
            tensor_to_pointer_map[to_uid] = static_cast<void *>(static_cast<char *>(it->second) + start_offset);
        }
        return {error_code_t::OK, ""};
    }

    int64_t
    get_max_cudnn_workspace_size() const {
        return get_max_cudnn_workspace_size_node();
    }

    // Key: uid to replace in variant pack
    // Value: uid to replace with, start offset to add to pointer
    std::unordered_map<Tensor_attributes::uid_t, std::pair<Tensor_attributes::uid_t, int64_t>>
        variant_pack_replacements;

    error_t
    run_auxiliary_kernels(
        cudnnHandle_t handle,
        void *fe_workspace,
        std::unordered_map<uid_t, std::tuple<int64_t, int64_t, std::vector<float>>> &workspace_modifications) const {
        cudaStream_t stream;
        CHECK_CUDNN_ERROR(detail::get_stream(handle, &stream));
        char *workspace = static_cast<char *>(fe_workspace);

        for (auto [uid, data] : workspace_modifications) {
            (void)uid;
            if (std::get<0>(data) == 0) {
                auto &vec_data = std::get<2>(data);
                CHECK_CUDA_ERROR(detail::cuda_mem_cpy_async(workspace + std::get<1>(data),
                                                            vec_data.data(),
                                                            vec_data.size() * sizeof(float),
                                                            cudaMemcpyHostToDevice,
                                                            stream));
            } else if (std::get<0>(data) == 1) {
                int64_t memset_size = (int64_t)std::get<2>(data)[0];
                CHECK_CUDA_ERROR(detail::cuda_mem_set_async(workspace + std::get<1>(data), 0, memset_size, stream));
            }
        }
        return {error_code_t::OK, ""};
    }

    size_t
    key(bool remove_shape) {
#ifndef CUDNN_FRONTEND_SKIP_JSON_LIB
        json j;
        serialize(j);
        if (remove_shape) {
            for (auto &tensor : j["tensors"]) {
                tensor["dim"].clear();
                tensor["stride"].clear();
            }
        }
        return std::hash<json>{}(j);
#else
        CUDNN_FRONTEND_UNUSED(remove_shape);
        return 1;
#endif
    }

   public:
    Graph() : INode(detail::Context{}) {}

    error_t
    update_cuda_graph(cudnnHandle_t handle,
                      std::unordered_map<std::shared_ptr<Tensor_attributes>, void *> &tensor_to_pointer_map,
                      void *workspace,
                      cudaGraph_t cudnn_cuda_graph) {
        // First get all the uids from the map
        std::unordered_map<Tensor_attributes::uid_t, void *> tensor_uid_to_pointer_map;
        tensor_uid_to_pointer_map.reserve(tensor_to_pointer_map.size());
        for (auto const &[tensor, pointer] : tensor_to_pointer_map) {
            tensor_uid_to_pointer_map.emplace(tensor->get_uid(), pointer);
        }

        return update_cuda_graph(handle, tensor_uid_to_pointer_map, workspace, cudnn_cuda_graph);
    }

    error_t
    update_cuda_graph(cudnnHandle_t handle,
                      std::unordered_map<Tensor_attributes::uid_t, void *> &uid_to_device_ptrs,
                      void *workspace,
                      cudaGraph_t cudnn_cuda_graph) {
        // Initializes this cudnn graph
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            cudnn_cuda_graph == nullptr, error_code_t::INVALID_VALUE, "cudnn_cuda_graph should not be a nullptr");

        size_t num_root_nodes;
        CHECK_CUDA_ERROR(detail::cuda_graph_get_root_nodes(cudnn_cuda_graph, nullptr, &num_root_nodes));
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            num_root_nodes != 1, error_code_t::INVALID_VALUE, "cudnn_cuda_graph should have exactly 1 root node.");

        cudaGraphNode_t current_node = nullptr;
        CHECK_CUDA_ERROR(detail::cuda_graph_get_root_nodes(cudnn_cuda_graph, &current_node, &num_root_nodes));

        ///////////////////////////////////////
        //// PASS BY VALUE TENSOR HANDLING ////
        ///////////////////////////////////////
        // Add pass_by_value data pointers to uid_to_pointer map
        // object lifetime is controlled by tensor_to_pass_by_value which means the pointer should stay valid while
        // making the cuda graph. cuda graph will then keep a copy of the kernel parameters, meaning that at the time of
        // launching the cuda_graph executable, tensor_to_pass_by_value being deallocated does not affect these cpu
        // value's.
        // No cuda graph nodes are required for handling fe owned pass by value tensors.
        std::unordered_map<uid_t, pass_by_values_t> tensor_to_pass_by_value;
        CHECK_CUDNN_FRONTEND_ERROR(collect_pass_by_value_tensors_subtree(tensor_to_pass_by_value));
        CHECK_CUDNN_FRONTEND_ERROR(
            extend_tensor_map_with_pass_by_value_tensors_(uid_to_device_ptrs, tensor_to_pass_by_value));

        ////////////////////////////
        //// WORKSPACE HANDLING ////
        ////////////////////////////
        // Get all types of extra calls that FE has to do on user workspace.
        std::unordered_map<uid_t, std::tuple<int64_t, int64_t, std::vector<float>>> workspace_modifications;
        int64_t workspace_offset = 0;
        CHECK_CUDNN_FRONTEND_ERROR(collect_tensors_in_workspace_subtree(workspace_modifications, workspace_offset));

        for (auto const &[uid, data] : workspace_modifications) {
            const auto &[operation_type, offset, vec_data] = data;
            uid_to_device_ptrs[uid]                        = static_cast<char *>(workspace) + offset;

            // 0 means memcpy
            if (operation_type == 0) {
                CHECK_CUDA_ERROR(
                    detail::cuda_graph_add_memcpy_node_set_params_1D(current_node,
                                                                     static_cast<char *>(workspace) + offset,
                                                                     vec_data.data(),
                                                                     vec_data.size() * sizeof(float),
                                                                     cudaMemcpyHostToDevice));
            }
            // 1 means memset
            else if (operation_type == 1) {
                // offset from workspace
                void *device_ptr    = static_cast<char *>(workspace) + offset;
                int64_t memset_size = static_cast<int64_t>(vec_data[0]);

                cudaMemsetParams params;
                params.dst         = device_ptr;
                params.elementSize = sizeof(char);
                params.value       = 0x0;
                params.width       = memset_size;
                params.height      = 1;  // 1D memset currently
                params.pitch       = 0;  // unused

                CHECK_CUDA_ERROR(detail::cuda_graph_add_memset_node_set_params(current_node, &params));
            }
            // Other values do not correspond to CUDA graph nodes
            else {
                continue;
            }

            size_t num_dependent_nodes;
            CHECK_CUDA_ERROR(detail::cuda_graph_node_get_dependent_nodes(current_node, nullptr, &num_dependent_nodes));
            RETURN_CUDNN_FRONTEND_ERROR_IF(
                num_dependent_nodes != 1,
                error_code_t::INVALID_VALUE,
                "Each node of cudnn_cuda_graph before the backend graph node should have exactly 1 dependent node.");
            CHECK_CUDA_ERROR(
                detail::cuda_graph_node_get_dependent_nodes(current_node, &current_node, &num_dependent_nodes));
        }

        // Make sure device pointer is provided for all uids expected for this plan
        std::vector<void *> device_ptrs;
        std::vector<uid_t> uids;

        device_ptrs.reserve(variant_pack_uids.size());
        uids.reserve(variant_pack_uids.size());

        for (auto const &uid : variant_pack_uids) {
            auto search = uid_to_device_ptrs.find(uid);
            RETURN_CUDNN_FRONTEND_ERROR_IF(search == uid_to_device_ptrs.end(),
                                           error_code_t::INVALID_VARIANT_PACK,
                                           "Uid " + std::to_string(uid) + " does not exist in variant pack.");
            device_ptrs.push_back(search->second);
            uids.push_back(uid);
        }

        ///////////////////
        //// BE GRAPH ////
        ///////////////////
        cudaGraph_t backend_cuda_graph;
        CHECK_CUDA_ERROR(detail::cuda_graph_child_graph_node_get_graph(current_node, &backend_cuda_graph));

        detail::backend_descriptor variant_pack_descriptor(CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR);
        RETURN_CUDNN_FRONTEND_ERROR_IF(variant_pack_descriptor.get_status() != CUDNN_STATUS_SUCCESS,
                                       error_code_t::CUDNN_BACKEND_API_FAILED,
                                       "Failed to create variant pack's backend descriptor.");

        // offset workspace by the already used fe graph workspace
        // this is where cudnn backend can start using workspace for its execution plans
        void *cudnn_workspace = static_cast<char *>(workspace) + fe_workspace_size;
        CHECK_CUDNN_FRONTEND_ERROR(create_variant_pack(variant_pack_descriptor, device_ptrs, uids, cudnn_workspace));

        int64_t candidate = plans.candidate;
        CHECK_CUDNN_FRONTEND_ERROR(plans.is_plan_index_executable(candidate));
        CHECK_CUDNN_ERROR(detail::update_cuda_graph(handle,
                                                    plans.execution_plans[candidate]->get_raw_desc(),
                                                    variant_pack_descriptor.get_ptr(),
                                                    backend_cuda_graph));

        // There should be nothing after the backend graph
        size_t num_dependent_nodes;
        CHECK_CUDA_ERROR(detail::cuda_graph_node_get_dependent_nodes(current_node, nullptr, &num_dependent_nodes));
        RETURN_CUDNN_FRONTEND_ERROR_IF(num_dependent_nodes != 0,
                                       error_code_t::INVALID_VALUE,
                                       "cudnn_cuda_graph should have no graph nodes after the backend graph node.");

        return {error_code_t::OK, ""};
    }

    error_t
    populate_cuda_graph(cudnnHandle_t handle,
                        std::unordered_map<std::shared_ptr<Tensor_attributes>, void *> &tensor_to_pointer_map,
                        void *workspace,
                        cudaGraph_t cudnn_cuda_graph) {
        // First get all the uids from the map
        std::unordered_map<Tensor_attributes::uid_t, void *> tensor_uid_to_pointer_map;
        tensor_uid_to_pointer_map.reserve(tensor_to_pointer_map.size());
        for (auto const &[tensor, pointer] : tensor_to_pointer_map) {
            tensor_uid_to_pointer_map.emplace(tensor->get_uid(), pointer);
        }

        return populate_cuda_graph(handle, tensor_uid_to_pointer_map, workspace, cudnn_cuda_graph);
    }

    error_t
    populate_cuda_graph(cudnnHandle_t handle,
                        std::unordered_map<Tensor_attributes::uid_t, void *> &uid_to_device_ptrs,
                        void *workspace,
                        cudaGraph_t cudnn_cuda_graph) {
        // Check if the cuda graph is empty
        size_t numNodes = 0;
        CHECK_CU_ERROR(detail::cu_graph_get_nodes(cudnn_cuda_graph, nullptr, &numNodes));
        RETURN_CUDNN_FRONTEND_ERROR_IF(numNodes != 0,
                                       error_code_t::INVALID_VALUE,
                                       "cuda graph provided to populate is not empty. cuDNN requires it to be empty "
                                       "for the corresponding update APIs to work correctly.");

        // This function makes linear cuda graphs. And that makes it easy to walk
        // the graph when updating it.
        // So just keeping track of the last node in the cuda graph is sufficient.
        cudaGraphNode_t last_node = nullptr;

        ///////////////////////////////////////
        //// PASS BY VALUE TENSOR HANDLING ////
        ///////////////////////////////////////
        // Add pass_by_value data pointers to uid_to_pointer map
        // object lifetime is controlled by tensor_to_pass_by_value which means the pointer should stay valid while
        // making the cuda graph. cuda graph will then keep a copy of the kernel parameters, meaning that at the time of
        // launching the cuda_graph executable, tensor_to_pass_by_value being deallocated does not affect these cpu
        // value's.
        // No cuda graph nodes are required for handling fe owned pass by value tensors.
        std::unordered_map<uid_t, pass_by_values_t> tensor_to_pass_by_value;
        CHECK_CUDNN_FRONTEND_ERROR(collect_pass_by_value_tensors_subtree(tensor_to_pass_by_value));
        CHECK_CUDNN_FRONTEND_ERROR(
            extend_tensor_map_with_pass_by_value_tensors_(uid_to_device_ptrs, tensor_to_pass_by_value));

        /////////////////////////////////
        //// WORKSPACE HANDLING ////
        /////////////////////////////////
        // Get all types of extra calls that FE has to do on user workspace.
        std::unordered_map<uid_t, std::tuple<int64_t, int64_t, std::vector<float>>> workspace_modifications;
        int64_t workspace_offset = 0;
        CHECK_CUDNN_FRONTEND_ERROR(collect_tensors_in_workspace_subtree(workspace_modifications, workspace_offset));

        for (auto const &[uid, data] : workspace_modifications) {
            const auto &[operation_type, offset, vec_data] = data;
            uid_to_device_ptrs[uid]                        = static_cast<char *>(workspace) + offset;

            cudaGraphNode_t node = nullptr;

            // 0 means memcpy
            if (operation_type == 0) {
                CHECK_CUDA_ERROR(detail::cuda_graph_add_memcpy_node_1D(&node,
                                                                       cudnn_cuda_graph,
                                                                       &last_node,
                                                                       last_node != nullptr,
                                                                       static_cast<char *>(workspace) + offset,
                                                                       vec_data.data(),
                                                                       vec_data.size() * sizeof(float),
                                                                       cudaMemcpyHostToDevice));
            }
            // 1 means memset
            else if (operation_type == 1) {
                // offset from workspace
                void *device_ptr    = static_cast<char *>(workspace) + offset;
                int64_t memset_size = static_cast<int64_t>(vec_data[0]);

                cudaMemsetParams params;
                params.dst         = device_ptr;
                params.elementSize = sizeof(char);
                params.value       = 0x0;
                params.width       = memset_size;
                params.height      = 1;  // 1D memset currently
                params.pitch       = 0;  // unused

                CHECK_CUDA_ERROR(detail::cuda_graph_add_memset_node(
                    &node, cudnn_cuda_graph, &last_node, last_node != nullptr, &params));
            }
            // Other values do not correspond to CUDA graph nodes
            else {
                continue;
            }

            last_node = node;
        }

        //////////////
        // BE graph //
        //////////////

        // Get the BE's cuda graph

        // Make sure device pointer is provided for all uids expected for this plan
        std::vector<void *> device_ptrs;
        device_ptrs.reserve(variant_pack_uids.size());
        std::vector<uid_t> uids;
        uids.reserve(variant_pack_uids.size());
        for (auto const &uid : variant_pack_uids) {
            auto search = uid_to_device_ptrs.find(uid);
            RETURN_CUDNN_FRONTEND_ERROR_IF(search == uid_to_device_ptrs.end(),
                                           error_code_t::INVALID_VARIANT_PACK,
                                           "Uid " + std::to_string(uid) + " does not exist in variant pack.");
            device_ptrs.push_back(search->second);
            uids.push_back(uid);
        }

        // Create the variant pack to pass to backend
        detail::backend_descriptor variant_pack_descriptor(CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR);
        RETURN_CUDNN_FRONTEND_ERROR_IF(variant_pack_descriptor.get_status() != CUDNN_STATUS_SUCCESS,
                                       error_code_t::CUDNN_BACKEND_API_FAILED,
                                       "Failed to create variant pack's backend descriptor.");

        // offset workspace by the already used fe graph workspace
        // this is where cudnn backend can start using workspace for its execution plans
        void *cudnn_workspace = static_cast<char *>(workspace) + fe_workspace_size;
        CHECK_CUDNN_FRONTEND_ERROR(create_variant_pack(variant_pack_descriptor, device_ptrs, uids, cudnn_workspace));

        // Get the plan candidate. It only makes to sense to make cuda graph after execution plan has been built.
        // And in that case the candidate would have been set.
        int64_t candidate = plans.candidate;
        CHECK_CUDNN_FRONTEND_ERROR(plans.is_plan_index_executable(candidate));

        // Finally get the backend cuda graph.
        cudaGraph_t backend_cuda_graph;
        // Initialize the cudnn cuda graph.
        // The responsibility to destroy is on the user.
        detail::cu_graph_create(&backend_cuda_graph, 0);  // 0 is just what the API says to pass

        CHECK_CUDNN_ERROR(detail::populate_cuda_graph(handle,
                                                      plans.execution_plans[candidate]->get_raw_desc(),
                                                      variant_pack_descriptor.get_ptr(),
                                                      backend_cuda_graph));

        // Clone BE graph into a graph_node
        // This same call also places the newly created into FE's graph
        // TODO: BE graph is at the end, so put in appropriate dependencies
        cudaGraphNode_t backend_cuda_graph_node;
        detail::cuda_graph_add_child_graph_node(
            &backend_cuda_graph_node, cudnn_cuda_graph, &last_node, last_node != nullptr, backend_cuda_graph);

        // Destroy the BE graph as it now has been cloned into a node
        // It was initialized by internals of backend, but the responsibility to destroy it is on FE.
        CHECK_CUDA_ERROR(detail::cuda_graph_destroy(backend_cuda_graph));

        return {error_code_t::OK, ""};
    }

    error_t
    validate() {
        CUDNN_FE_LOG_LABEL_ENDL("");
        CUDNN_FE_LOG(*this << std::endl;);

        // First validate all inputs that the user set.
        for (auto const &input : full_graph_inputs) {
            CHECK_CUDNN_FRONTEND_ERROR(input->validate());
        }

        // Validate the nodes, which in turn also infers missing tensor attributes.
        CHECK_CUDNN_FRONTEND_ERROR(validate_subtree());

        // Validate all outputs, which should now have everything set to be lowered to backend.
        for (auto const &output : full_graph_outputs) {
            CHECK_CUDNN_FRONTEND_ERROR(output->validate());
        }

        // Get all the pre assigned uids
        CHECK_CUDNN_FRONTEND_ERROR(get_pre_assigned_uids(used_uids));

        // Clear state
        used_uids.clear();

        return {error_code_t::OK, ""};
    }

    error_t
    build_operation_graph(cudnnHandle_t handle) {
        // expand composite nodes
        CHECK_CUDNN_FRONTEND_ERROR(expand_subtree());

        // Get all the pre assigned uids
        CHECK_CUDNN_FRONTEND_ERROR(get_pre_assigned_uids(used_uids));

        Tensor_attributes::uid_t start_uid = 1;
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensors_subtree(uid_to_tensors, start_uid, used_uids));

        // INode keeps track of all uids that an operation graph uses.
        // This helps to return errors to user during execution, without relying on backend to do so.
        // Also, as uid in a variant pack have to be unique, keep a set of them.
        CHECK_CUDNN_FRONTEND_ERROR(
            create_cudnn_operations(variant_pack_uids, operations, raw_operations, uid_to_tensors));

        // Collect variant pack modifiers when lowering to backend.
        // The collected map is used everytime when execute is called.
        CHECK_CUDNN_FRONTEND_ERROR(collect_variant_pack_replacements_subtree(variant_pack_replacements));

        fe_workspace_size = get_fe_workspace_size_subtree();

        // The method here fuses all operations. There will be 1 operation graph in total.
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_operation_graph(handle));

        if (is_dynamic_shape_enabled && kernel_cache && !kernel_cache->is_finalized()) {
            CHECK_CUDNN_FRONTEND_ERROR(kernel_cache->build(operation_graph->get_raw_desc()));
        }

        return {error_code_t::OK, ""};
    }

    error_t
    get_plan_name(std::string &name) const {
        return get_plan_name_at_index(plans.candidate, name);
    }

    error_t
    get_plan_name_at_index(int64_t plan_index, std::string &name) const {
        auto ret_val = plans.get_name_at_index(plan_index, name);
        CUDNN_FE_LOG_LABEL_ENDL("INFO: get_plan_name_at_index(" << plan_index << ") is " + name);
        return ret_val;
    }

    error_t
    get_workspace_size(int64_t &cudnn_workspace_size) const {
        return get_workspace_size_plan_at_index(plans.candidate, cudnn_workspace_size);
    }

    error_t
    get_workspace_size_plan_at_index(int64_t plan_index, int64_t &cudnn_workspace_size) const {
        // There are two workspaces:
        // - cudnn execution plan workspace
        // - FE node workspace (example: alibiSlope for fmha)
        int64_t cudnn_ws = 0;
        CHECK_CUDNN_FRONTEND_ERROR(get_cudnn_workspace_size_node(plan_index, cudnn_ws));
        cudnn_workspace_size = cudnn_ws + fe_workspace_size;
        CUDNN_FE_LOG_LABEL_ENDL("INFO: get_workspace_size() is " << cudnn_workspace_size);
        return {error_code_t::OK, ""};
    }

    int64_t
    get_workspace_size() const {
        return get_workspace_size_plan_at_index(plans.candidate);
    }

    int64_t
    get_workspace_size_plan_at_index(int64_t plan_index) const {
        int64_t cudnn_workspace = 0;
        auto status             = get_workspace_size_plan_at_index(plan_index, cudnn_workspace);
        if (status.is_bad()) {
            CUDNN_FE_LOG_LABEL_ENDL("ERROR: Querying workspace failed.");
        }
        return cudnn_workspace;
    }

    int64_t
    get_autotune_workspace_size() const {
        // There are two workspaces:
        // - cudnn execution plan workspace
        // - FE node workspace (example: alibiSlope for fmha)
        return fe_workspace_size + get_max_cudnn_workspace_size();
    }

    error_t
    autotune(cudnnHandle_t handle,
             std::unordered_map<int64_t, void *> &tensor_uid_to_pointer_map,
             void *workspace,
             void *user_impl = nullptr) {
        // Add pass_by_value data pointers to tensor_uid_to_pointer map
        // object lifetime is controlled by tensor_to_pass_by_value which means the pointer should stay valid during
        // execute.
        std::unordered_map<uid_t, pass_by_values_t> tensor_to_pass_by_value;
        CHECK_CUDNN_FRONTEND_ERROR(collect_pass_by_value_tensors_subtree(tensor_to_pass_by_value));

        CHECK_CUDNN_FRONTEND_ERROR(
            extend_tensor_map_with_pass_by_value_tensors_(tensor_uid_to_pointer_map, tensor_to_pass_by_value));

        CHECK_CUDNN_FRONTEND_ERROR(
            make_variant_pack_replacements(tensor_uid_to_pointer_map, variant_pack_replacements));

        std::unordered_map<uid_t, std::tuple<int64_t, int64_t, std::vector<float>>> workspace_modifications;
        int64_t workspace_offset = 0;
        CHECK_CUDNN_FRONTEND_ERROR(collect_tensors_in_workspace_subtree(workspace_modifications, workspace_offset));

        CHECK_CUDNN_FRONTEND_ERROR(run_auxiliary_kernels(handle, workspace, workspace_modifications));

        CHECK_CUDNN_FRONTEND_ERROR(
            extend_tensor_map_with_workspace_tensors_(tensor_uid_to_pointer_map, workspace, workspace_modifications));

        // offset workspace by the already used fe graph workspace
        // this is where cudnn backend can start using workspace for its execution plans
        void *cudnn_workspace = static_cast<char *>(workspace) + fe_workspace_size;

        CHECK_CUDNN_FRONTEND_ERROR(plans.autotune(handle, tensor_uid_to_pointer_map, cudnn_workspace, user_impl));
        return {error_code_t::OK, ""};
    }

    error_t
    autotune(cudnnHandle_t handle,
             std::unordered_map<std::shared_ptr<Tensor_attributes>, void *> &tensor_to_pointer_map,
             void *workspace,
             void *user_impl = nullptr) {
        // First get all the uids from the map
        std::unordered_map<int64_t, void *> tensor_uid_to_pointer_map;
        for (auto const &[tensor, pointer] : tensor_to_pointer_map) {
            tensor_uid_to_pointer_map.emplace(tensor->get_uid(), pointer);
        }

        return autotune(handle, tensor_uid_to_pointer_map, workspace, user_impl);
    }

    error_t
    execute_plan_at_index(cudnnHandle_t handle,
                          std::unordered_map<std::shared_ptr<Tensor_attributes>, void *> &tensor_to_pointer_map,
                          void *workspace,
                          int64_t plan_index) const {
        // First get all the uids from the map
        std::unordered_map<int64_t, void *> tensor_uid_to_pointer_map;
        for (auto const &[tensor, pointer] : tensor_to_pointer_map) {
            tensor_uid_to_pointer_map.emplace(tensor->get_uid(), pointer);
        }

        return execute_plan_at_index(handle, tensor_uid_to_pointer_map, workspace, plan_index);
    }

    error_t
    execute(cudnnHandle_t handle,
            std::unordered_map<std::shared_ptr<Tensor_attributes>, void *> &tensor_to_pointer_map,
            void *workspace) const {
        // First get all the uids from the map
        std::unordered_map<int64_t, void *> tensor_uid_to_pointer_map;
        for (auto const &[tensor, pointer] : tensor_to_pointer_map) {
            tensor_uid_to_pointer_map.emplace(tensor->get_uid(), pointer);
        }

        return execute(handle, tensor_uid_to_pointer_map, workspace);
    }

    error_t
    execute_plan_at_index(cudnnHandle_t handle,
                          std::unordered_map<int64_t, void *> &tensor_uid_to_pointer_map,
                          void *workspace,
                          int64_t plan_index) const {
        // Add pass_by_value data pointers to uid_to_pointer map
        // object lifetime is controlled by tensor_to_pass_by_value which means the pointer should stay valid during
        // execute.
        std::unordered_map<uid_t, pass_by_values_t> tensor_to_pass_by_value;
        CHECK_CUDNN_FRONTEND_ERROR(collect_pass_by_value_tensors_subtree(tensor_to_pass_by_value));

        CHECK_CUDNN_FRONTEND_ERROR(
            extend_tensor_map_with_pass_by_value_tensors_(tensor_uid_to_pointer_map, tensor_to_pass_by_value));

        CHECK_CUDNN_FRONTEND_ERROR(
            make_variant_pack_replacements(tensor_uid_to_pointer_map, variant_pack_replacements));

        std::unordered_map<uid_t, std::tuple<int64_t, int64_t, std::vector<float>>> workspace_modifications;
        int64_t workspace_offset = 0;
        CHECK_CUDNN_FRONTEND_ERROR(collect_tensors_in_workspace_subtree(workspace_modifications, workspace_offset));

        CHECK_CUDNN_FRONTEND_ERROR(run_auxiliary_kernels(handle, workspace, workspace_modifications));

        CHECK_CUDNN_FRONTEND_ERROR(
            extend_tensor_map_with_workspace_tensors_(tensor_uid_to_pointer_map, workspace, workspace_modifications));
        // offset workspace by the already used fe graph workspace
        // this is where cudnn backend can start using workspace for its execution plans
        void *cudnn_workspace = static_cast<char *>(workspace) + fe_workspace_size;

        CHECK_CUDNN_FRONTEND_ERROR(
            execute_cudnn_plan_with_uid(handle, tensor_uid_to_pointer_map, cudnn_workspace, plan_index));

        return {error_code_t::OK, ""};
    }

    error_t
    execute(cudnnHandle_t handle,
            std::unordered_map<int64_t, void *> &tensor_uid_to_pointer_map,
            void *workspace) const {
        // Add pass_by_value data pointers to uid_to_pointer map
        // object lifetime is controlled by tensor_to_pass_by_value which means the pointer should stay valid during
        // execute.
        std::unordered_map<uid_t, pass_by_values_t> tensor_to_pass_by_value;
        CHECK_CUDNN_FRONTEND_ERROR(collect_pass_by_value_tensors_subtree(tensor_to_pass_by_value));

        CHECK_CUDNN_FRONTEND_ERROR(
            extend_tensor_map_with_pass_by_value_tensors_(tensor_uid_to_pointer_map, tensor_to_pass_by_value));
        CHECK_CUDNN_FRONTEND_ERROR(
            make_variant_pack_replacements(tensor_uid_to_pointer_map, variant_pack_replacements));

        std::unordered_map<uid_t, std::tuple<int64_t, int64_t, std::vector<float>>> workspace_modifications;
        int64_t workspace_offset = 0;
        CHECK_CUDNN_FRONTEND_ERROR(collect_tensors_in_workspace_subtree(workspace_modifications, workspace_offset));

        CHECK_CUDNN_FRONTEND_ERROR(run_auxiliary_kernels(handle, workspace, workspace_modifications));

        CHECK_CUDNN_FRONTEND_ERROR(
            extend_tensor_map_with_workspace_tensors_(tensor_uid_to_pointer_map, workspace, workspace_modifications));
        // offset workspace by the already used fe graph workspace
        // this is where cudnn backend can start using workspace for its execution plans
        void *cudnn_workspace = static_cast<char *>(workspace) + fe_workspace_size;

        CHECK_CUDNN_FRONTEND_ERROR(
            execute_cudnn_plan_with_uid(handle, tensor_uid_to_pointer_map, cudnn_workspace, plans.candidate));

        return {error_code_t::OK, ""};
    }

    error_t
    serialize(std::vector<uint8_t> &data) const {
#ifndef CUDNN_FRONTEND_SKIP_JSON_LIB
        json j;
        serialize(j);

        auto const candidate = plans.candidate;
        auto execution_plan  = plans.execution_plans[candidate];
        if (execution_plan != nullptr) {
            auto serialized_plan    = execution_plan->getJsonRepresentation();
            j["cudnn_backend_data"] = serialized_plan;
            j["variant_pack_uids"]  = variant_pack_uids;
        }

        j["behavior_notes"] = plans.behavior_notes;

        std::unordered_map<uid_t, pass_by_values_t> tensor_to_pass_by_value;
        CHECK_CUDNN_FRONTEND_ERROR(collect_pass_by_value_tensors_subtree(tensor_to_pass_by_value));
        j["pass_by_values"] = tensor_to_pass_by_value;

        std::unordered_map<uid_t, std::tuple<int64_t, int64_t, std::vector<float>>> workspace_modifications;
        int64_t workspace_offset = 0;
        CHECK_CUDNN_FRONTEND_ERROR(collect_tensors_in_workspace_subtree(workspace_modifications, workspace_offset));
        j["workspace_modifications"] = workspace_modifications;

        j["variant_pack_replacements"] = variant_pack_replacements;

        j["fe_workspace_size"] = fe_workspace_size;

        data = json::to_ubjson(j);
        return {error_code_t::OK, ""};
#else
        CUDNN_FRONTEND_UNUSED(data);
        return {error_code_t::GRAPH_NOT_SUPPORTED, "unavailable when compiled with CUDNN_FRONTEND_SKIP_JSON_LIB"};
#endif
    }

    error_t
    deserialize(cudnnHandle_t handle, std::vector<uint8_t> const &data) {
#ifndef CUDNN_FRONTEND_SKIP_JSON_LIB
        json j = json::from_ubjson(data);

        if (j.contains("tensors")) {
            auto tensor_map = j["tensors"].get<std::unordered_map<std::string, json>>();
            for (const auto &tensor_info : tensor_map) {
                auto tensor_attributes = std::make_shared<Tensor_attributes>();
                from_json(tensor_info.second, *tensor_attributes);
                deserialized_tensor_properties.insert(tensor_attributes);
            }
        }

        auto serialized_plan = j["cudnn_backend_data"];
        CHECK_CUDNN_FRONTEND_ERROR(plans.build_plans(handle, serialized_plan));

        plans.behavior_notes = j["behavior_notes"].get<std::vector<std::vector<BehaviorNote_t>>>();

        variant_pack_uids = j["variant_pack_uids"].get<std::unordered_set<graph::Tensor_attributes::uid_t>>();

        deserialized_pass_by_value = j["pass_by_values"];

        deserialized_workspace_modifications = j["workspace_modifications"];

        variant_pack_replacements = j["variant_pack_replacements"];

        fe_workspace_size = j["fe_workspace_size"];

        return {error_code_t::OK, ""};
#else
        CUDNN_FRONTEND_UNUSED(handle);
        CUDNN_FRONTEND_UNUSED(data);
        return {error_code_t::GRAPH_NOT_SUPPORTED, "unavailable when compiled with CUDNN_FRONTEND_SKIP_JSON_LIB"};
#endif
    }

    Type
    getType() override {
        return Type::COMPOSITE;
    }

    Graph &
    set_intermediate_data_type(DataType_t type);
    Graph &
    set_io_data_type(DataType_t type);
    Graph &
    set_compute_data_type(DataType_t type);
    Graph &
    set_dynamic_shape_enabled(bool is_enabled);
    Graph &
    set_sm_count(int32_t type);
    Graph &
    set_sm_version(int32_t version);
    Graph &
    set_kernel_cache(std::shared_ptr<KernelCache> cache);

    Graph &
    set_name(std::string const &name) {
        context.set_name(name);
        return *this;
    }

    error_t
    query_tensor_attributes_of_uid(int64_t const uid, Tensor_attributes &tensor) const;

    std::shared_ptr<Tensor_attributes>
    tensor(Tensor_attributes const &tensor);

    std::shared_ptr<Tensor_attributes>
    tensor_like(std::shared_ptr<Tensor_attributes> const &tensor, std::string const &name = std::string{});

    std::array<std::shared_ptr<Tensor_attributes>, 3> layernorm(std::shared_ptr<Tensor_attributes>,
                                                                std::shared_ptr<Tensor_attributes>,
                                                                std::shared_ptr<Tensor_attributes>,
                                                                Layernorm_attributes);

    std::array<std::shared_ptr<Tensor_attributes>, 3> instancenorm(std::shared_ptr<Tensor_attributes>,
                                                                   std::shared_ptr<Tensor_attributes>,
                                                                   std::shared_ptr<Tensor_attributes>,
                                                                   Instancenorm_attributes);

    std::array<std::shared_ptr<Tensor_attributes>, 5> batchnorm(std::shared_ptr<Tensor_attributes>,
                                                                std::shared_ptr<Tensor_attributes>,
                                                                std::shared_ptr<Tensor_attributes>,
                                                                Batchnorm_attributes);

    std::shared_ptr<Tensor_attributes> batchnorm_inference(std::shared_ptr<Tensor_attributes>,
                                                           std::shared_ptr<Tensor_attributes>,
                                                           std::shared_ptr<Tensor_attributes>,
                                                           std::shared_ptr<Tensor_attributes>,
                                                           std::shared_ptr<Tensor_attributes>,
                                                           Batchnorm_inference_attributes);

    std::array<std::shared_ptr<Tensor_attributes>, 6> bn_finalize(std::shared_ptr<Tensor_attributes>,
                                                                  std::shared_ptr<Tensor_attributes>,
                                                                  std::shared_ptr<Tensor_attributes>,
                                                                  std::shared_ptr<Tensor_attributes>,
                                                                  std::shared_ptr<Tensor_attributes>,
                                                                  std::shared_ptr<Tensor_attributes>,
                                                                  BN_finalize_attributes);

    std::shared_ptr<Tensor_attributes> conv_fprop(std::shared_ptr<Tensor_attributes>,
                                                  std::shared_ptr<Tensor_attributes>,
                                                  Conv_fprop_attributes);

    std::shared_ptr<Tensor_attributes> conv_dgrad(std::shared_ptr<Tensor_attributes>,
                                                  std::shared_ptr<Tensor_attributes>,
                                                  Conv_dgrad_attributes);

    std::shared_ptr<Tensor_attributes> conv_wgrad(std::shared_ptr<Tensor_attributes>,
                                                  std::shared_ptr<Tensor_attributes>,
                                                  Conv_wgrad_attributes);

    std::array<std::shared_ptr<Tensor_attributes>, 5> dbn_weight(std::shared_ptr<Tensor_attributes>,
                                                                 std::shared_ptr<Tensor_attributes>,
                                                                 std::shared_ptr<Tensor_attributes>,
                                                                 std::shared_ptr<Tensor_attributes>,
                                                                 std::shared_ptr<Tensor_attributes>,
                                                                 DBN_weight_attributes);

    std::array<std::shared_ptr<Tensor_attributes>, 3> batchnorm_backward(std::shared_ptr<Tensor_attributes>,
                                                                         std::shared_ptr<Tensor_attributes>,
                                                                         std::shared_ptr<Tensor_attributes>,
                                                                         Batchnorm_backward_attributes);

    std::array<std::shared_ptr<Tensor_attributes>, 3> layernorm_backward(std::shared_ptr<Tensor_attributes>,
                                                                         std::shared_ptr<Tensor_attributes>,
                                                                         std::shared_ptr<Tensor_attributes>,
                                                                         Layernorm_backward_attributes);

    std::array<std::shared_ptr<Tensor_attributes>, 3> instancenorm_backward(std::shared_ptr<Tensor_attributes>,
                                                                            std::shared_ptr<Tensor_attributes>,
                                                                            std::shared_ptr<Tensor_attributes>,
                                                                            Instancenorm_backward_attributes);
    std::array<std::shared_ptr<Tensor_attributes>, 2> genstats(std::shared_ptr<Tensor_attributes>, Genstats_attributes);

    std::array<std::shared_ptr<Tensor_attributes>, 2> rmsnorm(std::shared_ptr<Tensor_attributes>,
                                                              std::shared_ptr<Tensor_attributes>,
                                                              Rmsnorm_attributes);

    std::array<std::shared_ptr<Tensor_attributes>, 3> rmsnorm_backward(std::shared_ptr<Tensor_attributes>,
                                                                       std::shared_ptr<Tensor_attributes>,
                                                                       std::shared_ptr<Tensor_attributes>,
                                                                       std::shared_ptr<Tensor_attributes>,
                                                                       Rmsnorm_backward_attributes);

    std::array<std::shared_ptr<Tensor_attributes>, 2> sdpa(std::shared_ptr<Tensor_attributes>,
                                                           std::shared_ptr<Tensor_attributes>,
                                                           std::shared_ptr<Tensor_attributes>,
                                                           SDPA_attributes);

    std::array<std::shared_ptr<Tensor_attributes>, 4> sdpa_fp8(std::shared_ptr<Tensor_attributes>,
                                                               std::shared_ptr<Tensor_attributes>,
                                                               std::shared_ptr<Tensor_attributes>,
                                                               std::shared_ptr<Tensor_attributes>,
                                                               std::shared_ptr<Tensor_attributes>,
                                                               std::shared_ptr<Tensor_attributes>,
                                                               std::shared_ptr<Tensor_attributes>,
                                                               std::shared_ptr<Tensor_attributes>,
                                                               std::shared_ptr<Tensor_attributes>,
                                                               SDPA_fp8_attributes);

    inline std::array<std::shared_ptr<Tensor_attributes>, 7> sdpa_fp8_backward(std::shared_ptr<Tensor_attributes>,
                                                                               std::shared_ptr<Tensor_attributes>,
                                                                               std::shared_ptr<Tensor_attributes>,
                                                                               std::shared_ptr<Tensor_attributes>,
                                                                               std::shared_ptr<Tensor_attributes>,
                                                                               std::shared_ptr<Tensor_attributes>,
                                                                               std::shared_ptr<Tensor_attributes>,
                                                                               std::shared_ptr<Tensor_attributes>,
                                                                               std::shared_ptr<Tensor_attributes>,
                                                                               std::shared_ptr<Tensor_attributes>,
                                                                               std::shared_ptr<Tensor_attributes>,
                                                                               std::shared_ptr<Tensor_attributes>,
                                                                               std::shared_ptr<Tensor_attributes>,
                                                                               std::shared_ptr<Tensor_attributes>,
                                                                               std::shared_ptr<Tensor_attributes>,
                                                                               std::shared_ptr<Tensor_attributes>,
                                                                               std::shared_ptr<Tensor_attributes>,
                                                                               std::shared_ptr<Tensor_attributes>,
                                                                               SDPA_fp8_backward_attributes);

    std::array<std::shared_ptr<Tensor_attributes>, 3> sdpa_backward(std::shared_ptr<Tensor_attributes>,
                                                                    std::shared_ptr<Tensor_attributes>,
                                                                    std::shared_ptr<Tensor_attributes>,
                                                                    std::shared_ptr<Tensor_attributes>,
                                                                    std::shared_ptr<Tensor_attributes>,
                                                                    std::shared_ptr<Tensor_attributes>,
                                                                    SDPA_backward_attributes);

    std::shared_ptr<Tensor_attributes> slice(std::shared_ptr<Tensor_attributes>, Slice_attributes);

    std::array<std::shared_ptr<Tensor_attributes>, 2> block_scale_quantize(std::shared_ptr<Tensor_attributes>,
                                                                           Block_scale_quantize_attributes);

    std::shared_ptr<Tensor_attributes> block_scale_dequantize(std::shared_ptr<Tensor_attributes>,
                                                              std::shared_ptr<Tensor_attributes>,
                                                              Block_scale_dequantize_attributes);

    std::shared_ptr<Tensor_attributes> concatenate(std::vector<std::shared_ptr<Tensor_attributes>>,
                                                   Concatenate_attributes);

    [[deprecated]] std::array<std::shared_ptr<Tensor_attributes>, 2>
    scaled_dot_product_flash_attention(std::shared_ptr<Tensor_attributes> q,
                                       std::shared_ptr<Tensor_attributes> k,
                                       std::shared_ptr<Tensor_attributes> v,
                                       SDPA_attributes attributes) {
        return sdpa(q, k, v, attributes);
    }
    [[deprecated]] std::array<std::shared_ptr<Tensor_attributes>, 3>
    scaled_dot_product_flash_attention_backward(std::shared_ptr<Tensor_attributes> q,
                                                std::shared_ptr<Tensor_attributes> k,
                                                std::shared_ptr<Tensor_attributes> v,
                                                std::shared_ptr<Tensor_attributes> o,
                                                std::shared_ptr<Tensor_attributes> dO,
                                                std::shared_ptr<Tensor_attributes> stats,
                                                SDPA_backward_attributes attributes) {
        return sdpa_backward(q, k, v, o, dO, stats, attributes);
    }

    error_t
    create_execution_plans(std::vector<HeurMode_t> const &mode);

    error_t
    create_execution_plan(int64_t const engine_id, std::unordered_map<KnobType_t, int64_t> const &knobs);

    int64_t
    get_execution_plan_count() const;

    inline error_t
    get_engine_count(int64_t &count);

    inline error_t
    get_knobs_for_engine(int64_t const engine, std::vector<Knob> &);

    error_t
    check_support(cudnnHandle_t h) {
        CHECK_CUDNN_FRONTEND_ERROR(plans.check_support(h));
        return {error_code_t::OK, ""};
    }

    error_t
    build(cudnnHandle_t const &handle,
          std::vector<HeurMode_t> const &mode,
          BuildPlanPolicy_t const policy     = BuildPlanPolicy_t::HEURISTICS_CHOICE,
          bool const do_multithreaded_builds = false);

    error_t
    build_plans(cudnnHandle_t const &handle,
                BuildPlanPolicy_t const policy     = BuildPlanPolicy_t::HEURISTICS_CHOICE,
                bool const do_multithreaded_builds = false);

    error_t
    build_plan_at_index(cudnnHandle_t const &handle, int64_t index);

    Graph &
    deselect_workspace_greater_than(int64_t const workspace) {
        plans.set_max_workspace_allowed(workspace);
        return *this;
    }

    Graph &
    deselect_shared_mem_greater_than(int64_t const workspace) {
        plans.set_max_shared_mem_allowed(workspace);
        return *this;
    }

    Graph &
    deselect_engines(std::vector<std::string> const &engine_names) {
        plans.set_barred_names(engine_names);
        return *this;
    }

    Graph &
    select_behavior_notes(std::vector<BehaviorNote_t> const &notes) {
        auto status = plans.filter_behavior_notes(notes, true);
        if (status.is_bad()) {
            CUDNN_FE_LOG(status.get_message() << std::endl);
        }
        return *this;
    }

    Graph &
    select_numeric_notes(std::vector<NumericalNote_t> const &notes) {
        auto status = plans.filter_numeric_notes(notes, true);
        if (status.is_bad()) {
            CUDNN_FE_LOG(status.get_message() << std::endl);
        }
        return *this;
    }

    Graph &
    deselect_behavior_notes(std::vector<BehaviorNote_t> const &notes) {
        auto status = plans.filter_behavior_notes(notes, false);
        if (status.is_bad()) {
            CUDNN_FE_LOG(status.get_message() << std::endl);
        }
        return *this;
    }

    Graph &
    deselect_numeric_notes(std::vector<NumericalNote_t> const &notes) {
        auto status = plans.filter_numeric_notes(notes, false);
        if (status.is_bad()) {
            CUDNN_FE_LOG(status.get_message() << std::endl);
        }
        return *this;
    }

    error_t
    get_behavior_notes_for_plan_at_index(int64_t const index, std::vector<BehaviorNote_t> &notes) const;

    error_t
    get_behavior_notes(std::vector<BehaviorNote_t> &notes) const;

#ifndef CUDNN_FRONTEND_SKIP_JSON_LIB
    virtual void
    serialize(json &j) const override final {
        // Different from serialization of other INodes.
        // Go over each subnode and serialize them.
        json full_json;

        full_json["context"]["name"]                   = context.get_name();
        full_json["context"]["compute_data_type"]      = context.get_compute_data_type();
        full_json["context"]["intermediate_data_type"] = context.get_intermediate_data_type();
        full_json["context"]["io_data_type"]           = context.get_io_data_type();
        full_json["context"]["sm_count"]               = context.get_target_sm_count();

        full_json.update(R"( {"tag": "GRAPH"})"_json);
        full_json["nodes"];
        for (auto const &sub_node : sub_nodes) {
            json j_sub_node;
            sub_node->serialize(j_sub_node);
            full_json["nodes"].push_back(j_sub_node);
        }

        j["context"] = full_json["context"];

        j["json_version"]           = "1.0";
        j["cudnn_backend_version"]  = detail::get_backend_version_string();
        j["cudnn_frontend_version"] = CUDNN_FRONTEND_VERSION;
        j["nodes"];
        j["tensors"];
        std::unordered_set<std::string> tensors;

        for (const auto &sub_node : full_json["nodes"]) {
            // Create a short version of the node
            auto short_node       = sub_node;
            short_node["inputs"]  = {};
            short_node["outputs"] = {};

            auto node_name = sub_node["tag"].get<std::string>();
            auto i         = 0;
            // Process node inputs
            for (const auto &input : sub_node["inputs"]) {
                std::string port_name;
                json tensor_info;

                if (node_name == "CONCATENATE") {
                    // Extract port_name and tensor_name
                    port_name   = std::to_string(i);
                    tensor_info = input;
                    i++;
                } else {
                    // Extract port_name and tensor_name
                    port_name   = input[0].get<std::string>();
                    tensor_info = input[1];
                }

                if (tensor_info.is_null()) {
                    continue;
                }

                std::string tensor_name = tensor_info["name"].get<std::string>();

                // Update short_node inputs
                short_node["inputs"][port_name] = tensor_name;

                // Check if the tensor is already in the tensors map
                if (tensors.find(tensor_name) == tensors.end()) {
                    // If not, add it to the j["tensors"]
                    j["tensors"][tensor_name] = tensor_info;
                }
            }

            // Process node outputs
            for (const auto &output : sub_node["outputs"]) {
                // Extract port_name and tensor_name
                auto port_name   = output[0].get<std::string>();
                auto tensor_info = output[1];

                if (tensor_info.is_null()) {
                    continue;
                }

                std::string tensor_name = tensor_info["name"].get<std::string>();

                // Update short_node outputs
                short_node["outputs"][port_name] = tensor_name;

                // Check if the tensor is already in the tensors map
                if (tensors.find(tensor_name) == tensors.end()) {
                    // If not, add it to the j["tensors"]
                    j["tensors"][tensor_name] = tensor_info;
                }
            }

            // Add the short_node to j["nodes"]
            j["nodes"].push_back(short_node);
        }
    };
#endif

    size_t
    key() override final {
        return key(is_dynamic_shape_enabled);
    }

    // TODO: temparorily placed in graphs class. This function needs to be a free standing function.
#ifndef CUDNN_FRONTEND_SKIP_JSON_LIB
    error_t
    deserialize(const json &j) {
        if (j.contains("context")) {
            const auto &j_context = j["context"];
            if (j_context.contains("compute_data_type") && !j_context["compute_data_type"].is_null()) {
                context.set_compute_data_type(j_context["compute_data_type"].get<DataType_t>());
            }
            if (j_context.contains("intermediate_data_type") && !j_context["intermediate_data_type"].is_null()) {
                context.set_intermediate_data_type(j_context["intermediate_data_type"].get<DataType_t>());
            }
            if (j_context.contains("io_data_type") && !j_context["io_data_type"].is_null()) {
                context.set_io_data_type(j_context["io_data_type"].get<DataType_t>());
            }
            if (j_context.contains("name") && !j_context["name"].is_null()) {
                context.set_name(j_context["name"].get<std::string>());
            }
            if (j_context.contains("sm_count") && !j_context["sm_count"].is_null()) {
                context.set_target_sm_count(j_context["sm_count"].get<int32_t>());
            }
        }

        std::map<std::string, std::shared_ptr<Tensor_attributes>> created_tensors;
        // Iterate through each sub-node in the full JSON
        if (j.contains("nodes") && j["nodes"].is_array()) {
            for (auto j_sub_node : j["nodes"]) {
                // Create a JSON object for inputs
                json inputs;

                // Iterate through each input of the sub-node
                if (j_sub_node.contains("inputs") && j_sub_node["inputs"].is_object()) {
                    for (auto &[port_name, tensor_name] : j_sub_node["inputs"].items()) {
                        if (j.contains("tensors") && j["tensors"].contains(tensor_name)) {
                            // Add the input to the inputs JSON object
                            inputs.push_back({port_name, j["tensors"][tensor_name]});
                        }
                    }
                }

                // Create a JSON object for outputs
                json outputs;

                // Iterate through each output of the sub-node
                if (j_sub_node.contains("outputs") && j_sub_node["outputs"].is_object()) {
                    for (auto &[port_name, tensor_name] : j_sub_node["outputs"].items()) {
                        if (j.contains("tensors") && j["tensors"].contains(tensor_name)) {
                            // Add the output to the outputs JSON object
                            outputs.push_back({port_name, j["tensors"][tensor_name]});
                        }
                    }
                }

                // Replace the original inputs and outputs of the sub-node with the new JSON objects
                j_sub_node["inputs"]  = inputs;
                j_sub_node["outputs"] = outputs;

                auto check_if_pre_created_tensor = [&created_tensors](std::shared_ptr<Tensor_attributes> t) {
                    if (t == nullptr) {
                        return t;
                    }

                    if (created_tensors.find(t->get_name()) == created_tensors.end()) {
                        created_tensors.insert({t->get_name(), t});
                        return t;
                    } else {
                        return created_tensors[t->get_name()];
                    }
                };

#define CHECK_TENSORS(attributes)                                      \
    for (const auto &[key, tensor] : attributes.inputs) {              \
        attributes.inputs[key] = check_if_pre_created_tensor(tensor);  \
    }                                                                  \
    for (const auto &[key, tensor] : attributes.outputs) {             \
        attributes.outputs[key] = check_if_pre_created_tensor(tensor); \
    }

#define FILL_GLOBAL_IO_TENSOR_MAP(attributes)                              \
    for (auto input_name_to_attr_pair : attributes.inputs) {               \
        if (input_name_to_attr_pair.second != nullptr &&                   \
            (input_name_to_attr_pair.second->get_is_virtual() == false)) { \
            full_graph_inputs.emplace(input_name_to_attr_pair.second);     \
        }                                                                  \
    }                                                                      \
    for (auto output_name_to_attr_pair : attributes.outputs) {             \
        if (output_name_to_attr_pair.second != nullptr) {                  \
            full_graph_outputs.emplace(output_name_to_attr_pair.second);   \
        }                                                                  \
    }
                if (j_sub_node.contains("tag") && j_sub_node["tag"].is_string()) {
                    auto tag = j_sub_node["tag"].get<std::string>();
                    if (tag == "CONV_FPROP") {
                        auto conv_fprop_attributes = j_sub_node.get<Conv_fprop_attributes>();
                        CHECK_TENSORS(conv_fprop_attributes);
                        FILL_GLOBAL_IO_TENSOR_MAP(conv_fprop_attributes);
                        sub_nodes.emplace_back(
                            std::make_unique<ConvolutionNode>(std::move(conv_fprop_attributes), context));
                    } else if (tag == "POINTWISE") {
                        auto pointwise_attributes = j_sub_node.get<Pointwise_attributes>();
                        CHECK_TENSORS(pointwise_attributes);
                        FILL_GLOBAL_IO_TENSOR_MAP(pointwise_attributes);
                        sub_nodes.emplace_back(
                            std::make_unique<PointwiseNode>(std::move(pointwise_attributes), context));
                    } else if (tag == "REDUCTION") {
                        auto reduction_attributes = j_sub_node.get<Reduction_attributes>();
                        CHECK_TENSORS(reduction_attributes);
                        FILL_GLOBAL_IO_TENSOR_MAP(reduction_attributes);
                        sub_nodes.emplace_back(
                            std::make_unique<ReductionNode>(std::move(reduction_attributes), context));
                    } else if (tag == "SDPA_FWD") {
                        auto sdpa_attributes = j_sub_node.get<SDPA_attributes>();
                        CHECK_TENSORS(sdpa_attributes);
                        FILL_GLOBAL_IO_TENSOR_MAP(sdpa_attributes);
                        sub_nodes.emplace_back(std::make_unique<SDPANode>(std::move(sdpa_attributes), context));
                    } else if (tag == "SDPA_BWD") {
                        auto sdpa_bwd_attributes = j_sub_node.get<SDPA_backward_attributes>();
                        CHECK_TENSORS(sdpa_bwd_attributes);
                        FILL_GLOBAL_IO_TENSOR_MAP(sdpa_bwd_attributes);
                        sub_nodes.emplace_back(
                            std::make_unique<SDPABackwardNode>(std::move(sdpa_bwd_attributes), context));
                    } else if (tag == "MATMUL") {
                        auto matmul_attributes = j_sub_node.get<Matmul_attributes>();
                        CHECK_TENSORS(matmul_attributes);
                        FILL_GLOBAL_IO_TENSOR_MAP(matmul_attributes);
                        sub_nodes.emplace_back(std::make_unique<MatmulNode>(std::move(matmul_attributes), context));
                    } else if (tag == "SLICE") {
                        auto slice_attributes = j_sub_node.get<Slice_attributes>();
                        CHECK_TENSORS(slice_attributes);
                        FILL_GLOBAL_IO_TENSOR_MAP(slice_attributes);
                        sub_nodes.emplace_back(std::make_unique<SliceNode>(std::move(slice_attributes), context));
                    } else if (tag == "SDPA_FP8_FWD") {
                        auto sdpa_fp8_attributes = j_sub_node.get<SDPA_fp8_attributes>();
                        CHECK_TENSORS(sdpa_fp8_attributes);
                        FILL_GLOBAL_IO_TENSOR_MAP(sdpa_fp8_attributes);
                        sub_nodes.emplace_back(std::make_unique<SDPAFP8Node>(std::move(sdpa_fp8_attributes), context));
                    } else if (tag == "RESAMPLE") {
                        auto resample_attributes = j_sub_node.get<Resample_attributes>();
                        CHECK_TENSORS(resample_attributes);
                        FILL_GLOBAL_IO_TENSOR_MAP(resample_attributes);
                        sub_nodes.emplace_back(std::make_unique<ResampleNode>(std::move(resample_attributes), context));
                    } else if (tag == "CONV_DGRAD") {
                        auto dgrad_attributes = j_sub_node.get<Conv_dgrad_attributes>();
                        CHECK_TENSORS(dgrad_attributes);
                        FILL_GLOBAL_IO_TENSOR_MAP(dgrad_attributes);
                        sub_nodes.emplace_back(std::make_unique<DgradNode>(std::move(dgrad_attributes), context));
                    } else if (tag == "CONV_WGRAD") {
                        auto wgrad_attributes = j_sub_node.get<Conv_wgrad_attributes>();
                        CHECK_TENSORS(wgrad_attributes);
                        FILL_GLOBAL_IO_TENSOR_MAP(wgrad_attributes);
                        sub_nodes.emplace_back(std::make_unique<WgradNode>(std::move(wgrad_attributes), context));
                    }
                }
#undef CHECK_TENSORS
            }
        }

        return {error_code_t::OK, ""};
    }
#endif

    std::string
    print(void) const {
#ifndef CUDNN_FRONTEND_SKIP_JSON_LIB
        std::stringstream ss;
        json j = *this;
        ss << j;
        return ss.str();
#else
        return "print is unavailable when compiled with CUDNN_FRONTEND_SKIP_JSON_LIB";
#endif
    }
};

inline error_t
Graph::get_behavior_notes_for_plan_at_index(int64_t const index, std::vector<BehaviorNote_t> &notes) const {
    CHECK_CUDNN_FRONTEND_ERROR(plans.get_behavior_notes_at_index(index, notes));
    return {error_code_t::OK, ""};
}

inline error_t
Graph::get_behavior_notes(std::vector<BehaviorNote_t> &notes) const {
    int64_t const candidate = plans.candidate;
    RETURN_CUDNN_FRONTEND_ERROR_IF(
        candidate == -1,
        error_code_t::INVALID_VALUE,
        "No candiate plan set for the graph. You can set one by building a plan, which in turn sets the "
        "candidate internally. Do note that you also query behaviour notes for a created-but-not-built plan by using "
        "get_behavior_notes_for_plan_at_index API.");

    CHECK_CUDNN_FRONTEND_ERROR(get_behavior_notes_for_plan_at_index(candidate, notes));
    return {error_code_t::OK, ""};
}

inline int64_t
Graph::get_execution_plan_count() const {
    return plans.execution_plans.size();
}

inline error_t
Graph::get_engine_count(int64_t &count) {
    CHECK_CUDNN_ERROR(detail::get_attribute(operation_graph->get_raw_desc(),
                                            CUDNN_ATTR_OPERATIONGRAPH_ENGINE_GLOBAL_COUNT,
                                            CUDNN_TYPE_INT64,
                                            1,
                                            nullptr,
                                            &count));

    return {error_code_t::OK, ""};
}

inline error_t
Graph::get_knobs_for_engine(int64_t const engine, std::vector<Knob> &knobs) {
    CHECK_CUDNN_FRONTEND_ERROR(detail::query_knobs(engine, operation_graph->get_raw_desc(), knobs));

    return {error_code_t::OK, ""};
}

inline error_t
Graph::create_execution_plans(std::vector<HeurMode_t> const &mode) {
    EngineConfigList op_graph_to_configs;
    CHECK_CUDNN_FRONTEND_ERROR(
        detail::query_cudnn_heuristics_impl(operation_graph, op_graph_to_configs, mode, context.get_target_sm_count()));

    CUDNN_FE_LOG_LABEL_ENDL("INFO: Extracting engine configs.");

    plans.set_tag(operation_graph->getTag());
    plans.enqueue_engine_configs(op_graph_to_configs);
    plans.set_kernel_cache(kernel_cache);

    CUDNN_FE_LOG_LABEL_ENDL("INFO: Querying engine config properties.");
    CHECK_CUDNN_FRONTEND_ERROR(plans.query_properties());

    return {error_code_t::OK, ""};
}

inline error_t
Graph::create_execution_plan(int64_t const engine_id, std::unordered_map<KnobType_t, int64_t> const &user_knobs) {
    // first create the engine
    // this just uses the global engine id and operation graph
    detail::backend_descriptor engine(CUDNN_BACKEND_ENGINE_DESCRIPTOR);
    RETURN_CUDNN_FRONTEND_ERROR_IF(engine.get_status() != CUDNN_STATUS_SUCCESS,
                                   error_code_t::CUDNN_BACKEND_API_FAILED,
                                   "Failed to create engine's backend descriptor.");
    CHECK_CUDNN_FRONTEND_ERROR(detail::create_engine(engine, engine_id, operation_graph->get_raw_desc()));

    // Create an array of knob choices
    std::vector<detail::backend_descriptor> knob_choices;
    CHECK_CUDNN_FRONTEND_ERROR(detail::set_knob_choices(user_knobs, knob_choices));

    auto engine_config = make_shared_backend_pointer((cudnnBackendDescriptorType_t)CUDNN_BACKEND_ENGINECFG_DESCRIPTOR);
    CHECK_CUDNN_FRONTEND_ERROR(detail::create_engine_config(engine_config, engine, knob_choices));
    plans.enqueue_engine_configs({engine_config});
    CHECK_CUDNN_FRONTEND_ERROR(plans.query_properties());

    return {error_code_t::OK, ""};
}

inline error_t
Graph::build_plan_at_index(cudnnHandle_t const &handle, int64_t plan_index) {
    CHECK_CUDNN_FRONTEND_ERROR(plans.build_plan_at_index(handle, plan_index));
    return {error_code_t::OK, ""};
}

inline error_t
Graph::build_plans(cudnnHandle_t const &handle, BuildPlanPolicy_t const policy, bool const do_multithreaded_builds) {
    CHECK_CUDNN_FRONTEND_ERROR(plans.build_plans(handle, policy, do_multithreaded_builds));
    return {error_code_t::OK, ""};
}

inline error_t
Graph::build(cudnnHandle_t const &handle,
             std::vector<HeurMode_t> const &modes,
             BuildPlanPolicy_t const policy,
             bool const do_multithreaded_builds) {
    CHECK_CUDNN_FRONTEND_ERROR(this->validate());
    CHECK_CUDNN_FRONTEND_ERROR(this->build_operation_graph(handle));
    CHECK_CUDNN_FRONTEND_ERROR(this->create_execution_plans(modes));
    CHECK_CUDNN_FRONTEND_ERROR(this->check_support(handle));
    CHECK_CUDNN_FRONTEND_ERROR(this->build_plans(handle, policy, do_multithreaded_builds));
    return {error_code_t::OK, ""};
}

inline Graph &
Graph::set_intermediate_data_type(DataType_t const type) {
    context.set_intermediate_data_type(type);
    return *this;
}

inline Graph &
Graph::set_io_data_type(DataType_t const type) {
    context.set_io_data_type(type);
    return *this;
}

inline Graph &
Graph::set_compute_data_type(DataType_t const type) {
    context.set_compute_data_type(type);
    return *this;
}

inline Graph &
Graph::set_dynamic_shape_enabled(bool is_enabled) {
    is_dynamic_shape_enabled = is_enabled;
    return *this;
}

inline Graph &
Graph::set_kernel_cache(std::shared_ptr<KernelCache> cache) {
    kernel_cache = cache;
    return *this;
}

inline Graph &
Graph::set_sm_count(int32_t count) {
    context.set_target_sm_count(count);
    return *this;
}

inline Graph &
Graph::set_sm_version(int32_t version) {
    context.set_sm_version(version);
    return *this;
}

inline std::shared_ptr<Tensor_attributes>
Graph::tensor(Tensor_attributes const &tensor) {
    auto tensor_ptr = std::make_shared<Tensor_attributes>(tensor);
    full_graph_inputs.emplace(tensor_ptr);
    return tensor_ptr;
}

inline error_t
Graph::query_tensor_attributes_of_uid(int64_t const uid, Tensor_attributes &tensor) const {
    for (auto const &o_tensor : full_graph_outputs) {
        if (uid == o_tensor->get_uid()) {
            tensor = *o_tensor;
            return {error_code_t::OK, ""};
        }
    }

    for (auto const &i_tensor : full_graph_inputs) {
        if (uid == i_tensor->get_uid()) {
            tensor = *i_tensor;
            return {error_code_t::OK, ""};
        }
    }

    for (auto const &d_tensor : deserialized_tensor_properties) {
        if (uid == d_tensor->get_uid()) {
            tensor = *d_tensor;
            return {error_code_t::OK, ""};
        }
    }

    return {error_code_t::INVALID_VALUE, "No matching tensor for this UID"};
}

// tensor_like is meant to create "useable" copies of a tensor.
// By usable, it means not copying over the uids, as uids are FE-level(internal) detail.
// It also means not copying over names, which are user-level(external) detail. But user is given option to provide a
// new name.
inline std::shared_ptr<Tensor_attributes>
Graph::tensor_like(std::shared_ptr<Tensor_attributes> const &tensor, std::string const &name) {
    auto tensor_ptr = std::make_shared<Tensor_attributes>(*tensor);

    // reset the uid of the cloned tensor
    // uids are not meant to be copied by tensor_like
    // When lowering to cudnn backend, both tensors involved here will get unique uids.
    tensor_ptr->clear_uid();

    // reset the name too. Defaults to empty string.
    tensor_ptr->set_name(name);
    full_graph_inputs.emplace(tensor_ptr);

    return tensor_ptr;
}

inline std::array<std::shared_ptr<Tensor_attributes>, 6>
Graph::bn_finalize(std::shared_ptr<Tensor_attributes> sum,
                   std::shared_ptr<Tensor_attributes> sq_sum,
                   std::shared_ptr<Tensor_attributes> scale,
                   std::shared_ptr<Tensor_attributes> bias,
                   std::shared_ptr<Tensor_attributes> epsilon,
                   std::shared_ptr<Tensor_attributes> accum_count,
                   BN_finalize_attributes attributes) {
    // Set outputs
    auto EQ_SCALE = attributes.outputs[BN_finalize_attributes::output_names::EQ_SCALE] =
        output_tensor(attributes.name + "::EQ_SCALE");
    auto EQ_BIAS = attributes.outputs[BN_finalize_attributes::output_names::EQ_BIAS] =
        output_tensor(attributes.name + "::EQ_BIAS");
    auto MEAN = attributes.outputs[BN_finalize_attributes::output_names::MEAN] =
        output_tensor(attributes.name + "::MEAN");
    auto INV_VARIANCE = attributes.outputs[BN_finalize_attributes::output_names::INV_VARIANCE] =
        output_tensor(attributes.name + "::INV_VARIANCE");
    std::shared_ptr<Tensor_attributes> NEXT_RUNNING_MEAN = nullptr;
    std::shared_ptr<Tensor_attributes> NEXT_RUNNING_VAR  = nullptr;
    if (attributes.inputs[BN_finalize_attributes::input_names::PREV_RUNNING_MEAN] &&
        attributes.inputs[BN_finalize_attributes::input_names::PREV_RUNNING_VAR] &&
        attributes.inputs[BN_finalize_attributes::input_names::MOMENTUM]) {
        NEXT_RUNNING_MEAN = output_tensor(attributes.name + "::NEXT_RUNNING_MEAN");
        NEXT_RUNNING_VAR  = output_tensor(attributes.name + "::NEXT_RUNNING_VAR");
    }
    attributes.outputs[BN_finalize_attributes::output_names::NEXT_RUNNING_MEAN] = NEXT_RUNNING_MEAN;
    attributes.outputs[BN_finalize_attributes::output_names::NEXT_RUNNING_VAR]  = NEXT_RUNNING_VAR;

    // Set inputs
    attributes.inputs[BN_finalize_attributes::input_names::SUM]         = sum;
    attributes.inputs[BN_finalize_attributes::input_names::SQ_SUM]      = sq_sum;
    attributes.inputs[BN_finalize_attributes::input_names::SCALE]       = scale;
    attributes.inputs[BN_finalize_attributes::input_names::BIAS]        = bias;
    attributes.inputs[BN_finalize_attributes::input_names::EPSILON]     = epsilon;
    attributes.inputs[BN_finalize_attributes::input_names::ACCUM_COUNT] = accum_count;

    sub_nodes.emplace_back(std::make_unique<BatchNormFinalizeNode>(std::move(attributes), context));

    return {EQ_SCALE, EQ_BIAS, MEAN, INV_VARIANCE, NEXT_RUNNING_MEAN, NEXT_RUNNING_VAR};
}

inline std::array<std::shared_ptr<Tensor_attributes>, 3>
Graph::layernorm(std::shared_ptr<Tensor_attributes> x,
                 std::shared_ptr<Tensor_attributes> scale,
                 std::shared_ptr<Tensor_attributes> bias,
                 Layernorm_attributes attributes) {
    // Set outputs
    auto Y = attributes.outputs[Layernorm_attributes::output_names::Y] = output_tensor(attributes.name + "::Y");
    std::shared_ptr<Tensor_attributes> MEAN                            = nullptr;
    std::shared_ptr<Tensor_attributes> INV_VARIANCE                    = nullptr;
    if (attributes.forward_phase == NormFwdPhase_t::TRAINING) {
        MEAN = attributes.outputs[Layernorm_attributes::output_names::MEAN] = output_tensor(attributes.name + "::MEAN");
        INV_VARIANCE = attributes.outputs[Layernorm_attributes::output_names::INV_VARIANCE] =
            output_tensor(attributes.name + "::INV_VARIANCE");
    }
    // Set inputs
    attributes.inputs[Layernorm_attributes::input_names::X]     = x;
    attributes.inputs[Layernorm_attributes::input_names::SCALE] = scale;
    attributes.inputs[Layernorm_attributes::input_names::BIAS]  = bias;

    sub_nodes.emplace_back(std::make_unique<LayerNormNode>(std::move(attributes), context));

    return {Y, MEAN, INV_VARIANCE};
}

inline std::array<std::shared_ptr<Tensor_attributes>, 3>
Graph::instancenorm(std::shared_ptr<Tensor_attributes> x,
                    std::shared_ptr<Tensor_attributes> scale,
                    std::shared_ptr<Tensor_attributes> bias,
                    Instancenorm_attributes attributes) {
    // Set outputs
    auto Y = attributes.outputs[Instancenorm_attributes::output_names::Y] = output_tensor(attributes.name + "::Y");
    std::shared_ptr<Tensor_attributes> MEAN                               = nullptr;
    std::shared_ptr<Tensor_attributes> INV_VARIANCE                       = nullptr;
    if (attributes.forward_phase == NormFwdPhase_t::TRAINING) {
        MEAN = attributes.outputs[Instancenorm_attributes::output_names::MEAN] =
            output_tensor(attributes.name + "::MEAN");
        INV_VARIANCE = attributes.outputs[Instancenorm_attributes::output_names::INV_VARIANCE] =
            output_tensor(attributes.name + "::INV_VARIANCE");
    }
    // Set inputs
    attributes.inputs[Instancenorm_attributes::input_names::X]     = x;
    attributes.inputs[Instancenorm_attributes::input_names::SCALE] = scale;
    attributes.inputs[Instancenorm_attributes::input_names::BIAS]  = bias;

    sub_nodes.emplace_back(std::make_unique<InstanceNormNode>(std::move(attributes), context));

    return {Y, MEAN, INV_VARIANCE};
}

inline std::array<std::shared_ptr<Tensor_attributes>, 5>
Graph::batchnorm(std::shared_ptr<Tensor_attributes> x,
                 std::shared_ptr<Tensor_attributes> scale,
                 std::shared_ptr<Tensor_attributes> bias,
                 Batchnorm_attributes attributes) {
    // Set outputs
    auto Y = attributes.outputs[Batchnorm_attributes::output_names::Y] = output_tensor(attributes.name + "::Y");
    auto MEAN = attributes.outputs[Batchnorm_attributes::output_names::MEAN] =
        output_tensor(attributes.name + "::MEAN");
    auto INV_VARIANCE = attributes.outputs[Batchnorm_attributes::output_names::INV_VARIANCE] =
        output_tensor(attributes.name + "::INV_VARIANCE");
    std::shared_ptr<Tensor_attributes> NEXT_RUNNING_MEAN = nullptr;
    std::shared_ptr<Tensor_attributes> NEXT_RUNNING_VAR  = nullptr;
    if (attributes.inputs[Batchnorm_attributes::input_names::PREV_RUNNING_MEAN] &&
        attributes.inputs[Batchnorm_attributes::input_names::PREV_RUNNING_VAR] &&
        attributes.inputs[Batchnorm_attributes::input_names::MOMENTUM]) {
        NEXT_RUNNING_MEAN = output_tensor(attributes.name + "::NEXT_RUNNING_MEAN");
        NEXT_RUNNING_VAR  = output_tensor(attributes.name + "::NEXT_RUNNING_VAR");
    }
    attributes.outputs[Batchnorm_attributes::output_names::NEXT_RUNNING_MEAN] = NEXT_RUNNING_MEAN;
    attributes.outputs[Batchnorm_attributes::output_names::NEXT_RUNNING_VAR]  = NEXT_RUNNING_VAR;

    // Set inputs
    attributes.inputs[Batchnorm_attributes::input_names::X]     = x;
    attributes.inputs[Batchnorm_attributes::input_names::SCALE] = scale;
    attributes.inputs[Batchnorm_attributes::input_names::BIAS]  = bias;

    sub_nodes.emplace_back(std::make_unique<BatchNormNode>(std::move(attributes), context));

    return {Y, MEAN, INV_VARIANCE, NEXT_RUNNING_MEAN, NEXT_RUNNING_VAR};
}

inline std::shared_ptr<Tensor_attributes>
Graph::batchnorm_inference(std::shared_ptr<Tensor_attributes> x,
                           std::shared_ptr<Tensor_attributes> mean,
                           std::shared_ptr<Tensor_attributes> inv_variance,
                           std::shared_ptr<Tensor_attributes> scale,
                           std::shared_ptr<Tensor_attributes> bias,
                           Batchnorm_inference_attributes attributes) {
    // Set outputs
    auto Y = attributes.outputs[Batchnorm_inference_attributes::output_names::Y] =
        output_tensor(attributes.name + "::Y");

    // Set inputs
    attributes.inputs[Batchnorm_inference_attributes::input_names::X]            = x;
    attributes.inputs[Batchnorm_inference_attributes::input_names::MEAN]         = mean;
    attributes.inputs[Batchnorm_inference_attributes::input_names::INV_VARIANCE] = inv_variance;
    attributes.inputs[Batchnorm_inference_attributes::input_names::SCALE]        = scale;
    attributes.inputs[Batchnorm_inference_attributes::input_names::BIAS]         = bias;

    sub_nodes.emplace_back(std::make_unique<BatchnormInferenceNode>(std::move(attributes), context));

    return Y;
}

inline std::array<std::shared_ptr<Tensor_attributes>, 3>
Graph::batchnorm_backward(std::shared_ptr<Tensor_attributes> dy,
                          std::shared_ptr<Tensor_attributes> x,
                          std::shared_ptr<Tensor_attributes> scale,
                          Batchnorm_backward_attributes attributes) {
    // Set outputs
    auto DX = attributes.outputs[Batchnorm_backward_attributes::output_names::DX] =
        output_tensor(attributes.name + "::DX");
    auto DSCALE = attributes.outputs[Batchnorm_backward_attributes::output_names::DSCALE] =
        output_tensor(attributes.name + "::DSCALE");
    auto DBIAS = attributes.outputs[Batchnorm_backward_attributes::output_names::DBIAS] =
        output_tensor(attributes.name + "::DBIAS");

    // Set inputs
    attributes.inputs[Batchnorm_backward_attributes::input_names::DY]    = dy;
    attributes.inputs[Batchnorm_backward_attributes::input_names::X]     = x;
    attributes.inputs[Batchnorm_backward_attributes::input_names::SCALE] = scale;

    sub_nodes.emplace_back(std::make_unique<DBNNode>(std::move(attributes), context));

    return {DX, DSCALE, DBIAS};
}

inline std::array<std::shared_ptr<Tensor_attributes>, 3>
Graph::instancenorm_backward(std::shared_ptr<Tensor_attributes> dy,
                             std::shared_ptr<Tensor_attributes> x,
                             std::shared_ptr<Tensor_attributes> scale,
                             Instancenorm_backward_attributes attributes) {
    // Set outputs
    auto DX = attributes.outputs[Instancenorm_backward_attributes::output_names::DX] =
        output_tensor(attributes.name + "::DX");
    auto DSCALE = attributes.outputs[Instancenorm_backward_attributes::output_names::DSCALE] =
        output_tensor(attributes.name + "::DSCALE");
    auto DBIAS = attributes.outputs[Instancenorm_backward_attributes::output_names::DBIAS] =
        output_tensor(attributes.name + "::DBIAS");

    // Set inputs
    attributes.inputs[Instancenorm_backward_attributes::input_names::DY]    = dy;
    attributes.inputs[Instancenorm_backward_attributes::input_names::X]     = x;
    attributes.inputs[Instancenorm_backward_attributes::input_names::SCALE] = scale;

    sub_nodes.emplace_back(std::make_unique<DINNode>(std::move(attributes), context));

    return {DX, DSCALE, DBIAS};
}

inline std::array<std::shared_ptr<Tensor_attributes>, 3>
Graph::layernorm_backward(std::shared_ptr<Tensor_attributes> dy,
                          std::shared_ptr<Tensor_attributes> x,
                          std::shared_ptr<Tensor_attributes> scale,
                          Layernorm_backward_attributes attributes) {
    // Set outputs
    auto DX = attributes.outputs[Layernorm_backward_attributes::output_names::DX] =
        output_tensor(attributes.name + "::DX");
    auto DSCALE = attributes.outputs[Layernorm_backward_attributes::output_names::DSCALE] =
        output_tensor(attributes.name + "::DSCALE");
    auto DBIAS = attributes.outputs[Layernorm_backward_attributes::output_names::DBIAS] =
        output_tensor(attributes.name + "::DBIAS");

    // Set inputs
    attributes.inputs[Layernorm_backward_attributes::input_names::DY]    = dy;
    attributes.inputs[Layernorm_backward_attributes::input_names::X]     = x;
    attributes.inputs[Layernorm_backward_attributes::input_names::SCALE] = scale;

    sub_nodes.emplace_back(std::make_unique<DLNNode>(std::move(attributes), context));

    return {DX, DSCALE, DBIAS};
}

inline std::shared_ptr<Tensor_attributes>
Graph::conv_fprop(std::shared_ptr<Tensor_attributes> x,
                  std::shared_ptr<Tensor_attributes> w,
                  Conv_fprop_attributes attributes) {
    // Make required output tensors
    if (attributes.name.empty()) {
        attributes.name += std::to_string(sub_nodes.size());
    }
    auto Y                                                     = output_tensor(attributes.name + "::Y");
    attributes.outputs[Conv_fprop_attributes::output_names::Y] = Y;

    // Set inputs
    attributes.inputs[Conv_fprop_attributes::input_names::X] = x;
    attributes.inputs[Conv_fprop_attributes::input_names::W] = w;

    sub_nodes.emplace_back(std::make_unique<ConvolutionNode>(std::move(attributes), context));

    return Y;
}

inline std::array<std::shared_ptr<Tensor_attributes>, 5>
Graph::dbn_weight(std::shared_ptr<Tensor_attributes> dy,
                  std::shared_ptr<Tensor_attributes> x,
                  std::shared_ptr<Tensor_attributes> mean,
                  std::shared_ptr<Tensor_attributes> inv_variance,
                  std::shared_ptr<Tensor_attributes> scale,
                  DBN_weight_attributes attributes) {
    if (attributes.name.empty()) {
        attributes.name += std::to_string(sub_nodes.size());
    }
    // Make required output tensors
    auto DBIAS = attributes.outputs[DBN_weight_attributes::output_names::DBIAS] =
        output_tensor(attributes.name + "::DBIAS");
    auto DSCALE = attributes.outputs[DBN_weight_attributes::output_names::DSCALE] =
        output_tensor(attributes.name + "::DSCALE");
    auto EQ_BIAS = attributes.outputs[DBN_weight_attributes::output_names::EQ_BIAS] =
        output_tensor(attributes.name + "::EQ_BIAS");
    auto EQ_SCALE_DY = attributes.outputs[DBN_weight_attributes::output_names::EQ_SCALE_DY] =
        output_tensor(attributes.name + "::EQ_SCALE_DY");
    auto EQ_SCALE_X = attributes.outputs[DBN_weight_attributes::output_names::EQ_SCALE_X] =
        output_tensor(attributes.name + "::EQ_SCALE_X");

    // Set inputs
    attributes.inputs[DBN_weight_attributes::input_names::DY]           = dy;
    attributes.inputs[DBN_weight_attributes::input_names::X]            = x;
    attributes.inputs[DBN_weight_attributes::input_names::SCALE]        = scale;
    attributes.inputs[DBN_weight_attributes::input_names::MEAN]         = mean;
    attributes.inputs[DBN_weight_attributes::input_names::INV_VARIANCE] = inv_variance;

    sub_nodes.emplace_back(std::make_unique<DBNWeightNode>(std::move(attributes), context));

    return {DSCALE, DBIAS, EQ_SCALE_DY, EQ_SCALE_X, EQ_BIAS};
}

inline std::shared_ptr<Tensor_attributes>
Graph::conv_dgrad(std::shared_ptr<Tensor_attributes> dy,
                  std::shared_ptr<Tensor_attributes> w,
                  Conv_dgrad_attributes attributes) {
    // Make required output tensors
    auto DX = attributes.outputs[Conv_dgrad_attributes::output_names::DX] = output_tensor(attributes.name + "::DX");

    // Set inputs
    attributes.inputs[Conv_dgrad_attributes::input_names::DY] = dy;
    attributes.inputs[Conv_dgrad_attributes::input_names::W]  = w;

    sub_nodes.emplace_back(std::make_unique<DgradNode>(std::move(attributes), context));

    return DX;
}

inline std::array<std::shared_ptr<Tensor_attributes>, 2>
Graph::genstats(std::shared_ptr<Tensor_attributes> x, Genstats_attributes attributes) {
    // Set outputs
    auto SUM = attributes.outputs[Genstats_attributes::output_names::SUM] =
        output_tensor(attributes.name + "_sum_output");
    auto SQ_SUM = attributes.outputs[Genstats_attributes::output_names::SQ_SUM] =
        output_tensor(attributes.name + "_sq_sum_output");

    // Set inputs
    attributes.inputs[Genstats_attributes::input_names::X] = x;

    sub_nodes.emplace_back(std::make_unique<GenstatsNode>(std::move(attributes), context));

    return {SUM, SQ_SUM};
}

inline std::shared_ptr<Tensor_attributes>
Graph::conv_wgrad(std::shared_ptr<Tensor_attributes> dy,
                  std::shared_ptr<Tensor_attributes> x,
                  Conv_wgrad_attributes attributes) {
    // Make required output tensors
    auto DW = attributes.outputs[Conv_wgrad_attributes::output_names::DW] = output_tensor(attributes.name + "::DW");

    // Set inputs
    attributes.inputs[Conv_wgrad_attributes::input_names::X]  = x;
    attributes.inputs[Conv_wgrad_attributes::input_names::DY] = dy;

    sub_nodes.emplace_back(std::make_unique<WgradNode>(std::move(attributes), context));

    return DW;
}

inline std::array<std::shared_ptr<Tensor_attributes>, 2>
Graph::rmsnorm(std::shared_ptr<Tensor_attributes> x,
               std::shared_ptr<Tensor_attributes> scale,
               Rmsnorm_attributes attributes) {
    // Set outputs
    auto Y = attributes.outputs[Rmsnorm_attributes::output_names::Y] = output_tensor(attributes.name + "::Y");
    std::shared_ptr<Tensor_attributes> INV_VARIANCE                  = nullptr;
    if (attributes.forward_phase == NormFwdPhase_t::TRAINING) {
        INV_VARIANCE = attributes.outputs[Rmsnorm_attributes::output_names::INV_VARIANCE] =
            output_tensor(attributes.name + "::INV_VARIANCE");
    }
    // Set inputs
    attributes.inputs[Rmsnorm_attributes::input_names::X]     = x;
    attributes.inputs[Rmsnorm_attributes::input_names::SCALE] = scale;

    sub_nodes.emplace_back(std::make_unique<RMSNormNode>(std::move(attributes), context));

    return {Y, INV_VARIANCE};
}

inline std::array<std::shared_ptr<Tensor_attributes>, 3>
Graph::rmsnorm_backward(std::shared_ptr<Tensor_attributes> dy,
                        std::shared_ptr<Tensor_attributes> x,
                        std::shared_ptr<Tensor_attributes> scale,
                        std::shared_ptr<Tensor_attributes> inv_variance,
                        Rmsnorm_backward_attributes attributes) {
    // Set outputs
    auto DX = attributes.outputs[Rmsnorm_backward_attributes::output_names::DX] =
        output_tensor(attributes.name + "::DX");
    auto DScale = attributes.outputs[Rmsnorm_backward_attributes::output_names::DSCALE] =
        output_tensor(attributes.name + "::Dscale");
    std::shared_ptr<Tensor_attributes> DBias = nullptr;
    if (attributes.use_dbias.value_or(true)) {
        DBias = attributes.outputs[Rmsnorm_backward_attributes::output_names::DBIAS] =
            output_tensor(attributes.name + "::Dbias");
    }

    // Set inputs
    attributes.inputs[Rmsnorm_backward_attributes::input_names::DY]           = dy;
    attributes.inputs[Rmsnorm_backward_attributes::input_names::X]            = x;
    attributes.inputs[Rmsnorm_backward_attributes::input_names::SCALE]        = scale;
    attributes.inputs[Rmsnorm_backward_attributes::input_names::INV_VARIANCE] = inv_variance;

    sub_nodes.emplace_back(std::make_unique<DRMSNormNode>(std::move(attributes), context));

    return {DX, DScale, DBias};
}

inline std::array<std::shared_ptr<Tensor_attributes>, 2>
Graph::sdpa(std::shared_ptr<Tensor_attributes> q,
            std::shared_ptr<Tensor_attributes> k,
            std::shared_ptr<Tensor_attributes> v,
            SDPA_attributes attributes) {
    // Make required output tensors
    auto O = attributes.outputs[SDPA_attributes::output_names::O] = output_tensor(attributes.name + "::O");

    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> Stats = nullptr;
    if (attributes.is_inference == false) {
        Stats = attributes.outputs[SDPA_attributes::output_names::Stats] = output_tensor(attributes.name + "::Stats");
    }

    // Set inputs
    attributes.inputs[SDPA_attributes::input_names::Q] = q;
    attributes.inputs[SDPA_attributes::input_names::K] = k;
    attributes.inputs[SDPA_attributes::input_names::V] = v;

    sub_nodes.emplace_back(std::make_unique<SDPANode>(std::move(attributes), context));

    return {O, Stats};
}

inline std::array<std::shared_ptr<Tensor_attributes>, 4>
Graph::sdpa_fp8(std::shared_ptr<Tensor_attributes> q,
                std::shared_ptr<Tensor_attributes> k,
                std::shared_ptr<Tensor_attributes> v,
                std::shared_ptr<Tensor_attributes> descale_q,
                std::shared_ptr<Tensor_attributes> descale_k,
                std::shared_ptr<Tensor_attributes> descale_v,
                std::shared_ptr<Tensor_attributes> descale_s,
                std::shared_ptr<Tensor_attributes> scale_s,
                std::shared_ptr<Tensor_attributes> scale_o,
                SDPA_fp8_attributes attributes) {
    // Make required output tensors
    auto O = attributes.outputs[SDPA_fp8_attributes::output_names::O] = output_tensor(attributes.name + "::O");

    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> Stats = nullptr;
    if (attributes.is_inference == false) {
        Stats = attributes.outputs[SDPA_fp8_attributes::output_names::Stats] =
            output_tensor(attributes.name + "::Stats");
    }

    auto Amax_S = attributes.outputs[SDPA_fp8_attributes::output_names::Amax_S] =
        output_tensor(attributes.name + "::Amax_S");
    auto Amax_O = attributes.outputs[SDPA_fp8_attributes::output_names::Amax_O] =
        output_tensor(attributes.name + "::Amax_O");

    // Set inputs
    attributes.inputs[SDPA_fp8_attributes::input_names::Q] = q;
    attributes.inputs[SDPA_fp8_attributes::input_names::K] = k;
    attributes.inputs[SDPA_fp8_attributes::input_names::V] = v;

    attributes.inputs[SDPA_fp8_attributes::input_names::Descale_Q] = descale_q;
    attributes.inputs[SDPA_fp8_attributes::input_names::Descale_K] = descale_k;
    attributes.inputs[SDPA_fp8_attributes::input_names::Descale_V] = descale_v;
    attributes.inputs[SDPA_fp8_attributes::input_names::Descale_S] = descale_s;
    attributes.inputs[SDPA_fp8_attributes::input_names::Scale_S]   = scale_s;
    attributes.inputs[SDPA_fp8_attributes::input_names::Scale_O]   = scale_o;

    sub_nodes.emplace_back(std::make_unique<SDPAFP8Node>(std::move(attributes), context));

    return {O, Stats, Amax_S, Amax_O};
}

inline std::array<std::shared_ptr<Tensor_attributes>, 7>
Graph::sdpa_fp8_backward(std::shared_ptr<Tensor_attributes> q,
                         std::shared_ptr<Tensor_attributes> k,
                         std::shared_ptr<Tensor_attributes> v,
                         std::shared_ptr<Tensor_attributes> o,
                         std::shared_ptr<Tensor_attributes> dO,
                         std::shared_ptr<Tensor_attributes> Stats,
                         std::shared_ptr<Tensor_attributes> descale_q,
                         std::shared_ptr<Tensor_attributes> descale_k,
                         std::shared_ptr<Tensor_attributes> descale_v,
                         std::shared_ptr<Tensor_attributes> descale_o,
                         std::shared_ptr<Tensor_attributes> descale_do,
                         std::shared_ptr<Tensor_attributes> descale_s,
                         std::shared_ptr<Tensor_attributes> descale_dp,
                         std::shared_ptr<Tensor_attributes> scale_s,
                         std::shared_ptr<Tensor_attributes> scale_dq,
                         std::shared_ptr<Tensor_attributes> scale_dk,
                         std::shared_ptr<Tensor_attributes> scale_dv,
                         std::shared_ptr<Tensor_attributes> scale_dp,
                         SDPA_fp8_backward_attributes attributes) {
    // Make required output tensors
    auto dQ = attributes.outputs[SDPA_fp8_backward_attributes::output_names::dQ] =
        output_tensor(attributes.name + "::dQ");
    auto dK = attributes.outputs[SDPA_fp8_backward_attributes::output_names::dK] =
        output_tensor(attributes.name + "::dK");
    auto dV = attributes.outputs[SDPA_fp8_backward_attributes::output_names::dV] =
        output_tensor(attributes.name + "::dV");
    auto Amax_dQ = attributes.outputs[SDPA_fp8_backward_attributes::output_names::Amax_dQ] =
        output_tensor(attributes.name + "::Amax_dQ");
    auto Amax_dK = attributes.outputs[SDPA_fp8_backward_attributes::output_names::Amax_dK] =
        output_tensor(attributes.name + "::Amax_dK");
    auto Amax_dV = attributes.outputs[SDPA_fp8_backward_attributes::output_names::Amax_dV] =
        output_tensor(attributes.name + "::Amax_dV");
    auto Amax_dP = attributes.outputs[SDPA_fp8_backward_attributes::output_names::Amax_dP] =
        output_tensor(attributes.name + "::Amax_dP");

    // Set inputs
    attributes.inputs[SDPA_fp8_backward_attributes::input_names::Q]     = q;
    attributes.inputs[SDPA_fp8_backward_attributes::input_names::K]     = k;
    attributes.inputs[SDPA_fp8_backward_attributes::input_names::V]     = v;
    attributes.inputs[SDPA_fp8_backward_attributes::input_names::O]     = o;
    attributes.inputs[SDPA_fp8_backward_attributes::input_names::Stats] = Stats;
    attributes.inputs[SDPA_fp8_backward_attributes::input_names::dO]    = dO;

    attributes.inputs[SDPA_fp8_backward_attributes::input_names::Descale_Q]  = descale_q;
    attributes.inputs[SDPA_fp8_backward_attributes::input_names::Descale_K]  = descale_k;
    attributes.inputs[SDPA_fp8_backward_attributes::input_names::Descale_V]  = descale_v;
    attributes.inputs[SDPA_fp8_backward_attributes::input_names::Descale_S]  = descale_s;
    attributes.inputs[SDPA_fp8_backward_attributes::input_names::Descale_O]  = descale_o;
    attributes.inputs[SDPA_fp8_backward_attributes::input_names::Descale_dO] = descale_do;
    attributes.inputs[SDPA_fp8_backward_attributes::input_names::Descale_dP] = descale_dp;

    attributes.inputs[SDPA_fp8_backward_attributes::input_names::Scale_dQ] = scale_dq;
    attributes.inputs[SDPA_fp8_backward_attributes::input_names::Scale_dK] = scale_dk;
    attributes.inputs[SDPA_fp8_backward_attributes::input_names::Scale_dV] = scale_dv;
    attributes.inputs[SDPA_fp8_backward_attributes::input_names::Scale_S]  = scale_s;
    attributes.inputs[SDPA_fp8_backward_attributes::input_names::Scale_dP] = scale_dp;

    sub_nodes.emplace_back(std::make_unique<SDPAFP8BackwardNode>(std::move(attributes), context));

    return {dQ, dK, dV, Amax_dQ, Amax_dK, Amax_dV, Amax_dP};
}

inline std::array<std::shared_ptr<Tensor_attributes>, 3>
Graph::sdpa_backward(std::shared_ptr<Tensor_attributes> q,
                     std::shared_ptr<Tensor_attributes> k,
                     std::shared_ptr<Tensor_attributes> v,
                     std::shared_ptr<Tensor_attributes> o,
                     std::shared_ptr<Tensor_attributes> dO,
                     std::shared_ptr<Tensor_attributes> stats,
                     SDPA_backward_attributes attributes) {
    // Set inputs
    attributes.inputs[SDPA_backward_attributes::input_names::Q]     = q;
    attributes.inputs[SDPA_backward_attributes::input_names::K]     = k;
    attributes.inputs[SDPA_backward_attributes::input_names::V]     = v;
    attributes.inputs[SDPA_backward_attributes::input_names::O]     = o;
    attributes.inputs[SDPA_backward_attributes::input_names::dO]    = dO;
    attributes.inputs[SDPA_backward_attributes::input_names::Stats] = stats;

    // Make required output tensors
    auto dQ = attributes.outputs[SDPA_backward_attributes::output_names::dQ] = output_tensor(attributes.name + "::dQ");
    auto dK = attributes.outputs[SDPA_backward_attributes::output_names::dK] = output_tensor(attributes.name + "::dK");
    auto dV = attributes.outputs[SDPA_backward_attributes::output_names::dV] = output_tensor(attributes.name + "::dV");

    sub_nodes.emplace_back(std::make_unique<SDPABackwardNode>(std::move(attributes), context));

    return {dQ, dK, dV};
}

inline std::shared_ptr<Tensor_attributes>
Graph::slice(std::shared_ptr<Tensor_attributes> input, Slice_attributes attributes) {
    attributes.inputs[Slice_attributes::input_names::X] = input;
    auto Y = attributes.outputs[Slice_attributes::output_names::Y] = output_tensor(attributes.name + "::Y");

    sub_nodes.emplace_back(std::make_unique<SliceNode>(std::move(attributes), context));
    return Y;
}

inline std::array<std::shared_ptr<Tensor_attributes>, 2>
Graph::block_scale_quantize(std::shared_ptr<Tensor_attributes> x, Block_scale_quantize_attributes attributes) {
    // Set outputs
    auto Y = attributes.outputs[Block_scale_quantize_attributes::output_names::Y] =
        output_tensor(attributes.name + "::Y");
    auto scale = attributes.outputs[Block_scale_quantize_attributes::output_names::scale] =
        output_tensor(attributes.name + "::scale");

    // Set inputs
    attributes.inputs[Block_scale_quantize_attributes::input_names::X] = x;

    sub_nodes.emplace_back(std::make_unique<BlockScaleQuantizeNode>(std::move(attributes), context));

    return {Y, scale};
}

inline std::shared_ptr<Tensor_attributes>
Graph::block_scale_dequantize(std::shared_ptr<Tensor_attributes> x,
                              std::shared_ptr<Tensor_attributes> scale,
                              Block_scale_dequantize_attributes attributes) {
    // Set outputs
    auto Y = attributes.outputs[Block_scale_dequantize_attributes::output_names::Y] =
        output_tensor(attributes.name + "::Y");

    // Set inputs
    attributes.inputs[Block_scale_dequantize_attributes::input_names::X]     = x;
    attributes.inputs[Block_scale_dequantize_attributes::input_names::scale] = scale;

    sub_nodes.emplace_back(std::make_unique<BlockScaleDequantizeNode>(std::move(attributes), context));

    return Y;
}

inline std::shared_ptr<Tensor_attributes>
Graph::concatenate(std::vector<std::shared_ptr<Tensor_attributes>> x, Concatenate_attributes attributes) {
    if (attributes.name.empty()) {
        attributes.name += std::to_string(sub_nodes.size());
    }

    // Set outputs
    auto Y = attributes.outputs[Concatenate_attributes::output_names::Y] = output_tensor(attributes.name + "::Y");

    // Set inputs
    for (auto &element : x) {
        attributes.inputs.push_back(element);
    }

    sub_nodes.emplace_back(std::make_unique<ConcatenateNode>(std::move(attributes), context));

    return Y;
}

static inline std::ostream &
operator<<(std::ostream &os, Graph const &graph) {
    os << graph.print();
    return os;
}

}  // namespace cudnn_frontend::graph
