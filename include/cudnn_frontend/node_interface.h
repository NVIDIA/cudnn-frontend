#pragma once

#include <memory>
#include <vector>
#include <unordered_map>
#include <variant>
#include <limits>

#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "../cudnn_frontend_Tensor.h"
#include "../cudnn_frontend_Operation.h"
#include "../cudnn_frontend_OperationGraph.h"
#include "../cudnn_frontend_ExecutionPlan.h"
#include "../cudnn_frontend_VariantPack.h"
#include "../cudnn_frontend_shim.h"

#include "cudnn_interface.h"

#include "graph_properties.h"

namespace cudnn_frontend {

namespace graph {

class BatchNormNode;
class DBNNode;
class MatmulNode;
class MatmulFP8Node;
class PointwiseNode;
class ReductionNode;
class ResampleNode;
class ReshapeNode;
class RngNode;
class SoftmaxNode;

// Interface for all nodes to follow.
class INode : public ICudnn {
   public:
    // A closed set of types that are allowed to be passed by value today
    using pass_by_values_t = Tensor_attributes::pass_by_values_t;

    detail::Context context;

   private:
    int64_t fe_workspace_size = 0;

    std::shared_ptr<Tensor_attributes>
    output_tensor(std::string const& name) {
        auto tensor = std::make_shared<Tensor_attributes>();
        tensor->set_name(name).set_is_virtual(true);
        return tensor;
    }

    virtual error_t
    pre_validate_node() const = 0;

    virtual error_t
    expand_and_infer_properties_node() = 0;

    virtual error_t
    post_validate_node() const = 0;

    virtual int64_t
    get_fe_workspace_size_node() const {
        // Mostly no FE nodes have require workspace initiailized to 0
        return fe_workspace_size;
    }

    int64_t
    get_cudnn_workspace_size(int64_t plan_index) const {
        int64_t cudnn_workspace_size = 0;

        auto status = get_cudnn_workspace_size_node(plan_index, cudnn_workspace_size);
        if (status.is_bad()) {
            getLogger() << "[cudnn_frontend] ERROR: Querying workspace failed." << std::endl;
        }

        return cudnn_workspace_size;
    }

    int64_t
    get_max_cudnn_workspace_size() const {
        int64_t cudnn_workspace_size = get_max_cudnn_workspace_size_node();
        for (auto const& sub_node : sub_nodes) {
            cudnn_workspace_size = std::max(cudnn_workspace_size, sub_node->get_max_cudnn_workspace_size());
        }
        return cudnn_workspace_size;
    }

    int64_t
    get_fe_workspace_size() const {
        int64_t fe_workspace_size = get_fe_workspace_size_node();
        for (auto const& sub_node : sub_nodes) {
            fe_workspace_size += sub_node->get_fe_workspace_size();
        }
        return fe_workspace_size;
    }

    virtual error_t
    pass_by_value_tensors_(std::unordered_map<uid_t, pass_by_values_t>& pass_by_values) const = 0;

    virtual error_t
    collect_pre_assigned_uids_(std::unordered_set<int64_t>& pre_assigned_uids) const = 0;

    virtual error_t
    set_uids_(int64_t& potential_uid, std::unordered_set<int64_t> const& pre_assigned_uids) const = 0;

    virtual error_t
    create_cudnn_tensors_(
        std::unordered_map<int64_t, std::shared_ptr<cudnn_frontend::Tensor>>& uid_to_backend_tensors) const = 0;

    error_t
    run_auxiliary_kernels(
        cudnnHandle_t handle,
        void* fe_workspace,
        std::unordered_map<uid_t, std::tuple<int64_t, int64_t, std::vector<float>>>& workspace_modifications) const {
        cudaStream_t stream;
        CHECK_CUDNN_ERROR(detail::get_stream(handle, &stream));
        char* workspace = static_cast<char*>(fe_workspace);

        for (auto [uid, data] : workspace_modifications) {
            (void)uid;
            if (std::get<0>(data) == 0) {
                auto& vec_data = std::get<2>(data);
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

    error_t
    gather_pass_by_value_tensors_(std::unordered_map<uid_t, pass_by_values_t>& tensor_to_pass_by_value) const {
        CHECK_CUDNN_FRONTEND_ERROR(pass_by_value_tensors_(tensor_to_pass_by_value));
        for (auto const& sub_node : sub_nodes) {
            CHECK_CUDNN_FRONTEND_ERROR(sub_node->gather_pass_by_value_tensors_(tensor_to_pass_by_value));
        }
        return {error_code_t::OK, ""};
    }

    virtual error_t
    workspace_modifications_tensors_(
        std::unordered_map<uid_t, std::tuple<int64_t, int64_t, std::vector<float>>>& worskspace_modifications,
        int64_t&) const {
        for (auto [uid, value] : deserialized_workspace_modifications) {
            worskspace_modifications.emplace(uid, value);
        }
        return {error_code_t::OK, ""};
    }

    error_t
    gather_workspace_modifications(
        std::unordered_map<uid_t, std::tuple<int64_t, int64_t, std::vector<float>>>& worskspace_modifications,
        int64_t& offset) const {
        CHECK_CUDNN_FRONTEND_ERROR(workspace_modifications_tensors_(worskspace_modifications, offset));
        offset = get_fe_workspace_size_node();
        for (auto const& sub_node : sub_nodes) {
            CHECK_CUDNN_FRONTEND_ERROR(sub_node->gather_workspace_modifications(worskspace_modifications, offset));
            offset += sub_node->get_fe_workspace_size_node();
        }
        return {error_code_t::OK, ""};
    }

    error_t
    extend_tensor_map_with_workspace_tensors_(
        std::unordered_map<int64_t, void*>& tensor_to_pointer_map,
        void* workspace,
        std::unordered_map<uid_t, std::tuple<int64_t, int64_t, std::vector<float>>> const& worskspace_modifications)
        const {
        for (auto const& [uid, data] : worskspace_modifications) {
            tensor_to_pointer_map.emplace(uid, static_cast<char*>(workspace) + std::get<1>(data));
        }
        return {error_code_t::OK, ""};
    }

    error_t
    extend_tensor_map_with_pass_by_value_tensors_(
        std::unordered_map<int64_t, void*>& tensor_to_pointer_map,
        std::unordered_map<uid_t, pass_by_values_t>& tensor_to_pass_by_value) const {
        for (auto& [uid, value] : tensor_to_pass_by_value) {
            if (half* half_value_ptr = std::get_if<half>(&value)) {
                tensor_to_pointer_map.emplace(uid, half_value_ptr);
            } else if (nv_bfloat16* nv_bfloat16_value_ptr = std::get_if<nv_bfloat16>(&value)) {
                tensor_to_pointer_map.emplace(uid, nv_bfloat16_value_ptr);
            } else if (int32_t* int32_t_value_ptr = std::get_if<int32_t>(&value)) {
                tensor_to_pointer_map.emplace(uid, int32_t_value_ptr);
            } else if (float* float_value_ptr = std::get_if<float>(&value)) {
                tensor_to_pointer_map.emplace(uid, float_value_ptr);
            } else {
                RETURN_CUDNN_FRONTEND_ERROR_IF(
                    true, error_code_t::INVALID_VARIANT_PACK, "Unexpected type for pass by value tensor.");
            }
        }
        return {error_code_t::OK, ""};
    }

   protected:
    std::unordered_map<uid_t, pass_by_values_t> deserialized_pass_by_value;
    std::unordered_map<uid_t, std::tuple<int64_t, int64_t, std::vector<float>>> deserialized_workspace_modifications;

    // Type of each node. Nodes can either be a composite (value COMPOSITE) or
    // one of the other primitive types. Primitives types are nothing but
    // cudnn operations.
    enum class Type {
        COMPOSITE,
        BATCHNORM,
        BATCHNORM_INFERENCE,
        BN_FINALIZE,
        CONVOLUTION,
        DBN,
        DBN_WEIGHT,
        DLN,
        DIN,
        DGRAD,
        DRMSNorm,
        GENSTATS,
        LAYERNORM,
        INSTANCENORM,
        MATMUL,
        POINTWISE,
        REDUCTION,
        RESAMPLE,
        RESHAPE,
        RMSNORM,
        RNG,
        SCALED_DOT_PRODUCT_ATTENTION,
        WGRAD
    };
    Type tag;

    inline void
    matmul(std::shared_ptr<Tensor_attributes> a,
           std::shared_ptr<Tensor_attributes> b,
           Matmul_attributes attributes,
           std::shared_ptr<Tensor_attributes> c);

    void
    matmul_fp8(std::shared_ptr<Tensor_attributes> a,
               std::shared_ptr<Tensor_attributes> b,
               std::shared_ptr<Tensor_attributes> descale_a,
               std::shared_ptr<Tensor_attributes> descale_b,
               std::shared_ptr<Tensor_attributes> scale_c,
               Matmul_fp8_attributes attributes,
               std::shared_ptr<Tensor_attributes> c,
               std::shared_ptr<Tensor_attributes> amax_c);

    void
    softmax(std::shared_ptr<Tensor_attributes> p,
            Softmax_attributes attributes,
            std::shared_ptr<Tensor_attributes> s,
            std::shared_ptr<Tensor_attributes> stats);

    void
    softmax(std::shared_ptr<Tensor_attributes> p,
            Softmax_attributes attributes,
            std::shared_ptr<Tensor_attributes> s,
            std::shared_ptr<Tensor_attributes> m,
            std::shared_ptr<Tensor_attributes> zinv);

    void
    pointwise(std::shared_ptr<Tensor_attributes> a,
              Pointwise_attributes attributes,
              std::shared_ptr<Tensor_attributes> c);

    void
    pointwise(std::shared_ptr<Tensor_attributes> a,
              std::shared_ptr<Tensor_attributes> b,
              Pointwise_attributes attributes,
              std::shared_ptr<Tensor_attributes> c);

    void
    reduction(std::shared_ptr<Tensor_attributes> a,
              Reduction_attributes attributes,
              std::shared_ptr<Tensor_attributes> c);

    void
    rng(std::shared_ptr<Tensor_attributes> seed,
        std::shared_ptr<Tensor_attributes> offset,
        Rng_attributes attributes,
        std::shared_ptr<Tensor_attributes> y);

    error_t
    pre_validate_and_expand_node() {
        // pre validate to catch errors early
        // Otherwise code reability decreases in expand_and_infer
        CHECK_CUDNN_FRONTEND_ERROR(pre_validate_node());
        CHECK_CUDNN_FRONTEND_ERROR(expand_and_infer_properties_node());
        for (auto const& sub_node : sub_nodes) {
            CHECK_CUDNN_FRONTEND_ERROR(sub_node->pre_validate_and_expand_node());
        }
        return {error_code_t::OK, ""};
    }

    error_t
    post_validate() const {
        // Validate self
        CHECK_CUDNN_FRONTEND_ERROR(post_validate_node());
        for (auto const& sub_node : sub_nodes) {
            CHECK_CUDNN_FRONTEND_ERROR(sub_node->post_validate());
        }
        return {error_code_t::OK, ""};
    }

    // Creates cudnn tensors for each node (and its sub nodes)
    error_t
    create_cudnn_tensors(
        std::unordered_map<int64_t, std::shared_ptr<cudnn_frontend::Tensor>>& uid_to_backend_tensors) const {
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensors_(uid_to_backend_tensors));
        for (auto const& sub_node : sub_nodes) {
            CHECK_CUDNN_FRONTEND_ERROR(sub_node->create_cudnn_tensors(uid_to_backend_tensors));
        }
        return {error_code_t::OK, ""};
    }

    // Creates cudnn operation for each node (and its sub nodes)
    // Only INode that map to a primitive cudnn operation need to specialize.
    virtual error_t
    create_cudnn_operations(
        std::unordered_set<uid_t>& uids_involved_in_operation,
        std::vector<std::shared_ptr<cudnn_frontend::Operation>>& backend_operations,
        std::unordered_map<int64_t, std::shared_ptr<cudnn_frontend::Tensor>>& uid_to_backend_tensors) const {
        for (auto const& sub_node : sub_nodes) {
            CHECK_CUDNN_FRONTEND_ERROR(sub_node->create_cudnn_operations(
                uids_involved_in_operation, backend_operations, uid_to_backend_tensors));
        }
        return {error_code_t::OK, ""};
    }

    error_t
    collect_pre_assigned_uids(std::unordered_set<int64_t>& pre_assigned_uids) const {
        // Collect uids from current node
        CHECK_CUDNN_FRONTEND_ERROR(collect_pre_assigned_uids_(pre_assigned_uids));
        for (auto const& sub_node : sub_nodes) {
            CHECK_CUDNN_FRONTEND_ERROR(sub_node->collect_pre_assigned_uids(pre_assigned_uids));
        }
        return {error_code_t::OK, ""};
    }

    error_t
    set_uids(int64_t& potential_uid, std::unordered_set<int64_t> const& pre_assigned_uids) const {
        // Collect uids from current node
        CHECK_CUDNN_FRONTEND_ERROR(set_uids_(potential_uid, pre_assigned_uids));
        for (auto const& sub_node : sub_nodes) {
            CHECK_CUDNN_FRONTEND_ERROR(sub_node->set_uids(potential_uid, pre_assigned_uids));
        }
        return {error_code_t::OK, ""};
    }

    // An implicitly topological-sorted vector of sub nodes.
    // The sorted order is a side effect of functional API.
    std::vector<std::shared_ptr<INode>> sub_nodes;

   public:
    virtual Type
    getType() = 0;

    std::shared_ptr<Tensor_attributes> matmul(std::shared_ptr<Tensor_attributes>,
                                              std::shared_ptr<Tensor_attributes>,
                                              Matmul_attributes);

    std::shared_ptr<Tensor_attributes> pointwise(std::shared_ptr<Tensor_attributes>, Pointwise_attributes);
    std::shared_ptr<Tensor_attributes> pointwise(std::shared_ptr<Tensor_attributes>,
                                                 std::shared_ptr<Tensor_attributes>,
                                                 Pointwise_attributes);
    std::shared_ptr<Tensor_attributes> pointwise(std::shared_ptr<Tensor_attributes>,
                                                 std::shared_ptr<Tensor_attributes>,
                                                 std::shared_ptr<Tensor_attributes>,
                                                 Pointwise_attributes);

    std::shared_ptr<Tensor_attributes> reduction(std::shared_ptr<Tensor_attributes>, Reduction_attributes);
    std::array<std::shared_ptr<Tensor_attributes>, 2> resample(std::shared_ptr<Tensor_attributes>, Resample_attributes);
    std::shared_ptr<Tensor_attributes> reshape(std::shared_ptr<Tensor_attributes>, Reshape_attributes);

    std::shared_ptr<Tensor_attributes> rng(std::shared_ptr<Tensor_attributes>,
                                           std::shared_ptr<Tensor_attributes>,
                                           Rng_attributes);
    error_t
    validate() {
        // infer_properties self
        CHECK_CUDNN_FRONTEND_ERROR(pre_validate_and_expand_node());

        // assign uids as part of validation
        // This helps catch whether user has assigned duplicated uids to tensors
        // Each time a backend tensor is created, uid will be incremented by 1, ensuring uniqueness.
        std::unordered_set<int64_t> pre_assigned_uids;
        CHECK_CUDNN_FRONTEND_ERROR(collect_pre_assigned_uids(pre_assigned_uids));

        Tensor_attributes::uid_t start_uid = 1;
        CHECK_CUDNN_FRONTEND_ERROR(set_uids(start_uid, pre_assigned_uids));

        // validate the full tree again
        CHECK_CUDNN_FRONTEND_ERROR(post_validate());

        return {error_code_t::OK, ""};
    }

    error_t
    build_operation_graph(cudnnHandle_t handle) {
        // Lower each sub node to cudnn backend.
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensors(uid_to_tensors));

        // INode keeps track of all uids that an operation graph uses.
        // This helps to return errors to user during execution, without relying on backend to do so.
        // Also, as uid in a variant pack have to be unique, keep a set of them.
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_operations(variant_pack_uids, operations, uid_to_tensors));

        // The method here fuses all operations. There will be 1 operation graph in total.
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_operation_graph(handle));

        return {error_code_t::OK, ""};
    }

    int64_t
    get_workspace_size() const {
        // There are two workspaces:
        // - cudnn execution plan workspace
        // - FE node workspace (example: alibiSlope for fmha)
        return get_fe_workspace_size() + get_cudnn_workspace_size(plans.candidate);
    }

    int64_t
    get_workspace_size_plan_at_index(int64_t plan_index) const {
        // There are two workspaces:
        // - cudnn execution plan workspace
        // - FE node workspace (example: alibiSlope for fmha)
        return get_fe_workspace_size() + get_cudnn_workspace_size(plan_index);
    }

    int64_t
    get_autotune_workspace_size() const {
        // There are two workspaces:
        // - cudnn execution plan workspace
        // - FE node workspace (example: alibiSlope for fmha)
        return get_fe_workspace_size() + get_max_cudnn_workspace_size();
    }

    error_t
    autotune(cudnnHandle_t handle,
             std::unordered_map<int64_t, void*>& tensor_uid_to_pointer_map,
             void* workspace,
             void* user_impl = nullptr) {
        // Add pass_by_value data pointers to tensor_uid_to_pointer map
        // object lifetime is controlled by tensor_to_pass_by_value which means the pointer should stay valid during
        // execute.
        std::unordered_map<uid_t, pass_by_values_t> tensor_to_pass_by_value;
        CHECK_CUDNN_FRONTEND_ERROR(gather_pass_by_value_tensors_(tensor_to_pass_by_value));

        CHECK_CUDNN_FRONTEND_ERROR(
            extend_tensor_map_with_pass_by_value_tensors_(tensor_uid_to_pointer_map, tensor_to_pass_by_value));

        std::unordered_map<uid_t, std::tuple<int64_t, int64_t, std::vector<float>>> workspace_modifications;
        int64_t workspace_offset = 0;
        CHECK_CUDNN_FRONTEND_ERROR(gather_workspace_modifications(workspace_modifications, workspace_offset));

        CHECK_CUDNN_FRONTEND_ERROR(run_auxiliary_kernels(handle, workspace, workspace_modifications));

        CHECK_CUDNN_FRONTEND_ERROR(
            extend_tensor_map_with_workspace_tensors_(tensor_uid_to_pointer_map, workspace, workspace_modifications));

        // offset workspace by the already used fe graph workspace
        // this is where cudnn backend can start using workspace for its execution plans
        void* cudnn_workspace = static_cast<char*>(workspace) + get_fe_workspace_size();

        CHECK_CUDNN_FRONTEND_ERROR(plans.autotune(handle, tensor_uid_to_pointer_map, cudnn_workspace, user_impl));
        return {error_code_t::OK, ""};
    }

    error_t
    autotune(cudnnHandle_t handle,
             std::unordered_map<std::shared_ptr<Tensor_attributes>, void*>& tensor_to_pointer_map,
             void* workspace,
             void* user_impl = nullptr) {
        // First get all the uids from the map
        std::unordered_map<int64_t, void*> tensor_uid_to_pointer_map;
        for (auto const& [tensor, pointer] : tensor_to_pointer_map) {
            tensor_uid_to_pointer_map.emplace(tensor->get_uid(), pointer);
        }

        return autotune(handle, tensor_uid_to_pointer_map, workspace, user_impl);
    }

    error_t
    execute_plan_at_index(cudnnHandle_t handle,
                          std::unordered_map<std::shared_ptr<Tensor_attributes>, void*>& tensor_to_pointer_map,
                          void* workspace,
                          int64_t plan_index) const {
        // First get all the uids from the map
        std::unordered_map<int64_t, void*> tensor_uid_to_pointer_map;
        for (auto const& [tensor, pointer] : tensor_to_pointer_map) {
            tensor_uid_to_pointer_map.emplace(tensor->get_uid(), pointer);
        }

        return execute_plan_at_index(handle, tensor_uid_to_pointer_map, workspace, plan_index);
    }

    error_t
    execute(cudnnHandle_t handle,
            std::unordered_map<std::shared_ptr<Tensor_attributes>, void*>& tensor_to_pointer_map,
            void* workspace) const {
        // First get all the uids from the map
        std::unordered_map<int64_t, void*> tensor_uid_to_pointer_map;
        for (auto const& [tensor, pointer] : tensor_to_pointer_map) {
            tensor_uid_to_pointer_map.emplace(tensor->get_uid(), pointer);
        }

        return execute(handle, tensor_uid_to_pointer_map, workspace);
    }

    error_t
    execute_plan_at_index(cudnnHandle_t handle,
                          std::unordered_map<int64_t, void*>& tensor_uid_to_pointer_map,
                          void* workspace,
                          int64_t plan_index) const {
        // Add pass_by_value data pointers to uid_to_pointer map
        // object lifetime is controlled by tensor_to_pass_by_value which means the pointer should stay valid during
        // execute.
        std::unordered_map<uid_t, pass_by_values_t> tensor_to_pass_by_value;
        CHECK_CUDNN_FRONTEND_ERROR(gather_pass_by_value_tensors_(tensor_to_pass_by_value));

        CHECK_CUDNN_FRONTEND_ERROR(
            extend_tensor_map_with_pass_by_value_tensors_(tensor_uid_to_pointer_map, tensor_to_pass_by_value));

        std::unordered_map<uid_t, std::tuple<int64_t, int64_t, std::vector<float>>> workspace_modifications;
        int64_t workspace_offset = 0;
        CHECK_CUDNN_FRONTEND_ERROR(gather_workspace_modifications(workspace_modifications, workspace_offset));

        CHECK_CUDNN_FRONTEND_ERROR(run_auxiliary_kernels(handle, workspace, workspace_modifications));

        CHECK_CUDNN_FRONTEND_ERROR(
            extend_tensor_map_with_workspace_tensors_(tensor_uid_to_pointer_map, workspace, workspace_modifications));
        // offset workspace by the already used fe graph workspace
        // this is where cudnn backend can start using workspace for its execution plans
        void* cudnn_workspace = static_cast<char*>(workspace) + get_fe_workspace_size();

        CHECK_CUDNN_FRONTEND_ERROR(
            execute_cudnn_plan_with_uid(handle, tensor_uid_to_pointer_map, cudnn_workspace, plan_index));

        return {error_code_t::OK, ""};
    }

    error_t
    execute(cudnnHandle_t handle,
            std::unordered_map<int64_t, void*>& tensor_uid_to_pointer_map,
            void* workspace) const {
        // Add pass_by_value data pointers to uid_to_pointer map
        // object lifetime is controlled by tensor_to_pass_by_value which means the pointer should stay valid during
        // execute.
        std::unordered_map<uid_t, pass_by_values_t> tensor_to_pass_by_value;
        CHECK_CUDNN_FRONTEND_ERROR(gather_pass_by_value_tensors_(tensor_to_pass_by_value));

        CHECK_CUDNN_FRONTEND_ERROR(
            extend_tensor_map_with_pass_by_value_tensors_(tensor_uid_to_pointer_map, tensor_to_pass_by_value));

        std::unordered_map<uid_t, std::tuple<int64_t, int64_t, std::vector<float>>> workspace_modifications;
        int64_t workspace_offset = 0;
        CHECK_CUDNN_FRONTEND_ERROR(gather_workspace_modifications(workspace_modifications, workspace_offset));

        CHECK_CUDNN_FRONTEND_ERROR(run_auxiliary_kernels(handle, workspace, workspace_modifications));

        CHECK_CUDNN_FRONTEND_ERROR(
            extend_tensor_map_with_workspace_tensors_(tensor_uid_to_pointer_map, workspace, workspace_modifications));
        // offset workspace by the already used fe graph workspace
        // this is where cudnn backend can start using workspace for its execution plans
        void* cudnn_workspace = static_cast<char*>(workspace) + get_fe_workspace_size();

        CHECK_CUDNN_FRONTEND_ERROR(
            execute_cudnn_plan_with_uid(handle, tensor_uid_to_pointer_map, cudnn_workspace, plans.candidate));

        return {error_code_t::OK, ""};
    }

    error_t
    deserialize(cudnnHandle_t handle, std::vector<uint8_t> const& data) {
#ifndef CUDNN_FRONTEND_SKIP_JSON_LIB
        json j = json::from_ubjson(data);

        auto serialized_plan = j["cudnn_backend_data"];
        CHECK_CUDNN_FRONTEND_ERROR(plans.build_plans(handle, serialized_plan));

        variant_pack_uids = j["variant_pack_uids"].get<std::unordered_set<graph::Tensor_attributes::uid_t>>();

        deserialized_pass_by_value = j["pass_by_values"];

        deserialized_workspace_modifications = j["workspace_modifications"];

        fe_workspace_size = j["fe_workspace_size"];

        return {error_code_t::OK, ""};
#else
        CUDNN_FRONTEND_UNUSED(handle);
        CUDNN_FRONTEND_UNUSED(data);
        return {error_code_t::GRAPH_NOT_SUPPORTED, "unavailable when compiled with CUDNN_FRONTEND_SKIP_JSON_LIB"};
#endif
    }

    error_t
    serialize(std::vector<uint8_t>& data) const {
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

        std::unordered_map<uid_t, pass_by_values_t> tensor_to_pass_by_value;
        CHECK_CUDNN_FRONTEND_ERROR(gather_pass_by_value_tensors_(tensor_to_pass_by_value));
        j["pass_by_values"] = tensor_to_pass_by_value;

        std::unordered_map<uid_t, std::tuple<int64_t, int64_t, std::vector<float>>> workspace_modifications;
        int64_t workspace_offset = 0;
        CHECK_CUDNN_FRONTEND_ERROR(gather_workspace_modifications(workspace_modifications, workspace_offset));
        j["workspace_modifications"] = workspace_modifications;

        j["fe_workspace_size"] = get_fe_workspace_size();

        data = json::to_ubjson(j);
        return {error_code_t::OK, ""};
#else
        CUDNN_FRONTEND_UNUSED(data);
        return {error_code_t::GRAPH_NOT_SUPPORTED, "unavailable when compiled with CUDNN_FRONTEND_SKIP_JSON_LIB"};
#endif
    }

    INode(detail::Context const& context) : context(context) {}

    // Make sure each node implements a public serialize function
#ifndef CUDNN_FRONTEND_SKIP_JSON_LIB
    virtual void
    serialize(json& j) const = 0;
#endif

    size_t
    key() {
#ifndef CUDNN_FRONTEND_SKIP_JSON_LIB
        json j;
        serialize(j);
        return std::hash<json>{}(j);
#else
        return 1;
#endif
    }

    virtual ~INode() = default;
};

#ifndef CUDNN_FRONTEND_SKIP_JSON_LIB
[[maybe_unused]] static void
to_json(json& j, const INode& p) {
    p.serialize(j);
}
#endif

template <typename DerivedT>
class NodeCRTP : public INode {
    DerivedT&
    self() {
        return *static_cast<DerivedT*>(this);
    }
    DerivedT const&
    self() const {
        return *static_cast<DerivedT const*>(this);
    }

    error_t
    pass_by_value_tensors_(
        std::unordered_map<Tensor_attributes::uid_t, pass_by_values_t>& tensor_to_pass_by_value) const override final {
        CHECK_CUDNN_FRONTEND_ERROR(self().attributes.fill_pass_by_value(tensor_to_pass_by_value));

        return {error_code_t::OK, ""};
    }

    error_t
    collect_pre_assigned_uids_(std::unordered_set<int64_t>& pre_assigned_uids) const override final {
        CHECK_CUDNN_FRONTEND_ERROR(self().attributes.get_prefilled_uids(pre_assigned_uids));

        return {error_code_t::OK, ""};
    }

    error_t
    set_uids_(int64_t& potential_uid, std::unordered_set<int64_t> const& pre_assigned_uids) const override final {
        CHECK_CUDNN_FRONTEND_ERROR(self().attributes.set_uids(potential_uid, pre_assigned_uids));

        return {error_code_t::OK, ""};
    }

    error_t
    create_cudnn_tensors_(
        std::unordered_map<int64_t, std::shared_ptr<cudnn_frontend::Tensor>>& tensors) const override final {
        getLogger() << "[cudnn_frontend] INFO: Creating cudnn tensors for node named '" << self().attributes.name
                    << "':" << std::endl;
        for (auto const& [name, tensor] : self().attributes.inputs) {
            (void)name;
            if (tensor) {
                CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(tensor, tensors));
            }
        }
        for (auto const& [name, tensor] : self().attributes.outputs) {
            (void)name;
            if (tensor) {
                CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(tensor, tensors));
            }
        }

        // Handle special case of BN where peer_stats is also an input
        if constexpr (std::is_same_v<DerivedT, DBNNode> || std::is_same_v<DerivedT, BatchNormNode>) {
            // Special case in BN where peer stats is also an input but is not present in inputs map
            for (auto const& tensor : self().attributes.peer_stats) {
                if (tensor) {
                    CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(tensor, tensors));
                }
            }
        }

        return {error_code_t::OK, ""};
    }

   protected:
    using INode::INode;
};

#define CUDNN_FE_VALIDATE_TENSOR_(port, map_)                                                      \
    {                                                                                              \
        auto t           = map_.find(port);                                                        \
        bool const has_t = (t != map_.end()) && (t->second != nullptr);                            \
        RETURN_CUDNN_FRONTEND_ERROR_IF(                                                            \
            !has_t, error_code_t::ATTRIBUTE_NOT_SET, std::string("Tensor ") + #port + " not set"); \
    }

#define CUDNN_FE_VALIDATE_AND_ASSIGN_TENSOR_(tensor, port, map_)                                   \
    auto tensor = map_.find(port);                                                                 \
    {                                                                                              \
        bool const has_t = (tensor != map_.end()) && (tensor->second != nullptr);                  \
        RETURN_CUDNN_FRONTEND_ERROR_IF(                                                            \
            !has_t, error_code_t::ATTRIBUTE_NOT_SET, std::string("Tensor ") + #port + " not set"); \
    }

#define CUDNN_FE_VALIDATE_INPUT_TENSOR(port) CUDNN_FE_VALIDATE_TENSOR_(port, attributes.inputs)

#define CUDNN_FE_VALIDATE_OUTPUT_TENSOR(port) CUDNN_FE_VALIDATE_TENSOR_(port, attributes.outputs)

#define CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(tensor, port) \
    CUDNN_FE_VALIDATE_AND_ASSIGN_TENSOR_(tensor, port, attributes.inputs)

#define CUDNN_FE_VALIDATE_AND_ASSIGN_OUTPUT_TENSOR(tensor, port) \
    CUDNN_FE_VALIDATE_AND_ASSIGN_TENSOR_(tensor, port, attributes.outputs)

}  // namespace graph

}  // namespace cudnn_frontend
