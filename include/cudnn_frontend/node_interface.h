#pragma once

#include <memory>
#include <vector>
#include <unordered_map>
#include <variant>
#include <limits>

#include <cuda_fp16.h>

#include "../cudnn_frontend_Tensor.h"
#include "../cudnn_frontend_Operation.h"
#include "../cudnn_frontend_OperationGraph.h"
#include "../cudnn_frontend_ExecutionPlan.h"
#include "../cudnn_frontend_VariantPack.h"

#include "cudnn_interface.h"

#include "graph_properties.h"

namespace cudnn_frontend {

namespace graph {

class MatmulNode;
class PointwiseNode;
class ReductionNode;
class ReshapeNode;
class RngNode;
class SoftmaxNode;

// Interface for all nodes to follow.
class INode : public ICudnn {
   public:
    // A closed set of types that are allowed to be passed by value today
    using pass_by_values_t = std::variant<int32_t, half, float, void*>;

    detail::Context context;

   private:
    std::unordered_map<uid_t, pass_by_values_t> deserialized_pass_by_value;
    std::unordered_map<uid_t, std::tuple<int64_t, int64_t, std::vector<float>>> deserialized_workspace_modifications;
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
    expand_and_infer_properties() = 0;

    virtual error_t
    post_validate_node() const = 0;

    virtual int64_t
    get_fe_workspace_size_node() const {
        // Mostly no FE nodes have require workspace initiailized to 0
        return fe_workspace_size;
    }

    int64_t
    get_cudnn_workspace_size(int64_t plan_index = -1) const {
        int64_t cudnn_workspace_size = 0;

        auto status = get_cudnn_workspace_size_node(plan_index, cudnn_workspace_size);
        if (status.is_bad()) {
            getLogger() << "[cudnn_frontend] ERROR: Querying workspace failed." << std::endl;
        }

        for (auto const& sub_node : sub_nodes) {
            cudnn_workspace_size = std::max(cudnn_workspace_size, sub_node->get_cudnn_workspace_size(plan_index));
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
    pass_by_value_tensors_(std::unordered_map<uid_t, pass_by_values_t>& pass_by_values) const {
        for (auto [uid, value] : deserialized_pass_by_value) {
            pass_by_values.emplace(uid, value);
        }
        return {error_code_t::OK, ""};
    }

    error_t
    run_auxiliary_kernels(
        cudnnHandle_t handle,
        void* fe_workspace,
        std::unordered_map<uid_t, std::tuple<int64_t, int64_t, std::vector<float>>>& workspace_modifications) const {
        cudaStream_t stream;
        CHECK_CUDNN_ERROR(cudnnGetStream(handle, &stream));
        char* workspace = static_cast<char*>(fe_workspace);

        for (auto [uid, data] : workspace_modifications) {
            (void)uid;
            if (std::get<0>(data) == 0) {
                auto& vec_data = std::get<2>(data);
                CHECK_CUDA_ERROR(cudaMemcpyAsync(workspace + std::get<1>(data),
                                                 vec_data.data(),
                                                 vec_data.size() * sizeof(float),
                                                 cudaMemcpyHostToDevice,
                                                 stream));
            } else if (std::get<0>(data) == 1) {
                int64_t memset_size = (int64_t)std::get<2>(data)[0];
                CHECK_CUDA_ERROR(cudaMemsetAsync(workspace + std::get<1>(data), 0, memset_size, stream));
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
            } else if (int32_t* int32_t_value_ptr = std::get_if<int32_t>(&value)) {
                tensor_to_pointer_map.emplace(uid, int32_t_value_ptr);
            } else if (float* float_value_ptr = std::get_if<float>(&value)) {
                tensor_to_pointer_map.emplace(uid, float_value_ptr);
            } else if (void** void_value_ptr = std::get_if<void*>(&value)) {
                tensor_to_pointer_map.emplace(uid, *void_value_ptr);
            } else {
                RETURN_CUDNN_FRONTEND_ERROR_IF(
                    true, error_code_t::INVALID_VARIANT_PACK, "Unexpected type for pass by value tensor.");
            }
        }
        return {error_code_t::OK, ""};
    }

   protected:
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

    void
    matmul(std::shared_ptr<Tensor_attributes> a,
           std::shared_ptr<Tensor_attributes> b,
           Matmul_attributes attributes,
           std::shared_ptr<Tensor_attributes> c) {
        attributes.inputs[Matmul_attributes::input_names::A]   = a;
        attributes.inputs[Matmul_attributes::input_names::B]   = b;
        attributes.outputs[Matmul_attributes::output_names::C] = c;
        sub_nodes.emplace_back(std::make_unique<MatmulNode>(std::move(attributes), context));
    }

    void
    softmax(std::shared_ptr<Tensor_attributes> p,
            Softmax_attributes attributes,
            std::shared_ptr<Tensor_attributes> s,
            std::shared_ptr<Tensor_attributes> stats) {
        attributes.inputs[Softmax_attributes::input_names::P]       = p;
        attributes.outputs[Softmax_attributes::output_names::S]     = s;
        attributes.outputs[Softmax_attributes::output_names::Stats] = stats;
        sub_nodes.emplace_back(std::make_unique<SoftmaxNode>(std::move(attributes), context));
    }

    void
    softmax(std::shared_ptr<Tensor_attributes> p,
            Softmax_attributes attributes,
            std::shared_ptr<Tensor_attributes> s,
            std::shared_ptr<Tensor_attributes> m,
            std::shared_ptr<Tensor_attributes> zinv) {
        attributes.inputs[Softmax_attributes::input_names::P]      = p;
        attributes.outputs[Softmax_attributes::output_names::S]    = s;
        attributes.outputs[Softmax_attributes::output_names::M]    = m;
        attributes.outputs[Softmax_attributes::output_names::Zinv] = zinv;
        sub_nodes.emplace_back(std::make_unique<SoftmaxNode>(std::move(attributes), context));
    }

    void
    pointwise(std::shared_ptr<Tensor_attributes> a,
              Pointwise_attributes attributes,
              std::shared_ptr<Tensor_attributes> c) {
        attributes.inputs[Pointwise_attributes::input_names::IN_0]    = a;
        attributes.outputs[Pointwise_attributes::output_names::OUT_0] = c;
        sub_nodes.emplace_back(std::make_unique<PointwiseNode>(std::move(attributes), context));
    }

    void
    pointwise(std::shared_ptr<Tensor_attributes> a,
              std::shared_ptr<Tensor_attributes> b,
              Pointwise_attributes attributes,
              std::shared_ptr<Tensor_attributes> c) {
        attributes.inputs[Pointwise_attributes::input_names::IN_0]    = a;
        attributes.inputs[Pointwise_attributes::input_names::IN_1]    = b;
        attributes.outputs[Pointwise_attributes::output_names::OUT_0] = c;
        sub_nodes.emplace_back(std::make_unique<PointwiseNode>(std::move(attributes), context));
    }

    void
    reduction(std::shared_ptr<Tensor_attributes> a,
              Reduction_attributes attributes,
              std::shared_ptr<Tensor_attributes> c) {
        attributes.inputs[Reduction_attributes::input_names::X]   = a;
        attributes.outputs[Reduction_attributes::output_names::Y] = c;
        sub_nodes.emplace_back(std::make_unique<ReductionNode>(std::move(attributes), context));
    }

    void
    rng(std::shared_ptr<Tensor_attributes> seed,
        std::shared_ptr<Tensor_attributes> offset,
        Rng_attributes attributes,
        std::shared_ptr<Tensor_attributes> y) {
        attributes.inputs[Rng_attributes::input_names::Seed]   = seed;
        attributes.inputs[Rng_attributes::input_names::Offset] = offset;
        attributes.outputs[Rng_attributes::output_names::Y]    = y;
        sub_nodes.emplace_back(std::make_unique<RngNode>(std::move(attributes), context));
    }

    // Creates cudnn tensors for each node (and its sub nodes)
    virtual error_t
    create_cudnn_tensors(int64_t& uid,
                         std::unordered_map<int64_t, std::shared_ptr<cudnn_frontend::Tensor>>& uid_to_backend_tensors,
                         std::unordered_set<int64_t> const& invalid_uids) const {
        for (auto const& sub_node : sub_nodes) {
            CHECK_CUDNN_FRONTEND_ERROR(sub_node->create_cudnn_tensors(uid, uid_to_backend_tensors, invalid_uids));
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

    virtual error_t
    collect_pre_assigned_uids(std::unordered_set<int64_t>& pre_assigned_uids) const {
        for (auto const& sub_node : sub_nodes) {
            auto x = sub_node->collect_pre_assigned_uids(pre_assigned_uids);
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
    std::shared_ptr<Tensor_attributes> reshape(std::shared_ptr<Tensor_attributes>, Reshape_attributes);

    std::shared_ptr<Tensor_attributes> rng(std::shared_ptr<Tensor_attributes>,
                                           std::shared_ptr<Tensor_attributes>,
                                           Rng_attributes);
    error_t
    validate() {
        // validate self
        CHECK_CUDNN_FRONTEND_ERROR(pre_validate_node());

        // infer_properties self
        CHECK_CUDNN_FRONTEND_ERROR(expand_and_infer_properties());

        // validate sub nodes
        for (auto const& sub_node : sub_nodes) {
            CHECK_CUDNN_FRONTEND_ERROR(sub_node->validate());
        }

        // validate self
        CHECK_CUDNN_FRONTEND_ERROR(post_validate_node());

        return {error_code_t::OK, ""};
    }

    error_t
    build_operation_graph(cudnnHandle_t handle) {
        // Starting uid for operation graph.
        // Each time a backend tensor is created, uid will be incremented by 1, ensuring uniqueness.
        // TODO: Maybe just use uid_to_tensors size as uid each time?
        int64_t uid = 1;

        std::unordered_set<int64_t> pre_assigned_uids;
        CHECK_CUDNN_FRONTEND_ERROR(collect_pre_assigned_uids(pre_assigned_uids));
        while (pre_assigned_uids.find(uid) != pre_assigned_uids.end()) {
            uid++;
        }

        // Lower each sub node to cudnn backend.
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensors(uid, uid_to_tensors, pre_assigned_uids));

        // INode needs to keep track of all uids that an operation graph uses.
        // This is because cudnn backend will not accept extra tensors in variant pack.
        // But FE users provide 1 large list of tensors.
        // So internally FE assigns subset of the usre-provided tensor list to each operation graph.
        // Also, as uid in a variant pack have to be unique, keep a set of them.
        std::unordered_set<uid_t> uids_involved_in_operation;
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_operations(uids_involved_in_operation, operations, uid_to_tensors));

        // The method here fuses all operations. There will be 1 operation graph in total.
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_operation_graphs(handle));
        variant_pack_uids.push_back(std::move(uids_involved_in_operation));

        return {error_code_t::OK, ""};
    }

    int64_t
    get_workspace_size() const {
        // There are two workspaces:
        // - cudnn execution plan workspace
        // - FE node workspace (example: alibiSlope for fmha)
        return get_fe_workspace_size() + get_cudnn_workspace_size();
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

        for (auto& plan_list : plans) {
            CHECK_CUDNN_FRONTEND_ERROR(
                plan_list.autotune(handle, tensor_uid_to_pointer_map, cudnn_workspace, user_impl));
        }
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
            execute_cudnn_plans_with_uid(handle, tensor_uid_to_pointer_map, cudnn_workspace, plan_index));

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

        CHECK_CUDNN_FRONTEND_ERROR(execute_cudnn_plans_with_uid(handle, tensor_uid_to_pointer_map, cudnn_workspace));

        return {error_code_t::OK, ""};
    }

    error_t
    deserialize(cudnnHandle_t handle, std::vector<uint8_t> const& data) {
        json j                = json::from_ubjson(data);
        auto serialized_plans = j["cudnn_backend_data"];
        if (serialized_plans.size() == 0) {
            return {error_code_t::GRAPH_EXECUTION_PLAN_CREATION_FAILED, "No plans in the serialized json"};
        }

        auto index = 0;
        for (auto const& serialized_plan : serialized_plans) {
            Execution_plan_list plan_list;
            CHECK_CUDNN_FRONTEND_ERROR(plan_list.build_plans(handle, serialized_plan));
            plans.emplace_back(std::move(plan_list));
            std::unordered_set<uid_t>&& opgraph_variant_packs = j["variant_pack_uids"][index];
            variant_pack_uids.emplace_back(opgraph_variant_packs);
            index++;
        }

        std::unordered_map<uid_t, int32_t> integer_pass_by_values;
        std::unordered_map<uid_t, float> half_pass_by_values;
        std::unordered_map<uid_t, float> float_pass_by_values;

        auto pass_by_value_tensors = j["pass_by_values"];
        for (auto i = 0u; i < pass_by_value_tensors.size(); i++) {
            if (i == 0) {
                integer_pass_by_values = pass_by_value_tensors[i].get<std::unordered_map<uid_t, int32_t>>();
            } else if (i == 1) {
                half_pass_by_values = pass_by_value_tensors[i].get<std::unordered_map<uid_t, float>>();
            } else if (i == 2) {
                float_pass_by_values = pass_by_value_tensors[i].get<std::unordered_map<uid_t, float>>();
            }
        }

        for (auto const& [uid, value] : integer_pass_by_values) {
            deserialized_pass_by_value.emplace(uid, value);
        }
        for (auto const& [uid, value] : half_pass_by_values) {
            deserialized_pass_by_value.emplace(uid, __float2half(value));
        }
        for (auto const& [uid, value] : float_pass_by_values) {
            deserialized_pass_by_value.emplace(uid, value);
        }

        deserialized_workspace_modifications = j["workspace_modifications"];

        fe_workspace_size = j["fe_workspace_size"];

        return {error_code_t::OK, ""};
    }

    error_t
    serialize(std::vector<uint8_t>& data) const {
        json j;
        serialize(j);
        j["cudnn_backend_data"];
        int index = 0;
        for (auto& plan_list : plans) {
            auto const candidate = plan_list.candidate;
            auto execution_plan  = plan_list.execution_plans[candidate];
            if (execution_plan != nullptr) {
                auto serialized_plan = execution_plan->getJsonRepresentation();
                j["cudnn_backend_data"].push_back(serialized_plan);
                j["variant_pack_uids"].push_back(variant_pack_uids[index]);
                index++;
            }
        }

        std::unordered_map<uid_t, pass_by_values_t> tensor_to_pass_by_value;
        CHECK_CUDNN_FRONTEND_ERROR(gather_pass_by_value_tensors_(tensor_to_pass_by_value));

        j["pass_by_values"];
        std::unordered_map<uid_t, int32_t> integer_pass_by_values;
        std::unordered_map<uid_t, float> half_pass_by_values;
        std::unordered_map<uid_t, float> float_pass_by_values;
        // std::unordered_map<uid_t, void *>  void_ptr_pass_by_values;
        for (auto const& [uid, pass_by_value] : tensor_to_pass_by_value) {
            if (pass_by_value.index() == 0) {
                integer_pass_by_values.emplace(uid, std::get<0>(pass_by_value));
            } else if (pass_by_value.index() == 1) {
                half_pass_by_values.emplace(uid, __half2float(std::get<1>(pass_by_value)));
            } else if (pass_by_value.index() == 2) {
                float_pass_by_values.emplace(uid, std::get<2>(pass_by_value));
            }
        }
        // json j = half_pass_by_values;
        j["pass_by_values"].push_back(integer_pass_by_values);
        j["pass_by_values"].push_back(half_pass_by_values);
        j["pass_by_values"].push_back(float_pass_by_values);

        std::unordered_map<uid_t, std::tuple<int64_t, int64_t, std::vector<float>>> workspace_modifications;
        int64_t workspace_offset = 0;
        CHECK_CUDNN_FRONTEND_ERROR(gather_workspace_modifications(workspace_modifications, workspace_offset));

        j["workspace_modifications"] = workspace_modifications;

        j["fe_workspace_size"] = get_fe_workspace_size();

        data = json::to_ubjson(j);
        return {error_code_t::OK, ""};
    }

    INode(detail::Context const& context) : context(context) {}

    // Make sure each node implements a public serialize function
    virtual void
    serialize(json& j) const = 0;

    size_t
    key() {
        json j;
        serialize(j);
        return std::hash<json>{}(j);
    }

    virtual ~INode() = default;
};

[[maybe_unused]] static void
to_json(json& j, const INode& p) {
    p.serialize(j);
}

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

inline std::shared_ptr<Tensor_attributes>
INode::matmul(std::shared_ptr<Tensor_attributes> a,
              std::shared_ptr<Tensor_attributes> b,
              Matmul_attributes attributes) {
    attributes.inputs[Matmul_attributes::input_names::A] = a;
    attributes.inputs[Matmul_attributes::input_names::B] = b;
    auto C = attributes.outputs[Matmul_attributes::output_names::C] = output_tensor(attributes.name + "::C");

    sub_nodes.emplace_back(std::make_unique<MatmulNode>(std::move(attributes), context));
    return C;
}

inline std::shared_ptr<Tensor_attributes>
INode::pointwise(std::shared_ptr<Tensor_attributes> a, Pointwise_attributes attributes) {
    attributes.inputs[Pointwise_attributes::input_names::IN_0] = a;
    auto OUT_0 = attributes.outputs[Pointwise_attributes::output_names::OUT_0] =
        output_tensor(attributes.name + "::OUT_0");

    sub_nodes.emplace_back(std::make_unique<PointwiseNode>(std::move(attributes), context));
    return OUT_0;
}

inline std::shared_ptr<Tensor_attributes>
INode::pointwise(std::shared_ptr<Tensor_attributes> a,
                 std::shared_ptr<Tensor_attributes> b,
                 Pointwise_attributes attributes) {
    attributes.inputs[Pointwise_attributes::input_names::IN_0] = a;
    attributes.inputs[Pointwise_attributes::input_names::IN_1] = b;
    auto OUT_0 = attributes.outputs[Pointwise_attributes::output_names::OUT_0] =
        output_tensor(attributes.name + "::OUT_0");

    sub_nodes.emplace_back(std::make_unique<PointwiseNode>(std::move(attributes), context));
    return OUT_0;
}

inline std::shared_ptr<Tensor_attributes>
INode::pointwise(std::shared_ptr<Tensor_attributes> a,
                 std::shared_ptr<Tensor_attributes> b,
                 std::shared_ptr<Tensor_attributes> c,
                 Pointwise_attributes attributes) {
    attributes.inputs[Pointwise_attributes::input_names::IN_0] = a;
    attributes.inputs[Pointwise_attributes::input_names::IN_1] = b;
    attributes.inputs[Pointwise_attributes::input_names::IN_2] = c;
    auto OUT_0 = attributes.outputs[Pointwise_attributes::output_names::OUT_0] =
        output_tensor(attributes.name + "::OUT_0");

    sub_nodes.emplace_back(std::make_unique<PointwiseNode>(std::move(attributes), context));
    return OUT_0;
}

inline std::shared_ptr<Tensor_attributes>
INode::reduction(std::shared_ptr<Tensor_attributes> input, Reduction_attributes attributes) {
    attributes.inputs[Reduction_attributes::input_names::X] = input;
    auto Y = attributes.outputs[Reduction_attributes::output_names::Y] = output_tensor(attributes.name + "::Y");

    sub_nodes.emplace_back(std::make_unique<ReductionNode>(std::move(attributes), context));
    return Y;
}

inline std::shared_ptr<Tensor_attributes>
INode::reshape(std::shared_ptr<Tensor_attributes> input, Reshape_attributes attributes) {
    attributes.inputs[Reshape_attributes::input_names::X] = input;
    auto Y = attributes.outputs[Reshape_attributes::output_names::Y] = output_tensor(attributes.name + "::Y");

    sub_nodes.emplace_back(std::make_unique<ReshapeNode>(std::move(attributes), context));
    return Y;
}

inline std::shared_ptr<Tensor_attributes>
INode::rng(std::shared_ptr<Tensor_attributes> seed,
           std::shared_ptr<Tensor_attributes> offset,
           Rng_attributes attributes) {
    attributes.inputs[Rng_attributes::input_names::Seed]   = seed;
    attributes.inputs[Rng_attributes::input_names::Offset] = offset;
    auto Y = attributes.outputs[Rng_attributes::output_names::Y] = output_tensor(attributes.name + "::Y");

    sub_nodes.emplace_back(std::make_unique<RngNode>(std::move(attributes), context));
    return Y;
}

}  // namespace graph

}  // namespace cudnn_frontend