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

#include "cudnn_frontend_cudnn_interface.h"

#include "cudnn_frontend_graph_properties.h"

namespace cudnn_frontend {

namespace graph {

// Interface for all nodes to follow.
class INode : public ICudnn {
   public:
    // A closed set of types that are allowed to be passed by value today
    using pass_by_values_t = std::variant<half, float, void*>;

    // Stores workspace size in bytes required by FE node
    // It does NOT include cudnn backend workspace
    size_t workspace_size;

    detail::Context context;

   private:
    virtual error_t
    assign_uids_node() {
        return {error_code_t::OK, ""};
    };

    virtual error_t
    infer_properties_node() {
        return {error_code_t::OK, ""};
    };

    bool has_validation_checked = false;
    virtual error_t
    validate_node() const {
        return {error_code_t::OK, ""};
    };

    error_t
    assign_uids() {
        CHECK_CUDNN_FRONTEND_ERROR(assign_uids_node());
        for (auto const& sub_node : sub_nodes) {
            auto status = sub_node->assign_uids();
            if (status.is_bad()) {
                return status;
            }
        }
        return {error_code_t::OK, ""};
    }

    virtual int64_t
    get_fe_workspace_size_node() const {
        // Mostly no FE nodes have require workspace
        return 0;
    }

    int64_t
    get_cudnn_workspace_size() const {
        int64_t cudnn_workspace_size = get_cudnn_workspace_size_node();
        for (auto const& sub_node : sub_nodes) {
            cudnn_workspace_size += sub_node->get_cudnn_workspace_size();
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
    pass_by_value_tensors_(std::unordered_map<std::shared_ptr<Tensor_attributes>, pass_by_values_t>&,
                           [[maybe_unused]] void* node_workspace) {
        return {error_code_t::OK, ""};
    }

    error_t
    gather_pass_by_value_tensors(
        std::unordered_map<std::shared_ptr<Tensor_attributes>, pass_by_values_t>& tensor_to_pass_by_value,
        void* fe_workspace) {
        void* node_workspace = fe_workspace;
        CHECK_CUDNN_FRONTEND_ERROR(pass_by_value_tensors_(tensor_to_pass_by_value, node_workspace));
        node_workspace = static_cast<char*>(node_workspace) + get_fe_workspace_size_node();
        for (auto const& sub_node : sub_nodes) {
            auto status    = sub_node->gather_pass_by_value_tensors(tensor_to_pass_by_value, node_workspace);
            node_workspace = static_cast<char*>(node_workspace) + sub_node->get_fe_workspace_size_node();
            if (status.get_code() != error_code_t::OK) {
                return status;
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
        DGRAD,
        GENSTATS,
        LAYERNORM,
        MATMUL,
        POINTWISE,
        REDUCTION,
        RESAMPLE,
        RESHAPE,
        RNG,
        SCALED_DOT_PRODUCT_ATTENTION,
        WGRAD
    };
    Type tag;

    virtual error_t
    createTensors() {
        for (auto const& sub_node : sub_nodes) {
            auto status = sub_node->createTensors();
            if (status.get_code() != error_code_t::OK) {
                return status;
            }
        }
        return {error_code_t::OK, ""};
    }

    virtual error_t
    createOperationGraphs(cudnnHandle_t) {
        return {error_code_t::GRAPH_NOT_SUPPORTED, ""};
    }

    virtual error_t
    createOperations() {
        for (auto const& sub_node : sub_nodes) {
            auto status = sub_node->createOperations();
            if (status.is_bad()) {
                return status;
            }

            // Roll up operations to parent node, so that parent can too partition operation graphs.
            for (auto&& operation_with_uids : sub_node->operations) {
                operations.push_back(std::move(operation_with_uids));
            }
        }
        return {error_code_t::OK, ""};
    }

    std::vector<std::unique_ptr<INode>> sub_nodes;

   public:
    virtual Type
    getType() = 0;

    error_t
    validate() {
        if (has_validation_checked) {
            return {error_code_t::OK, ""};
        }

        // validate self
        auto status = validate_node();
        if (status.is_bad()) {
            getLogger() << "[cudnn_frontend] ERROR: Validation failed." << std::endl;
            return status;
        }

        // infer_properties self
        status = infer_properties_node();
        if (status.is_bad()) {
            getLogger() << "[cudnn_frontend] ERROR: Infer properties failed." << std::endl;
            return status;
        }

        // validate sub nodes
        for (auto const& sub_node : sub_nodes) {
            status = sub_node->validate();
            if (status.is_bad()) {
                getLogger() << "[cudnn_frontend] ERROR: Validation failed." << std::endl;
                return status;
            }
        }

        has_validation_checked = true;
        return {error_code_t::OK, ""};
    }

    error_t
    build_operation_graph(cudnnHandle_t handle) {
        auto status = validate();
        if (status.is_bad()) {
            getLogger() << "[cudnn_frontend] ERROR: Failed to build." << std::endl;
            return status;
        }

        status = assign_uids();
        if (status.is_bad()) {
            getLogger() << "[cudnn_frontend] ERROR: Failed to build." << std::endl;
            return status;
        }

        status = createTensors();
        if (status.is_bad()) {
            getLogger() << "[cudnn_frontend] ERROR: Failed to build." << std::endl;
            return status;
        }

        status = createOperations();
        if (status.is_bad()) {
            getLogger() << "[cudnn_frontend] ERROR: Failed to build." << std::endl;
            return status;
        }

        status = createOperationGraphs(handle);
        if (status.is_bad()) {
            getLogger() << "[cudnn_frontend] ERROR: Failed to build." << std::endl;
            return status;
        }

        return {error_code_t::OK, ""};
    }

    int64_t
    get_workspace_size() const {
        // There are two workspaces:
        // - cudnn execution plan workspace
        // - FE node workspace (example: alibiSlope for fmha)
        return get_fe_workspace_size() + get_cudnn_workspace_size();
    }

    error_t
    execute(cudnnHandle_t handle,
            std::unordered_map<std::shared_ptr<Tensor_attributes>, void*> const& tensor_to_pointer_map,
            void* workspace) {
        std::unordered_map<int64_t, void*> tensor_uid_to_pointer_map;
        for (auto const& [tensor, pointer] : tensor_to_pointer_map) {
            tensor_uid_to_pointer_map.emplace(tensor->get_uid(), pointer);
        }

        std::unordered_map<std::shared_ptr<Tensor_attributes>, pass_by_values_t> tensor_to_pass_by_value;
        void* fe_workspace    = workspace;
        void* cudnn_workspace = static_cast<char*>(fe_workspace) + get_fe_workspace_size();

        auto status = gather_pass_by_value_tensors(tensor_to_pass_by_value, fe_workspace);
        if (status.is_bad()) {
            getLogger() << "[cudnn_frontend] ERROR: Failed to gather_pass_by_value_tensors." << std::endl;
            return status;
        }

        // Add pass_by_value data pointers to tensor_uid_to_pointer map
        // object lifetime is controlled by tensor_to_pass_by_value which means the pointer should stay valid during
        // execute
        for (auto& [tensor, value] : tensor_to_pass_by_value) {
            if (half* half_value_ptr = std::get_if<half>(&value)) {
                tensor_uid_to_pointer_map.emplace(tensor->get_uid(), half_value_ptr);
            } else if (float* float_value_ptr = std::get_if<float>(&value)) {
                tensor_uid_to_pointer_map.emplace(tensor->get_uid(), float_value_ptr);
            } else if (void** void_value_ptr = std::get_if<void*>(&value)) {
                tensor_uid_to_pointer_map.emplace(tensor->get_uid(), *void_value_ptr);
            } else {
                status.code    = error_code_t::INVALID_VARIANT_PACK;
                status.err_msg = "[cudnn_frontend] ERROR: Unexpected type for pass by value tensor.";
                return status;
            }
        }

        status = execute_cudnn_plans(handle, tensor_uid_to_pointer_map, cudnn_workspace);
        if (status.is_bad()) {
            getLogger() << "[cudnn_frontend] ERROR: Execution failed." << std::endl;
            return status;
        }

        return status;
    }

    INode(detail::Context const& context) : context(context) {}

    virtual void
    serialize(json& j) const {
        j["nodes"];
        for (auto const& sub_node : sub_nodes) {
            json j_sub_node;
            sub_node->serialize(j_sub_node);
            j["nodes"].push_back(j_sub_node);
        }
    };

    virtual ~INode(){};
};

[[maybe_unused]] static void
to_json(json& j, const INode& p) {
    p.serialize(j);
}

class Execution_plan_list {
    std::string operation_tag;
    EngineConfigList engine_configs;
    std::vector<std::vector<cudnnBackendNumericalNote_t>> numeric_notes;
    std::vector<std::vector<cudnnBackendNumericalNote_t>> behavior_notes;

    std::vector<std::shared_ptr<ExecutionPlan>> execution_plans;

    std::vector<bool> filtered_indices;
    int64_t max_workspace_allowed = std::numeric_limits<int64_t>::max();

   public:
    void
    set_tag(std::string const& tag) {
        operation_tag = tag;
    }
    void
    set_engine_configs(EngineConfigList list) {
        engine_configs = list;
    }

    std::shared_ptr<ExecutionPlan> const
    get_candidate() const {
        return (execution_plans.size() ? execution_plans.front() : nullptr);
    }

    std::vector<std::shared_ptr<ExecutionPlan>>&
    get_execution_plans() {
        return execution_plans;
    }

    error_t
    query_properties() {
        numeric_notes.reserve(engine_configs.size());
        behavior_notes.reserve(engine_configs.size());
        filtered_indices.resize(engine_configs.size());
        for (auto& engine_config : engine_configs) {
            int64_t elem_count = 0;
            std::vector<cudnnBackendNumericalNote_t> numerics;
            std::vector<cudnnBackendNumericalNote_t> behavior;

            ManagedOpaqueDescriptor extractedEngine   = make_shared_backend_pointer(CUDNN_BACKEND_ENGINE_DESCRIPTOR);
            cudnnBackendDescriptor_t extractedEngine_ = extractedEngine->get_backend_descriptor();
            auto status = cudnnBackendGetAttribute(engine_config->get_backend_descriptor(),
                                                   CUDNN_ATTR_ENGINECFG_ENGINE,
                                                   CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                   1,
                                                   &elem_count,
                                                   &extractedEngine_);
            if (status != CUDNN_STATUS_SUCCESS) {
                return {error_code_t::HEURISTIC_QUERY_FAILED, "Heuristic query Engine failed."};
            }

            status = cudnnBackendGetAttribute(extractedEngine_,
                                              CUDNN_ATTR_ENGINE_NUMERICAL_NOTE,
                                              CUDNN_TYPE_NUMERICAL_NOTE,
                                              CUDNN_NUMERICAL_NOTE_TYPE_COUNT,
                                              &elem_count,
                                              nullptr);
            if (status != CUDNN_STATUS_SUCCESS) {
                return {error_code_t::HEURISTIC_QUERY_FAILED, "Heuristic query Numerical Note failed"};
            }
            numerics.resize(static_cast<size_t>(elem_count));
            status = cudnnBackendGetAttribute(extractedEngine_,
                                              CUDNN_ATTR_ENGINE_NUMERICAL_NOTE,
                                              CUDNN_TYPE_NUMERICAL_NOTE,
                                              CUDNN_NUMERICAL_NOTE_TYPE_COUNT,
                                              &elem_count,
                                              numerics.data());
            if (status != CUDNN_STATUS_SUCCESS) {
                return {error_code_t::HEURISTIC_QUERY_FAILED, "Heuristic query Numerical Notes failed"};
            }
            status = cudnnBackendGetAttribute(extractedEngine_,
                                              CUDNN_ATTR_ENGINE_BEHAVIOR_NOTE,
                                              CUDNN_TYPE_BEHAVIOR_NOTE,
                                              CUDNN_BEHAVIOR_NOTE_TYPE_COUNT,
                                              &elem_count,
                                              nullptr);
            if (status != CUDNN_STATUS_SUCCESS) {
                return {error_code_t::HEURISTIC_QUERY_FAILED, "Heuristic query Behavior Note failed"};
            }
            behavior.resize(static_cast<size_t>(elem_count));
            status = cudnnBackendGetAttribute(extractedEngine_,
                                              CUDNN_ATTR_ENGINE_BEHAVIOR_NOTE,
                                              CUDNN_TYPE_BEHAVIOR_NOTE,
                                              CUDNN_BEHAVIOR_NOTE_TYPE_COUNT,
                                              &elem_count,
                                              behavior.data());
            if (status != CUDNN_STATUS_SUCCESS) {
                return {error_code_t::HEURISTIC_QUERY_FAILED, "Heuristic query Behavior Notes failed"};
            }
            numeric_notes.emplace_back(numerics);
            behavior_notes.emplace_back(behavior);
        }
        return {error_code_t::OK, ""};
    }

    error_t
    filter_out_numeric_notes(std::vector<cudnnBackendNumericalNote_t> const& notes) {
        for (auto note : notes) {
            for (auto i = 0u; i < engine_configs.size(); i++) {
                if (std::find(numeric_notes[i].begin(), numeric_notes[i].end(), note) != numeric_notes[i].end()) {
                    filtered_indices[i] = true;
                }
            }
        }
        return {error_code_t::OK, ""};
    }

    error_t
    filter_out_behavior_notes(std::vector<cudnnBackendBehaviorNote_t> const& notes) {
        for (auto note : notes) {
            for (auto i = 0u; i < engine_configs.size(); i++) {
                if (std::find(behavior_notes[i].begin(), behavior_notes[i].end(), note) != behavior_notes[i].end()) {
                    filtered_indices[i] = true;
                }
            }
        }
        return {error_code_t::OK, ""};
    }

    error_t
    set_max_workspace_allowed(int64_t const workspace_allowed) {
        max_workspace_allowed = workspace_allowed;
        return {error_code_t::OK, ""};
    }

    EngineConfigList
    get_filtered_engine_configs() {
        EngineConfigList filtered_engine_configs;
        getLogger() << "[cudnn_frontend] INFO: "
                    << " Filtering engine_configs ..." << engine_configs.size() << std::endl;
        for (auto i = 0u; i < engine_configs.size(); i++) {
            if (filtered_indices[i] == false) {
                filtered_engine_configs.push_back(engine_configs[i]);
            }
        }
        getLogger() << "[cudnn_frontend] INFO: "
                    << " Filtered engine_configs ..." << filtered_engine_configs.size() << std::endl;
        return filtered_engine_configs;
    }

    error_t
    check_support(cudnnHandle_t handle) {
        error_t status = {error_code_t::OK, ""};
        auto configs   = get_filtered_engine_configs();
        for (auto& config : configs) {
#ifndef NV_CUDNN_DISABLE_EXCEPTION
            try {
#endif
                auto plan = cudnn_frontend::ExecutionPlanBuilder()
                                .setHandle(handle)
                                .setEngineConfig(config, operation_tag)
                                .build();
                if (plan.get_status() != CUDNN_STATUS_SUCCESS) {
                    getLogger() << "[cudnn_frontend] ERROR: "
                                << "Config failed with " << plan.get_error() << std::endl;
                    continue;
                }
                getLogger() << "[cudnn_frontend] INFO: "
                            << "Config succeeded! Plan has built!" << std::endl;
                getLogger() << "[cudnn_frontend] INFO: " << plan.describe() << std::endl;

                if (plan.getWorkspaceSize() <= max_workspace_allowed) {
                    execution_plans.push_back(std::make_shared<ExecutionPlan>(std::move(plan)));
                    return status;
                }

#ifndef NV_CUDNN_DISABLE_EXCEPTION
            } catch (cudnn_frontend::cudnnException& e) {
                getLogger() << "[cudnn_frontend] ERROR: "
                            << "Config failed with " << e.getCudnnStatus() << " " << e.what() << std::endl;
                continue;
            }
#endif
        }

        if (execution_plans.empty()) {
            return {error_code_t::GRAPH_EXECUTION_PLAN_CREATION_FAILED,
                    "[cudnn_frontend] Error: No execution plans built successfully."};
        }
        return status;
    }

    error_t
    build_all_plans(cudnnHandle_t handle) {
        auto configs = get_filtered_engine_configs();
        for (auto& config : configs) {
#ifndef NV_CUDNN_DISABLE_EXCEPTION
            try {
#endif
                auto plan = cudnn_frontend::ExecutionPlanBuilder()
                                .setHandle(handle)
                                .setEngineConfig(config, operation_tag)
                                .build();
                if (plan.get_status() != CUDNN_STATUS_SUCCESS) {
                    getLogger() << "[cudnn_frontend] ERROR: "
                                << "Config failed with " << plan.get_error() << std::endl;
                    continue;
                }
                getLogger() << "[cudnn_frontend] INFO: "
                            << "Config succeeded! Plan has built!" << std::endl;
                getLogger() << "[cudnn_frontend] INFO: " << plan.describe() << std::endl;

                if (plan.getWorkspaceSize() <= max_workspace_allowed) {
                    execution_plans.push_back(std::make_shared<ExecutionPlan>(std::move(plan)));
                }

#ifndef NV_CUDNN_DISABLE_EXCEPTION
            } catch (cudnn_frontend::cudnnException& e) {
                getLogger() << "[cudnn_frontend] ERROR: "
                            << "Config failed with " << e.getCudnnStatus() << " " << e.what() << std::endl;
                continue;
            }
#endif
        }

        if (execution_plans.empty()) {
            return {error_code_t::GRAPH_NOT_SUPPORTED,
                    "[cudnn_frontend] Error: No execution plans finalized successfully. Hence, not supported."};
        }
        return {error_code_t::OK, ""};
    }

    int64_t
    get_max_workspace_size() {
        int64_t max_size = 0;
        for (auto& plan : execution_plans) {
            max_size = std::max(max_size, plan->getWorkspaceSize());
        }
        return max_size;
    }
};

}  // namespace graph

}  // namespace cudnn_frontend