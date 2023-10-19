#pragma once

#include <memory>
#include <vector>
#include <unordered_map>

#include "../cudnn_frontend_Tensor.h"
#include "../cudnn_frontend_Operation.h"
#include "../cudnn_frontend_OperationGraph.h"
#include "../cudnn_frontend_EngineConfig.h"
#include "../cudnn_frontend_ExecutionPlan.h"
#include "../cudnn_frontend_VariantPack.h"

#include "graph_properties.h"
#include "graph_helpers.h"

namespace cudnn_frontend {

class ICudnn {
   public:
    using uid_t = int64_t;

    static uid_t
    create_new_uid() {
        static uid_t uid = 0;
        uid++;
        return uid;
    }

   protected:
    inline static std::unordered_map<uid_t, std::shared_ptr<cudnn_frontend::Tensor>> tensors;

    struct operation_with_uids {
        cudnn_frontend::Operation_v8 operation;
        std::vector<uid_t> uids;
    };
    std::vector<operation_with_uids> operations;

    std::vector<std::shared_ptr<OperationGraph_v8>> operation_graphs;
    std::vector<std::shared_ptr<ExecutionPlan>> execution_plans;

    // uid_t in a variant pack have to be unique, so keep a set of them.
    std::vector<std::unordered_set<uid_t>> variant_pack_uids;

    error_t
    create_cudnn_tensor(std::shared_ptr<graph::Tensor_attributes> const& props) {
        // Check whether tensor already created
        auto const uid = props->get_uid();
        if (tensors.find(uid) != tensors.end()) {
            getLogger() << "[cudnn_frontend] INFO: Backend tensor already created for Id: " << uid << ".\n";
            return {error_code_t::OK, ""};
        }

        // Create new backend tensor
        auto tensor = cudnn_frontend::TensorBuilder()
                          .setDim(props->get_dim().size(), props->get_dim().data())
                          .setStrides(props->get_stride().size(), props->get_stride().data())
                          .setId(uid)
                          .setAlignment(16)
                          .setDataType(props->get_data_type())
                          .setVirtual(props->get_is_virtual())
                          .setByValue(props->get_is_pass_by_value())
                          .setReorderType(props->get_reordering_type())
                          .build();
        tensors.emplace(uid, std::make_shared<Tensor>(std::move(tensor)));

        return {error_code_t::OK, ""};
    }

    error_t
    create_cudnn_operation_graphs(cudnnHandle_t handle) {
        std::vector<Operation const*> cudnn_operations;
        for (auto const& operation_with_uid : operations) {
            cudnn_operations.push_back(&(operation_with_uid.operation));
        }
        auto cudnn_operation_graph = cudnn_frontend::OperationGraphBuilder()
                                         .setHandle(handle)
                                         .setOperationGraph(cudnn_operations.size(), cudnn_operations.data())
                                         .build();

        operation_graphs.push_back(std::make_shared<OperationGraph_v8>(std::move(cudnn_operation_graph)));
        getLogger() << "[cudnn_frontend] INFO: Successfully built Operation Graphs." << std::endl;

        // Push variant pack tensors required for this operation graph
        std::unordered_set<uid_t> variant_pack_for_operation_graph;
        for (auto const& operation_with_uid : operations) {
            variant_pack_for_operation_graph.insert(std::begin(operation_with_uid.uids),
                                                    std::end(operation_with_uid.uids));
        }
        variant_pack_uids.emplace_back(variant_pack_for_operation_graph);

        return {error_code_t::OK, ""};
    }

   public:
    int64_t
    get_cudnn_workspace_size_node() const {
        int64_t current_workspace_size = 0;
        for (auto const& execution_plan : execution_plans) {
            current_workspace_size += execution_plan->getWorkspaceSize();
        }
        return current_workspace_size;
    }

    error_t
    execute_cudnn_plans(cudnnHandle_t handle,
                        std::unordered_map<uid_t, void*> const& tensor_uid_to_pointer_map,
                        void* workspace_ptr) {
        getLogger() << "[cudnn_frontend] INFO: Executing " << execution_plans.size() << " Plans." << std::endl;

        for (size_t i = 0; i < execution_plans.size(); ++i) {
            auto const& execution_plan   = execution_plans[i];
            auto const& variant_pack_uid = variant_pack_uids[i];

            getLogger() << "[cudnn_frontend] INFO: Executing " << execution_plan->getTag() << "..." << std::endl;

            std::vector<void*> device_ptrs;
            std::vector<uid_t> uids;
            for (auto const& uid : variant_pack_uid) {
                if (auto search = tensor_uid_to_pointer_map.find(uid); search == tensor_uid_to_pointer_map.end()) {
                    std::string message =
                        "[cudnn_frontend] ERROR: " + std::to_string(uid) + " does not exist in variant pack.";
                    return {error_code_t::INVALID_VARIANT_PACK, message};
                }
                device_ptrs.push_back(tensor_uid_to_pointer_map.at(uid));
                uids.push_back(uid);
            }
            auto variant_pack = VariantPackBuilder()
                                    .setDataPointers(device_ptrs.size(), device_ptrs.data())
                                    .setUids(uids.size(), uids.data())
                                    .setWorkspacePointer(workspace_ptr)
                                    .build();
            if (variant_pack.get_status() != CUDNN_STATUS_SUCCESS) {
                std::string message = "[cudnn_frontend] ERROR: Variant pack creation failed with " +
                                      std::string(variant_pack.get_error());
                return {error_code_t::INVALID_VARIANT_PACK, message};
            }
            getLogger() << "[cudnn_frontend] INFO: Built variant pack for " << execution_plan->getTag() << "..."
                        << std::endl;

            auto status = cudnnBackendExecute(handle, execution_plan->get_raw_desc(), variant_pack.get_raw_desc());
            if (status != CUDNN_STATUS_SUCCESS) {
                std::string message = "[cudnn_frontend] ERROR: Graph execution failed.";
                return {error_code_t::GRAPH_EXECUTION_FAILED, message};
            }
            getLogger() << "[cudnn_frontend] INFO: Executed " << execution_plan->getTag() << "." << std::endl;
        }

        return {error_code_t::OK, ""};
    }
};

namespace detail {
inline error_t
query_cudnn_heuristics_impl(std::shared_ptr<OperationGraph_v8> const& operation_graph,
                            cudnn_frontend::EngineConfigList& configs,
                            std::vector<HeurMode_t> const& modes) {
    auto const& operation_graph_tag = operation_graph->getTag();
    getLogger() << "[cudnn_frontend] INFO: "
                << " Getting plan from heuristics for " << operation_graph_tag << " ..." << std::endl;

    auto statuses = cudnn_frontend::get_heuristics_list(modes, *operation_graph, allowAllConfig, configs, true);

    getLogger() << "[cudnn_frontend] INFO: get_heuristics_list statuses: ";
    for (size_t i = 0; i < statuses.size(); i++) {
        getLogger() << cudnn_frontend::to_string(statuses[i]) << " ";
    }
    getLogger() << std::endl;

    getLogger() << "[cudnn_frontend] INFO: config list has " << configs.size() << " configurations." << std::endl;

    if (configs.empty()) {
        getLogger() << "[cudnn_frontend] ERROR: No valid engine configs returned from heuristics.";
        return {error_code_t::HEURISTIC_QUERY_FAILED, "No valid engine configs for " + operation_graph_tag};
    }
    return {error_code_t::OK, ""};
}

inline error_t
query_heuristics(std::vector<std::shared_ptr<OperationGraph_v8>> const& operation_graphs,
                 std::unordered_map<std::string, EngineConfigList>& op_graph_to_configs,
                 std::vector<HeurMode_t> const& modes) {
    for (auto const& operation_graph : operation_graphs) {
        cudnn_frontend::EngineConfigList configs;
        CHECK_CUDNN_FRONTEND_ERROR(detail::query_cudnn_heuristics_impl(operation_graph, configs, modes));
        op_graph_to_configs.emplace(operation_graph->getTag(), configs);
    }
    return {error_code_t::OK, ""};
}

inline error_t
create_cudnn_execution_plan(std::shared_ptr<ExecutionPlan>& plan,
                            ManagedOpaqueDescriptor& config,
                            std::string const& operation_graph_tag,
                            cudnnHandle_t handle) {
#ifndef NV_CUDNN_DISABLE_EXCEPTION
    try {
#endif
        auto built_plan = cudnn_frontend::ExecutionPlanBuilder()
                              .setHandle(handle)
                              .setEngineConfig(config, operation_graph_tag)
                              .build();
        if (built_plan.get_status() != CUDNN_STATUS_SUCCESS) {
            getLogger() << "[cudnn_frontend] ERROR: "
                        << "Config failed with " << built_plan.get_error() << std::endl;
            return {error_code_t::GRAPH_EXECUTION_PLAN_CREATION_FAILED, "Couldn't build plan from Config."};
        }

        getLogger() << "[cudnn_frontend] INFO: Config succeeded! Plan has built!\n";
        getLogger() << "[cudnn_frontend] INFO: " << built_plan.describe() << std::endl;
        plan = std::make_shared<ExecutionPlan>(std::move(built_plan));

#ifndef NV_CUDNN_DISABLE_EXCEPTION
    } catch (cudnn_frontend::cudnnException& e) {
        getLogger() << "[cudnn_frontend] ERROR: "
                    << "Config failed with " << e.getCudnnStatus() << " " << e.what() << std::endl;
        return {error_code_t::GRAPH_EXECUTION_PLAN_CREATION_FAILED, "Couldn't build plan from Config."};
    }
#endif

    return {error_code_t::OK, ""};
}

}  // namespace detail
}  // namespace cudnn_frontend
