#pragma once

#include <memory>
#include <vector>
#include <unordered_map>

#include "../cudnn_frontend_Tensor.h"
#include "../cudnn_frontend_Operation.h"
#include "../cudnn_frontend_OperationGraph.h"
#include "../cudnn_frontend_ExecutionPlan.h"
#include "../cudnn_frontend_VariantPack.h"

#include "cudnn_frontend_graph_properties.h"

namespace cudnn_frontend {

using op_graph_to_engine_configs = std::unordered_map<std::string, EngineConfigList>;

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

    op_graph_to_engine_configs engine_configs;

    // uid_t in a variant pack have to be unique, so keep a set of them.
    std::vector<std::unordered_set<uid_t>> variant_pack_uids;

    error_t
    create_cudnn_tensor(std::shared_ptr<graph::Tensor_attributes> const& props) {
        // Check whether tensor already created
        if (tensors.find(props->get_uid()) != tensors.end()) {
            return {error_code_t::OK, ""};
        }

        // Create new backend tensor
        auto tensor = cudnn_frontend::TensorBuilder()
                          .setDim(props->get_dim().size(), props->get_dim().data())
                          .setStrides(props->get_stride().size(), props->get_stride().data())
                          .setId(props->get_uid())
                          .setAlignment(16)
                          .setDataType(props->get_data_type())
                          .setVirtual(props->get_is_virtual())
                          .setByValue(props->get_is_pass_by_value())
                          .setReorderType(props->get_reordering_type())
                          .build();
        tensors.emplace(props->get_uid(), std::make_shared<Tensor>(std::move(tensor)));

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

    error_t
    query_heuristics(HeurMode_t mode) {
        for (auto const& op_graph : operation_graphs) {
            getLogger() << "[cudnn_frontend] INFO: "
                        << " Getting plan from heuristics for " << op_graph->getTag() << " ..." << std::endl;

            cudnn_frontend::EngineConfigList configs;

            switch (mode) {
                case HeurMode_t::HEUR_MODE_A: {
                    auto statuses = cudnn_frontend::get_heuristics_list<1>(
                        {"heuristics_mode_a"}, *op_graph, allowAllConfig, configs, true);

                    getLogger() << "[cudnn_frontend] INFO: "
                                << "mode_a get_heuristics_list statuses: ";
                    for (size_t i = 0; i < statuses.size(); i++) {
                        getLogger() << cudnn_frontend::to_string(statuses[i]) << " ";
                    }
                    getLogger() << std::endl;
                    break;
                }
                case HeurMode_t::HEUR_MODE_B: {
                    auto statuses = cudnn_frontend::get_heuristics_list<1>(
                        {"heuristics_mode_b"}, *op_graph, allowAllConfig, configs, true);

                    getLogger() << "[cudnn_frontend] INFO: "
                                << "mode_b get_heuristics_list statuses: ";
                    for (size_t i = 0; i < statuses.size(); i++) {
                        getLogger() << cudnn_frontend::to_string(statuses[i]) << " ";
                    }
                    getLogger() << std::endl;
                    break;
                }
                case HeurMode_t::HEUR_MODE_FALLBACK: {
                    auto statuses = cudnn_frontend::get_heuristics_list<1>(
                        {"heuristics_fallback"}, *op_graph, allowAllConfig, configs, true);

                    getLogger() << "[cudnn_frontend] INFO: "
                                << "fallback get_heuristics_list statuses: ";
                    for (size_t i = 0; i < statuses.size(); i++) {
                        getLogger() << cudnn_frontend::to_string(statuses[i]) << " ";
                    }
                    getLogger() << std::endl;
                    break;
                }
            }

            getLogger() << "[cudnn_frontend] INFO: "
                        << "Mode " << json{mode} << " config list has " << configs.size() << " configurations."
                        << std::endl;

            if (configs.size() > 0) {
                engine_configs.emplace(op_graph->getTag(), configs);
            }
        }
        return {error_code_t::OK, ""};
    }

    error_t
    create_cudnn_execution_plan(cudnnHandle_t handle) {
        for (auto const& filtered_configs : engine_configs) {
            for (size_t i = 0; i < filtered_configs.second.size(); i++) {
                getLogger() << "[cudnn_frontend] INFO: "
                            << "Trying config: " << i << std::endl;

#ifndef NV_CUDNN_DISABLE_EXCEPTION
                try {
#endif

                    auto configs = filtered_configs.second;
                    auto plan    = cudnn_frontend::ExecutionPlanBuilder()
                                    .setHandle(handle)
                                    .setEngineConfig(configs[i], filtered_configs.first)
                                    .build();
                    if (plan.get_status() != CUDNN_STATUS_SUCCESS) {
                        getLogger() << "[cudnn_frontend] ERROR: "
                                    << "Config " << i << " failed with " << plan.get_error() << std::endl;
                        // If last config, return error
                        // or else continue to the next config
                        if (i == filtered_configs.second.size() - 1) {
                            return {error_code_t::GRAPH_EXECUTION_PLAN_CREATION_FAILED, "No successful plan built."};
                        }
                        continue;
                    }
                    getLogger() << "[cudnn_frontend] INFO: "
                                << "Config " << i << " succeeded! Plan has built!" << std::endl;
                    getLogger() << "[cudnn_frontend] INFO: " << plan.describe() << std::endl;

                    execution_plans.push_back(std::make_shared<ExecutionPlan>(std::move(plan)));
                    getLogger() << "[cudnn_frontend] INFO: "
                                << " Successfully built plan." << std::endl;

                    // Getting here means plan successfully built
                    // move onto next operation graph
                    break;

#ifndef NV_CUDNN_DISABLE_EXCEPTION
                } catch (cudnn_frontend::cudnnException& e) {
                    // The last config didn't work (E.g. all configs didn't work)
                    getLogger() << "[cudnn_frontend] ERROR: "
                                << "Config " << i << " failed with " << e.getCudnnStatus() << " " << e.what()
                                << std::endl;
                    if (i == filtered_configs.second.size() - 1) {
                        return {error_code_t::GRAPH_EXECUTION_PLAN_CREATION_FAILED, "All plan creation failed"};
                    }
                    continue;
                }
#endif
            }
        }

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

}  // namespace cudnn_frontend
