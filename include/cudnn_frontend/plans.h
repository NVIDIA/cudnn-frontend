#pragma once

#include <string>
#include <vector>

#include "../cudnn_frontend_EngineConfig.h"
#include "../cudnn_frontend_Logging.h"

namespace cudnn_frontend {

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

        cudnn_frontend::EngineConfigList good_configs;

        for (auto& engine_config : configs) {
            int64_t elem_count                        = 0;
            ManagedOpaqueDescriptor extractedEngine   = make_shared_backend_pointer(CUDNN_BACKEND_ENGINE_DESCRIPTOR);
            cudnnBackendDescriptor_t extractedEngine_ = extractedEngine->get_backend_descriptor();
            auto status = cudnnBackendGetAttribute(engine_config->get_backend_descriptor(),
                                                   CUDNN_ATTR_ENGINECFG_ENGINE,
                                                   CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                   1,
                                                   &elem_count,
                                                   &extractedEngine_);
            if (status == CUDNN_STATUS_SUCCESS) {
                good_configs.push_back(engine_config);
            }
        }

        getLogger() << "[cudnn_frontend] INFO: config list has " << good_configs.size() << " good configurations."
                    << std::endl;
        op_graph_to_configs.emplace(operation_graph->getTag(), good_configs);
    }
    return {error_code_t::OK, ""};
}

inline error_t
create_cudnn_execution_plan(std::shared_ptr<ExecutionPlan>& plan,
                            ManagedOpaqueDescriptor const& config,
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

namespace graph {
class Execution_plan_list {
    std::string operation_tag;
    EngineConfigList engine_configs;
    std::vector<std::vector<cudnnBackendNumericalNote_t>> numeric_notes;
    std::vector<std::vector<cudnnBackendNumericalNote_t>> behavior_notes;

    std::vector<bool> filtered_indices;
    int64_t max_workspace_allowed = std::numeric_limits<int64_t>::max();

    std::shared_ptr<ExecutionPlan> candidate;

   public:
    std::vector<std::shared_ptr<ExecutionPlan>>
        execution_plans;  // Filtered engine configs that have been made as plans

    void
    set_tag(std::string const& tag) {
        operation_tag = tag;
    }
    void
    set_engine_configs(EngineConfigList list) {
        engine_configs = list;
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
            RETURN_CUDNN_FRONTEND_ERROR_IF((status != CUDNN_STATUS_SUCCESS),
                                           error_code_t::HEURISTIC_QUERY_FAILED,
                                           "Heuristic query Engine failed.");

            status = cudnnBackendGetAttribute(extractedEngine_,
                                              CUDNN_ATTR_ENGINE_NUMERICAL_NOTE,
                                              CUDNN_TYPE_NUMERICAL_NOTE,
                                              CUDNN_NUMERICAL_NOTE_TYPE_COUNT,
                                              &elem_count,
                                              nullptr);
            RETURN_CUDNN_FRONTEND_ERROR_IF((status != CUDNN_STATUS_SUCCESS),
                                           error_code_t::HEURISTIC_QUERY_FAILED,
                                           "Heuristic query Numerical Note failed");

            numerics.resize(static_cast<size_t>(elem_count));
            status = cudnnBackendGetAttribute(extractedEngine_,
                                              CUDNN_ATTR_ENGINE_NUMERICAL_NOTE,
                                              CUDNN_TYPE_NUMERICAL_NOTE,
                                              CUDNN_NUMERICAL_NOTE_TYPE_COUNT,
                                              &elem_count,
                                              numerics.data());
            RETURN_CUDNN_FRONTEND_ERROR_IF((status != CUDNN_STATUS_SUCCESS),
                                           error_code_t::HEURISTIC_QUERY_FAILED,
                                           "Heuristic query Numerical Note failed");
            status = cudnnBackendGetAttribute(extractedEngine_,
                                              CUDNN_ATTR_ENGINE_BEHAVIOR_NOTE,
                                              CUDNN_TYPE_BEHAVIOR_NOTE,
                                              CUDNN_BEHAVIOR_NOTE_TYPE_COUNT,
                                              &elem_count,
                                              nullptr);
            RETURN_CUDNN_FRONTEND_ERROR_IF((status != CUDNN_STATUS_SUCCESS),
                                           error_code_t::HEURISTIC_QUERY_FAILED,
                                           "Heuristic query Behavior Note failed");

            behavior.resize(static_cast<size_t>(elem_count));
            status = cudnnBackendGetAttribute(extractedEngine_,
                                              CUDNN_ATTR_ENGINE_BEHAVIOR_NOTE,
                                              CUDNN_TYPE_BEHAVIOR_NOTE,
                                              CUDNN_BEHAVIOR_NOTE_TYPE_COUNT,
                                              &elem_count,
                                              behavior.data());
            RETURN_CUDNN_FRONTEND_ERROR_IF((status != CUDNN_STATUS_SUCCESS),
                                           error_code_t::HEURISTIC_QUERY_FAILED,
                                           "Heuristic query Behavior Note failed");
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

    void
    set_max_workspace_allowed(int64_t const workspace_allowed) {
        max_workspace_allowed = workspace_allowed;
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
        auto const& configs = get_filtered_engine_configs();
        for (auto const& config : configs) {
            std::shared_ptr<ExecutionPlan> plan;
            auto const& fe_status = detail::create_cudnn_execution_plan(plan, config, operation_tag, handle);

            if (fe_status.is_good() && plan->getWorkspaceSize() <= max_workspace_allowed) {
                RETURN_CUDNN_FRONTEND_ERROR_IF(execution_plans.size(),
                                               error_code_t::GRAPH_EXECUTION_PLAN_CREATION_FAILED,
                                               "[cudnn_frontend] Check support or build called already.");

                // No plans should be pushed here.
                // But check_support in v8 incurs compilation cost.
                // If not pushed, build_plans will incur compilation cost again.
                // TODO: Uncomment after https://nvbugswb.nvidia.com/NvBugs5/SWBug.aspx?bugid=4299195&cmtNo=
                // if(cudnnGetVersion() < 9100)
                { execution_plans.push_back(std::move(plan)); }
                return {error_code_t::OK, ""};
            }
        }

        return {error_code_t::GRAPH_EXECUTION_PLAN_CREATION_FAILED,
                "[cudnn_frontend] Error: No execution plans built successfully."};
    }

    error_t
    build_plans(cudnnHandle_t handle, BuildPlanPolicy_t const policy, bool const do_multithreaded_builds) {
        RETURN_CUDNN_FRONTEND_ERROR_IF(do_multithreaded_builds,
                                       error_code_t::GRAPH_EXECUTION_PLAN_CREATION_FAILED,
                                       "Doing multithreaded builds is not yet supported.");

        auto const& configs = get_filtered_engine_configs();

        switch (policy) {
            case BuildPlanPolicy_t::HEURISTICS_CHOICE:
                // short circuit in case a plan was already created.
                // This happens as check_support for v8 builds a plan.
                // Should not happen in v9.
                // TODO: Uncomment after https://nvbugswb.nvidia.com/NvBugs5/SWBug.aspx?bugid=4299195&cmtNo=
                // if(cudnnGetVersion() < 9100)
                {
                    if (execution_plans.size() > 0) {
                        return {error_code_t::OK, ""};
                    }
                }

                for (auto const& config : configs) {
                    std::shared_ptr<ExecutionPlan> plan;
                    auto const& fe_status = detail::create_cudnn_execution_plan(plan, config, operation_tag, handle);

                    if (fe_status.is_good() && plan->getWorkspaceSize() <= max_workspace_allowed) {
                        execution_plans.push_back(std::move(plan));
                        break;
                    }
                }
                break;
            case BuildPlanPolicy_t::ALL:
                for (auto const& config : configs) {
                    std::shared_ptr<ExecutionPlan> plan;
                    auto const& fe_status = detail::create_cudnn_execution_plan(plan, config, operation_tag, handle);

                    if (fe_status.is_good() && plan->getWorkspaceSize() <= max_workspace_allowed) {
                        execution_plans.push_back(std::move(plan));
                    }
                }
                break;
        }

        RETURN_CUDNN_FRONTEND_ERROR_IF(execution_plans.empty(),
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "No execution plans finalized successfully. Hence, not supported.");

        return {error_code_t::OK, ""};
    }

    int64_t
    get_autotune_workspace() const {
        int64_t max_size = 0;
        for (auto& plan : execution_plans) {
            max_size = std::max(max_size, plan->getWorkspaceSize());
        }
        return max_size;
    }

    std::shared_ptr<ExecutionPlan>
    get_best_candidate() const {
        if (execution_plans.empty()) return nullptr;
        return execution_plans.front();
    }

    static error_t
    autotune_default_impl(std::vector<std::shared_ptr<ExecutionPlan>>& execution_plans,
                          cudnnHandle_t handle,
                          std::unordered_map<std::shared_ptr<Tensor_attributes>, void*> variants,
                          void* workspace,
                          void*) {
        // Create the variant pack for all the plans to use.
        std::vector<int64_t> uids;
        std::vector<void*> ptrs;
        for (auto it : variants) {
            if (it.first != nullptr) {
                uids.push_back(it.first->get_uid());
                ptrs.push_back(it.second);
            }
        }

        auto variantPack = VariantPackBuilder()
                               .setDataPointers(ptrs.size(), ptrs.data())
                               .setUids(uids.size(), uids.data())
                               .setWorkspacePointer(workspace)
                               .build();

        std::vector<std::shared_ptr<ExecutionPlan>> time_sorted_plans;

        auto plan_cmp = [](std::shared_ptr<ExecutionPlan> a, std::shared_ptr<ExecutionPlan> b) {
            return a->getExecutionTime() < b->getExecutionTime();
        };

        std::set<std::shared_ptr<ExecutionPlan>, decltype(plan_cmp)> timed_execution_plans(plan_cmp);

        const int maxIterCount         = 100;
        const float threshhold         = 0.95f;
        uint64_t successful_plan_count = 0;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaDeviceSynchronize();

        cudaStream_t stream = nullptr;
        cudnnGetStream(handle, &stream);

        for (auto plan : execution_plans) {
            float time_ms       = 0.0f;
            float final_time_ms = 0.0f;
            float min_time_ms   = std::numeric_limits<float>::max();

            // Warm-up run
            auto warmup_status = cudnnBackendExecute(handle, plan->get_raw_desc(), variantPack.get_raw_desc());
            if (warmup_status != CUDNN_STATUS_SUCCESS) {
                getLogger() << "[cudnn_frontend] Plan " << plan->getTag() << " failed with " << to_string(warmup_status)
                            << std::endl;
                continue;
            }
            successful_plan_count++;
            cudaDeviceSynchronize();

            for (int i = 0; i < maxIterCount; i++) {
                cudaEventRecord(start, stream);

                cudnnBackendExecute(handle, plan->get_raw_desc(), variantPack.get_raw_desc());

                cudaEventRecord(stop, stream);
                cudaEventSynchronize(stop);
                cudaEventElapsedTime(&time_ms, start, stop);

                final_time_ms = std::min(min_time_ms, time_ms);
                if (time_ms / min_time_ms < threshhold) {
                    min_time_ms = final_time_ms;
                } else {
                    break;
                }
            }

            getLogger() << "[cudnn_frontend] Plan " << plan->getTag() << " took " << std::setw(10) << final_time_ms
                        << std::endl;
            plan->setExecutionTime(final_time_ms);
            timed_execution_plans.insert(plan);
        }

        execution_plans.clear();
        for (auto sorted_plan : timed_execution_plans) {
            execution_plans.push_back(sorted_plan);
        }

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        getLogger() << "Autotuned " << successful_plan_count << " plans." << std::endl;
        return {error_code_t::OK, ""};
    }

    std::function<error_t(std::vector<std::shared_ptr<ExecutionPlan>>&,
                          cudnnHandle_t,
                          std::unordered_map<std::shared_ptr<Tensor_attributes>, void*>,
                          void*,
                          void*)>
        autotune_impl = &Execution_plan_list::autotune_default_impl;

    error_t
    autotune(cudnnHandle_t handle,
             std::unordered_map<std::shared_ptr<Tensor_attributes>, void*> variants,
             void* workspace,
             void* user_impl = nullptr) {
        auto error = autotune_impl(execution_plans, handle, variants, workspace, user_impl);
        return error;
    }
};

}  // namespace graph
}  // namespace cudnn_frontend