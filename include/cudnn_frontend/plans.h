#pragma once

#include <string>
#include <vector>

#include "../cudnn_frontend_EngineConfig.h"
#include "../cudnn_frontend_Logging.h"

namespace cudnn_frontend::graph {

class Execution_plan_list {
    std::string operation_tag;
    EngineConfigList engine_configs;
    std::vector<std::vector<cudnnBackendNumericalNote_t>> numeric_notes;
    std::vector<std::vector<cudnnBackendNumericalNote_t>> behavior_notes;

    std::vector<bool> filtered_indices;
    int64_t max_workspace_allowed = std::numeric_limits<int64_t>::max();

   public:
    std::vector<std::shared_ptr<ExecutionPlan>> execution_plans;

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
        auto const& configs = get_filtered_engine_configs();
        for (auto config : configs) {
            std::shared_ptr<ExecutionPlan> plan;
            auto const& fe_status = detail::create_cudnn_execution_plan(plan, config, operation_tag, handle);

            if (fe_status.is_good() && plan->getWorkspaceSize() <= max_workspace_allowed) {
                execution_plans.push_back(plan);
                return {error_code_t::OK, ""};
            }
        }

        return {error_code_t::GRAPH_EXECUTION_PLAN_CREATION_FAILED,
                "[cudnn_frontend] Error: No execution plans built successfully."};
    }

    error_t
    build_all_plans(cudnnHandle_t handle) {
        auto const& configs = get_filtered_engine_configs();
        for (auto config : configs) {
            std::shared_ptr<ExecutionPlan> plan;
            auto const& fe_status = detail::create_cudnn_execution_plan(plan, config, operation_tag, handle);

            if (fe_status.is_good() && plan->getWorkspaceSize() <= max_workspace_allowed) {
                execution_plans.push_back(plan);
            }
        }

        RETURN_CUDNN_FRONTEND_ERROR_IF(execution_plans.empty(),
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "No execution plans finalized successfully. Hence, not supported.");

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

class Plans {
   public:
    Execution_plan_list list_of_engine_configs;

    Plans&
    filter_out_numeric_notes(std::vector<cudnnBackendNumericalNote_t> const&);
    Plans&
    filter_out_behavior_notes(std::vector<cudnnBackendBehaviorNote_t> const&);
    Plans&
    filter_out_workspace_greater_than(int64_t const workspace) {
        list_of_engine_configs.set_max_workspace_allowed(workspace);
        return *this;
    }

    error_t build_all_plans(cudnnHandle_t);

    inline error_t
    check_support(cudnnHandle_t h) {
        CHECK_CUDNN_FRONTEND_ERROR(list_of_engine_configs.check_support(h));
        return {error_code_t::OK, ""};
    }

    int64_t
    get_max_workspace_size();

    static error_t
    autotune_default_impl(Plans* plans,
                          cudnnHandle_t handle,
                          std::unordered_map<std::shared_ptr<Tensor_attributes>, void*> variants,
                          void* workspace,
                          void*) {
        auto& execution_plans = plans->list_of_engine_configs.get_execution_plans();

        // Create the variant pack for all the plans to use.
        std::vector<int64_t> uids;
        std::vector<void*> ptrs;
        for (auto it : variants) {
            uids.push_back(it.first->get_uid());
            ptrs.push_back(it.second);
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

        for (auto plan : plans->list_of_engine_configs.get_execution_plans()) {
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

    std::function<
        error_t(Plans*, cudnnHandle_t, std::unordered_map<std::shared_ptr<Tensor_attributes>, void*>, void*, void*)>
        autotune_impl = &Plans::autotune_default_impl;

    error_t
    autotune(cudnnHandle_t handle,
             std::unordered_map<std::shared_ptr<Tensor_attributes>, void*> variants,
             void* workspace,
             void* user_impl = nullptr) {
        auto error = autotune_impl(this, handle, variants, workspace, user_impl);
        return error;
    }
};

inline Plans&
Plans::filter_out_behavior_notes(std::vector<cudnnBackendBehaviorNote_t> const& notes) {
    // TODO: The error returned is not propagate to user.
    // Should the return value be changed to error_code_t too?
    auto status = list_of_engine_configs.filter_out_behavior_notes(notes);
    if (status.is_bad()) {
        getLogger() << "[cudnn_frontend] ERROR: Filtering by behavioural notes failed." << std::endl;
    }
    return *this;
}

inline Plans&
Plans::filter_out_numeric_notes(std::vector<cudnnBackendNumericalNote_t> const& notes) {
    // TODO: The error returned is not propagate to user.
    // Should the return value be changed to error_code_t too?
    auto status = list_of_engine_configs.filter_out_numeric_notes(notes);
    if (status.is_bad()) {
        getLogger() << "[cudnn_frontend] ERROR: Filtering by numerical notes failed." << std::endl;
    }
    return *this;
}

inline error_t
Plans::build_all_plans(cudnnHandle_t h) {
    CHECK_CUDNN_FRONTEND_ERROR(list_of_engine_configs.build_all_plans(h));
    return {error_code_t::OK, ""};
}

inline int64_t
Plans::get_max_workspace_size() {
    return list_of_engine_configs.get_max_workspace_size();
}

}  // namespace cudnn_frontend::graph