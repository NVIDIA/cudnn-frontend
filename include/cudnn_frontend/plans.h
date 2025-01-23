#pragma once

#include <optional>
#include <string>
#include <vector>

#include "../cudnn_frontend_EngineConfig.h"
#include "../cudnn_frontend_Logging.h"
#include "graph_helpers.h"

#include "backend/execution_helpers.h"
#include "backend/plan_helpers.h"

namespace cudnn_frontend {

namespace detail {

inline error_t
execute(cudnnHandle_t handle,
        ExecutionPlan* plan,
        std::vector<void*>& device_ptrs,
        std::vector<int64_t> const& uids,
        void* workspace_ptr) {
    // TODO: below line fails with MSVC. warning C4127: conditional expression is constant
    // RETURN_CUDNN_FRONTEND_ERROR_IF(!plan, error_code_t::GRAPH_EXECUTION_FAILED, "No plan found to execute!!");
    CUDNN_FE_LOG_LABEL_ENDL("INFO: Executing " << plan->getTag() << "...");

    backend_descriptor variant_pack_descriptor(CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR);
    RETURN_CUDNN_FRONTEND_ERROR_IF(variant_pack_descriptor.get_status() != CUDNN_STATUS_SUCCESS,
                                   error_code_t::CUDNN_BACKEND_API_FAILED,
                                   "Failed to create variant pack's backend descriptor.");

    CHECK_CUDNN_FRONTEND_ERROR(create_variant_pack(variant_pack_descriptor, device_ptrs, uids, workspace_ptr));
    CHECK_CUDNN_ERROR(execute(handle, plan->get_raw_desc(), variant_pack_descriptor.get_ptr()));

    CUDNN_FE_LOG_LABEL_ENDL("INFO: Executed " << plan->getTag() << ".");

    return {error_code_t::OK, ""};
}

inline error_t
query_cudnn_heuristics_impl(std::shared_ptr<OperationGraph_v8> const& operation_graph,
                            cudnn_frontend::EngineConfigList& configs,
                            std::vector<HeurMode_t> const& modes,
                            int32_t sm_count) {
    RETURN_CUDNN_FRONTEND_ERROR_IF(
        operation_graph == nullptr,
        error_code_t::HEURISTIC_QUERY_FAILED,
        "Empty operation graph provided. Did you forget to call graph.build_operation_graph()?");

    auto const& operation_graph_tag = operation_graph->getTag();
    CUDNN_FE_LOG_LABEL_ENDL("INFO: " << " Getting plan from heuristics for " << operation_graph_tag << " ...");

    std::vector<cudnnStatus_t> statuses;
#ifdef NV_CUDNN_DISABLE_EXCEPTION
    // disable exception macro is defined. Calling build will not throw.
    // Check status of desc and return error.
    statuses = cudnn_frontend::get_heuristics_list(modes, *operation_graph, allowAllConfig, configs, true, sm_count);
#else
    // build() can throw
    // wrap in try catch
    try {
        statuses =
            cudnn_frontend::get_heuristics_list(modes, *operation_graph, allowAllConfig, configs, true, sm_count);
    } catch (cudnn_frontend::cudnnException& e) {
        // Silly MSVC error that thinks below condition is constexpr
        // RETURN_CUDNN_FRONTEND_ERROR_IF(
        //     e.getCudnnStatus() != CUDNN_STATUS_SUCCESS, error_code_t::HEURISTIC_QUERY_FAILED, e.what());
        CUDNN_FE_LOG_LABEL("ERROR: " << e.what() << ". ");
        CUDNN_FE_LOG(error_code_t::HEURISTIC_QUERY_FAILED << " because querying heuristics failed at " << __FILE__
                                                          << ":" << __LINE__ << "\n");
        return {error_code_t::HEURISTIC_QUERY_FAILED, e.what()};
    }
#endif

    CUDNN_FE_LOG_LABEL("INFO: get_heuristics_list statuses: ");
    for (size_t i = 0; i < statuses.size(); i++) {
        CUDNN_FE_LOG(cudnn_frontend::to_string(statuses[i]) << " ");
    }
    CUDNN_FE_LOG(std::endl);

    CUDNN_FE_LOG_LABEL_ENDL("INFO: config list has " << configs.size() << " configurations.");

    if (configs.empty()) {
        std::string err_msg = detail::get_last_error_string_();
        CUDNN_FE_LOG_LABEL_ENDL("ERROR: No valid engine configs returned from heuristics.\n" << err_msg);
        return {error_code_t::HEURISTIC_QUERY_FAILED,
                "No valid engine configs for " + operation_graph_tag + "\n" + err_msg};
    }
    return {error_code_t::OK, ""};
}

inline error_t
create_cudnn_execution_plan(std::shared_ptr<ExecutionPlan>& plan,
                            std::string const& serialized_data,
                            cudnnHandle_t handle) {
    auto&& plan_builder = cudnn_frontend::ExecutionPlanBuilder();

    plan_builder.setHandle(handle);

#ifdef NV_CUDNN_DISABLE_EXCEPTION
    // disable exception macro is defined. Calling build will not throw.
    // Check status of desc and return error.
    auto built_plan = plan_builder.loadFromJson(serialized_data);
    RETURN_CUDNN_FRONTEND_ERROR_IF(built_plan.get_status() != CUDNN_STATUS_SUCCESS,
                                   error_code_t::GRAPH_EXECUTION_PLAN_CREATION_FAILED,
                                   built_plan.get_error());
    plan = std::make_shared<ExecutionPlan>(std::move(built_plan));
#else
    // build() can throw
    // wrap in try catch
    try {
        auto built_plan = plan_builder.loadFromJson(serialized_data);
        plan            = std::make_shared<ExecutionPlan>(std::move(built_plan));
    } catch (cudnn_frontend::cudnnException& e) {
        // Silly MSVC error that thinks below condition is constexpr
        // RETURN_CUDNN_FRONTEND_ERROR_IF(
        //     e.getCudnnStatus() != CUDNN_STATUS_SUCCESS, error_code_t::GRAPH_EXECUTION_PLAN_CREATION_FAILED,
        //     e.what());
        CUDNN_FE_LOG_LABEL(" ERROR: " << e.what() << ". ");
        CUDNN_FE_LOG(error_code_t::GRAPH_EXECUTION_PLAN_CREATION_FAILED << " because plan building failed at "
                                                                        << __FILE__ << ":" << __LINE__ << "\n");
        return {error_code_t::GRAPH_EXECUTION_PLAN_CREATION_FAILED, e.what()};
    }
#endif

    return {error_code_t::OK, ""};
}

inline error_t
create_cudnn_execution_plan(std::shared_ptr<ExecutionPlan>& plan,
                            ManagedOpaqueDescriptor const& config,
                            std::string const& operation_graph_tag,
                            std::shared_ptr<KernelCache> kernel_cache,
                            cudnnHandle_t handle) {
    auto&& plan_builder = cudnn_frontend::ExecutionPlanBuilder();

    plan_builder.setHandle(handle).setEngineConfig(config, operation_graph_tag).setKernelCache(kernel_cache);

#ifdef NV_CUDNN_DISABLE_EXCEPTION
    // disable exception macro is defined. Calling build will not throw.
    // Check status of desc and return error.
    auto built_plan = plan_builder.build();
    RETURN_CUDNN_FRONTEND_ERROR_IF(built_plan.get_status() != CUDNN_STATUS_SUCCESS,
                                   error_code_t::GRAPH_EXECUTION_PLAN_CREATION_FAILED,
                                   built_plan.get_error());
    plan = std::make_shared<ExecutionPlan>(std::move(built_plan));
#else
    // build() can throw
    // wrap in try catch
    try {
        auto built_plan = plan_builder.build();
        plan            = std::make_shared<ExecutionPlan>(std::move(built_plan));
    } catch (cudnn_frontend::cudnnException& e) {
        // Silly MSVC error that thinks below condition is constexpr
        // RETURN_CUDNN_FRONTEND_ERROR_IF(
        //     e.getCudnnStatus() != CUDNN_STATUS_SUCCESS, error_code_t::GRAPH_EXECUTION_PLAN_CREATION_FAILED,
        //     e.what());
        CUDNN_FE_LOG_LABEL("ERROR: " << e.what() << ". ");
        CUDNN_FE_LOG(error_code_t::GRAPH_EXECUTION_PLAN_CREATION_FAILED << " because plan building failed at "
                                                                        << __FILE__ << ":" << __LINE__ << "\n");
        return {error_code_t::GRAPH_EXECUTION_PLAN_CREATION_FAILED, e.what()};
    }
#endif

    return {error_code_t::OK, ""};
}

}  // namespace detail

namespace graph {
class Execution_plan_list {
    std::string operation_tag;

    std::vector<bool> barred_indices;
    std::shared_ptr<KernelCache> kernel_cache;

    int64_t max_workspace_allowed  = std::numeric_limits<int64_t>::max();
    int64_t max_shared_mem_allowed = 1024 * 1024 * 1024;  // Crazy high number (2GB) which will never be hit

    std::vector<std::string> barred_engine_names = {};
    EngineConfigList engine_configs;

    error_t
    _build_plan_at_index_impl(cudnnHandle_t handle, int64_t index) {
        if (execution_plans[index] == nullptr) {
            CHECK_CUDNN_FRONTEND_ERROR(detail::create_cudnn_execution_plan(
                execution_plans[index], engine_configs[index], operation_tag, kernel_cache, handle));
        }

        auto is_blocked = [](std::string const& full_name, std::vector<std::string> const& blocked_names) -> bool {
            for (auto const& blocked_name : blocked_names) {
                if (full_name.find(blocked_name) != std::string::npos) {
                    return true;
                }
            }
            return false;
        };
        auto const& plan_tag = execution_plans[index]->getTag();
        if (is_blocked(plan_tag, barred_engine_names)) {
            barred_indices[index] = true;

            return {error_code_t::GRAPH_EXECUTION_PLAN_CREATION_FAILED,
                    "[cudnn_frontend] Error: Deselecting execution plan with name " + plan_tag + " at position " +
                        std::to_string(index)};
        }

        // workspace check for 9.2+ is already done at engine config level
        if (detail::get_backend_version() < 90200 || detail::get_compiled_version() < 90200) {
            if (execution_plans[index]->getWorkspaceSize() > max_workspace_allowed) {
                barred_indices[index] = true;
                return {error_code_t::GRAPH_EXECUTION_PLAN_CREATION_FAILED,
                        "[cudnn_frontend] Error: Workspace size is too large."};
            }
        }

        // Sets candidate in case user does not call execute with plan_index later.
        candidate = index;

        return {error_code_t::OK, ""};
    }

   public:
    std::vector<std::vector<NumericalNote_t>> numeric_notes;
    std::vector<std::vector<BehaviorNote_t>> behavior_notes;

    std::vector<std::shared_ptr<ExecutionPlan>>
        execution_plans;  // a built plan corresponding to each engine config, irrespective of whether config is
                          // selected or deselected.

    // Stores position of best plan in above vector of execution plan
    int64_t candidate = -1;

    void
    set_tag(std::string const& tag) {
        operation_tag = tag;
    }
    void
    enqueue_engine_configs(EngineConfigList list) {
        std::move(list.begin(), list.end(), back_inserter(engine_configs));
    }
    void
    set_kernel_cache(std::shared_ptr<KernelCache> kernel_cache_) {
        kernel_cache = kernel_cache_;
    }

    std::vector<std::shared_ptr<ExecutionPlan>>&
    get_execution_plans() {
        return execution_plans;
    }

    error_t
    query_properties() {
        numeric_notes.reserve(engine_configs.size());
        behavior_notes.reserve(engine_configs.size());

        barred_indices.resize(engine_configs.size(), 0);
        execution_plans.resize(engine_configs.size());

        for (auto& engine_config : engine_configs) {
            int64_t elem_count = 0;
            std::vector<cudnnBackendNumericalNote_t> numeric;
            std::vector<cudnnBackendBehaviorNote_t> behavior;

            ManagedOpaqueDescriptor extractedEngine   = make_shared_backend_pointer(CUDNN_BACKEND_ENGINE_DESCRIPTOR);
            cudnnBackendDescriptor_t extractedEngine_ = extractedEngine->get_backend_descriptor();
            auto status                               = detail::get_attribute(engine_config->get_backend_descriptor(),
                                                CUDNN_ATTR_ENGINECFG_ENGINE,
                                                CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                1,
                                                &elem_count,
                                                &extractedEngine_);
            RETURN_CUDNN_FRONTEND_ERROR_IF((status != CUDNN_STATUS_SUCCESS),
                                           error_code_t::HEURISTIC_QUERY_FAILED,
                                           "Heuristic query Engine failed.");

            status = detail::get_attribute(extractedEngine_,
                                           CUDNN_ATTR_ENGINE_NUMERICAL_NOTE,
                                           CUDNN_TYPE_NUMERICAL_NOTE,
                                           CUDNN_NUMERICAL_NOTE_TYPE_COUNT,
                                           &elem_count,
                                           nullptr);
            RETURN_CUDNN_FRONTEND_ERROR_IF((status != CUDNN_STATUS_SUCCESS),
                                           error_code_t::HEURISTIC_QUERY_FAILED,
                                           "Heuristic query Numerical Note failed");

            numeric.resize(static_cast<size_t>(elem_count));
            status = detail::get_attribute(extractedEngine_,
                                           CUDNN_ATTR_ENGINE_NUMERICAL_NOTE,
                                           CUDNN_TYPE_NUMERICAL_NOTE,
                                           CUDNN_NUMERICAL_NOTE_TYPE_COUNT,
                                           &elem_count,
                                           numeric.data());
            RETURN_CUDNN_FRONTEND_ERROR_IF((status != CUDNN_STATUS_SUCCESS),
                                           error_code_t::HEURISTIC_QUERY_FAILED,
                                           "Heuristic query Numerical Note failed");
            status = detail::get_attribute(extractedEngine_,
                                           CUDNN_ATTR_ENGINE_BEHAVIOR_NOTE,
                                           CUDNN_TYPE_BEHAVIOR_NOTE,
                                           CUDNN_BEHAVIOR_NOTE_TYPE_COUNT,
                                           &elem_count,
                                           nullptr);
            RETURN_CUDNN_FRONTEND_ERROR_IF((status != CUDNN_STATUS_SUCCESS),
                                           error_code_t::HEURISTIC_QUERY_FAILED,
                                           "Heuristic query Behavior Note failed");

            behavior.resize(static_cast<size_t>(elem_count));
            status = detail::get_attribute(extractedEngine_,
                                           CUDNN_ATTR_ENGINE_BEHAVIOR_NOTE,
                                           CUDNN_TYPE_BEHAVIOR_NOTE,
                                           CUDNN_BEHAVIOR_NOTE_TYPE_COUNT,
                                           &elem_count,
                                           behavior.data());
            RETURN_CUDNN_FRONTEND_ERROR_IF((status != CUDNN_STATUS_SUCCESS),
                                           error_code_t::HEURISTIC_QUERY_FAILED,
                                           "Heuristic query Behavior Note failed");

            std::vector<NumericalNote_t> numerics;
            numerics.resize(numeric.size());
            for (auto& note : numeric) {
                numerics.push_back(detail::convert_from_cudnn_type(note));
            }
            numeric_notes.emplace_back(std::move(numerics));

            std::vector<BehaviorNote_t> behaviors;
            behaviors.reserve(behaviors.size());
            for (auto& note : behavior) {
                behaviors.push_back(detail::convert_from_cudnn_type(note));
            }
            behavior_notes.emplace_back(std::move(behaviors));
        }
        return {error_code_t::OK, ""};
    }

    error_t
    filter_numeric_notes(std::vector<NumericalNote_t> const& notes, bool const keep) {
        for (auto& note : notes) {
            for (auto i = 0u; i < engine_configs.size(); i++) {
                bool has_barred_note =
                    std::find(numeric_notes[i].begin(), numeric_notes[i].end(), note) != numeric_notes[i].end();

                barred_indices[i] = barred_indices[i] || (has_barred_note ? !keep : keep);
            }
        }
        return {error_code_t::OK, ""};
    }

    error_t
    filter_behavior_notes(std::vector<BehaviorNote_t> const& notes, bool const keep) {
        for (auto& note : notes) {
            for (auto i = 0u; i < engine_configs.size(); i++) {
                bool has_barred_note =
                    std::find(behavior_notes[i].begin(), behavior_notes[i].end(), note) != behavior_notes[i].end();

                barred_indices[i] = barred_indices[i] || (has_barred_note ? !keep : keep);
            }
        }
        return {error_code_t::OK, ""};
    }

    void
    set_max_workspace_allowed(int64_t const workspace_allowed) {
        max_workspace_allowed = workspace_allowed;
    }

    void
    set_max_shared_mem_allowed(int64_t const smem_allowed) {
        max_shared_mem_allowed = smem_allowed;
    }

    void
    set_barred_names(std::vector<std::string> const& engine_names) {
        barred_engine_names = engine_names;
    }

    EngineConfigList
    get_barred_engine_configs() {
        EngineConfigList barred_engine_configs;
        CUDNN_FE_LOG_LABEL_ENDL("INFO: " << " Filtering engine_configs ..." << engine_configs.size());
        for (auto i = 0u; i < engine_configs.size(); i++) {
            if (barred_indices[i] == false) {
                barred_engine_configs.push_back(engine_configs[i]);
            }
        }
        CUDNN_FE_LOG_LABEL_ENDL("INFO: " << " barred engine_configs ..." << barred_engine_configs.size());
        return barred_engine_configs;
    }

    error_t
    get_name_at_index(int64_t index, std::string& name) const {
        name = detail::get_engine_tag(engine_configs[index]);
        return {error_code_t::OK, ""};
    }

    error_t
    check_support_at_index(cudnnHandle_t handle, int64_t index) {
        // Ignore if the engine config was deselected.
        // This usually happens when user deselects by numerical and behavioural notes.

        RETURN_CUDNN_FRONTEND_ERROR_IF((index < 0) || (static_cast<int64_t>(barred_indices.size()) <= index),
                                       error_code_t::GRAPH_EXECUTION_FAILED,
                                       "Plan index " + std::to_string(index) + " is invalid.");

        if (barred_indices[index] == true) {
            CUDNN_FE_LOG_LABEL_ENDL("Deselecting execution plan at position " << index);
        }

        RETURN_CUDNN_FRONTEND_ERROR_IF(barred_indices[index] == true,
                                       error_code_t::GRAPH_EXECUTION_PLAN_CREATION_FAILED,
                                       "Deselecting execution plan");

        // Ignore if engine name was specified to be ignored by the user.
        auto is_blocked = [](std::string const& full_name, std::vector<std::string> const& blocked_names) -> bool {
            for (auto const& blocked_name : blocked_names) {
                if (full_name.find(blocked_name) != std::string::npos) {
                    return true;
                }
            }
            return false;
        };
        auto cfg_tag = detail::get_engine_tag(engine_configs[index]);
        if (is_blocked(cfg_tag, barred_engine_names)) {
            barred_indices[index] = true;
            return {error_code_t::GRAPH_EXECUTION_PLAN_CREATION_FAILED,
                    "[cudnn_frontend] Error: Deselecting execution plan with name " + cfg_tag + " at position " +
                        std::to_string(index)};
        }

        if (detail::get_backend_version() >= 90200 && detail::get_compiled_version() >= 90200) {
            // Ignore kernels that require larger than tolerable shared memory.
            int32_t shared_memory_size = INT32_MAX;
            auto status                = detail::get_shared_memory_size(engine_configs[index], shared_memory_size);
            if (status.is_bad()) {
                CUDNN_FE_LOG_LABEL_ENDL("WARN: Unknown Shared memory size, so not deselecting plan at position "
                                        << index);
            } else if (shared_memory_size > max_shared_mem_allowed) {
                barred_indices[index] = true;
                return {error_code_t::GRAPH_EXECUTION_PLAN_CREATION_FAILED,
                        "[cudnn_frontend] Error: Skipping plan since shared memory violation. Requires " +
                            std::to_string(shared_memory_size)};
            }

            // Filter by workspace can happen at this engine config stage itself.
            int64_t workspace_size = INT64_MAX;
            CHECK_CUDNN_FRONTEND_ERROR(detail::get_workspace_size(engine_configs[index], workspace_size));
            if (workspace_size > max_workspace_allowed) {
                barred_indices[index] = true;
                return {error_code_t::GRAPH_EXECUTION_PLAN_CREATION_FAILED,
                        "[cudnn_frontend] Error: Skipping plan since workspace violation. Requires " +
                            std::to_string(workspace_size)};
            }
        }
        // Else we need to build the config. A successful execution plan build means that check_support succeeded.
        else {
            CHECK_CUDNN_FRONTEND_ERROR(_build_plan_at_index_impl(handle, index));
        }

        CUDNN_FE_LOG_LABEL_ENDL("Check support for index " << index << " passed with cfg " << cfg_tag);
        // All checks passed for this config, so return success.
        return {error_code_t::OK, ""};
    }

    error_t
    check_support(cudnnHandle_t handle) {
        // Go over each engine config and return true when you find the first one that is supported.
        for (auto i = 0u; i < engine_configs.size(); i++) {
            auto status = check_support_at_index(handle, i);
            if (status.is_good()) {
                return {error_code_t::OK, ""};
            }
        }

        std::string err_msg = detail::get_last_error_string_();
        CUDNN_FE_LOG_LABEL_ENDL("ERROR: No valid engine configs returned from heuristics.\n" << err_msg);
        return {error_code_t::GRAPH_EXECUTION_PLAN_CREATION_FAILED,
                "[cudnn_frontend] Error: No execution plans support the graph." + err_msg};
    }

    error_t
    get_behavior_notes_at_index(int64_t const index, std::vector<BehaviorNote_t>& notes) const {
        RETURN_CUDNN_FRONTEND_ERROR_IF((index < 0) || (static_cast<int64_t>(behavior_notes.size()) <= index),
                                       error_code_t::GRAPH_EXECUTION_FAILED,
                                       "Plan index " + std::to_string(index) + " is invalid.");

        notes = behavior_notes[index];

        return {error_code_t::OK, ""};
    }

    error_t
    build_plans(cudnnHandle_t handle, std::string const& json) {
        execution_plans.resize(1);
        auto const& fe_status = detail::create_cudnn_execution_plan(execution_plans[0], json, handle);

        if (fe_status.is_good()) {
            candidate = 0;
        }

        return fe_status;
    }

    error_t
    build_plan_at_index(cudnnHandle_t handle, int64_t index) {
        CHECK_CUDNN_FRONTEND_ERROR(check_support_at_index(handle, index));
        CHECK_CUDNN_FRONTEND_ERROR(_build_plan_at_index_impl(handle, index));

        return {error_code_t::OK, ""};
    }

    error_t
    build_plans(cudnnHandle_t handle, BuildPlanPolicy_t const policy, bool const do_multithreaded_builds) {
        RETURN_CUDNN_FRONTEND_ERROR_IF(do_multithreaded_builds,
                                       error_code_t::GRAPH_EXECUTION_PLAN_CREATION_FAILED,
                                       "Doing multithreaded builds is not yet supported.");

        // short circuit in case a plan was already created.
        // This happens as check_support for v8 builds a plan.
        if (policy == BuildPlanPolicy_t::HEURISTICS_CHOICE && candidate != -1) {
            return {error_code_t::OK, ""};
        }

        for (auto i = 0u; i < engine_configs.size(); i++) {
            auto status = build_plan_at_index(handle, i);
            if (status.is_bad()) {
                CUDNN_FE_LOG_LABEL_ENDL("WARN: Failed to build plan at " << i);
                continue;
            }

            // Only set the candidate the first time, as the order of iteration is from highest to lowest priority
            if (candidate == -1) {
                candidate = static_cast<int64_t>(i);
                CUDNN_FE_LOG_LABEL_ENDL("INFO: Candidate set as " << i);
            }

            // Return from this function as first successfully built plan is found.
            if (policy == BuildPlanPolicy_t::HEURISTICS_CHOICE) {
                return {error_code_t::OK, ""};
            }
        }

        // Return an error if no execution plans could be built
        RETURN_CUDNN_FRONTEND_ERROR_IF(candidate == -1,
                                       error_code_t::GRAPH_EXECUTION_PLAN_CREATION_FAILED,
                                       "[cudnn_frontend] Error: No valid execution plans built.");

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

    static error_t
    autotune_default_impl(std::vector<std::shared_ptr<ExecutionPlan>>& execution_plans,
                          cudnnHandle_t handle,
                          std::unordered_map<int64_t, void*> const& tensor_to_pointer_map,
                          void* workspace_ptr,
                          void*) {
        // Create the variant pack for all the plans to use.
        std::vector<int64_t> uids;
        std::vector<void*> ptrs;
        for (auto it : tensor_to_pointer_map) {
            uids.push_back(it.first);
            ptrs.push_back(it.second);
        }

        std::vector<std::shared_ptr<ExecutionPlan>> time_sorted_plans;

        auto plan_cmp = [](std::shared_ptr<ExecutionPlan> a, std::shared_ptr<ExecutionPlan> b) {
            return a->getExecutionTime() < b->getExecutionTime();
        };

        std::multiset<std::shared_ptr<ExecutionPlan>, decltype(plan_cmp)> timed_execution_plans(plan_cmp);

        const int maxIterCount         = 100;
        const float threshhold         = 0.95f;
        uint64_t successful_plan_count = 0;
        cudaEvent_t start, stop;
        detail::cuda_event_create(&start);
        detail::cuda_event_create(&stop);
        detail::cuda_device_synchronize();

        cudaStream_t stream = nullptr;
        detail::get_stream(handle, &stream);

        for (auto plan : execution_plans) {
            float time_ms       = 0.0f;
            float final_time_ms = 0.0f;
            float min_time_ms   = std::numeric_limits<float>::max();

            // Warm-up run
            CHECK_CUDNN_FRONTEND_ERROR(detail::execute(handle, plan.get(), ptrs, uids, workspace_ptr));
            successful_plan_count++;
            detail::cuda_device_synchronize();

            for (int i = 0; i < maxIterCount; i++) {
                detail::cuda_event_record(start, stream);

                auto status = detail::execute(handle, plan.get(), ptrs, uids, workspace_ptr);

                detail::cuda_event_record(stop, stream);
                detail::cuda_event_synchronize(stop);
                detail::cuda_event_elapsed_time(&time_ms, start, stop);

                final_time_ms = std::min(min_time_ms, time_ms);
                if (time_ms / min_time_ms < threshhold) {
                    min_time_ms = final_time_ms;
                } else {
                    break;
                }
            }

            CUDNN_FE_LOG_LABEL_ENDL("Plan " << plan->getTag() << " took " << std::setw(10) << final_time_ms);
            plan->setExecutionTime(final_time_ms);
            timed_execution_plans.insert(plan);
        }

        execution_plans.clear();
        for (auto sorted_plan : timed_execution_plans) {
            execution_plans.push_back(sorted_plan);
        }

        detail::cuda_event_destroy(start);
        detail::cuda_event_destroy(stop);

        CUDNN_FE_LOG_LABEL_ENDL("Autotuned " << successful_plan_count << " plans.");
        return {error_code_t::OK, ""};
    }

    std::function<error_t(std::vector<std::shared_ptr<ExecutionPlan>>&,
                          cudnnHandle_t,
                          std::unordered_map<int64_t, void*> const&,
                          void*,
                          void*)>
        autotune_impl = &Execution_plan_list::autotune_default_impl;

    error_t
    autotune(cudnnHandle_t handle,
             std::unordered_map<int64_t, void*> const& tensor_to_pointer_map,
             void* workspace,
             void* user_impl = nullptr) {
        auto error = autotune_impl(execution_plans, handle, tensor_to_pointer_map, workspace, user_impl);
        return error;
    }

    error_t
    is_plan_index_executable(int64_t const index) const {
        RETURN_CUDNN_FRONTEND_ERROR_IF((index < 0) || (static_cast<int64_t>(execution_plans.size()) <= index),
                                       error_code_t::GRAPH_EXECUTION_FAILED,
                                       "Plan index " + std::to_string(index) + " is invalid.");

        RETURN_CUDNN_FRONTEND_ERROR_IF(execution_plans[index] == nullptr,
                                       error_code_t::GRAPH_EXECUTION_FAILED,
                                       "Plan index " + std::to_string(index) + " did not build.");

        return {error_code_t::OK, ""};
    }
};

}  // namespace graph
}  // namespace cudnn_frontend
