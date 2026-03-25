#pragma once

#include <optional>
#include <string>
#include <vector>

#include "../cudnn_frontend_EngineConfig.h"
#include "../cudnn_frontend_Logging.h"
#include "graph_helpers.h"

#include "backend/execution_helpers.h"
#include "backend/plan_helpers.h"
#include "experimental/sm90_sdpa_prefill_engine.h"
#include "experimental/sm100_sdpa_prefill_engine.h"
#include "experimental/sm100_rms_norm_silu_engine.h"

namespace cudnn_frontend {

namespace detail {

inline error_t
execute(cudnnHandle_t handle,
        ExecutionPlan* plan,
        std::vector<void*>& device_ptrs,
        std::vector<int64_t> const& uids,
        void* workspace_ptr,
        std::vector<int64_t> const& override_uids,
        std::vector<std::vector<int64_t>> const& override_shapes,
        std::vector<std::vector<int64_t>> const& override_strides) {
    // TODO: below line fails with MSVC. warning C4127: conditional expression is constant
    // RETURN_CUDNN_FRONTEND_ERROR_IF(!plan, error_code_t::GRAPH_EXECUTION_FAILED, "No plan found to execute!!");
    CUDNN_FE_LOG_LABEL_ENDL("INFO: Executing " << plan->getTag() << "...");

    backend_descriptor variant_pack_descriptor(CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR);
    RETURN_CUDNN_FRONTEND_ERROR_IF(variant_pack_descriptor.get_status() != CUDNN_STATUS_SUCCESS,
                                   error_code_t::CUDNN_BACKEND_API_FAILED,
                                   "Failed to create variant pack's backend descriptor.");

    CHECK_CUDNN_FRONTEND_ERROR(create_variant_pack(
        variant_pack_descriptor, device_ptrs, uids, workspace_ptr, override_uids, override_shapes, override_strides));
    _CUDNN_CHECK_CUDNN_ERROR(execute(handle, plan->get_raw_desc(), variant_pack_descriptor.get_ptr()));

    CUDNN_FE_LOG_LABEL_ENDL("INFO: Executed " << plan->getTag() << ".");

    return {error_code_t::OK, ""};
}

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
    _CUDNN_CHECK_CUDNN_ERROR(execute(handle, plan->get_raw_desc(), variant_pack_descriptor.get_ptr()));

    CUDNN_FE_LOG_LABEL_ENDL("INFO: Executed " << plan->getTag() << ".");

    return {error_code_t::OK, ""};
}

inline error_t
query_cudnn_heuristics_impl(std::shared_ptr<OperationGraph_v8> const& operation_graph,
                            cudnn_frontend::EngineConfigList& configs,
                            std::vector<HeurMode_t> const& modes,
                            int32_t sm_count,
                            std::shared_ptr<const DeviceProperties> device_properties = nullptr) {
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
    statuses = cudnn_frontend::get_heuristics_list(
        modes, *operation_graph, allowAllConfig, configs, true, sm_count, device_properties);
#else
    // build() can throw
    // wrap in try catch
    try {
        statuses = cudnn_frontend::get_heuristics_list(
            modes, *operation_graph, allowAllConfig, configs, true, sm_count, device_properties);
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
                            std::shared_ptr<KernelCache> kernel_cache) {
    auto&& plan_builder = cudnn_frontend::ExecutionPlanBuilder();

    plan_builder.setEngineConfig(config, operation_graph_tag).setKernelCache(kernel_cache);

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
    std::shared_ptr<KernelCache> kernel_cache = nullptr;

    int64_t max_workspace_allowed  = std::numeric_limits<int64_t>::max();
    int64_t max_shared_mem_allowed = 1024 * 1024 * 1024;  // Crazy high number (2GB) which will never be hit

    std::vector<std::string> barred_engine_names = {};
    EngineConfigList engine_configs;

    error_t
    _build_plan_at_index_impl(int64_t index) {
        if (execution_plans[index] == nullptr) {
            CHECK_CUDNN_FRONTEND_ERROR(detail::create_cudnn_execution_plan(
                execution_plans[index], engine_configs[index], operation_tag, kernel_cache));
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
    check_support_at_index(int64_t index) {
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
            CHECK_CUDNN_FRONTEND_ERROR(_build_plan_at_index_impl(index));
        }

        CUDNN_FE_LOG_LABEL_ENDL("Check support for index " << index << " passed with cfg " << cfg_tag);
        // All checks passed for this config, so return success.
        return {error_code_t::OK, ""};
    }

    error_t
    check_support() {
        // Go over each engine config and return true when you find the first one that is supported.
        for (auto i = 0u; i < engine_configs.size(); i++) {
            auto status = check_support_at_index(i);
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
    build_plan_at_index(int64_t index) {
        CHECK_CUDNN_FRONTEND_ERROR(check_support_at_index(index));
        CHECK_CUDNN_FRONTEND_ERROR(_build_plan_at_index_impl(index));

        return {error_code_t::OK, ""};
    }

    error_t
    build_plans(BuildPlanPolicy_t const policy, bool const do_multithreaded_builds) {
        RETURN_CUDNN_FRONTEND_ERROR_IF(do_multithreaded_builds,
                                       error_code_t::GRAPH_EXECUTION_PLAN_CREATION_FAILED,
                                       "Doing multithreaded builds is not yet supported.");

        // short circuit in case a plan was already created.
        // This happens as check_support for v8 builds a plan.
        if (policy == BuildPlanPolicy_t::HEURISTICS_CHOICE && candidate != -1) {
            return {error_code_t::OK, ""};
        }

        for (auto i = 0u; i < engine_configs.size(); i++) {
            auto status = build_plan_at_index(i);
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
        // OSS SDPA engine path
        if (index == OSS_ENGINE_CANDIDATE) {
            RETURN_CUDNN_FRONTEND_ERROR_IF(
                !oss_engine_built_, error_code_t::GRAPH_EXECUTION_FAILED, "OSS SDPA engine not built.");
            return {error_code_t::OK, ""};
        }

        // OSS RmsNorm+SiLU engine path
        if (index == OSS_RMS_NORM_SILU_ENGINE_CANDIDATE) {
            RETURN_CUDNN_FRONTEND_ERROR_IF(
                !oss_rms_norm_silu_built_, error_code_t::GRAPH_EXECUTION_FAILED, "OSS RmsNorm+SiLU engine not built.");
            return {error_code_t::OK, ""};
        }

        RETURN_CUDNN_FRONTEND_ERROR_IF((index < 0) || (static_cast<int64_t>(execution_plans.size()) <= index),
                                       error_code_t::GRAPH_EXECUTION_FAILED,
                                       "Plan index " + std::to_string(index) + " is invalid.");

        RETURN_CUDNN_FRONTEND_ERROR_IF(execution_plans[index] == nullptr,
                                       error_code_t::GRAPH_EXECUTION_FAILED,
                                       "Plan index " + std::to_string(index) + " did not build.");

        return {error_code_t::OK, ""};
    }

    // ================================================================
    // Open-source NVRTC engine support
    // ================================================================

    static constexpr int64_t OSS_ENGINE_CANDIDATE = -2;

    // Context cached from the Graph for OSS engine execution
    struct OssEngineContext {
        int64_t batch = 0, heads_q = 0, heads_kv = 0, seq_q = 0, seq_kv = 0, d = 0;
        int64_t q_uid = -1, k_uid = -1, v_uid = -1, o_uid = -1, max_uid = -1, sum_exp_uid = -1;
        std::vector<int64_t> q_stride, k_stride, v_stride, o_stride;
        std::vector<int64_t> max_stride, sum_exp_stride;
        std::optional<float> attn_scale;
    };

    void
    set_oss_engine(std::shared_ptr<experimental::IOssSdpaEngine> engine) {
        oss_engine_ = std::move(engine);
    }

    void
    set_oss_engine_context(OssEngineContext ctx) {
        oss_ctx_ = std::move(ctx);
    }

    bool
    has_oss_engine() const {
        return oss_engine_ != nullptr;
    }

    bool
    is_oss_candidate() const {
        return candidate == OSS_ENGINE_CANDIDATE;
    }

    error_t
    check_oss_engine_support(int64_t sm_version) {
        RETURN_CUDNN_FRONTEND_ERROR_IF(!oss_engine_, error_code_t::GRAPH_NOT_SUPPORTED, "No OSS engine registered");
        cudnn_frontend::experimental::AttentionShape_t shape = {
            static_cast<uint32_t>(oss_ctx_.batch),
            static_cast<uint32_t>(oss_ctx_.heads_q),
            static_cast<uint32_t>(oss_ctx_.heads_kv),
            static_cast<uint32_t>(oss_ctx_.heads_kv),
            static_cast<uint32_t>(oss_ctx_.seq_q),
            static_cast<uint32_t>(oss_ctx_.seq_kv),
            static_cast<uint32_t>(oss_ctx_.d),
            static_cast<uint32_t>(oss_ctx_.d),
        };
        auto status = oss_engine_->check_support(shape, sm_version);
        if (status.is_good()) {
            oss_engine_supported_ = true;
            candidate             = OSS_ENGINE_CANDIDATE;
        }
        return status;
    }

    error_t
    build_oss_engine() {
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            !oss_engine_supported_, error_code_t::GRAPH_NOT_SUPPORTED, "OSS engine not supported");
        auto status = oss_engine_->build();
        if (status.is_good()) {
            oss_engine_built_ = true;
            candidate         = OSS_ENGINE_CANDIDATE;
        }
        return status;
    }

    error_t
    execute_oss_engine(std::unordered_map<int64_t, void*> const& tensor_uid_to_pointer_map,
                       void* workspace,
                       int device,
                       cudaStream_t stream) const {
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            !oss_engine_built_, error_code_t::GRAPH_EXECUTION_FAILED, "OSS engine not built");

        auto get_ptr = [&](int64_t uid) -> void* {
            auto it = tensor_uid_to_pointer_map.find(uid);
            return (it != tensor_uid_to_pointer_map.end()) ? it->second : nullptr;
        };

        void* q_ptr       = get_ptr(oss_ctx_.q_uid);
        void* k_ptr       = get_ptr(oss_ctx_.k_uid);
        void* v_ptr       = get_ptr(oss_ctx_.v_uid);
        void* o_ptr       = get_ptr(oss_ctx_.o_uid);
        void* max_ptr     = get_ptr(oss_ctx_.max_uid);
        void* sum_exp_ptr = get_ptr(oss_ctx_.sum_exp_uid);

        RETURN_CUDNN_FRONTEND_ERROR_IF(!q_ptr || !k_ptr || !v_ptr || !o_ptr,
                                       error_code_t::INVALID_VARIANT_PACK,
                                       "Missing Q/K/V/O pointers in variant pack for OSS engine");
        RETURN_CUDNN_FRONTEND_ERROR_IF(!max_ptr || !sum_exp_ptr,
                                       error_code_t::INVALID_VARIANT_PACK,
                                       "Missing max/sum_exp pointers in variant pack for OSS engine");

        return oss_engine_->execute(static_cast<int>(oss_ctx_.batch),
                                    static_cast<int>(oss_ctx_.heads_q),
                                    static_cast<int>(oss_ctx_.heads_kv),
                                    static_cast<int>(oss_ctx_.seq_q),
                                    static_cast<int>(oss_ctx_.seq_kv),
                                    static_cast<int>(oss_ctx_.d),
                                    q_ptr,
                                    oss_ctx_.q_stride,
                                    k_ptr,
                                    oss_ctx_.k_stride,
                                    v_ptr,
                                    oss_ctx_.v_stride,
                                    o_ptr,
                                    oss_ctx_.o_stride,
                                    max_ptr,
                                    oss_ctx_.max_stride,
                                    sum_exp_ptr,
                                    oss_ctx_.sum_exp_stride,
                                    workspace,
                                    device,
                                    stream,
                                    oss_ctx_.attn_scale);
    }

    // Dynamic-shape overload: resolve overridden dims/strides from override vectors,
    // then delegate to the engine's execute() with the resolved values.
    error_t
    execute_oss_engine(std::unordered_map<int64_t, void*> const& tensor_uid_to_pointer_map,
                       void* workspace,
                       int device,
                       cudaStream_t stream,
                       std::vector<int64_t> const& override_uids,
                       std::vector<std::vector<int64_t>> const& override_shapes,
                       std::vector<std::vector<int64_t>> const& override_strides) const {
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            !oss_engine_built_, error_code_t::GRAPH_EXECUTION_FAILED, "OSS engine not built");
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            override_uids.size() != override_shapes.size() || override_uids.size() != override_strides.size(),
            error_code_t::INVALID_VALUE,
            "override_uids/shapes/strides must have the same size");

        // Build uid → index lookup
        std::unordered_map<int64_t, size_t> uid_to_idx;
        for (size_t i = 0; i < override_uids.size(); ++i) {
            uid_to_idx[override_uids[i]] = i;
        }

        // Helper: resolve shape/stride for a given tensor UID
        auto resolve_shape = [&](int64_t uid,
                                 std::vector<int64_t> const& default_shape) -> std::vector<int64_t> const& {
            auto it = uid_to_idx.find(uid);
            return (it != uid_to_idx.end()) ? override_shapes[it->second] : default_shape;
        };
        auto resolve_stride = [&](int64_t uid,
                                  std::vector<int64_t> const& default_stride) -> std::vector<int64_t> const& {
            auto it = uid_to_idx.find(uid);
            return (it != uid_to_idx.end()) ? override_strides[it->second] : default_stride;
        };

        // Resolve Q shape → extract batch, heads_q, seq_q, d
        std::vector<int64_t> q_default_shape = {oss_ctx_.batch, oss_ctx_.heads_q, oss_ctx_.seq_q, oss_ctx_.d};
        auto const& q_shape                  = resolve_shape(oss_ctx_.q_uid, q_default_shape);
        auto const& q_stride                 = resolve_stride(oss_ctx_.q_uid, oss_ctx_.q_stride);

        RETURN_CUDNN_FRONTEND_ERROR_IF(
            q_shape.size() < 4, error_code_t::INVALID_VALUE, "Q override shape must have at least 4 elements");

        int64_t batch   = q_shape[0];
        int64_t heads_q = q_shape[1];
        int64_t seq_q   = q_shape[2];
        int64_t d       = q_shape[3];

        // d must not change (kernel is compiled for a specific d)
        RETURN_CUDNN_FRONTEND_ERROR_IF(d != oss_ctx_.d,
                                       error_code_t::INVALID_VALUE,
                                       "Cannot change d dimension dynamically (compiled kernel is d=" +
                                           std::to_string(oss_ctx_.d) + ", override d=" + std::to_string(d) + ")");

        // Resolve K shape → extract heads_kv, seq_kv
        std::vector<int64_t> k_default_shape = {oss_ctx_.batch, oss_ctx_.heads_kv, oss_ctx_.seq_kv, oss_ctx_.d};
        auto const& k_shape                  = resolve_shape(oss_ctx_.k_uid, k_default_shape);
        auto const& k_stride                 = resolve_stride(oss_ctx_.k_uid, oss_ctx_.k_stride);

        int64_t heads_kv = k_shape[1];
        int64_t seq_kv   = k_shape[2];

        // Resolve V, O, max, sum_exp strides
        auto const& v_stride   = resolve_stride(oss_ctx_.v_uid, oss_ctx_.v_stride);
        auto const& o_stride   = resolve_stride(oss_ctx_.o_uid, oss_ctx_.o_stride);
        auto const& max_stride = resolve_stride(oss_ctx_.max_uid, oss_ctx_.max_stride);
        auto const& se_stride  = resolve_stride(oss_ctx_.sum_exp_uid, oss_ctx_.sum_exp_stride);

        // Look up device pointers
        auto get_ptr = [&](int64_t uid) -> void* {
            auto it = tensor_uid_to_pointer_map.find(uid);
            return (it != tensor_uid_to_pointer_map.end()) ? it->second : nullptr;
        };

        void* q_ptr       = get_ptr(oss_ctx_.q_uid);
        void* k_ptr       = get_ptr(oss_ctx_.k_uid);
        void* v_ptr       = get_ptr(oss_ctx_.v_uid);
        void* o_ptr       = get_ptr(oss_ctx_.o_uid);
        void* max_ptr     = get_ptr(oss_ctx_.max_uid);
        void* sum_exp_ptr = get_ptr(oss_ctx_.sum_exp_uid);

        RETURN_CUDNN_FRONTEND_ERROR_IF(!q_ptr || !k_ptr || !v_ptr || !o_ptr,
                                       error_code_t::INVALID_VARIANT_PACK,
                                       "Missing Q/K/V/O pointers in variant pack for OSS engine");
        RETURN_CUDNN_FRONTEND_ERROR_IF(!max_ptr || !sum_exp_ptr,
                                       error_code_t::INVALID_VARIANT_PACK,
                                       "Missing max/sum_exp pointers in variant pack for OSS engine");

        return oss_engine_->execute(static_cast<int>(batch),
                                    static_cast<int>(heads_q),
                                    static_cast<int>(heads_kv),
                                    static_cast<int>(seq_q),
                                    static_cast<int>(seq_kv),
                                    static_cast<int>(d),
                                    q_ptr,
                                    q_stride,
                                    k_ptr,
                                    k_stride,
                                    v_ptr,
                                    v_stride,
                                    o_ptr,
                                    o_stride,
                                    max_ptr,
                                    max_stride,
                                    sum_exp_ptr,
                                    se_stride,
                                    workspace,
                                    device,
                                    stream,
                                    oss_ctx_.attn_scale);
    }

    // ================================================================
    // Open-source NVRTC engine support: RmsNorm + SiLU
    // ================================================================

    static constexpr int64_t OSS_RMS_NORM_SILU_ENGINE_CANDIDATE = -3;

    // Context cached from the Graph for RmsNorm+SiLU engine execution
    struct OssRmsNormSiluContext {
        int64_t num_tokens = 0;
        int64_t C          = 0;

        // Tensor UIDs for pointer lookup from variant pack
        int64_t x_uid       = -1;  // input [num_tokens, C]
        int64_t y_uid       = -1;  // output [num_tokens, C] (after SiLU)
        int64_t scale_uid   = -1;  // gamma weights [C]
        int64_t bias_uid    = -1;  // beta bias [C], optional (-1 if absent)
        int64_t epsilon_uid = -1;  // epsilon scalar

        // FP8 output: optional scale/scale_inv/amax tensor UIDs
        int64_t fp8_scale_uid     = -1;
        int64_t fp8_scale_inv_uid = -1;
        int64_t fp8_amax_uid      = -1;

        // NVFP4 output: row-wise block-scale tensor UID
        int64_t nvfp4_scale_row_uid = -1;

        experimental::RmsNormSiluDtype output_dtype = experimental::RmsNormSiluDtype::BF16;
    };

    void
    set_oss_rms_norm_silu_engine(std::shared_ptr<experimental::IOssNormEngine> engine) {
        oss_rms_norm_silu_engine_ = std::move(engine);
    }

    void
    set_oss_rms_norm_silu_context(OssRmsNormSiluContext ctx) {
        oss_rms_norm_silu_ctx_ = std::move(ctx);
    }

    bool
    has_oss_rms_norm_silu_engine() const {
        return oss_rms_norm_silu_engine_ != nullptr;
    }

    bool
    is_oss_rms_norm_silu_candidate() const {
        return candidate == OSS_RMS_NORM_SILU_ENGINE_CANDIDATE;
    }

    error_t
    check_oss_rms_norm_silu_support(int64_t sm_version) {
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            !oss_rms_norm_silu_engine_, error_code_t::GRAPH_NOT_SUPPORTED, "No RmsNorm+SiLU OSS engine registered");
        experimental::NormSiluShape_t shape;
        shape.C            = static_cast<int>(oss_rms_norm_silu_ctx_.C);
        shape.num_tokens   = static_cast<int>(oss_rms_norm_silu_ctx_.num_tokens);
        shape.output_dtype = oss_rms_norm_silu_ctx_.output_dtype;
        auto status        = oss_rms_norm_silu_engine_->check_support(shape, static_cast<int>(sm_version));
        if (status.is_good()) {
            oss_rms_norm_silu_supported_ = true;
            candidate                    = OSS_RMS_NORM_SILU_ENGINE_CANDIDATE;
        }
        return status;
    }

    error_t
    build_oss_rms_norm_silu_engine() {
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            !oss_rms_norm_silu_supported_, error_code_t::GRAPH_NOT_SUPPORTED, "RmsNorm+SiLU OSS engine not supported");
        auto status = oss_rms_norm_silu_engine_->build();
        if (status.is_good()) {
            oss_rms_norm_silu_built_ = true;
            candidate                = OSS_RMS_NORM_SILU_ENGINE_CANDIDATE;
        }
        return status;
    }

    int64_t
    get_oss_rms_norm_silu_workspace_size() const {
        if (!oss_rms_norm_silu_engine_) return 0;
        return oss_rms_norm_silu_engine_->get_workspace_size();
    }

    error_t
    execute_oss_rms_norm_silu_engine(std::unordered_map<int64_t, void*> const& tensor_uid_to_pointer_map,
                                     void* workspace,
                                     int device,
                                     cudaStream_t stream) const {
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            !oss_rms_norm_silu_built_, error_code_t::GRAPH_EXECUTION_FAILED, "RmsNorm+SiLU OSS engine not built");

        auto get_ptr = [&](int64_t uid) -> void* {
            auto it = tensor_uid_to_pointer_map.find(uid);
            return (it != tensor_uid_to_pointer_map.end()) ? it->second : nullptr;
        };

        void* x_ptr     = get_ptr(oss_rms_norm_silu_ctx_.x_uid);
        void* y_ptr     = get_ptr(oss_rms_norm_silu_ctx_.y_uid);
        void* scale_ptr = get_ptr(oss_rms_norm_silu_ctx_.scale_uid);
        void* bias_ptr  = (oss_rms_norm_silu_ctx_.bias_uid >= 0) ? get_ptr(oss_rms_norm_silu_ctx_.bias_uid) : nullptr;
        void* eps_ptr   = get_ptr(oss_rms_norm_silu_ctx_.epsilon_uid);

        RETURN_CUDNN_FRONTEND_ERROR_IF(!x_ptr || !y_ptr || !scale_ptr,
                                       error_code_t::INVALID_VARIANT_PACK,
                                       "Missing X/Y/SCALE pointers in variant pack for RmsNorm+SiLU OSS engine");
        RETURN_CUDNN_FRONTEND_ERROR_IF(!eps_ptr,
                                       error_code_t::INVALID_VARIANT_PACK,
                                       "Missing EPSILON pointer in variant pack for RmsNorm+SiLU OSS engine");

        // Read epsilon value from the tensor pointer (it's a scalar tensor)
        float epsilon = *static_cast<float const*>(eps_ptr);

        // Populate FP8 / NVFP4 extra params from variant pack
        experimental::RmsNormSiluExtraParams extra;
        auto opt_ptr          = [&](int64_t uid) -> void* { return (uid >= 0) ? get_ptr(uid) : nullptr; };
        extra.fp8_scale       = opt_ptr(oss_rms_norm_silu_ctx_.fp8_scale_uid);
        extra.fp8_scale_inv   = opt_ptr(oss_rms_norm_silu_ctx_.fp8_scale_inv_uid);
        extra.fp8_amax        = opt_ptr(oss_rms_norm_silu_ctx_.fp8_amax_uid);
        extra.nvfp4_scale_row = opt_ptr(oss_rms_norm_silu_ctx_.nvfp4_scale_row_uid);

        return oss_rms_norm_silu_engine_->execute(x_ptr,
                                                  y_ptr,
                                                  scale_ptr,
                                                  bias_ptr,
                                                  static_cast<int>(oss_rms_norm_silu_ctx_.num_tokens),
                                                  static_cast<int>(oss_rms_norm_silu_ctx_.C),
                                                  epsilon,
                                                  workspace,
                                                  device,
                                                  stream,
                                                  extra);
    }

   private:
    std::shared_ptr<experimental::IOssSdpaEngine> oss_engine_;
    bool oss_engine_supported_ = false;
    bool oss_engine_built_     = false;
    OssEngineContext oss_ctx_;

    std::shared_ptr<experimental::IOssNormEngine> oss_rms_norm_silu_engine_;
    bool oss_rms_norm_silu_supported_ = false;
    bool oss_rms_norm_silu_built_     = false;
    OssRmsNormSiluContext oss_rms_norm_silu_ctx_;
};

}  // namespace graph
}  // namespace cudnn_frontend
