#pragma once

#include <optional>
#include <string>
#include <vector>

#include "../cudnn_frontend_EngineConfig.h"
#include "../cudnn_frontend_Logging.h"
#include "graph_helpers.h"

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
    getLogger() << "[cudnn_frontend] INFO: Executing " << plan->getTag() << "..." << std::endl;

    auto&& variant_pack_builder = VariantPackBuilder();
    variant_pack_builder.setDataPointers(device_ptrs.size(), device_ptrs.data())
        .setUids(uids.size(), uids.data())
        .setWorkspacePointer(workspace_ptr);

    cudnnBackendDescriptor_t raw_variant_pack = nullptr;
#ifdef NV_CUDNN_DISABLE_EXCEPTION
    // disable exception macro is defined. Calling build will not throw.
    // Check status of desc and return error.
    auto variant_pack = variant_pack_builder.build();
    RETURN_CUDNN_FRONTEND_ERROR_IF(variant_pack.get_status() != CUDNN_STATUS_SUCCESS,
                                   error_code_t::INVALID_VARIANT_PACK,
                                   variant_pack.get_error());
    raw_variant_pack = variant_pack.get_raw_desc();
#else
    // build() can throw
    // wrap in try catch
    try {
        auto variant_pack = variant_pack_builder.build();
        raw_variant_pack  = variant_pack.get_raw_desc();
    } catch (cudnn_frontend::cudnnException& e) {
        // Silly MSVC error that thinks below condition is constexpr
        // RETURN_CUDNN_FRONTEND_ERROR_IF(
        //     e.getCudnnStatus() != CUDNN_STATUS_SUCCESS, error_code_t::INVALID_VARIANT_PACK, e.what());
        getLogger() << "[cudnn_frontend] ERROR: " << e.what() << ". ";
        getLogger() << error_code_t::INVALID_VARIANT_PACK << " because variant packing building failed at " << __FILE__
                    << ":" << __LINE__ << "\n";
        return {error_code_t::INVALID_VARIANT_PACK, e.what()};
    }
#endif

    auto status = cudnnBackendExecute(handle, plan->get_raw_desc(), raw_variant_pack);
    if (status != CUDNN_STATUS_SUCCESS) {
        std::string message = "[cudnn_frontend] ERROR: Graph execution failed.";
        return {error_code_t::GRAPH_EXECUTION_FAILED, message};
    }
    getLogger() << "[cudnn_frontend] INFO: Executed " << plan->getTag() << "." << std::endl;

    return {error_code_t::OK, ""};
}

inline error_t
query_cudnn_heuristics_impl(std::shared_ptr<OperationGraph_v8> const& operation_graph,
                            cudnn_frontend::EngineConfigList& configs,
                            std::vector<HeurMode_t> const& modes) {
    auto const& operation_graph_tag = operation_graph->getTag();
    getLogger() << "[cudnn_frontend] INFO: "
                << " Getting plan from heuristics for " << operation_graph_tag << " ..." << std::endl;

    std::vector<cudnnStatus_t> statuses;
#ifdef NV_CUDNN_DISABLE_EXCEPTION
    // disable exception macro is defined. Calling build will not throw.
    // Check status of desc and return error.
    statuses = cudnn_frontend::get_heuristics_list(modes, *operation_graph, allowAllConfig, configs, true);
#else
    // build() can throw
    // wrap in try catch
    try {
        statuses = cudnn_frontend::get_heuristics_list(modes, *operation_graph, allowAllConfig, configs, true);
    } catch (cudnn_frontend::cudnnException& e) {
        // Silly MSVC error that thinks below condition is constexpr
        // RETURN_CUDNN_FRONTEND_ERROR_IF(
        //     e.getCudnnStatus() != CUDNN_STATUS_SUCCESS, error_code_t::HEURISTIC_QUERY_FAILED, e.what());
        getLogger() << "[cudnn_frontend] ERROR: " << e.what() << ". ";
        getLogger() << error_code_t::HEURISTIC_QUERY_FAILED << " because querying heuristics failed at " << __FILE__
                    << ":" << __LINE__ << "\n";
        return {error_code_t::HEURISTIC_QUERY_FAILED, e.what()};
    }
#endif

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
        getLogger() << "[cudnn_frontend] ERROR: " << e.what() << ". ";
        getLogger() << error_code_t::GRAPH_EXECUTION_PLAN_CREATION_FAILED << " because plan building failed at "
                    << __FILE__ << ":" << __LINE__ << "\n";
        return {error_code_t::GRAPH_EXECUTION_PLAN_CREATION_FAILED, e.what()};
    }
#endif

    return {error_code_t::OK, ""};
}

inline error_t
create_cudnn_execution_plan(std::shared_ptr<ExecutionPlan>& plan,
                            ManagedOpaqueDescriptor const& config,
                            std::string const& operation_graph_tag,
                            cudnnHandle_t handle) {
    auto&& plan_builder = cudnn_frontend::ExecutionPlanBuilder();

    plan_builder.setHandle(handle).setEngineConfig(config, operation_graph_tag);

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
        getLogger() << "[cudnn_frontend] ERROR: " << e.what() << ". ";
        getLogger() << error_code_t::GRAPH_EXECUTION_PLAN_CREATION_FAILED << " because plan building failed at "
                    << __FILE__ << ":" << __LINE__ << "\n";
        return {error_code_t::GRAPH_EXECUTION_PLAN_CREATION_FAILED, e.what()};
    }
#endif

    return {error_code_t::OK, ""};
}

}  // namespace detail

namespace graph {
class Execution_plan_list {
    std::string operation_tag;
    std::vector<std::vector<cudnnBackendNumericalNote_t>> numeric_notes;
    std::vector<std::vector<cudnnBackendNumericalNote_t>> behavior_notes;
    std::vector<bool> filtered_indices;

    int64_t max_workspace_allowed = std::numeric_limits<int64_t>::max();

    EngineConfigList engine_configs;

   public:
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

        filtered_indices.resize(engine_configs.size(), 0);
        execution_plans.resize(engine_configs.size());

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
    deselect_numeric_notes(std::vector<NumericalNote_t> const& notes) {
        for (auto& note : notes) {
            cudnnBackendNumericalNote_t backend_note;

            RETURN_CUDNN_FRONTEND_ERROR_IF(detail::convert_to_cudnn_type(note, backend_note) != CUDNN_STATUS_SUCCESS,
                                           error_code_t::CUDNN_BACKEND_API_FAILED,
                                           "Unexpected behaviour note provided.");

            for (auto i = 0u; i < engine_configs.size(); i++) {
                if (std::find(numeric_notes[i].begin(), numeric_notes[i].end(), backend_note) !=
                    numeric_notes[i].end()) {
                    filtered_indices[i] = true;
                }
            }
        }
        return {error_code_t::OK, ""};
    }

    error_t
    deselect_behavior_notes(std::vector<BehaviorNote_t> const& notes) {
        for (auto& note : notes) {
            cudnnBackendBehaviorNote_t backend_note;

            RETURN_CUDNN_FRONTEND_ERROR_IF(detail::convert_to_cudnn_type(note, backend_note) != CUDNN_STATUS_SUCCESS,
                                           error_code_t::CUDNN_BACKEND_API_FAILED,
                                           "Unexpected behaviour note provided.");

            for (auto i = 0u; i < engine_configs.size(); i++) {
                if (std::find(behavior_notes[i].begin(), behavior_notes[i].end(), backend_note) !=
                    behavior_notes[i].end()) {
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
        for (auto i = 0u; i < engine_configs.size(); i++) {
            if (filtered_indices[i]) {
                getLogger() << "[cudnn_frontend] INFO: Deselecting execution plan at position " << i << std::endl;
                continue;
            }

            auto const& config = engine_configs[i];
            auto fe_status     = detail::create_cudnn_execution_plan(execution_plans[i], config, operation_tag, handle);
            getLogger() << "[cudnn_frontend] INFO: Building plan at index " << i << " gave " << fe_status.get_code()
                        << " with message: " << fe_status.get_message() << std::endl;

            // If a plan is built successfully, set it as a candidate
            if (fe_status.is_good()) {
                // Filter out execution plans with workspace greater than whats available from user
                if (execution_plans[i]->getWorkspaceSize() > max_workspace_allowed) {
                    filtered_indices[i] = true;
                    getLogger() << "[cudnn_frontend] INFO: Deselecting execution plan at position " << i << std::endl;
                    continue;
                }

                candidate = static_cast<int64_t>(i);
                return {error_code_t::OK, ""};
            }
        }

        // No plans were able to be built. Return error.
        return {error_code_t::GRAPH_EXECUTION_PLAN_CREATION_FAILED,
                "[cudnn_frontend] Error: No execution plans built successfully."};
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
        RETURN_CUDNN_FRONTEND_ERROR_IF(filtered_indices[index] == true,
                                       error_code_t::GRAPH_EXECUTION_PLAN_CREATION_FAILED,
                                       "Chosen plan index has been deselected.");

        if (execution_plans[index] != nullptr && execution_plans[index]->getWorkspaceSize() <= max_workspace_allowed) {
            return {error_code_t::OK, ""};
        };

        auto fe_status =
            detail::create_cudnn_execution_plan(execution_plans[index], engine_configs[index], operation_tag, handle);

        getLogger() << "[cudnn_frontend] INFO: Building plan at index " << index << " gave " << fe_status.get_code()
                    << " with message: " << fe_status.get_message() << std::endl;

        // Sets candidate in case user does not call execute with plan_index later.
        if (fe_status.is_good()) {
            if (execution_plans[index]->getWorkspaceSize() <= max_workspace_allowed) {
                candidate = index;
            } else {
                filtered_indices[index] = true;
                return {error_code_t::GRAPH_EXECUTION_PLAN_CREATION_FAILED,
                        "[cudnn_frontend] Error: Workspace size is too large."};
            }
        }

        return fe_status;
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
            if (filtered_indices[i]) {
                getLogger() << "[cudnn_frontend] INFO: Skipping deselected engine plan at index " << i << std::endl;
                continue;
            }

            auto fe_status =
                detail::create_cudnn_execution_plan(execution_plans[i], engine_configs[i], operation_tag, handle);
            getLogger() << "[cudnn_frontend] INFO: Building plan at index " << i << " gave " << fe_status.get_code()
                        << " with message: " << fe_status.get_message() << std::endl;

            if (fe_status.is_good()) {
                if (execution_plans[i]->getWorkspaceSize() > max_workspace_allowed) {
                    filtered_indices[i] = true;
                    continue;
                }
                // Only set the candidate the first time, as the order of iteration is from highest to lowest priority
                if (candidate == -1) {
                    candidate = static_cast<int64_t>(i);
                }

                // Return from this function as first successfully built plan is found.
                if (policy == BuildPlanPolicy_t::HEURISTICS_CHOICE) {
                    return {error_code_t::OK, ""};
                }
            }
        }

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
            CHECK_CUDNN_FRONTEND_ERROR(detail::execute(handle, plan.get(), ptrs, uids, workspace_ptr));
            successful_plan_count++;
            cudaDeviceSynchronize();

            for (int i = 0; i < maxIterCount; i++) {
                cudaEventRecord(start, stream);

                auto status = detail::execute(handle, plan.get(), ptrs, uids, workspace_ptr);

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
};

}  // namespace graph
}  // namespace cudnn_frontend