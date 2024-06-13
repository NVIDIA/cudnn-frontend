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
#include "plans.h"

namespace cudnn_frontend {

class ICudnn {
   protected:
    using uid_t = int64_t;

    //// Store tensors and operations as they (probably?) need to be kept alive.
    //
    // The tensor mapping from fe::Tensor to be::Tensor.
    //
    // sub nodes share fe::Tensor. Example, in a conv-bias graph, conv output Y and bias input IN_0 are the same
    // fe::Tensor. But both sub ndoes need to work together to make sure only one be::Tensor is created. Hence this
    // uid_to_backend_tensors acts as the global registry for each sub node to use.
    //
    // Key cannot be fe::Tensor, or shared_ptr<fe::Tensor>, or underlying object address of fe::Tensor.
    // Hence using uid, as that uniquely identifies both types of tensors.
    std::unordered_map<int64_t, std::shared_ptr<cudnn_frontend::Tensor>> uid_to_tensors;
    std::vector<std::shared_ptr<cudnn_frontend::Operation>> operations;

    std::shared_ptr<OperationGraph_v8> operation_graph;
    std::unordered_set<graph::Tensor_attributes::uid_t> variant_pack_uids;

    graph::Execution_plan_list plans;

    // TODO: Always returns OK. Can the status and error message be accessed from tensor descriptor?
    error_t
    create_cudnn_tensor(std::shared_ptr<graph::Tensor_attributes> const& props,
                        std::unordered_map<uid_t, std::shared_ptr<cudnn_frontend::Tensor>>& tensors) const {
        // TODO: uid check has to be moved to validate stage.
        RETURN_CUDNN_FRONTEND_ERROR_IF(props->has_uid() == false,
                                       error_code_t::ATTRIBUTE_NOT_SET,
                                       "Tensor named '" + props->get_name() + "' has no uid assigned.");

        // Check whether tensor already created
        auto tensor_uid = props->get_uid();
        if (tensors.find(tensor_uid) != tensors.end()) {
            getLogger() << "[cudnn_frontend] INFO: Backend Tensor named '" << props->get_name() << "' with UID "
                        << tensor_uid << " already created." << std::endl;
            return {error_code_t::OK, ""};
        }

        auto&& tensor_builder = cudnn_frontend::TensorBuilder();

        tensor_builder.setDim(props->get_dim().size(), props->get_dim().data())
            .setStrides(props->get_stride().size(), props->get_stride().data())
            .setId(tensor_uid)
            .setAlignment(16)
            .setDataType(props->get_data_type())
            .setVirtual(props->get_is_virtual())
            .setByValue(props->get_is_pass_by_value())
            .setReorderType(props->get_reordering_type());

        if (auto ragged_offset_props = props->get_ragged_offset()) {
            CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(ragged_offset_props, tensors));
            tensor_builder.setRaggedOffset(tensors.at(ragged_offset_props->get_uid()));
        }

#ifdef NV_CUDNN_DISABLE_EXCEPTION
        // disable exception macro is defined. Calling build will not throw.
        // Check status of desc and return error.
        auto tensor = tensor_builder.build();
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            tensor.get_status() != CUDNN_STATUS_SUCCESS, error_code_t::CUDNN_BACKEND_API_FAILED, tensor.get_error());
        tensors.emplace(tensor_uid, std::make_shared<Tensor>(std::move(tensor)));
#else
        // build() can throw
        // wrap in try catch
        try {
            auto tensor = tensor_builder.build();
            tensors.emplace(tensor_uid, std::make_shared<Tensor>(std::move(tensor)));
        } catch (cudnn_frontend::cudnnException& e) {
            RETURN_CUDNN_FRONTEND_ERROR_IF(
                e.getCudnnStatus() != CUDNN_STATUS_SUCCESS, error_code_t::CUDNN_BACKEND_API_FAILED, e.what());
        }
#endif

        return {error_code_t::OK, ""};
    }

    error_t
    create_cudnn_operation_graph(cudnnHandle_t handle) {
        std::vector<Operation const*> cudnn_operations;
        for (std::shared_ptr<cudnn_frontend::Operation> operation : operations) {
            cudnn_operations.push_back(operation.get());
        }

        auto&& cudnn_operation_graph_builder = cudnn_frontend::OperationGraphBuilder();
        cudnn_operation_graph_builder.setHandle(handle).setOperationGraph(cudnn_operations.size(),
                                                                          cudnn_operations.data());

#ifdef NV_CUDNN_DISABLE_EXCEPTION
        // disable exception macro is defined. Calling build will not throw.
        // Check status of desc and return error.
        auto cudnn_operation_graph = cudnn_operation_graph_builder.build();
        RETURN_CUDNN_FRONTEND_ERROR_IF(cudnn_operation_graph.get_status() != CUDNN_STATUS_SUCCESS,
                                       error_code_t::CUDNN_BACKEND_API_FAILED,
                                       cudnn_operation_graph.get_error());
        operation_graph = std::make_shared<OperationGraph_v8>(std::move(cudnn_operation_graph));
#else
        // build() can throw
        // wrap in try catch
        try {
            auto cudnn_operation_graph = cudnn_operation_graph_builder.build();
            operation_graph            = std::make_shared<OperationGraph_v8>(std::move(cudnn_operation_graph));
        } catch (cudnn_frontend::cudnnException& e) {
            RETURN_CUDNN_FRONTEND_ERROR_IF(
                e.getCudnnStatus() != CUDNN_STATUS_SUCCESS, error_code_t::CUDNN_BACKEND_API_FAILED, e.what());
        }
#endif
        return {error_code_t::OK, "Successfully built Operation Graphs."};
    }

   public:
    error_t
    get_cudnn_workspace_size_node(int64_t const plan_index, int64_t& cudnn_workspace_size) const {
        CHECK_CUDNN_FRONTEND_ERROR(plans.is_plan_index_executable(plan_index));

        cudnn_workspace_size = std::max(cudnn_workspace_size, plans.execution_plans[plan_index]->getWorkspaceSize());

        return {error_code_t::OK, ""};
    }

    int64_t
    get_max_cudnn_workspace_size_node() const {
        return plans.get_autotune_workspace();
    }

    error_t
    execute_cudnn_plan_with_uid(cudnnHandle_t handle,
                                std::unordered_map<int64_t, void*> const& tensor_uid_to_pointer_map,
                                void* workspace_ptr,
                                int64_t plan_index) const {
        // Make sure device pointer is provided for all uids expected for this plan
        std::vector<void*> device_ptrs;
        std::vector<uid_t> uids;
        for (auto const& uid : variant_pack_uids) {
            auto search = tensor_uid_to_pointer_map.find(uid);
            RETURN_CUDNN_FRONTEND_ERROR_IF(search == tensor_uid_to_pointer_map.end(),
                                           error_code_t::INVALID_VARIANT_PACK,
                                           "Uid " + std::to_string(uid) + " does not exist in variant pack.");
            device_ptrs.push_back(search->second);
            uids.push_back(uid);
        }

        CHECK_CUDNN_FRONTEND_ERROR(plans.is_plan_index_executable(plan_index));

        getLogger() << "[cudnn_frontend] INFO: Executing plan at index " << plan_index << "." << std::endl;

        CHECK_CUDNN_FRONTEND_ERROR(
            detail::execute(handle, plans.execution_plans[plan_index].get(), device_ptrs, uids, workspace_ptr));

        return {error_code_t::OK, ""};
    }
};

}  // namespace cudnn_frontend
