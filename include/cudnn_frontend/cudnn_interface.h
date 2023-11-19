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
    std::vector<cudnn_frontend::Operation> operations;

    std::vector<std::shared_ptr<OperationGraph_v8>> operation_graphs;
    std::vector<std::unordered_set<uid_t>> variant_pack_uids;

    std::vector<graph::Execution_plan_list> plans;

    // TODO: Always returns OK. Can the status and error message be accessed from tensor descriptor?
    error_t
    create_cudnn_tensor(std::shared_ptr<graph::Tensor_attributes> const& props,
                        int64_t& uid,
                        std::unordered_map<int64_t, std::shared_ptr<cudnn_frontend::Tensor>>& tensors) const {
        // Check whether tensor already created
        // TODO: Do not reply on uid being 0?
        if (props->get_uid() == 0) {
            // Make sure no other tensor somehow already has claimed uid.
            RETURN_CUDNN_FRONTEND_ERROR_IF(tensors.find(uid) != tensors.end(),
                                           error_code_t::ATTRIBUTE_NOT_SET,
                                           "Trying to assign same uid to possibily two different tensors.");
            props->set_uid(uid);
            uid++;

            auto&& tensor_builder = cudnn_frontend::TensorBuilder();

            tensor_builder.setDim(props->get_dim().size(), props->get_dim().data())
                .setStrides(props->get_stride().size(), props->get_stride().data())
                .setId(props->get_uid())
                .setAlignment(16)
                .setDataType(props->get_data_type())
                .setVirtual(props->get_is_virtual())
                .setByValue(props->get_is_pass_by_value())
                .setReorderType(props->get_reordering_type());

            if (auto ragged_offset_props = props->get_ragged_offset()) {
                CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(ragged_offset_props, uid, tensors));
                tensor_builder.setRaggedOffset(tensors.at(ragged_offset_props->get_uid()));
            }

            auto tensor = tensor_builder.build();
            tensors.emplace(props->get_uid(), std::make_shared<Tensor>(std::move(tensor)));

        } else {
            // Make sure tensor's uid is present in backend tensor registry.
            RETURN_CUDNN_FRONTEND_ERROR_IF(
                tensors.find(props->get_uid()) == tensors.end(),
                error_code_t::ATTRIBUTE_NOT_SET,
                "Backend tensor already not found for non-zero Id: " + std::to_string(props->get_uid()));

            getLogger() << "[cudnn_frontend] INFO: Backend tensor already created for Id: " +
                               std::to_string(props->get_uid())
                        << std::endl;
        }

        return {error_code_t::OK, ""};
    }

    error_t
    create_cudnn_operation_graphs(cudnnHandle_t handle) {
        std::vector<Operation const*> cudnn_operations;
        for (auto const& operation : operations) {
            cudnn_operations.push_back(&operation);
        }
        auto cudnn_operation_graph = cudnn_frontend::OperationGraphBuilder()
                                         .setHandle(handle)
                                         .setOperationGraph(cudnn_operations.size(), cudnn_operations.data())
                                         .build();

        operation_graphs.push_back(std::make_shared<OperationGraph_v8>(std::move(cudnn_operation_graph)));
        getLogger() << "[cudnn_frontend] INFO: Successfully built Operation Graphs." << std::endl;

        return {error_code_t::OK, ""};
    }

   public:
    int64_t
    get_cudnn_workspace_size_node() const {
        int64_t current_workspace_size = 0;
        for (auto const& execution_plan_list : plans) {
            current_workspace_size =
                std::max(current_workspace_size, execution_plan_list.get_best_candidate()->getWorkspaceSize());
        }
        return current_workspace_size;
    }

    int64_t
    get_max_cudnn_workspace_size_node() const {
        int64_t current_workspace_size = 0;
        for (auto const& execution_plan_list : plans) {
            current_workspace_size = std::max(current_workspace_size, execution_plan_list.get_autotune_workspace());
        }
        return current_workspace_size;
    }

    error_t
    execute_cudnn_plans(cudnnHandle_t handle,
                        std::unordered_map<uid_t, void*> const& tensor_uid_to_pointer_map,
                        void* workspace_ptr) const {
        getLogger() << "[cudnn_frontend] INFO: Executing " << plans.size() << " Plans." << std::endl;

        for (size_t i = 0; i < plans.size(); ++i) {
            auto const& execution_plan = plans[i].get_best_candidate();
            RETURN_CUDNN_FRONTEND_ERROR_IF(
                execution_plan == nullptr, error_code_t::GRAPH_EXECUTION_FAILED, "No plan found to execute!!");
            auto const& variant_pack_uid = variant_pack_uids[i];

            getLogger() << "[cudnn_frontend] INFO: Executing " << execution_plan->getTag() << "..." << std::endl;

            std::vector<void*> device_ptrs;
            std::vector<uid_t> uids;
            for (auto const& uid : variant_pack_uid) {
                auto search = tensor_uid_to_pointer_map.find(uid);
                RETURN_CUDNN_FRONTEND_ERROR_IF(search == tensor_uid_to_pointer_map.end(),
                                               error_code_t::INVALID_VARIANT_PACK,
                                               "Uid " + std::to_string(uid) + " does not exist in variant pack.");
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
