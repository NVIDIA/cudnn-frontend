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

#include "cudnn_interface.h"

#include "graph_properties.h"

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
            CHECK_CUDNN_FRONTEND_ERROR(sub_node->assign_uids());
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
    pass_by_value_tensors_(cudnnHandle_t,
                           std::unordered_map<std::shared_ptr<Tensor_attributes>, pass_by_values_t>&,
                           void*) {
        return {error_code_t::OK, ""};
    }

    error_t
    gather_pass_by_value_tensors(
        cudnnHandle_t const& handle,
        std::unordered_map<std::shared_ptr<Tensor_attributes>, pass_by_values_t>& tensor_to_pass_by_value,
        void* fe_workspace) {
        void* node_workspace = fe_workspace;
        CHECK_CUDNN_FRONTEND_ERROR(pass_by_value_tensors_(handle, tensor_to_pass_by_value, node_workspace));
        node_workspace = static_cast<char*>(node_workspace) + get_fe_workspace_size_node();
        for (auto const& sub_node : sub_nodes) {
            CHECK_CUDNN_FRONTEND_ERROR(
                sub_node->gather_pass_by_value_tensors(handle, tensor_to_pass_by_value, node_workspace));
            node_workspace = static_cast<char*>(node_workspace) + sub_node->get_fe_workspace_size_node();
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
        DIN,
        DGRAD,
        DRMSNorm,
        GENSTATS,
        LAYERNORM,
        INSTANCENORM,
        MATMUL,
        POINTWISE,
        REDUCTION,
        RESAMPLE,
        RESHAPE,
        RMSNORM,
        RNG,
        SCALED_DOT_PRODUCT_ATTENTION,
        WGRAD
    };
    Type tag;

    virtual error_t
    createTensors() {
        for (auto const& sub_node : sub_nodes) {
            CHECK_CUDNN_FRONTEND_ERROR(sub_node->createTensors());
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
            CHECK_CUDNN_FRONTEND_ERROR(sub_node->createOperations());

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
        CHECK_CUDNN_FRONTEND_ERROR(validate_node());

        // infer_properties self
        CHECK_CUDNN_FRONTEND_ERROR(infer_properties_node());

        // validate sub nodes
        for (auto const& sub_node : sub_nodes) {
            CHECK_CUDNN_FRONTEND_ERROR(sub_node->validate());
        }

        has_validation_checked = true;
        return {error_code_t::OK, ""};
    }

    error_t
    build_operation_graph(cudnnHandle_t handle) {
        CHECK_CUDNN_FRONTEND_ERROR(validate());
        CHECK_CUDNN_FRONTEND_ERROR(assign_uids());
        CHECK_CUDNN_FRONTEND_ERROR(createTensors());
        CHECK_CUDNN_FRONTEND_ERROR(createOperations());
        CHECK_CUDNN_FRONTEND_ERROR(createOperationGraphs(handle));
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

        CHECK_CUDNN_FRONTEND_ERROR(gather_pass_by_value_tensors(handle, tensor_to_pass_by_value, fe_workspace));

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
                RETURN_CUDNN_FRONTEND_ERROR_IF(
                    true, error_code_t::INVALID_VARIANT_PACK, "Unexpected type for pass by value tensor.");
            }
        }

        CHECK_CUDNN_FRONTEND_ERROR(execute_cudnn_plans(handle, tensor_uid_to_pointer_map, cudnn_workspace));

        return {error_code_t::OK, ""};
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

}  // namespace graph

}  // namespace cudnn_frontend