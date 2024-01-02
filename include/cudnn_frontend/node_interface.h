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

class MatmulNode;
class PointwiseNode;
class ReductionNode;
class ReshapeNode;
class RngNode;
class SoftmaxNode;

// Interface for all nodes to follow.
class INode : public ICudnn {
   public:
    // A closed set of types that are allowed to be passed by value today
    using pass_by_values_t = std::variant<int32_t, half, float, void*>;

    detail::Context context;

   private:
    std::shared_ptr<Tensor_attributes>
    output_tensor(std::string const& name) {
        auto tensor = std::make_shared<Tensor_attributes>();
        tensor->set_name(name).set_is_virtual(true);
        return tensor;
    }

    virtual error_t
    pre_validate_node() const = 0;

    virtual error_t
    expand_and_infer_properties() = 0;

    virtual error_t
    post_validate_node() const = 0;

    virtual int64_t
    get_fe_workspace_size_node() const {
        // Mostly no FE nodes have require workspace
        return 0;
    }

    int64_t
    get_cudnn_workspace_size() const {
        int64_t cudnn_workspace_size = get_cudnn_workspace_size_node();
        for (auto const& sub_node : sub_nodes) {
            cudnn_workspace_size = std::max(cudnn_workspace_size, sub_node->get_cudnn_workspace_size());
        }
        return cudnn_workspace_size;
    }

    int64_t
    get_max_cudnn_workspace_size() const {
        int64_t cudnn_workspace_size = get_max_cudnn_workspace_size_node();
        for (auto const& sub_node : sub_nodes) {
            cudnn_workspace_size = std::max(cudnn_workspace_size, sub_node->get_max_cudnn_workspace_size());
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
                           std::unordered_map<std::shared_ptr<Tensor_attributes>, void*> const&,
                           std::unordered_map<std::shared_ptr<Tensor_attributes>, pass_by_values_t>&,
                           void*) const {
        return {error_code_t::OK, ""};
    }

    error_t
    gather_pass_by_value_tensors(
        cudnnHandle_t const& handle,
        std::unordered_map<std::shared_ptr<Tensor_attributes>, void*> const& tensor_to_pointer_map,
        std::unordered_map<std::shared_ptr<Tensor_attributes>, pass_by_values_t>& tensor_to_pass_by_value,
        void* fe_workspace) const {
        void* node_workspace = fe_workspace;
        CHECK_CUDNN_FRONTEND_ERROR(
            pass_by_value_tensors_(handle, tensor_to_pointer_map, tensor_to_pass_by_value, node_workspace));
        node_workspace = static_cast<char*>(node_workspace) + get_fe_workspace_size_node();
        for (auto const& sub_node : sub_nodes) {
            CHECK_CUDNN_FRONTEND_ERROR(sub_node->gather_pass_by_value_tensors(
                handle, tensor_to_pointer_map, tensor_to_pass_by_value, node_workspace));
            node_workspace = static_cast<char*>(node_workspace) + sub_node->get_fe_workspace_size_node();
        }
        return {error_code_t::OK, ""};
    }

    error_t
    extend_tensor_map_with_pass_by_value_tensors_(
        std::unordered_map<int64_t, void*>& tensor_to_pointer_map,
        std::unordered_map<std::shared_ptr<Tensor_attributes>, pass_by_values_t>& tensor_to_pass_by_value) const {
        for (auto& [tensor, value] : tensor_to_pass_by_value) {
            if (half* half_value_ptr = std::get_if<half>(&value)) {
                tensor_to_pointer_map.emplace(tensor->get_uid(), half_value_ptr);
            } else if (int32_t* int32_t_value_ptr = std::get_if<int32_t>(&value)) {
                tensor_to_pointer_map.emplace(tensor->get_uid(), int32_t_value_ptr);
            } else if (float* float_value_ptr = std::get_if<float>(&value)) {
                tensor_to_pointer_map.emplace(tensor->get_uid(), float_value_ptr);
            } else if (void** void_value_ptr = std::get_if<void*>(&value)) {
                tensor_to_pointer_map.emplace(tensor->get_uid(), *void_value_ptr);
            } else {
                RETURN_CUDNN_FRONTEND_ERROR_IF(
                    true, error_code_t::INVALID_VARIANT_PACK, "Unexpected type for pass by value tensor.");
            }
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

    void
    matmul(std::shared_ptr<Tensor_attributes> a,
           std::shared_ptr<Tensor_attributes> b,
           Matmul_attributes attributes,
           std::shared_ptr<Tensor_attributes> c) {
        attributes.inputs[Matmul_attributes::input_names::A]   = a;
        attributes.inputs[Matmul_attributes::input_names::B]   = b;
        attributes.outputs[Matmul_attributes::output_names::C] = c;
        sub_nodes.emplace_back(std::make_unique<MatmulNode>(std::move(attributes), context));
    }

    void
    softmax(std::shared_ptr<Tensor_attributes> p,
            Softmax_attributes attributes,
            std::shared_ptr<Tensor_attributes> s,
            std::shared_ptr<Tensor_attributes> stats) {
        attributes.inputs[Softmax_attributes::input_names::P]       = p;
        attributes.outputs[Softmax_attributes::output_names::S]     = s;
        attributes.outputs[Softmax_attributes::output_names::Stats] = stats;
        sub_nodes.emplace_back(std::make_unique<SoftmaxNode>(std::move(attributes), context));
    }

    void
    softmax(std::shared_ptr<Tensor_attributes> p,
            Softmax_attributes attributes,
            std::shared_ptr<Tensor_attributes> s,
            std::shared_ptr<Tensor_attributes> m,
            std::shared_ptr<Tensor_attributes> zinv) {
        attributes.inputs[Softmax_attributes::input_names::P]      = p;
        attributes.outputs[Softmax_attributes::output_names::S]    = s;
        attributes.outputs[Softmax_attributes::output_names::M]    = m;
        attributes.outputs[Softmax_attributes::output_names::Zinv] = zinv;
        sub_nodes.emplace_back(std::make_unique<SoftmaxNode>(std::move(attributes), context));
    }

    void
    pointwise(std::shared_ptr<Tensor_attributes> a,
              Pointwise_attributes attributes,
              std::shared_ptr<Tensor_attributes> c) {
        attributes.inputs[Pointwise_attributes::input_names::IN_0]    = a;
        attributes.outputs[Pointwise_attributes::output_names::OUT_0] = c;
        sub_nodes.emplace_back(std::make_unique<PointwiseNode>(std::move(attributes), context));
    }

    void
    pointwise(std::shared_ptr<Tensor_attributes> a,
              std::shared_ptr<Tensor_attributes> b,
              Pointwise_attributes attributes,
              std::shared_ptr<Tensor_attributes> c) {
        attributes.inputs[Pointwise_attributes::input_names::IN_0]    = a;
        attributes.inputs[Pointwise_attributes::input_names::IN_1]    = b;
        attributes.outputs[Pointwise_attributes::output_names::OUT_0] = c;
        sub_nodes.emplace_back(std::make_unique<PointwiseNode>(std::move(attributes), context));
    }

    void
    reduction(std::shared_ptr<Tensor_attributes> a,
              Reduction_attributes attributes,
              std::shared_ptr<Tensor_attributes> c) {
        attributes.inputs[Reduction_attributes::input_names::X]   = a;
        attributes.outputs[Reduction_attributes::output_names::Y] = c;
        sub_nodes.emplace_back(std::make_unique<ReductionNode>(std::move(attributes), context));
    }

    void
    rng(std::shared_ptr<Tensor_attributes> seed,
        std::shared_ptr<Tensor_attributes> offset,
        Rng_attributes attributes,
        std::shared_ptr<Tensor_attributes> y) {
        attributes.inputs[Rng_attributes::input_names::Seed]   = seed;
        attributes.inputs[Rng_attributes::input_names::Offset] = offset;
        attributes.outputs[Rng_attributes::output_names::Y]    = y;
        sub_nodes.emplace_back(std::make_unique<RngNode>(std::move(attributes), context));
    }

    // Creates cudnn tensors for each node (and its sub nodes)
    virtual error_t
    create_cudnn_tensors(
        int64_t& uid,
        std::unordered_map<int64_t, std::shared_ptr<cudnn_frontend::Tensor>>& uid_to_backend_tensors) const {
        for (auto const& sub_node : sub_nodes) {
            CHECK_CUDNN_FRONTEND_ERROR(sub_node->create_cudnn_tensors(uid, uid_to_backend_tensors));
        }
        return {error_code_t::OK, ""};
    }

    // Creates cudnn operation for each node (and its sub nodes)
    // Only INode that map to a primitive cudnn operation need to specialize.
    virtual error_t
    create_cudnn_operations(
        std::unordered_set<uid_t>& uids_involved_in_operation,
        std::vector<std::shared_ptr<cudnn_frontend::Operation>>& backend_operations,
        std::unordered_map<int64_t, std::shared_ptr<cudnn_frontend::Tensor>>& uid_to_backend_tensors) const {
        for (auto const& sub_node : sub_nodes) {
            CHECK_CUDNN_FRONTEND_ERROR(sub_node->create_cudnn_operations(
                uids_involved_in_operation, backend_operations, uid_to_backend_tensors));
        }
        return {error_code_t::OK, ""};
    }

    // An implicitly topological-sorted vector of sub nodes.
    // The sorted order is a side effect of functional API.
    std::vector<std::shared_ptr<INode>> sub_nodes;

   public:
    virtual Type
    getType() = 0;

    std::shared_ptr<Tensor_attributes> matmul(std::shared_ptr<Tensor_attributes>,
                                              std::shared_ptr<Tensor_attributes>,
                                              Matmul_attributes);

    std::shared_ptr<Tensor_attributes> pointwise(std::shared_ptr<Tensor_attributes>, Pointwise_attributes);
    std::shared_ptr<Tensor_attributes> pointwise(std::shared_ptr<Tensor_attributes>,
                                                 std::shared_ptr<Tensor_attributes>,
                                                 Pointwise_attributes);
    std::shared_ptr<Tensor_attributes> pointwise(std::shared_ptr<Tensor_attributes>,
                                                 std::shared_ptr<Tensor_attributes>,
                                                 std::shared_ptr<Tensor_attributes>,
                                                 Pointwise_attributes);

    std::shared_ptr<Tensor_attributes> reduction(std::shared_ptr<Tensor_attributes>, Reduction_attributes);
    std::shared_ptr<Tensor_attributes> reshape(std::shared_ptr<Tensor_attributes>, Reshape_attributes);

    std::shared_ptr<Tensor_attributes> rng(std::shared_ptr<Tensor_attributes>,
                                           std::shared_ptr<Tensor_attributes>,
                                           Rng_attributes);
    error_t
    validate() {
        // validate self
        CHECK_CUDNN_FRONTEND_ERROR(pre_validate_node());

        // infer_properties self
        CHECK_CUDNN_FRONTEND_ERROR(expand_and_infer_properties());

        // validate sub nodes
        for (auto const& sub_node : sub_nodes) {
            CHECK_CUDNN_FRONTEND_ERROR(sub_node->validate());
        }

        // validate self
        CHECK_CUDNN_FRONTEND_ERROR(post_validate_node());

        return {error_code_t::OK, ""};
    }

    error_t
    build_operation_graph(cudnnHandle_t handle) {
        // Starting uid for operation graph.
        // Each time a backend tensor is created, uid will be incremented by 1, ensuring uniqueness.
        // TODO: Maybe just use uid_to_tensors size as uid each time?
        int64_t uid = 1;

        // Lower each sub node to cudnn backend.
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensors(uid, uid_to_tensors));

        // INode needs to keep track of all uids that an operation graph uses.
        // This is because cudnn backend will not accept extra tensors in variant pack.
        // But FE users provide 1 large list of tensors.
        // So internally FE assigns subset of the usre-provided tensor list to each operation graph.
        // Also, as uid in a variant pack have to be unique, keep a set of them.
        std::unordered_set<uid_t> uids_involved_in_operation;
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_operations(uids_involved_in_operation, operations, uid_to_tensors));

        // The method here fuses all operations. There will be 1 operation graph in total.
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_operation_graphs(handle));
        variant_pack_uids.push_back(std::move(uids_involved_in_operation));

        return {error_code_t::OK, ""};
    }

    int64_t
    get_workspace_size() const {
        // There are two workspaces:
        // - cudnn execution plan workspace
        // - FE node workspace (example: alibiSlope for fmha)
        return get_fe_workspace_size() + get_cudnn_workspace_size();
    }

    int64_t
    get_autotune_workspace_size() const {
        // There are two workspaces:
        // - cudnn execution plan workspace
        // - FE node workspace (example: alibiSlope for fmha)
        return get_fe_workspace_size() + get_max_cudnn_workspace_size();
    }

    error_t
    execute(cudnnHandle_t handle,
            std::unordered_map<std::shared_ptr<Tensor_attributes>, void*> const& tensor_to_pointer_map,
            void* workspace) const {
        std::unordered_map<int64_t, void*> tensor_uid_to_pointer_map;
        for (auto const& [tensor, pointer] : tensor_to_pointer_map) {
            tensor_uid_to_pointer_map.emplace(tensor->get_uid(), pointer);
        }

        std::unordered_map<std::shared_ptr<Tensor_attributes>, pass_by_values_t> tensor_to_pass_by_value;
        void* fe_workspace    = workspace;
        void* cudnn_workspace = static_cast<char*>(fe_workspace) + get_fe_workspace_size();

        CHECK_CUDNN_FRONTEND_ERROR(
            gather_pass_by_value_tensors(handle, tensor_to_pointer_map, tensor_to_pass_by_value, fe_workspace));

        // Add pass_by_value data pointers to tensor_uid_to_pointer map
        // object lifetime is controlled by tensor_to_pass_by_value which means the pointer should stay valid during
        // execute
        for (auto& [tensor, value] : tensor_to_pass_by_value) {
            if (half* half_value_ptr = std::get_if<half>(&value)) {
                tensor_uid_to_pointer_map.emplace(tensor->get_uid(), half_value_ptr);
            } else if (int32_t* int32_t_value_ptr = std::get_if<int32_t>(&value)) {
                tensor_uid_to_pointer_map.emplace(tensor->get_uid(), int32_t_value_ptr);
            } else if (float* float_value_ptr = std::get_if<float>(&value)) {
                tensor_uid_to_pointer_map.emplace(tensor->get_uid(), float_value_ptr);
            } else if (void** void_value_ptr = std::get_if<void*>(&value)) {
                tensor_uid_to_pointer_map.emplace(tensor->get_uid(), *void_value_ptr);
            } else {
                RETURN_CUDNN_FRONTEND_ERROR_IF(
                    true, error_code_t::INVALID_VARIANT_PACK, "Execute unexpected type for pass by value tensor.");
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

    size_t
    key() {
        json j;
        serialize(j);
        return std::hash<json>{}(j);
    }

    virtual ~INode() = default;
};

[[maybe_unused]] static void
to_json(json& j, const INode& p) {
    p.serialize(j);
}

#define CUDNN_FE_VALIDATE_TENSOR_(port, map_)                                                      \
    {                                                                                              \
        auto t           = map_.find(port);                                                        \
        bool const has_t = (t != map_.end()) && (t->second != nullptr);                            \
        RETURN_CUDNN_FRONTEND_ERROR_IF(                                                            \
            !has_t, error_code_t::ATTRIBUTE_NOT_SET, std::string("Tensor ") + #port + " not set"); \
    }

#define CUDNN_FE_VALIDATE_AND_ASSIGN_TENSOR_(tensor, port, map_)                                   \
    auto tensor = map_.find(port);                                                                 \
    {                                                                                              \
        bool const has_t = (tensor != map_.end()) && (tensor->second != nullptr);                  \
        RETURN_CUDNN_FRONTEND_ERROR_IF(                                                            \
            !has_t, error_code_t::ATTRIBUTE_NOT_SET, std::string("Tensor ") + #port + " not set"); \
    }

#define CUDNN_FE_VALIDATE_INPUT_TENSOR(port) CUDNN_FE_VALIDATE_TENSOR_(port, attributes.inputs)

#define CUDNN_FE_VALIDATE_OUTPUT_TENSOR(port) CUDNN_FE_VALIDATE_TENSOR_(port, attributes.outputs)

#define CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(tensor, port) \
    CUDNN_FE_VALIDATE_AND_ASSIGN_TENSOR_(tensor, port, attributes.inputs)

#define CUDNN_FE_VALIDATE_AND_ASSIGN_OUTPUT_TENSOR(tensor, port) \
    CUDNN_FE_VALIDATE_AND_ASSIGN_TENSOR_(tensor, port, attributes.outputs)

inline std::shared_ptr<Tensor_attributes>
INode::matmul(std::shared_ptr<Tensor_attributes> a,
              std::shared_ptr<Tensor_attributes> b,
              Matmul_attributes attributes) {
    attributes.inputs[Matmul_attributes::input_names::A] = a;
    attributes.inputs[Matmul_attributes::input_names::B] = b;
    auto C = attributes.outputs[Matmul_attributes::output_names::C] = output_tensor(attributes.name + "::C");

    sub_nodes.emplace_back(std::make_unique<MatmulNode>(std::move(attributes), context));
    return C;
}

inline std::shared_ptr<Tensor_attributes>
INode::pointwise(std::shared_ptr<Tensor_attributes> a, Pointwise_attributes attributes) {
    attributes.inputs[Pointwise_attributes::input_names::IN_0] = a;
    auto OUT_0 = attributes.outputs[Pointwise_attributes::output_names::OUT_0] =
        output_tensor(attributes.name + "::OUT_0");

    sub_nodes.emplace_back(std::make_unique<PointwiseNode>(std::move(attributes), context));
    return OUT_0;
}

inline std::shared_ptr<Tensor_attributes>
INode::pointwise(std::shared_ptr<Tensor_attributes> a,
                 std::shared_ptr<Tensor_attributes> b,
                 Pointwise_attributes attributes) {
    attributes.inputs[Pointwise_attributes::input_names::IN_0] = a;
    attributes.inputs[Pointwise_attributes::input_names::IN_1] = b;
    auto OUT_0 = attributes.outputs[Pointwise_attributes::output_names::OUT_0] =
        output_tensor(attributes.name + "::OUT_0");

    sub_nodes.emplace_back(std::make_unique<PointwiseNode>(std::move(attributes), context));
    return OUT_0;
}

inline std::shared_ptr<Tensor_attributes>
INode::pointwise(std::shared_ptr<Tensor_attributes> a,
                 std::shared_ptr<Tensor_attributes> b,
                 std::shared_ptr<Tensor_attributes> c,
                 Pointwise_attributes attributes) {
    attributes.inputs[Pointwise_attributes::input_names::IN_0] = a;
    attributes.inputs[Pointwise_attributes::input_names::IN_1] = b;
    attributes.inputs[Pointwise_attributes::input_names::IN_2] = c;
    auto OUT_0 = attributes.outputs[Pointwise_attributes::output_names::OUT_0] =
        output_tensor(attributes.name + "::OUT_0");

    sub_nodes.emplace_back(std::make_unique<PointwiseNode>(std::move(attributes), context));
    return OUT_0;
}

inline std::shared_ptr<Tensor_attributes>
INode::reduction(std::shared_ptr<Tensor_attributes> input, Reduction_attributes attributes) {
    attributes.inputs[Reduction_attributes::input_names::X] = input;
    auto Y = attributes.outputs[Reduction_attributes::output_names::Y] = output_tensor(attributes.name + "::Y");

    sub_nodes.emplace_back(std::make_unique<ReductionNode>(std::move(attributes), context));
    return Y;
}

inline std::shared_ptr<Tensor_attributes>
INode::reshape(std::shared_ptr<Tensor_attributes> input, Reshape_attributes attributes) {
    attributes.inputs[Reshape_attributes::input_names::X] = input;
    auto Y = attributes.outputs[Reshape_attributes::output_names::Y] = output_tensor(attributes.name + "::Y");

    sub_nodes.emplace_back(std::make_unique<ReshapeNode>(std::move(attributes), context));
    return Y;
}

inline std::shared_ptr<Tensor_attributes>
INode::rng(std::shared_ptr<Tensor_attributes> seed,
           std::shared_ptr<Tensor_attributes> offset,
           Rng_attributes attributes) {
    attributes.inputs[Rng_attributes::input_names::Seed]   = seed;
    attributes.inputs[Rng_attributes::input_names::Offset] = offset;
    auto Y = attributes.outputs[Rng_attributes::output_names::Y] = output_tensor(attributes.name + "::Y");

    sub_nodes.emplace_back(std::make_unique<RngNode>(std::move(attributes), context));
    return Y;
}

}  // namespace graph

}  // namespace cudnn_frontend