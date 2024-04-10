#pragma once

#include <unordered_map>

#include "node/batchnorm.h"
#include "node/batchnorm_inference.h"
#include "node/bn_finalize.h"
#include "node/conv_fprop.h"
#include "node/conv_dgrad.h"
#include "node/conv_wgrad.h"
#include "node/dbn.h"
#include "node/dln.h"
#include "node/dbn_weight.h"
#include "node/genstats.h"
#include "node/layernorm.h"
#include "node/instancenorm.h"
#include "node/rmsnorm.h"
#include "node/resample.h"
#include "node/reshape.h"
// #include "node/scaled_dot_product_attention.h"
#include "node/scaled_dot_product_flash_attention.h"
#include "node/sdpa_fp8.h"
#include "node/sdpa_fp8_bwd.h"

#include "plans.h"
#include "graph_helpers.h"

namespace cudnn_frontend::graph {

class Graph : public INode {
   private:
    std::unordered_set<std::shared_ptr<Tensor_attributes>> tensors;

    void
    add_to_tensor_map(std::shared_ptr<Tensor_attributes> tensor) {
        tensors.emplace(tensor);
    }

    std::shared_ptr<Tensor_attributes>
    output_tensor(std::string const &name) {
        auto tensor = std::make_shared<Tensor_attributes>();
        tensor->set_name(name).set_is_virtual(true);
        add_to_tensor_map(tensor);
        return tensor;
    }

    // This API is still work in progress and unverified.
    // std::array<std::shared_ptr<Tensor_attributes>, 2> scaled_dot_product_attention(
    //     std::shared_ptr<Tensor_attributes>,
    //     std::shared_ptr<Tensor_attributes>,
    //     std::shared_ptr<Tensor_attributes>,
    //     Scaled_dot_product_attention_attributes);

    error_t
    pre_validate_node() const override final {
        return {error_code_t::OK, ""};
    }

    error_t
    expand_and_infer_properties_node() override final {
        return {error_code_t::OK, ""};
    }

    error_t
    post_validate_node() const override final {
        return {error_code_t::OK, ""};
    }

    virtual error_t
    pass_by_value_tensors_(std::unordered_map<uid_t, pass_by_values_t> &pass_by_values) const override final {
        for (auto [uid, value] : deserialized_pass_by_value) {
            pass_by_values.emplace(uid, value);
        }
        return {error_code_t::OK, ""};
    }

    virtual error_t
    collect_pre_assigned_uids_([[maybe_unused]] std::unordered_set<int64_t> &pre_assigned_uids) const override final {
        return {error_code_t::OK, ""};
    }

    virtual error_t
    create_cudnn_tensors_([[maybe_unused]] std::unordered_map<int64_t, std::shared_ptr<cudnn_frontend::Tensor>>
                              &tensors) const override final {
        return {error_code_t::OK, ""};
    }

    virtual error_t
    set_uids_([[maybe_unused]] int64_t &potential_uid,
              [[maybe_unused]] std::unordered_set<int64_t> const &pre_assigned_uids) const override final {
        return {error_code_t::OK, ""};
    }

   public:
    Graph() : INode(detail::Context{}) {}

    Type
    getType() override {
        return Type::COMPOSITE;
    }

    Graph &
    set_intermediate_data_type(DataType_t type);
    Graph &
    set_io_data_type(DataType_t type);
    Graph &
    set_compute_data_type(DataType_t type);

    std::shared_ptr<Tensor_attributes>
    tensor(Tensor_attributes const &tensor);

    std::shared_ptr<Tensor_attributes>
    tensor_like(std::shared_ptr<Tensor_attributes> const &tensor, std::string const &name = std::string{});

    std::array<std::shared_ptr<Tensor_attributes>, 3> layernorm(std::shared_ptr<Tensor_attributes>,
                                                                std::shared_ptr<Tensor_attributes>,
                                                                std::shared_ptr<Tensor_attributes>,
                                                                Layernorm_attributes);

    std::array<std::shared_ptr<Tensor_attributes>, 3> instancenorm(std::shared_ptr<Tensor_attributes>,
                                                                   std::shared_ptr<Tensor_attributes>,
                                                                   std::shared_ptr<Tensor_attributes>,
                                                                   Instancenorm_attributes);

    std::array<std::shared_ptr<Tensor_attributes>, 5> batchnorm(std::shared_ptr<Tensor_attributes>,
                                                                std::shared_ptr<Tensor_attributes>,
                                                                std::shared_ptr<Tensor_attributes>,
                                                                Batchnorm_attributes);

    std::shared_ptr<Tensor_attributes> batchnorm_inference(std::shared_ptr<Tensor_attributes>,
                                                           std::shared_ptr<Tensor_attributes>,
                                                           std::shared_ptr<Tensor_attributes>,
                                                           std::shared_ptr<Tensor_attributes>,
                                                           std::shared_ptr<Tensor_attributes>,
                                                           Batchnorm_inference_attributes);

    std::array<std::shared_ptr<Tensor_attributes>, 6> bn_finalize(std::shared_ptr<Tensor_attributes>,
                                                                  std::shared_ptr<Tensor_attributes>,
                                                                  std::shared_ptr<Tensor_attributes>,
                                                                  std::shared_ptr<Tensor_attributes>,
                                                                  std::shared_ptr<Tensor_attributes>,
                                                                  std::shared_ptr<Tensor_attributes>,
                                                                  BN_finalize_attributes);

    std::shared_ptr<Tensor_attributes> conv_fprop(std::shared_ptr<Tensor_attributes>,
                                                  std::shared_ptr<Tensor_attributes>,
                                                  Conv_fprop_attributes);

    std::shared_ptr<Tensor_attributes> conv_dgrad(std::shared_ptr<Tensor_attributes>,
                                                  std::shared_ptr<Tensor_attributes>,
                                                  Conv_dgrad_attributes);

    std::shared_ptr<Tensor_attributes> conv_wgrad(std::shared_ptr<Tensor_attributes>,
                                                  std::shared_ptr<Tensor_attributes>,
                                                  Conv_wgrad_attributes);

    std::array<std::shared_ptr<Tensor_attributes>, 5> dbn_weight(std::shared_ptr<Tensor_attributes>,
                                                                 std::shared_ptr<Tensor_attributes>,
                                                                 std::shared_ptr<Tensor_attributes>,
                                                                 std::shared_ptr<Tensor_attributes>,
                                                                 std::shared_ptr<Tensor_attributes>,
                                                                 DBN_weight_attributes);

    std::array<std::shared_ptr<Tensor_attributes>, 3> batchnorm_backward(std::shared_ptr<Tensor_attributes>,
                                                                         std::shared_ptr<Tensor_attributes>,
                                                                         std::shared_ptr<Tensor_attributes>,
                                                                         Batchnorm_backward_attributes);

    std::array<std::shared_ptr<Tensor_attributes>, 3> layernorm_backward(std::shared_ptr<Tensor_attributes>,
                                                                         std::shared_ptr<Tensor_attributes>,
                                                                         std::shared_ptr<Tensor_attributes>,
                                                                         Layernorm_backward_attributes);

    std::array<std::shared_ptr<Tensor_attributes>, 3> instancenorm_backward(std::shared_ptr<Tensor_attributes>,
                                                                            std::shared_ptr<Tensor_attributes>,
                                                                            std::shared_ptr<Tensor_attributes>,
                                                                            Instancenorm_backward_attributes);
    std::array<std::shared_ptr<Tensor_attributes>, 2> genstats(std::shared_ptr<Tensor_attributes>, Genstats_attributes);

    std::array<std::shared_ptr<Tensor_attributes>, 2> rmsnorm(std::shared_ptr<Tensor_attributes>,
                                                              std::shared_ptr<Tensor_attributes>,
                                                              Rmsnorm_attributes);

    std::array<std::shared_ptr<Tensor_attributes>, 3> rmsnorm_backward(std::shared_ptr<Tensor_attributes>,
                                                                       std::shared_ptr<Tensor_attributes>,
                                                                       std::shared_ptr<Tensor_attributes>,
                                                                       std::shared_ptr<Tensor_attributes>,
                                                                       Rmsnorm_backward_attributes);

    std::array<std::shared_ptr<Tensor_attributes>, 2> sdpa(std::shared_ptr<Tensor_attributes>,
                                                           std::shared_ptr<Tensor_attributes>,
                                                           std::shared_ptr<Tensor_attributes>,
                                                           SDPA_attributes);

    std::array<std::shared_ptr<Tensor_attributes>, 4> sdpa_fp8(std::shared_ptr<Tensor_attributes>,
                                                               std::shared_ptr<Tensor_attributes>,
                                                               std::shared_ptr<Tensor_attributes>,
                                                               std::shared_ptr<Tensor_attributes>,
                                                               std::shared_ptr<Tensor_attributes>,
                                                               std::shared_ptr<Tensor_attributes>,
                                                               std::shared_ptr<Tensor_attributes>,
                                                               std::shared_ptr<Tensor_attributes>,
                                                               std::shared_ptr<Tensor_attributes>,
                                                               SDPA_fp8_attributes);

    inline std::array<std::shared_ptr<Tensor_attributes>, 7> sdpa_fp8_backward(std::shared_ptr<Tensor_attributes>,
                                                                               std::shared_ptr<Tensor_attributes>,
                                                                               std::shared_ptr<Tensor_attributes>,
                                                                               std::shared_ptr<Tensor_attributes>,
                                                                               std::shared_ptr<Tensor_attributes>,
                                                                               std::shared_ptr<Tensor_attributes>,
                                                                               std::shared_ptr<Tensor_attributes>,
                                                                               std::shared_ptr<Tensor_attributes>,
                                                                               std::shared_ptr<Tensor_attributes>,
                                                                               std::shared_ptr<Tensor_attributes>,
                                                                               std::shared_ptr<Tensor_attributes>,
                                                                               std::shared_ptr<Tensor_attributes>,
                                                                               std::shared_ptr<Tensor_attributes>,
                                                                               std::shared_ptr<Tensor_attributes>,
                                                                               std::shared_ptr<Tensor_attributes>,
                                                                               std::shared_ptr<Tensor_attributes>,
                                                                               std::shared_ptr<Tensor_attributes>,
                                                                               std::shared_ptr<Tensor_attributes>,
                                                                               SDPA_fp8_backward_attributes);

    std::array<std::shared_ptr<Tensor_attributes>, 3> sdpa_backward(std::shared_ptr<Tensor_attributes>,
                                                                    std::shared_ptr<Tensor_attributes>,
                                                                    std::shared_ptr<Tensor_attributes>,
                                                                    std::shared_ptr<Tensor_attributes>,
                                                                    std::shared_ptr<Tensor_attributes>,
                                                                    std::shared_ptr<Tensor_attributes>,
                                                                    SDPA_backward_attributes);

    [[deprecated]] std::array<std::shared_ptr<Tensor_attributes>, 2>
    scaled_dot_product_flash_attention(std::shared_ptr<Tensor_attributes> q,
                                       std::shared_ptr<Tensor_attributes> k,
                                       std::shared_ptr<Tensor_attributes> v,
                                       SDPA_attributes attributes) {
        return sdpa(q, k, v, attributes);
    }
    [[deprecated]] std::array<std::shared_ptr<Tensor_attributes>, 3>
    scaled_dot_product_flash_attention_backward(std::shared_ptr<Tensor_attributes> q,
                                                std::shared_ptr<Tensor_attributes> k,
                                                std::shared_ptr<Tensor_attributes> v,
                                                std::shared_ptr<Tensor_attributes> o,
                                                std::shared_ptr<Tensor_attributes> dO,
                                                std::shared_ptr<Tensor_attributes> stats,
                                                SDPA_backward_attributes attributes) {
        return sdpa_backward(q, k, v, o, dO, stats, attributes);
    }

    error_t
    create_execution_plans(std::vector<HeurMode_t> const &mode);

    int64_t
    get_execution_plan_count() const;

    error_t
    check_support(cudnnHandle_t h) {
        for (auto &plan_list : plans) {
            CHECK_CUDNN_FRONTEND_ERROR(plan_list.check_support(h));
        }
        return {error_code_t::OK, ""};
    }

    error_t
    build_plans(cudnnHandle_t const &handle,
                BuildPlanPolicy_t const policy     = BuildPlanPolicy_t::HEURISTICS_CHOICE,
                bool const do_multithreaded_builds = false);

    error_t
    build_plan_at_index(cudnnHandle_t const &handle, int64_t index);

    Graph &
    deselect_workspace_greater_than(int64_t const workspace) {
        for (auto &plan_list : plans) {
            plan_list.set_max_workspace_allowed(workspace);
        }
        return *this;
    }

    Graph &
    deselect_engines(std::vector<std::string> const &engine_names) {
        for (auto &plan_list : plans) {
            plan_list.set_barred_names(engine_names);
        }
        return *this;
    }

    Graph &
    select_behavior_notes(std::vector<BehaviorNote_t> const &notes) {
        for (auto &plan_list : plans) {
            auto status = plan_list.filter_behavior_notes(notes, true);
            if (status.is_bad()) {
                getLogger() << status.get_message() << std::endl;
            }
        }
        return *this;
    }

    Graph &
    select_numeric_notes(std::vector<NumericalNote_t> const &notes) {
        for (auto &plan_list : plans) {
            auto status = plan_list.filter_numeric_notes(notes, true);
            if (status.is_bad()) {
                getLogger() << status.get_message() << std::endl;
            }
        }
        return *this;
    }

    Graph &
    deselect_behavior_notes(std::vector<BehaviorNote_t> const &notes) {
        for (auto &plan_list : plans) {
            auto status = plan_list.filter_behavior_notes(notes, false);
            if (status.is_bad()) {
                getLogger() << status.get_message() << std::endl;
            }
        }
        return *this;
    }

    Graph &
    deselect_numeric_notes(std::vector<NumericalNote_t> const &notes) {
        for (auto &plan_list : plans) {
            auto status = plan_list.filter_numeric_notes(notes, false);
            if (status.is_bad()) {
                getLogger() << status.get_message() << std::endl;
            }
        }
        return *this;
    }

    using INode::deserialize;
    using INode::serialize;

    virtual void
    serialize(json &j) const override final {
        // Different from serialization of other INodes.
        // Go over each subnode and serialize them.
        j["nodes"];
        for (auto const &sub_node : sub_nodes) {
            json j_sub_node;
            sub_node->serialize(j_sub_node);
            j["nodes"].push_back(j_sub_node);
        }
    };

    // TODO: temparorily placed in graphs class. This function needs to be a free standing function.
    error_t
    deserialize(const json &j) {
        if (j.contains("nodes") && j["nodes"].is_array()) {
            for (const auto &j_sub_node : j["nodes"]) {
                if (j_sub_node.contains("tag") && j_sub_node["tag"].is_string()) {
                    auto tag = j_sub_node["tag"].get<std::string>();
                    if (tag == "CONV_FPROP") {
                        auto conv_fprop_attributes = j_sub_node.get<Conv_fprop_attributes>();
                        sub_nodes.emplace_back(
                            std::make_unique<ConvolutionNode>(std::move(conv_fprop_attributes), detail::Context()));
                    } else if (tag == "POINTWISE") {
                        auto pointwise_attributes = j_sub_node.get<Pointwise_attributes>();
                        sub_nodes.emplace_back(
                            std::make_unique<PointwiseNode>(std::move(pointwise_attributes), detail::Context()));
                    } else if (tag == "REDUCTION") {
                        auto reduction_attributes = j_sub_node.get<Reduction_attributes>();
                        sub_nodes.emplace_back(
                            std::make_unique<ReductionNode>(std::move(reduction_attributes), detail::Context()));
                    } else if (tag == "SDPA_FWD") {
                        auto sdpa_attributes = j_sub_node.get<SDPA_attributes>();
                        sub_nodes.emplace_back(
                            std::make_unique<SDPANode>(std::move(sdpa_attributes), detail::Context()));
                    } else if (tag == "SDPA_BWD") {
                        auto sdpa_bwd_attributes = j_sub_node.get<SDPA_backward_attributes>();
                        sub_nodes.emplace_back(
                            std::make_unique<SDPABackwardNode>(std::move(sdpa_bwd_attributes), detail::Context()));
                    }
                }
            }
        }

        return {error_code_t::OK, ""};
    }

    std::string
    print(void) const {
        std::stringstream ss;
        json j = *this;
        ss << j.dump(4);
        return ss.str();
    }
};

inline int64_t
Graph::get_execution_plan_count() const {
    int64_t plan_count = 0;
    for (auto &plan_list : plans) {
        plan_count += plan_list.execution_plans.size();
    }
    return plan_count;
}

inline error_t
Graph::create_execution_plans(std::vector<HeurMode_t> const &mode) {
    std::unordered_map<std::string, EngineConfigList> op_graph_to_configs;
    CHECK_CUDNN_FRONTEND_ERROR(detail::query_heuristics(operation_graphs, op_graph_to_configs, mode));

    getLogger() << "[cudnn_frontend] INFO: Extracting engine configs." << std::endl;

    for (auto const &op : op_graph_to_configs) {
        Execution_plan_list plan_list;

        plan_list.set_tag(op.first);
        plan_list.set_engine_configs(op.second);

        getLogger() << "[cudnn_frontend] INFO: Querying engine config properties\n";
        CHECK_CUDNN_FRONTEND_ERROR(plan_list.query_properties());

        plans.emplace_back(std::move(plan_list));
    }

    return {error_code_t::OK, ""};
}

inline error_t
Graph::build_plan_at_index(cudnnHandle_t const &handle, int64_t plan_index) {
    for (auto i = 0u; i < plans.size(); i++) {
        CHECK_CUDNN_FRONTEND_ERROR(plans[i].build_plan_at_index(handle, plan_index));
    }
    return {error_code_t::OK, ""};
}

inline error_t
Graph::build_plans(cudnnHandle_t const &handle, BuildPlanPolicy_t const policy, bool const do_multithreaded_builds) {
    for (auto &plan_list : plans) {
        CHECK_CUDNN_FRONTEND_ERROR(plan_list.build_plans(handle, policy, do_multithreaded_builds));
    }
    return {error_code_t::OK, ""};
}

inline Graph &
Graph::set_intermediate_data_type(DataType_t const type) {
    context.set_intermediate_data_type(type);
    return *this;
}

inline Graph &
Graph::set_io_data_type(DataType_t const type) {
    context.set_io_data_type(type);
    return *this;
}

inline Graph &
Graph::set_compute_data_type(DataType_t const type) {
    context.set_compute_data_type(type);
    return *this;
}

inline std::shared_ptr<Tensor_attributes>
Graph::tensor(Tensor_attributes const &tensor) {
    auto tensor_ptr = std::make_shared<Tensor_attributes>(tensor);
    add_to_tensor_map(tensor_ptr);
    return tensor_ptr;
}

// tensor_like is meant to create "useable" copies of a tensor.
// By usable, it means not copying over the uids, as uids are FE-level(internal) detail.
// It also means not copying over names, which are user-level(external) detail. But user is given option to provide a
// new name.
inline std::shared_ptr<Tensor_attributes>
Graph::tensor_like(std::shared_ptr<Tensor_attributes> const &tensor, std::string const &name) {
    auto tensor_ptr = std::make_shared<Tensor_attributes>(*tensor);

    // reset the uid of the cloned tensor
    // uids are not meant to be copied by tensor_like
    // When lowering to cudnn backend, both tensors involved here will get unique uids.
    tensor_ptr->clear_uid();

    // reset the name too. Defaults to empty string.
    tensor_ptr->set_name(name);

    return tensor_ptr;
}

inline std::array<std::shared_ptr<Tensor_attributes>, 6>
Graph::bn_finalize(std::shared_ptr<Tensor_attributes> sum,
                   std::shared_ptr<Tensor_attributes> sq_sum,
                   std::shared_ptr<Tensor_attributes> scale,
                   std::shared_ptr<Tensor_attributes> bias,
                   std::shared_ptr<Tensor_attributes> epsilon,
                   std::shared_ptr<Tensor_attributes> accum_count,
                   BN_finalize_attributes attributes) {
    // Set outputs
    auto EQ_SCALE = attributes.outputs[BN_finalize_attributes::output_names::EQ_SCALE] =
        output_tensor(attributes.name + "::EQ_SCALE");
    auto EQ_BIAS = attributes.outputs[BN_finalize_attributes::output_names::EQ_BIAS] =
        output_tensor(attributes.name + "::EQ_BIAS");
    auto MEAN = attributes.outputs[BN_finalize_attributes::output_names::MEAN] =
        output_tensor(attributes.name + "::MEAN");
    auto INV_VARIANCE = attributes.outputs[BN_finalize_attributes::output_names::INV_VARIANCE] =
        output_tensor(attributes.name + "::INV_VARIANCE");
    std::shared_ptr<Tensor_attributes> NEXT_RUNNING_MEAN = nullptr;
    std::shared_ptr<Tensor_attributes> NEXT_RUNNING_VAR  = nullptr;
    if (attributes.inputs[BN_finalize_attributes::input_names::PREV_RUNNING_MEAN] &&
        attributes.inputs[BN_finalize_attributes::input_names::PREV_RUNNING_VAR] &&
        attributes.inputs[BN_finalize_attributes::input_names::MOMENTUM]) {
        NEXT_RUNNING_MEAN = output_tensor(attributes.name + "::NEXT_RUNNING_MEAN");
        NEXT_RUNNING_VAR  = output_tensor(attributes.name + "::NEXT_RUNNING_VAR");
    }
    attributes.outputs[BN_finalize_attributes::output_names::NEXT_RUNNING_MEAN] = NEXT_RUNNING_MEAN;
    attributes.outputs[BN_finalize_attributes::output_names::NEXT_RUNNING_VAR]  = NEXT_RUNNING_VAR;

    // Set inputs
    attributes.inputs[BN_finalize_attributes::input_names::SUM]         = sum;
    attributes.inputs[BN_finalize_attributes::input_names::SQ_SUM]      = sq_sum;
    attributes.inputs[BN_finalize_attributes::input_names::SCALE]       = scale;
    attributes.inputs[BN_finalize_attributes::input_names::BIAS]        = bias;
    attributes.inputs[BN_finalize_attributes::input_names::EPSILON]     = epsilon;
    attributes.inputs[BN_finalize_attributes::input_names::ACCUM_COUNT] = accum_count;

    sub_nodes.emplace_back(std::make_unique<BatchNormFinalizeNode>(std::move(attributes), context));

    return {EQ_SCALE, EQ_BIAS, MEAN, INV_VARIANCE, NEXT_RUNNING_MEAN, NEXT_RUNNING_VAR};
}

inline std::array<std::shared_ptr<Tensor_attributes>, 3>
Graph::layernorm(std::shared_ptr<Tensor_attributes> x,
                 std::shared_ptr<Tensor_attributes> scale,
                 std::shared_ptr<Tensor_attributes> bias,
                 Layernorm_attributes attributes) {
    // Set outputs
    auto Y = attributes.outputs[Layernorm_attributes::output_names::Y] = output_tensor(attributes.name + "::Y");
    std::shared_ptr<Tensor_attributes> MEAN                            = nullptr;
    std::shared_ptr<Tensor_attributes> INV_VARIANCE                    = nullptr;
    if (attributes.forward_phase == NormFwdPhase_t::TRAINING) {
        MEAN = attributes.outputs[Layernorm_attributes::output_names::MEAN] = output_tensor(attributes.name + "::MEAN");
        INV_VARIANCE = attributes.outputs[Layernorm_attributes::output_names::INV_VARIANCE] =
            output_tensor(attributes.name + "::INV_VARIANCE");
    }
    // Set inputs
    attributes.inputs[Layernorm_attributes::input_names::X]     = x;
    attributes.inputs[Layernorm_attributes::input_names::SCALE] = scale;
    attributes.inputs[Layernorm_attributes::input_names::BIAS]  = bias;

    sub_nodes.emplace_back(std::make_unique<LayerNormNode>(std::move(attributes), context));

    return {Y, MEAN, INV_VARIANCE};
}

inline std::array<std::shared_ptr<Tensor_attributes>, 3>
Graph::instancenorm(std::shared_ptr<Tensor_attributes> x,
                    std::shared_ptr<Tensor_attributes> scale,
                    std::shared_ptr<Tensor_attributes> bias,
                    Instancenorm_attributes attributes) {
    // Set outputs
    auto Y = attributes.outputs[Instancenorm_attributes::output_names::Y] = output_tensor(attributes.name + "::Y");
    std::shared_ptr<Tensor_attributes> MEAN                               = nullptr;
    std::shared_ptr<Tensor_attributes> INV_VARIANCE                       = nullptr;
    if (attributes.forward_phase == NormFwdPhase_t::TRAINING) {
        MEAN = attributes.outputs[Instancenorm_attributes::output_names::MEAN] =
            output_tensor(attributes.name + "::MEAN");
        INV_VARIANCE = attributes.outputs[Instancenorm_attributes::output_names::INV_VARIANCE] =
            output_tensor(attributes.name + "::INV_VARIANCE");
    }
    // Set inputs
    attributes.inputs[Instancenorm_attributes::input_names::X]     = x;
    attributes.inputs[Instancenorm_attributes::input_names::SCALE] = scale;
    attributes.inputs[Instancenorm_attributes::input_names::BIAS]  = bias;

    sub_nodes.emplace_back(std::make_unique<InstanceNormNode>(std::move(attributes), context));

    return {Y, MEAN, INV_VARIANCE};
}

inline std::array<std::shared_ptr<Tensor_attributes>, 5>
Graph::batchnorm(std::shared_ptr<Tensor_attributes> x,
                 std::shared_ptr<Tensor_attributes> scale,
                 std::shared_ptr<Tensor_attributes> bias,
                 Batchnorm_attributes attributes) {
    // Set outputs
    auto Y = attributes.outputs[Batchnorm_attributes::output_names::Y] = output_tensor(attributes.name + "::Y");
    auto MEAN = attributes.outputs[Batchnorm_attributes::output_names::MEAN] =
        output_tensor(attributes.name + "::MEAN");
    auto INV_VARIANCE = attributes.outputs[Batchnorm_attributes::output_names::INV_VARIANCE] =
        output_tensor(attributes.name + "::INV_VARIANCE");
    std::shared_ptr<Tensor_attributes> NEXT_RUNNING_MEAN = nullptr;
    std::shared_ptr<Tensor_attributes> NEXT_RUNNING_VAR  = nullptr;
    if (attributes.inputs[Batchnorm_attributes::input_names::PREV_RUNNING_MEAN] &&
        attributes.inputs[Batchnorm_attributes::input_names::PREV_RUNNING_VAR] &&
        attributes.inputs[Batchnorm_attributes::input_names::MOMENTUM]) {
        NEXT_RUNNING_MEAN = output_tensor(attributes.name + "::NEXT_RUNNING_MEAN");
        NEXT_RUNNING_VAR  = output_tensor(attributes.name + "::NEXT_RUNNING_VAR");
    }
    attributes.outputs[Batchnorm_attributes::output_names::NEXT_RUNNING_MEAN] = NEXT_RUNNING_MEAN;
    attributes.outputs[Batchnorm_attributes::output_names::NEXT_RUNNING_VAR]  = NEXT_RUNNING_VAR;

    // Set inputs
    attributes.inputs[Batchnorm_attributes::input_names::X]     = x;
    attributes.inputs[Batchnorm_attributes::input_names::SCALE] = scale;
    attributes.inputs[Batchnorm_attributes::input_names::BIAS]  = bias;

    sub_nodes.emplace_back(std::make_unique<BatchNormNode>(std::move(attributes), context));

    return {Y, MEAN, INV_VARIANCE, NEXT_RUNNING_MEAN, NEXT_RUNNING_VAR};
}

inline std::shared_ptr<Tensor_attributes>
Graph::batchnorm_inference(std::shared_ptr<Tensor_attributes> x,
                           std::shared_ptr<Tensor_attributes> mean,
                           std::shared_ptr<Tensor_attributes> inv_variance,
                           std::shared_ptr<Tensor_attributes> scale,
                           std::shared_ptr<Tensor_attributes> bias,
                           Batchnorm_inference_attributes attributes) {
    // Set outputs
    auto Y = attributes.outputs[Batchnorm_inference_attributes::output_names::Y] =
        output_tensor(attributes.name + "::Y");

    // Set inputs
    attributes.inputs[Batchnorm_inference_attributes::input_names::X]            = x;
    attributes.inputs[Batchnorm_inference_attributes::input_names::MEAN]         = mean;
    attributes.inputs[Batchnorm_inference_attributes::input_names::INV_VARIANCE] = inv_variance;
    attributes.inputs[Batchnorm_inference_attributes::input_names::SCALE]        = scale;
    attributes.inputs[Batchnorm_inference_attributes::input_names::BIAS]         = bias;

    sub_nodes.emplace_back(std::make_unique<BatchnormInferenceNode>(std::move(attributes), context));

    return Y;
}

inline std::array<std::shared_ptr<Tensor_attributes>, 3>
Graph::batchnorm_backward(std::shared_ptr<Tensor_attributes> dy,
                          std::shared_ptr<Tensor_attributes> x,
                          std::shared_ptr<Tensor_attributes> scale,
                          Batchnorm_backward_attributes attributes) {
    // Set outputs
    auto DX = attributes.outputs[Batchnorm_backward_attributes::output_names::DX] =
        output_tensor(attributes.name + "::DX");
    auto DSCALE = attributes.outputs[Batchnorm_backward_attributes::output_names::DSCALE] =
        output_tensor(attributes.name + "::DSCALE");
    auto DBIAS = attributes.outputs[Batchnorm_backward_attributes::output_names::DBIAS] =
        output_tensor(attributes.name + "::DBIAS");

    // Set inputs
    attributes.inputs[Batchnorm_backward_attributes::input_names::DY]    = dy;
    attributes.inputs[Batchnorm_backward_attributes::input_names::X]     = x;
    attributes.inputs[Batchnorm_backward_attributes::input_names::SCALE] = scale;

    sub_nodes.emplace_back(std::make_unique<DBNNode>(std::move(attributes), context));

    return {DX, DSCALE, DBIAS};
}

inline std::array<std::shared_ptr<Tensor_attributes>, 3>
Graph::instancenorm_backward(std::shared_ptr<Tensor_attributes> dy,
                             std::shared_ptr<Tensor_attributes> x,
                             std::shared_ptr<Tensor_attributes> scale,
                             Instancenorm_backward_attributes attributes) {
    // Set outputs
    auto DX = attributes.outputs[Instancenorm_backward_attributes::output_names::DX] =
        output_tensor(attributes.name + "::DX");
    auto DSCALE = attributes.outputs[Instancenorm_backward_attributes::output_names::DSCALE] =
        output_tensor(attributes.name + "::DSCALE");
    auto DBIAS = attributes.outputs[Instancenorm_backward_attributes::output_names::DBIAS] =
        output_tensor(attributes.name + "::DBIAS");

    // Set inputs
    attributes.inputs[Instancenorm_backward_attributes::input_names::DY]    = dy;
    attributes.inputs[Instancenorm_backward_attributes::input_names::X]     = x;
    attributes.inputs[Instancenorm_backward_attributes::input_names::SCALE] = scale;

    sub_nodes.emplace_back(std::make_unique<DINNode>(std::move(attributes), context));

    return {DX, DSCALE, DBIAS};
}

inline std::array<std::shared_ptr<Tensor_attributes>, 3>
Graph::layernorm_backward(std::shared_ptr<Tensor_attributes> dy,
                          std::shared_ptr<Tensor_attributes> x,
                          std::shared_ptr<Tensor_attributes> scale,
                          Layernorm_backward_attributes attributes) {
    // Set outputs
    auto DX = attributes.outputs[Layernorm_backward_attributes::output_names::DX] =
        output_tensor(attributes.name + "::DX");
    auto DSCALE = attributes.outputs[Layernorm_backward_attributes::output_names::DSCALE] =
        output_tensor(attributes.name + "::DSCALE");
    auto DBIAS = attributes.outputs[Layernorm_backward_attributes::output_names::DBIAS] =
        output_tensor(attributes.name + "::DBIAS");

    // Set inputs
    attributes.inputs[Layernorm_backward_attributes::input_names::DY]    = dy;
    attributes.inputs[Layernorm_backward_attributes::input_names::X]     = x;
    attributes.inputs[Layernorm_backward_attributes::input_names::SCALE] = scale;

    sub_nodes.emplace_back(std::make_unique<DLNNode>(std::move(attributes), context));

    return {DX, DSCALE, DBIAS};
}

inline std::shared_ptr<Tensor_attributes>
Graph::conv_fprop(std::shared_ptr<Tensor_attributes> x,
                  std::shared_ptr<Tensor_attributes> w,
                  Conv_fprop_attributes attributes) {
    // Make required output tensors
    auto Y                                                     = output_tensor(attributes.name + "::Y");
    attributes.outputs[Conv_fprop_attributes::output_names::Y] = Y;

    // Set inputs
    attributes.inputs[Conv_fprop_attributes::input_names::X] = x;
    attributes.inputs[Conv_fprop_attributes::input_names::W] = w;

    sub_nodes.emplace_back(std::make_unique<ConvolutionNode>(std::move(attributes), context));

    return Y;
}

inline std::array<std::shared_ptr<Tensor_attributes>, 5>
Graph::dbn_weight(std::shared_ptr<Tensor_attributes> dy,
                  std::shared_ptr<Tensor_attributes> x,
                  std::shared_ptr<Tensor_attributes> mean,
                  std::shared_ptr<Tensor_attributes> inv_variance,
                  std::shared_ptr<Tensor_attributes> scale,
                  DBN_weight_attributes attributes) {
    // Make required output tensors
    auto DBIAS = attributes.outputs[DBN_weight_attributes::output_names::DBIAS] =
        output_tensor(attributes.name + "::DBIAS");
    auto DSCALE = attributes.outputs[DBN_weight_attributes::output_names::DSCALE] =
        output_tensor(attributes.name + "::DSCALE");
    auto EQ_BIAS = attributes.outputs[DBN_weight_attributes::output_names::EQ_BIAS] =
        output_tensor(attributes.name + "::EQ_BIAS");
    auto EQ_SCALE_DY = attributes.outputs[DBN_weight_attributes::output_names::EQ_SCALE_DY] =
        output_tensor(attributes.name + "::EQ_SCALE_DY");
    auto EQ_SCALE_X = attributes.outputs[DBN_weight_attributes::output_names::EQ_SCALE_X] =
        output_tensor(attributes.name + "::EQ_SCALE_X");

    // Set inputs
    attributes.inputs[DBN_weight_attributes::input_names::DY]           = dy;
    attributes.inputs[DBN_weight_attributes::input_names::X]            = x;
    attributes.inputs[DBN_weight_attributes::input_names::SCALE]        = scale;
    attributes.inputs[DBN_weight_attributes::input_names::MEAN]         = mean;
    attributes.inputs[DBN_weight_attributes::input_names::INV_VARIANCE] = inv_variance;

    sub_nodes.emplace_back(std::make_unique<DBNWeightNode>(std::move(attributes), context));

    return {DSCALE, DBIAS, EQ_SCALE_DY, EQ_SCALE_X, EQ_BIAS};
}

inline std::shared_ptr<Tensor_attributes>
Graph::conv_dgrad(std::shared_ptr<Tensor_attributes> dy,
                  std::shared_ptr<Tensor_attributes> w,
                  Conv_dgrad_attributes attributes) {
    // Make required output tensors
    auto DX = attributes.outputs[Conv_dgrad_attributes::output_names::DX] = output_tensor(attributes.name + "::DX");

    // Set inputs
    attributes.inputs[Conv_dgrad_attributes::input_names::DY] = dy;
    attributes.inputs[Conv_dgrad_attributes::input_names::W]  = w;

    sub_nodes.emplace_back(std::make_unique<DgradNode>(std::move(attributes), context));

    return DX;
}

inline std::array<std::shared_ptr<Tensor_attributes>, 2>
Graph::genstats(std::shared_ptr<Tensor_attributes> x, Genstats_attributes attributes) {
    // Set outputs
    auto SUM = attributes.outputs[Genstats_attributes::output_names::SUM] =
        output_tensor(attributes.name + "_sum_output");
    auto SQ_SUM = attributes.outputs[Genstats_attributes::output_names::SQ_SUM] =
        output_tensor(attributes.name + "_sq_sum_output");

    // Set inputs
    attributes.inputs[Genstats_attributes::input_names::X] = x;

    sub_nodes.emplace_back(std::make_unique<GenstatsNode>(std::move(attributes), context));

    return {SUM, SQ_SUM};
}

inline std::shared_ptr<Tensor_attributes>
Graph::conv_wgrad(std::shared_ptr<Tensor_attributes> dy,
                  std::shared_ptr<Tensor_attributes> x,
                  Conv_wgrad_attributes attributes) {
    // Make required output tensors
    auto DW = attributes.outputs[Conv_wgrad_attributes::output_names::DW] = output_tensor(attributes.name + "::DW");

    // Set inputs
    attributes.inputs[Conv_wgrad_attributes::input_names::X]  = x;
    attributes.inputs[Conv_wgrad_attributes::input_names::DY] = dy;

    sub_nodes.emplace_back(std::make_unique<WgradNode>(std::move(attributes), context));

    return DW;
}

inline std::array<std::shared_ptr<Tensor_attributes>, 2>
Graph::rmsnorm(std::shared_ptr<Tensor_attributes> x,
               std::shared_ptr<Tensor_attributes> scale,
               Rmsnorm_attributes attributes) {
    // Set outputs
    auto Y = attributes.outputs[Rmsnorm_attributes::output_names::Y] = output_tensor(attributes.name + "::Y");
    std::shared_ptr<Tensor_attributes> INV_VARIANCE                  = nullptr;
    if (attributes.forward_phase == NormFwdPhase_t::TRAINING) {
        INV_VARIANCE = attributes.outputs[Rmsnorm_attributes::output_names::INV_VARIANCE] =
            output_tensor(attributes.name + "::INV_VARIANCE");
    }
    // Set inputs
    attributes.inputs[Rmsnorm_attributes::input_names::X]     = x;
    attributes.inputs[Rmsnorm_attributes::input_names::SCALE] = scale;

    sub_nodes.emplace_back(std::make_unique<RMSNormNode>(std::move(attributes), context));

    return {Y, INV_VARIANCE};
}

inline std::array<std::shared_ptr<Tensor_attributes>, 3>
Graph::rmsnorm_backward(std::shared_ptr<Tensor_attributes> dy,
                        std::shared_ptr<Tensor_attributes> x,
                        std::shared_ptr<Tensor_attributes> scale,
                        std::shared_ptr<Tensor_attributes> inv_variance,
                        Rmsnorm_backward_attributes attributes) {
    // Set outputs
    auto DX = attributes.outputs[Rmsnorm_backward_attributes::output_names::DX] =
        output_tensor(attributes.name + "::DX");
    auto DScale = attributes.outputs[Rmsnorm_backward_attributes::output_names::DSCALE] =
        output_tensor(attributes.name + "::Dscale");
    std::shared_ptr<Tensor_attributes> DBias = nullptr;
    if (attributes.use_dbias.value_or(true)) {
        DBias = attributes.outputs[Rmsnorm_backward_attributes::output_names::DBIAS] =
            output_tensor(attributes.name + "::Dbias");
    }

    // Set inputs
    attributes.inputs[Rmsnorm_backward_attributes::input_names::DY]           = dy;
    attributes.inputs[Rmsnorm_backward_attributes::input_names::X]            = x;
    attributes.inputs[Rmsnorm_backward_attributes::input_names::SCALE]        = scale;
    attributes.inputs[Rmsnorm_backward_attributes::input_names::INV_VARIANCE] = inv_variance;

    sub_nodes.emplace_back(std::make_unique<DRMSNormNode>(std::move(attributes), context));

    return {DX, DScale, DBias};
}

// inline std::array<std::shared_ptr<Tensor_attributes>, 2>
// Graph::scaled_dot_product_attention(std::shared_ptr<Tensor_attributes> q,
//                                     std::shared_ptr<Tensor_attributes> k,
//                                     std::shared_ptr<Tensor_attributes> v,
//                                     Scaled_dot_product_attention_attributes options) {
//     // Make required output tensors
//     auto O = options.outputs.O = output_tensor(options.get_name() + "_output");
//     auto S = options.outputs.S = output_tensor(options.get_name() + "_softmax_output");

//     // Set inputs
//     options.inputs.Q = q;
//     options.inputs.K = k;
//     options.inputs.V = v;

//     sub_nodes.emplace_back(std::make_unique<ScaledDotProductAttentionNode>(std::move(options), context));

//     return {O, S};
// }

inline std::array<std::shared_ptr<Tensor_attributes>, 2>
Graph::sdpa(std::shared_ptr<Tensor_attributes> q,
            std::shared_ptr<Tensor_attributes> k,
            std::shared_ptr<Tensor_attributes> v,
            SDPA_attributes attributes) {
    // Make required output tensors
    auto O = attributes.outputs[SDPA_attributes::output_names::O] = output_tensor(attributes.name + "::O");

    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> Stats = nullptr;
    if (attributes.is_inference == false) {
        Stats = attributes.outputs[SDPA_attributes::output_names::Stats] = output_tensor(attributes.name + "::Stats");
    }

    // Set inputs
    attributes.inputs[SDPA_attributes::input_names::Q] = q;
    attributes.inputs[SDPA_attributes::input_names::K] = k;
    attributes.inputs[SDPA_attributes::input_names::V] = v;

    sub_nodes.emplace_back(std::make_unique<SDPANode>(std::move(attributes), context));

    return {O, Stats};
}

inline std::array<std::shared_ptr<Tensor_attributes>, 4>
Graph::sdpa_fp8(std::shared_ptr<Tensor_attributes> q,
                std::shared_ptr<Tensor_attributes> k,
                std::shared_ptr<Tensor_attributes> v,
                std::shared_ptr<Tensor_attributes> descale_q,
                std::shared_ptr<Tensor_attributes> descale_k,
                std::shared_ptr<Tensor_attributes> descale_v,
                std::shared_ptr<Tensor_attributes> descale_s,
                std::shared_ptr<Tensor_attributes> scale_s,
                std::shared_ptr<Tensor_attributes> scale_o,
                SDPA_fp8_attributes attributes) {
    // Make required output tensors
    auto O = attributes.outputs[SDPA_fp8_attributes::output_names::O] = output_tensor(attributes.name + "::O");

    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> Stats = nullptr;
    if (attributes.is_inference == false) {
        Stats = attributes.outputs[SDPA_fp8_attributes::output_names::Stats] =
            output_tensor(attributes.name + "::Stats");
    }

    auto Amax_S = attributes.outputs[SDPA_fp8_attributes::output_names::Amax_S] =
        output_tensor(attributes.name + "::Amax_S");
    auto Amax_O = attributes.outputs[SDPA_fp8_attributes::output_names::Amax_O] =
        output_tensor(attributes.name + "::Amax_O");

    // Set inputs
    attributes.inputs[SDPA_fp8_attributes::input_names::Q] = q;
    attributes.inputs[SDPA_fp8_attributes::input_names::K] = k;
    attributes.inputs[SDPA_fp8_attributes::input_names::V] = v;

    attributes.inputs[SDPA_fp8_attributes::input_names::Descale_Q] = descale_q;
    attributes.inputs[SDPA_fp8_attributes::input_names::Descale_K] = descale_k;
    attributes.inputs[SDPA_fp8_attributes::input_names::Descale_V] = descale_v;
    attributes.inputs[SDPA_fp8_attributes::input_names::Descale_S] = descale_s;
    attributes.inputs[SDPA_fp8_attributes::input_names::Scale_S]   = scale_s;
    attributes.inputs[SDPA_fp8_attributes::input_names::Scale_O]   = scale_o;

    sub_nodes.emplace_back(std::make_unique<SDPAFP8Node>(std::move(attributes), context));

    return {O, Stats, Amax_S, Amax_O};
}

inline std::array<std::shared_ptr<Tensor_attributes>, 7>
Graph::sdpa_fp8_backward(std::shared_ptr<Tensor_attributes> q,
                         std::shared_ptr<Tensor_attributes> k,
                         std::shared_ptr<Tensor_attributes> v,
                         std::shared_ptr<Tensor_attributes> o,
                         std::shared_ptr<Tensor_attributes> dO,
                         std::shared_ptr<Tensor_attributes> Stats,
                         std::shared_ptr<Tensor_attributes> descale_q,
                         std::shared_ptr<Tensor_attributes> descale_k,
                         std::shared_ptr<Tensor_attributes> descale_v,
                         std::shared_ptr<Tensor_attributes> descale_o,
                         std::shared_ptr<Tensor_attributes> descale_do,
                         std::shared_ptr<Tensor_attributes> descale_s,
                         std::shared_ptr<Tensor_attributes> descale_dp,
                         std::shared_ptr<Tensor_attributes> scale_s,
                         std::shared_ptr<Tensor_attributes> scale_dq,
                         std::shared_ptr<Tensor_attributes> scale_dk,
                         std::shared_ptr<Tensor_attributes> scale_dv,
                         std::shared_ptr<Tensor_attributes> scale_dp,
                         SDPA_fp8_backward_attributes attributes) {
    // Make required output tensors
    auto dQ = attributes.outputs[SDPA_fp8_backward_attributes::output_names::dQ] =
        output_tensor(attributes.name + "::dQ");
    auto dK = attributes.outputs[SDPA_fp8_backward_attributes::output_names::dK] =
        output_tensor(attributes.name + "::dK");
    auto dV = attributes.outputs[SDPA_fp8_backward_attributes::output_names::dV] =
        output_tensor(attributes.name + "::dV");
    auto Amax_dQ = attributes.outputs[SDPA_fp8_backward_attributes::output_names::Amax_dQ] =
        output_tensor(attributes.name + "::Amax_dQ");
    auto Amax_dK = attributes.outputs[SDPA_fp8_backward_attributes::output_names::Amax_dK] =
        output_tensor(attributes.name + "::Amax_dK");
    auto Amax_dV = attributes.outputs[SDPA_fp8_backward_attributes::output_names::Amax_dV] =
        output_tensor(attributes.name + "::Amax_dV");
    auto Amax_dP = attributes.outputs[SDPA_fp8_backward_attributes::output_names::Amax_dP] =
        output_tensor(attributes.name + "::Amax_dP");

    // Set inputs
    attributes.inputs[SDPA_fp8_backward_attributes::input_names::Q]     = q;
    attributes.inputs[SDPA_fp8_backward_attributes::input_names::K]     = k;
    attributes.inputs[SDPA_fp8_backward_attributes::input_names::V]     = v;
    attributes.inputs[SDPA_fp8_backward_attributes::input_names::O]     = o;
    attributes.inputs[SDPA_fp8_backward_attributes::input_names::Stats] = Stats;
    attributes.inputs[SDPA_fp8_backward_attributes::input_names::dO]    = dO;

    attributes.inputs[SDPA_fp8_backward_attributes::input_names::Descale_Q]  = descale_q;
    attributes.inputs[SDPA_fp8_backward_attributes::input_names::Descale_K]  = descale_k;
    attributes.inputs[SDPA_fp8_backward_attributes::input_names::Descale_V]  = descale_v;
    attributes.inputs[SDPA_fp8_backward_attributes::input_names::Descale_S]  = descale_s;
    attributes.inputs[SDPA_fp8_backward_attributes::input_names::Descale_O]  = descale_o;
    attributes.inputs[SDPA_fp8_backward_attributes::input_names::Descale_dO] = descale_do;
    attributes.inputs[SDPA_fp8_backward_attributes::input_names::Descale_dP] = descale_dp;

    attributes.inputs[SDPA_fp8_backward_attributes::input_names::Scale_dQ] = scale_dq;
    attributes.inputs[SDPA_fp8_backward_attributes::input_names::Scale_dK] = scale_dk;
    attributes.inputs[SDPA_fp8_backward_attributes::input_names::Scale_dV] = scale_dv;
    attributes.inputs[SDPA_fp8_backward_attributes::input_names::Scale_S]  = scale_s;
    attributes.inputs[SDPA_fp8_backward_attributes::input_names::Scale_dP] = scale_dp;

    sub_nodes.emplace_back(std::make_unique<SDPAFP8BackwardNode>(std::move(attributes), context));

    return {dQ, dK, dV, Amax_dQ, Amax_dK, Amax_dV, Amax_dP};
}

inline std::array<std::shared_ptr<Tensor_attributes>, 3>
Graph::sdpa_backward(std::shared_ptr<Tensor_attributes> q,
                     std::shared_ptr<Tensor_attributes> k,
                     std::shared_ptr<Tensor_attributes> v,
                     std::shared_ptr<Tensor_attributes> o,
                     std::shared_ptr<Tensor_attributes> dO,
                     std::shared_ptr<Tensor_attributes> stats,
                     SDPA_backward_attributes attributes) {
    // Set inputs
    attributes.inputs[SDPA_backward_attributes::input_names::Q]     = q;
    attributes.inputs[SDPA_backward_attributes::input_names::K]     = k;
    attributes.inputs[SDPA_backward_attributes::input_names::V]     = v;
    attributes.inputs[SDPA_backward_attributes::input_names::O]     = o;
    attributes.inputs[SDPA_backward_attributes::input_names::dO]    = dO;
    attributes.inputs[SDPA_backward_attributes::input_names::Stats] = stats;

    // Make required output tensors
    auto dQ = attributes.outputs[SDPA_backward_attributes::output_names::dQ] = output_tensor(attributes.name + "::dQ");
    auto dK = attributes.outputs[SDPA_backward_attributes::output_names::dK] = output_tensor(attributes.name + "::dK");
    auto dV = attributes.outputs[SDPA_backward_attributes::output_names::dV] = output_tensor(attributes.name + "::dV");

    sub_nodes.emplace_back(std::make_unique<SDPABackwardNode>(std::move(attributes), context));

    return {dQ, dK, dV};
}

static inline std::ostream &
operator<<(std::ostream &os, Graph const &graph) {
    os << graph.print();
    return os;
}

}  // namespace cudnn_frontend::graph