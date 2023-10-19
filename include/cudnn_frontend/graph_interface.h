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
#include "node/matmul.h"
#include "node/pointwise.h"
#include "node/reduction.h"
#include "node/reshape.h"
#include "node/rmsnorm.h"
#include "node/rng.h"
#include "node/scaled_dot_product_attention.h"
#include "node/scaled_dot_product_flash_attention.h"

#include "plans.h"
#include "graph_helpers.h"

namespace cudnn_frontend::graph {

class Graph : public INode {
   private:
    std::unordered_set<std::shared_ptr<Tensor_attributes>> tensors;

    std::shared_ptr<Tensor_attributes>
    output_tensor(std::string const &name) {
        auto tensor = std::make_shared<Tensor_attributes>();
        tensor->set_name(name).set_is_virtual(true);
        tensors.emplace(tensor);
        return tensor;
    }

    // This API is still work in progress and unverified.
    std::array<std::shared_ptr<Tensor_attributes>, 2> scaled_dot_product_attention(
        std::shared_ptr<Tensor_attributes>,
        std::shared_ptr<Tensor_attributes>,
        std::shared_ptr<Tensor_attributes>,
        Scaled_dot_product_attention_attributes);

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

    std::array<std::shared_ptr<Tensor_attributes>, 2> rmsnorm(std::shared_ptr<Tensor_attributes>,
                                                              std::shared_ptr<Tensor_attributes>,
                                                              Rmsnorm_attributes);

    std::array<std::shared_ptr<Tensor_attributes>, 3> rmsnorm_backward(std::shared_ptr<Tensor_attributes>,
                                                                       std::shared_ptr<Tensor_attributes>,
                                                                       std::shared_ptr<Tensor_attributes>,
                                                                       std::shared_ptr<Tensor_attributes>,
                                                                       Rmsnorm_backward_attributes);

    std::array<std::shared_ptr<Tensor_attributes>, 2> scaled_dot_product_flash_attention(
        std::shared_ptr<Tensor_attributes>,
        std::shared_ptr<Tensor_attributes>,
        std::shared_ptr<Tensor_attributes>,
        Scaled_dot_product_flash_attention_attributes);
    std::array<std::shared_ptr<Tensor_attributes>, 3> scaled_dot_product_flash_attention_backward(
        std::shared_ptr<Tensor_attributes>,
        std::shared_ptr<Tensor_attributes>,
        std::shared_ptr<Tensor_attributes>,
        std::shared_ptr<Tensor_attributes>,
        std::shared_ptr<Tensor_attributes>,
        std::shared_ptr<Tensor_attributes>,
        Scaled_dot_product_flash_attention_backward_attributes);

    Plans
    get_execution_plan_list(std::vector<HeurMode_t> const &mode);

    error_t
    set_execution_plans(Plans const &plan) {
        if (plan.list_of_engine_configs.get_candidate() == nullptr) {
            return {error_code_t::GRAPH_EXECUTION_PLAN_CREATION_FAILED,
                    "[cudnn_frontend] ERROR: No validate candidate for plan execution"};
        }
        execution_plans.emplace_back(plan.list_of_engine_configs.get_candidate());

        return {error_code_t::OK, ""};
    }

    error_t
    build(cudnnHandle_t const &handle, std::vector<HeurMode_t> const &mode);

    error_t
    createOperationGraphs(cudnnHandle_t handle) override final {
        getLogger() << "Operation Graph has " << operations.size() << " operations." << std::endl;

        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_operation_graphs(handle));

        return {error_code_t::OK, ""};
    }
};

inline Plans
Graph::get_execution_plan_list(std::vector<HeurMode_t> const &mode) {
    Plans plan_list;
    // TODO: The error returned is not propagate to user.
    // Should the return value be changed to error_code_t too?

    std::unordered_map<std::string, EngineConfigList> op_graph_to_configs;
    auto status = detail::query_heuristics(operation_graphs, op_graph_to_configs, mode);
    if (status.is_bad()) {
        getLogger() << "[cudnn_frontend] ERROR: Failed to build." << std::endl;
        return plan_list;
    }

    getLogger() << "[cudnn_frontend] INFO: Extracting engine configs." << std::endl;
    auto &engine_configs = plan_list.list_of_engine_configs;
    engine_configs.set_tag(op_graph_to_configs.begin()->first);
    engine_configs.set_engine_configs(op_graph_to_configs.begin()->second);

    getLogger() << "[cudnn_frontend] INFO: Querying engine config properties\n";
    status = engine_configs.query_properties();
    if (status.is_bad()) {
        getLogger() << "[cudnn_frontend] ERROR: Querying engine configs failed." << std::endl;
    }
    return plan_list;
}

inline error_t
Graph::build(cudnnHandle_t const &handle, std::vector<HeurMode_t> const &modes) {
    CHECK_CUDNN_FRONTEND_ERROR(validate());

    CHECK_CUDNN_FRONTEND_ERROR(build_operation_graph(handle));

    auto plans = get_execution_plan_list(modes);

    CHECK_CUDNN_FRONTEND_ERROR(plans.check_support(handle));

    CHECK_CUDNN_FRONTEND_ERROR(set_execution_plans(plans));

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
    tensors.emplace(tensor_ptr);
    return tensor_ptr;
}

inline std::array<std::shared_ptr<Tensor_attributes>, 6>
Graph::bn_finalize(std::shared_ptr<Tensor_attributes> sum,
                   std::shared_ptr<Tensor_attributes> sq_sum,
                   std::shared_ptr<Tensor_attributes> scale,
                   std::shared_ptr<Tensor_attributes> bias,
                   std::shared_ptr<Tensor_attributes> epsilon,
                   std::shared_ptr<Tensor_attributes> accum_count,
                   BN_finalize_attributes options) {
    // Set outputs
    auto EQ_SCALE = options.outputs.EQ_SCALE = output_tensor(options.get_name() + "::EQ_SCALE");
    auto EQ_BIAS = options.outputs.EQ_BIAS = output_tensor(options.get_name() + "::EQ_BIAS");
    auto MEAN = options.outputs.MEAN = output_tensor(options.get_name() + "::MEAN");
    auto INV_VARIANCE = options.outputs.INV_VARIANCE     = output_tensor(options.get_name() + "::INV_VARIANCE");
    std::shared_ptr<Tensor_attributes> NEXT_RUNNING_MEAN = nullptr;
    std::shared_ptr<Tensor_attributes> NEXT_RUNNING_VAR  = nullptr;
    if (options.inputs.PREV_RUNNING_MEAN && options.inputs.PREV_RUNNING_VAR && options.inputs.MOMENTUM) {
        NEXT_RUNNING_MEAN = output_tensor(options.get_name() + "::NEXT_RUNNING_MEAN");
        NEXT_RUNNING_VAR  = output_tensor(options.get_name() + "::NEXT_RUNNING_VAR");
    }
    options.outputs.NEXT_RUNNING_MEAN = NEXT_RUNNING_MEAN;
    options.outputs.NEXT_RUNNING_VAR  = NEXT_RUNNING_VAR;

    // Set inputs
    options.inputs.SUM         = sum;
    options.inputs.SQ_SUM      = sq_sum;
    options.inputs.SCALE       = scale;
    options.inputs.BIAS        = bias;
    options.inputs.EPSILON     = epsilon;
    options.inputs.ACCUM_COUNT = accum_count;

    sub_nodes.emplace_back(std::make_unique<BatchNormFinalizeNode>(std::move(options), context));

    return {EQ_SCALE, EQ_BIAS, MEAN, INV_VARIANCE, NEXT_RUNNING_MEAN, NEXT_RUNNING_VAR};
}

inline std::array<std::shared_ptr<Tensor_attributes>, 3>
Graph::layernorm(std::shared_ptr<Tensor_attributes> x,
                 std::shared_ptr<Tensor_attributes> scale,
                 std::shared_ptr<Tensor_attributes> bias,
                 Layernorm_attributes options) {
    // Set outputs
    auto Y = options.outputs.Y                      = output_tensor(options.get_name() + "::Y");
    std::shared_ptr<Tensor_attributes> MEAN         = nullptr;
    std::shared_ptr<Tensor_attributes> INV_VARIANCE = nullptr;
    if (options.forward_phase == NormFwdPhase_t::TRAINING) {
        MEAN = options.outputs.MEAN = output_tensor(options.get_name() + "::MEAN");
        INV_VARIANCE = options.outputs.INV_VARIANCE = output_tensor(options.get_name() + "::INV_VARIANCE");
    }
    // Set inputs
    options.inputs.X     = x;
    options.inputs.SCALE = scale;
    options.inputs.BIAS  = bias;

    sub_nodes.emplace_back(std::make_unique<LayerNormNode>(std::move(options), context));

    return {Y, MEAN, INV_VARIANCE};
}

inline std::array<std::shared_ptr<Tensor_attributes>, 3>
Graph::instancenorm(std::shared_ptr<Tensor_attributes> x,
                    std::shared_ptr<Tensor_attributes> scale,
                    std::shared_ptr<Tensor_attributes> bias,
                    Instancenorm_attributes options) {
    // Set outputs
    auto Y = options.outputs.Y                      = output_tensor(options.get_name() + "::Y");
    std::shared_ptr<Tensor_attributes> MEAN         = nullptr;
    std::shared_ptr<Tensor_attributes> INV_VARIANCE = nullptr;
    if (options.forward_phase == NormFwdPhase_t::TRAINING) {
        MEAN = options.outputs.MEAN = output_tensor(options.get_name() + "::MEAN");
        INV_VARIANCE = options.outputs.INV_VARIANCE = output_tensor(options.get_name() + "::INV_VARIANCE");
    }
    // Set inputs
    options.inputs.X     = x;
    options.inputs.SCALE = scale;
    options.inputs.BIAS  = bias;

    sub_nodes.emplace_back(std::make_unique<InstanceNormNode>(std::move(options), context));

    return {Y, MEAN, INV_VARIANCE};
}

inline std::array<std::shared_ptr<Tensor_attributes>, 5>
Graph::batchnorm(std::shared_ptr<Tensor_attributes> x,
                 std::shared_ptr<Tensor_attributes> scale,
                 std::shared_ptr<Tensor_attributes> bias,
                 Batchnorm_attributes options) {
    // Set outputs
    auto Y = options.outputs.Y = output_tensor(options.get_name() + "::Y");
    auto MEAN = options.outputs.MEAN = output_tensor(options.get_name() + "::MEAN");
    auto INV_VARIANCE = options.outputs.INV_VARIANCE     = output_tensor(options.get_name() + "::INV_VARIANCE");
    std::shared_ptr<Tensor_attributes> NEXT_RUNNING_MEAN = nullptr;
    std::shared_ptr<Tensor_attributes> NEXT_RUNNING_VAR  = nullptr;
    if (options.inputs.PREV_RUNNING_MEAN && options.inputs.PREV_RUNNING_VAR && options.inputs.MOMENTUM) {
        NEXT_RUNNING_MEAN = output_tensor(options.get_name() + "::NEXT_RUNNING_MEAN");
        NEXT_RUNNING_VAR  = output_tensor(options.get_name() + "::NEXT_RUNNING_VAR");
    }
    options.outputs.NEXT_RUNNING_MEAN = NEXT_RUNNING_MEAN;
    options.outputs.NEXT_RUNNING_VAR  = NEXT_RUNNING_VAR;

    // Set inputs
    options.inputs.X     = x;
    options.inputs.SCALE = scale;
    options.inputs.BIAS  = bias;

    sub_nodes.emplace_back(std::make_unique<BatchNormNode>(std::move(options), context));

    return {Y, MEAN, INV_VARIANCE, NEXT_RUNNING_MEAN, NEXT_RUNNING_VAR};
}

inline std::shared_ptr<Tensor_attributes>
Graph::batchnorm_inference(std::shared_ptr<Tensor_attributes> x,
                           std::shared_ptr<Tensor_attributes> mean,
                           std::shared_ptr<Tensor_attributes> inv_variance,
                           std::shared_ptr<Tensor_attributes> scale,
                           std::shared_ptr<Tensor_attributes> bias,
                           Batchnorm_inference_attributes options) {
    // Set outputs
    auto Y = options.outputs.Y = output_tensor(options.get_name() + "::Y");

    // Set inputs
    options.inputs.X            = x;
    options.inputs.MEAN         = mean;
    options.inputs.INV_VARIANCE = inv_variance;
    options.inputs.SCALE        = scale;
    options.inputs.BIAS         = bias;

    sub_nodes.emplace_back(std::make_unique<BatchnormInferenceNode>(std::move(options), context));

    return Y;
}

inline std::array<std::shared_ptr<Tensor_attributes>, 3>
Graph::batchnorm_backward(std::shared_ptr<Tensor_attributes> dy,
                          std::shared_ptr<Tensor_attributes> x,
                          std::shared_ptr<Tensor_attributes> scale,
                          Batchnorm_backward_attributes options) {
    // Set outputs
    options.make_outputs([this](std::string const &name) { return output_tensor(name); });
    auto return_outputs = options.outputs;

    // Set inputs
    options.inputs.DY    = dy;
    options.inputs.X     = x;
    options.inputs.SCALE = scale;

    sub_nodes.emplace_back(std::make_unique<DBNNode>(std::move(options), context));

    return {return_outputs.DX, return_outputs.DSCALE, return_outputs.DBIAS};
}

inline std::array<std::shared_ptr<Tensor_attributes>, 3>
Graph::instancenorm_backward(std::shared_ptr<Tensor_attributes> dy,
                             std::shared_ptr<Tensor_attributes> x,
                             std::shared_ptr<Tensor_attributes> scale,
                             Instancenorm_backward_attributes options) {
    // Set outputs
    options.make_outputs([this](std::string const &name) { return output_tensor(name); });
    auto return_outputs = options.outputs;

    // Set inputs
    options.inputs.DY    = dy;
    options.inputs.X     = x;
    options.inputs.SCALE = scale;

    sub_nodes.emplace_back(std::make_unique<DINNode>(std::move(options), context));

    return {return_outputs.DX, return_outputs.DSCALE, return_outputs.DBIAS};
}

inline std::array<std::shared_ptr<Tensor_attributes>, 3>
Graph::layernorm_backward(std::shared_ptr<Tensor_attributes> dy,
                          std::shared_ptr<Tensor_attributes> x,
                          std::shared_ptr<Tensor_attributes> scale,
                          Layernorm_backward_attributes options) {
    // Set outputs
    options.make_outputs([this](std::string const &name) { return output_tensor(name); });
    auto return_outputs = options.outputs;

    // Set inputs
    options.inputs.DY    = dy;
    options.inputs.X     = x;
    options.inputs.SCALE = scale;

    sub_nodes.emplace_back(std::make_unique<DLNNode>(std::move(options), context));

    return {return_outputs.DX, return_outputs.DSCALE, return_outputs.DBIAS};
}

inline std::shared_ptr<Tensor_attributes>
Graph::conv_fprop(std::shared_ptr<Tensor_attributes> x,
                  std::shared_ptr<Tensor_attributes> w,
                  Conv_fprop_attributes options) {
    // Make required output tensors
    auto Y            = output_tensor(options.get_name() + "_output");
    options.outputs.Y = Y;

    // Set inputs
    options.inputs.X = x;
    options.inputs.W = w;

    sub_nodes.emplace_back(std::make_unique<ConvolutionNode>(std::move(options), context));

    return Y;
}

inline std::array<std::shared_ptr<Tensor_attributes>, 5>
Graph::dbn_weight(std::shared_ptr<Tensor_attributes> dy,
                  std::shared_ptr<Tensor_attributes> x,
                  std::shared_ptr<Tensor_attributes> mean,
                  std::shared_ptr<Tensor_attributes> inv_variance,
                  std::shared_ptr<Tensor_attributes> scale,
                  DBN_weight_attributes options) {
    // Make required output tensors
    options.make_outputs([this](std::string const &name) { return output_tensor(name); });
    auto return_outputs = options.outputs;

    // Set inputs
    options.inputs.DY           = dy;
    options.inputs.X            = x;
    options.inputs.SCALE        = scale;
    options.inputs.MEAN         = mean;
    options.inputs.INV_VARIANCE = inv_variance;

    sub_nodes.emplace_back(std::make_unique<DBNWeightNode>(std::move(options), context));

    return {return_outputs.DSCALE,
            return_outputs.DBIAS,
            return_outputs.EQ_SCALE_DY,
            return_outputs.EQ_SCALE_X,
            return_outputs.EQ_BIAS};
}

inline std::shared_ptr<Tensor_attributes>
Graph::conv_dgrad(std::shared_ptr<Tensor_attributes> dy,
                  std::shared_ptr<Tensor_attributes> w,
                  Conv_dgrad_attributes options) {
    // Make required output tensors
    auto DX = options.outputs.DX = output_tensor(options.get_name() + "_output");

    // Set inputs
    options.inputs.DY = dy;
    options.inputs.W  = w;

    sub_nodes.emplace_back(std::make_unique<DgradNode>(std::move(options), context));

    return DX;
}

inline std::array<std::shared_ptr<Tensor_attributes>, 2>
Graph::genstats(std::shared_ptr<Tensor_attributes> x, Genstats_attributes options) {
    // Set outputs
    auto SUM = options.outputs.SUM = output_tensor(options.get_name() + "_sum_output");
    auto SQ_SUM = options.outputs.SQ_SUM = output_tensor(options.get_name() + "_sq_sum_output");

    // Set inputs
    options.inputs.X = x;

    sub_nodes.emplace_back(std::make_unique<GenstatsNode>(std::move(options), context));

    return {SUM, SQ_SUM};
}

inline std::shared_ptr<Tensor_attributes>
Graph::conv_wgrad(std::shared_ptr<Tensor_attributes> dy,
                  std::shared_ptr<Tensor_attributes> x,
                  Conv_wgrad_attributes options) {
    // Make required output tensors
    auto DW = options.outputs.DW = output_tensor(options.get_name() + "_output");

    // Set inputs
    options.inputs.X  = x;
    options.inputs.DY = dy;

    sub_nodes.emplace_back(std::make_unique<WgradNode>(std::move(options), context));

    return DW;
}

inline std::shared_ptr<Tensor_attributes>
Graph::pointwise(std::shared_ptr<Tensor_attributes> a, Pointwise_attributes options) {
    auto OUT_0 = options.outputs.OUT_0 = output_tensor(options.get_name() + "_output");

    // Set inputs
    options.inputs.IN_0 = a;

    sub_nodes.emplace_back(std::make_unique<PointwiseNode>(std::move(options), context));

    return OUT_0;
}

inline std::shared_ptr<Tensor_attributes>
Graph::pointwise(std::shared_ptr<Tensor_attributes> a,
                 std::shared_ptr<Tensor_attributes> b,
                 Pointwise_attributes options) {
    auto OUT_0 = options.outputs.OUT_0 = output_tensor(options.get_name() + "_output");

    // Set inputs
    options.inputs.IN_0 = a;
    options.inputs.IN_1 = b;

    sub_nodes.emplace_back(std::make_unique<PointwiseNode>(std::move(options), context));

    return OUT_0;
}

inline std::shared_ptr<Tensor_attributes>
Graph::pointwise(std::shared_ptr<Tensor_attributes> a,
                 std::shared_ptr<Tensor_attributes> b,
                 std::shared_ptr<Tensor_attributes> c,
                 Pointwise_attributes options) {
    auto OUT_0 = options.outputs.OUT_0 = output_tensor(options.get_name() + "_output");

    // Set inputs
    options.inputs.IN_0 = a;
    options.inputs.IN_1 = b;
    options.inputs.IN_2 = c;

    sub_nodes.emplace_back(std::make_unique<PointwiseNode>(std::move(options), context));

    return OUT_0;
}

inline std::shared_ptr<Tensor_attributes>
Graph::reduction(std::shared_ptr<Tensor_attributes> input, Reduction_attributes options) {
    auto Y = options.outputs.Y = output_tensor(options.get_name() + "_output");

    // Set inputs
    options.inputs.X = input;

    sub_nodes.emplace_back(std::make_unique<ReductionNode>(std::move(options), context));

    return Y;
}

inline std::array<std::shared_ptr<Tensor_attributes>, 2>
Graph::rmsnorm(std::shared_ptr<Tensor_attributes> x,
               std::shared_ptr<Tensor_attributes> scale,
               Rmsnorm_attributes options) {
    // Set outputs
    auto Y = options.outputs.Y                      = output_tensor(options.get_name() + "::Y");
    std::shared_ptr<Tensor_attributes> INV_VARIANCE = nullptr;
    if (options.forward_phase == NormFwdPhase_t::TRAINING) {
        INV_VARIANCE = options.outputs.INV_VARIANCE = output_tensor(options.get_name() + "::INV_VARIANCE");
    }
    // Set inputs
    options.inputs.X     = x;
    options.inputs.SCALE = scale;

    sub_nodes.emplace_back(std::make_unique<RMSNormNode>(std::move(options), context));

    return {Y, INV_VARIANCE};
}

inline std::array<std::shared_ptr<Tensor_attributes>, 3>
Graph::rmsnorm_backward(std::shared_ptr<Tensor_attributes> dy,
                        std::shared_ptr<Tensor_attributes> x,
                        std::shared_ptr<Tensor_attributes> scale,
                        std::shared_ptr<Tensor_attributes> inv_variance,
                        Rmsnorm_backward_attributes options) {
    // Set outputs
    auto DX = options.outputs.DX = output_tensor(options.get_name() + "::DX");
    auto DScale = options.outputs.DSCALE     = output_tensor(options.get_name() + "::Dscale");
    std::shared_ptr<Tensor_attributes> DBias = nullptr;
    if (options.use_dbias.value_or(true)) {
        DBias = options.outputs.DBIAS = output_tensor(options.get_name() + "::Dbias");
    }

    // Set inputs
    options.inputs.DY           = dy;
    options.inputs.X            = x;
    options.inputs.SCALE        = scale;
    options.inputs.INV_VARIANCE = inv_variance;

    sub_nodes.emplace_back(std::make_unique<DRMSNormNode>(std::move(options), context));

    return {DX, DScale, DBias};
}

inline std::shared_ptr<Tensor_attributes>
Graph::matmul(std::shared_ptr<Tensor_attributes> a, std::shared_ptr<Tensor_attributes> b, Matmul_attributes options) {
    auto C = options.outputs.C = output_tensor(options.get_name() + "_output");

    // Set inputs
    options.inputs.A = a;
    options.inputs.B = b;

    sub_nodes.emplace_back(std::make_unique<MatmulNode>(std::move(options), context));

    return C;
}

inline std::array<std::shared_ptr<Tensor_attributes>, 2>
Graph::scaled_dot_product_attention(std::shared_ptr<Tensor_attributes> q,
                                    std::shared_ptr<Tensor_attributes> k,
                                    std::shared_ptr<Tensor_attributes> v,
                                    Scaled_dot_product_attention_attributes options) {
    // Make required output tensors
    auto O = options.outputs.O = output_tensor(options.get_name() + "_output");
    auto S = options.outputs.S = output_tensor(options.get_name() + "_softmax_output");

    // Set inputs
    options.inputs.Q = q;
    options.inputs.K = k;
    options.inputs.V = v;

    sub_nodes.emplace_back(std::make_unique<ScaledDotProductAttentionNode>(std::move(options), context));

    return {O, S};
}

inline std::array<std::shared_ptr<Tensor_attributes>, 2>
Graph::scaled_dot_product_flash_attention(std::shared_ptr<Tensor_attributes> q,
                                          std::shared_ptr<Tensor_attributes> k,
                                          std::shared_ptr<Tensor_attributes> v,
                                          Scaled_dot_product_flash_attention_attributes options) {
    // Make required output tensors
    auto O = options.outputs.O = output_tensor(options.get_name() + "::O");

    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> Stats = nullptr;
    if (options.is_inference == false) {
        Stats = options.outputs.Stats = output_tensor(options.get_name() + "::Stats");
    }

    // Set inputs
    options.inputs.Q = q;
    options.inputs.K = k;
    options.inputs.V = v;

    sub_nodes.emplace_back(std::make_unique<ScaledDotProductFlashAttentionNode>(std::move(options), context));

    return {O, Stats};
}

inline std::array<std::shared_ptr<Tensor_attributes>, 3>
Graph::scaled_dot_product_flash_attention_backward(std::shared_ptr<Tensor_attributes> q,
                                                   std::shared_ptr<Tensor_attributes> k,
                                                   std::shared_ptr<Tensor_attributes> v,
                                                   std::shared_ptr<Tensor_attributes> o,
                                                   std::shared_ptr<Tensor_attributes> dO,
                                                   std::shared_ptr<Tensor_attributes> Stats,
                                                   Scaled_dot_product_flash_attention_backward_attributes options) {
    // Set inputs
    options.inputs.Q     = q;
    options.inputs.K     = k;
    options.inputs.V     = v;
    options.inputs.O     = o;
    options.inputs.dO    = dO;
    options.inputs.Stats = Stats;

    // Make required output tensors
    auto dQ = options.outputs.dQ = output_tensor(options.get_name() + "::dQ");
    auto dK = options.outputs.dK = output_tensor(options.get_name() + "::dK");
    auto dV = options.outputs.dV = output_tensor(options.get_name() + "::dV");

    sub_nodes.emplace_back(std::make_unique<ScaledDotProductFlashAttentionBackwardNode>(std::move(options), context));

    return {dQ, dK, dV};
}

}  // namespace cudnn_frontend::graph