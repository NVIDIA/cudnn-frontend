#pragma once

#include <unordered_map>

#include "cudnn_frontend/node/batchnorm.h"
#include "cudnn_frontend/node/batchnorm_inference.h"
#include "cudnn_frontend/node/bn_finalize.h"
#include "cudnn_frontend/node/conv_fprop.h"
#include "cudnn_frontend/node/conv_dgrad.h"
#include "cudnn_frontend/node/conv_wgrad.h"
#include "cudnn_frontend/node/dbn.h"
#include "cudnn_frontend/node/dln.h"
#include "cudnn_frontend/node/dbn_weight.h"
#include "cudnn_frontend/node/genstats.h"
#include "cudnn_frontend/node/layernorm.h"
#include "cudnn_frontend/node/matmul.h"
#include "cudnn_frontend/node/pointwise.h"
#include "cudnn_frontend/node/reduction.h"
#include "cudnn_frontend/node/rng.h"
#include "cudnn_frontend/node/reshape.h"
#include "cudnn_frontend/node/scaled_dot_product_attention.h"
#include "cudnn_frontend/node/scaled_dot_product_flash_attention.h"

#include "cudnn_frontend_graph_helpers.h"

namespace cudnn_frontend::graph {

class Plans {
    friend class Graph;
    Execution_plan_list list_of_engine_configs;

   public:
    Execution_plan_list &
    get_engine_configs() {
        return list_of_engine_configs;
    }

    Plans &
    filter_out_numeric_notes(std::vector<cudnnBackendNumericalNote_t> const &);
    Plans &
    filter_out_behavior_notes(std::vector<cudnnBackendBehaviorNote_t> const &);
    Plans &
    filter_out_workspace_greater_than(int64_t const workspace) {
        list_of_engine_configs.set_max_workspace_allowed(workspace);
        return *this;
    }

    error_t build_all_plans(cudnnHandle_t);

    inline error_t
    check_support(cudnnHandle_t h) {
        auto status = list_of_engine_configs.check_support(h);
        return status;
    }

    int64_t
    get_max_workspace_size();

    static error_t
    autotune_default_impl(Plans *plans,
                          cudnnHandle_t handle,
                          std::unordered_map<std::shared_ptr<Tensor_attributes>, void *> variants,
                          void *workspace,
                          void *) {
        auto &execution_plans = plans->get_engine_configs().get_execution_plans();

        // Create the variant pack for all the plans to use.
        std::vector<int64_t> uids;
        std::vector<void *> ptrs;
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

        for (auto plan : plans->get_engine_configs().get_execution_plans()) {
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
        error_t(Plans *, cudnnHandle_t, std::unordered_map<std::shared_ptr<Tensor_attributes>, void *>, void *, void *)>
        autotune_impl = &Plans::autotune_default_impl;

    error_t
    autotune(cudnnHandle_t handle,
             std::unordered_map<std::shared_ptr<Tensor_attributes>, void *> variants,
             void *workspace,
             void *user_impl = nullptr) {
        auto error = autotune_impl(this, handle, variants, workspace, user_impl);
        return error;
    }
};

inline Plans &
Plans::filter_out_behavior_notes(std::vector<cudnnBackendBehaviorNote_t> const &notes) {
    // TODO: The error returned is not propagate to user.
    // Should the return value be changed to error_code_t too?
    auto status = list_of_engine_configs.filter_out_behavior_notes(notes);
    if (status.is_bad()) {
        getLogger() << "[cudnn_frontend] ERROR: Filtering by behavioural notes failed." << std::endl;
    }
    return *this;
}

inline Plans &
Plans::filter_out_numeric_notes(std::vector<cudnnBackendNumericalNote_t> const &notes) {
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
    auto status = list_of_engine_configs.build_all_plans(h);
    return status;
}

inline int64_t
Plans::get_max_workspace_size() {
    return list_of_engine_configs.get_max_workspace_size();
}

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

    std::array<std::shared_ptr<Tensor_attributes>, 2> scaled_dot_product_flash_attention(
        std::shared_ptr<Tensor_attributes>,
        std::shared_ptr<Tensor_attributes>,
        std::shared_ptr<Tensor_attributes>,
        Scaled_dot_product_flash_attention_attributes);

    Plans
    get_execution_plan_list(HeurMode_t mode);

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
    get_engine_configs(Execution_plan_list &plan_list) {
        getLogger() << "[cudnn_frontend] INFO: Extracting engine configs." << std::endl;

        if (engine_configs.size() == 0) {
            return {error_code_t::HEURISTIC_QUERY_FAILED, "No valid engine configs for mode_a"};
        }
        plan_list.set_tag(engine_configs.begin()->first);
        plan_list.set_engine_configs(engine_configs.begin()->second);

        getLogger() << "[cudnn_frontend] INFO: Querying engine config properties for cfg_count "
                    << engine_configs.begin()->second.size() << std::endl;
        CHECK_CUDNN_FRONTEND_ERROR(plan_list.query_properties());

        return {error_code_t::OK, ""};
    }

    error_t
    createOperationGraphs(cudnnHandle_t handle) override final {
        getLogger() << "Operation Graph has " << operations.size() << " operations." << std::endl;

        auto status = create_cudnn_operation_graphs(handle);
        if (status.is_bad()) {
            getLogger() << "[cudnn_frontend] ERROR: " << status.get_code()
                        << " Failed to create execution plans for graph partitioning in FlatNode." << std::endl;
            return status;
        }

        return {error_code_t::OK, ""};
    }
};

inline Plans
Graph::get_execution_plan_list(HeurMode_t mode) {
    Plans plan_list;
    // TODO: The error returned is not propagate to user.
    // Should the return value be changed to error_code_t too?

    auto status = query_heuristics(mode);
    if (status.is_bad()) {
        getLogger() << "[cudnn_frontend] ERROR: Failed to build." << std::endl;
        return plan_list;
    }

    status = get_engine_configs(plan_list.list_of_engine_configs);
    if (status.is_bad()) {
        getLogger() << "[cudnn_frontend] ERROR: Querying engine configs failed." << std::endl;
    }
    return plan_list;
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
    Stats = options.outputs.Stats = output_tensor(options.get_name() + "::Stats");

    // Set inputs
    options.inputs.Q = q;
    options.inputs.K = k;
    options.inputs.V = v;

    sub_nodes.emplace_back(std::make_unique<ScaledDotProductFlashAttentionNode>(std::move(options), context));

    return {O, Stats};
}

}  // namespace cudnn_frontend::graph