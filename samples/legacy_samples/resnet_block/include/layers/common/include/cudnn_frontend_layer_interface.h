#ifndef __CUDNN__FRONTEND_LAYER_INTERFACE__
#define __CUDNN__FRONTEND_LAYER_INTERFACE__
#endif

#pragma once

#include "layers/common/cudnn_frontend_resnet_block_helpers.h"
#include "cudnn_frontend.h"

namespace cudnn_frontend {

/**
 * @brief A Problem Descriptor builder class to configure and and build problme descriptors such as convolution
 * descriptors, pointwise descriptors, etc. This class is used within the building blocks to configure and build problem
 * descriptors.
 *
 */
class ProblemDescBuilder {
   public:
    /**
     * Configure the math precision for the descriptor
     *
     * @param mathPrec The math precision to use. Uses cudnnDataType_t enum (e.g. CUDNN_DATA_FLOAT)
     */
    void
    configureMathPrec(const cudnnDataType_t &mathPrec) {
        mathPrec_ = mathPrec;
    };

    /**
     * Configure the pointwise mode for the descriptor. This could be ADD, MUL, RELU, etc.
     *
     * @param mode A cudnnPointWiseMode_t enum (e.g. CUDNN_POINTWISE_ADD)
     */
    void
    configurePointWiseMode(const cudnnPointwiseMode_t &mode) {
        pwMode_ = mode;
    };

    /**
     * @brief Builds the matmul descriptor. All is needed is the math precision configured. If
     * NV_CUDNN_DISABLE_EXCEPTION is not defined, it throws an exception if the problem descriptor failed to build.
     * Otherwise, it returns the descriptor with a bad status. If the problem descriptor successfully builds, it returns
     * normally.
     *
     * @return MatMulDesc A built matmul descriptor.
     */
    MatMulDesc
    buildMatmul() {
#ifndef NV_CUDNN_DISABLE_EXCEPTION
        try {
#endif

            return MatMulDescBuilder().setMathPrecision(mathPrec_).build();

#ifndef NV_CUDNN_DISABLE_EXCEPTION
        } catch (cudnn_frontend::cudnnException &e) {
            std::string error_message =
                std::string("Error in creating matmul problem descriptor. Error message: ") + e.what();
            throw cudnnException(error_message.c_str(), e.getCudnnStatus());
        }
#endif
    };

    /**
     * @brief Builds a convolution descriptor. If NV_CUDNN_DISABLE_EXCEPTION is not defined, it throws an exception if
     * the problem descriptor failed to build. Otherwise, it returns the descriptor with a bad status. If the problem
     * descriptor successfully builds, it returns normally.
     *
     * @param convParams A struct containing the convolution parameters.
     * @return ConvDesc_v8 Returns a built convolution descriptor.
     */
    ConvDesc_v8
    buildConv(const ConvParams &convParams) {
#ifndef NV_CUDNN_DISABLE_EXCEPTION
        try {
#endif
            return ConvDescBuilder_v8()
                .setDataType(mathPrec_)
                .setMathMode(convParams.convMode)
                .setNDims(convParams.nConvDims)
                .setStrides(convParams.nConvDims, convParams.convStride)
                .setPrePadding(convParams.nConvDims, convParams.convPad)
                .setPostPadding(convParams.nConvDims, convParams.convPad)
                .setDilation(convParams.nConvDims, convParams.convDilation)
                .build();

#ifndef NV_CUDNN_DISABLE_EXCEPTION
        } catch (cudnn_frontend::cudnnException &e) {
            std::string error_message =
                std::string("Error in creating convolution problem descriptor. Error message: ") + e.what();
            throw cudnnException(error_message.c_str(), e.getCudnnStatus());
        }
#endif
    };

    /**
     * @brief Builds a resampling descriptor. If NV_CUDNN_DISABLE_EXCEPTION is not defined, it throws an exception if
     * the problem descriptor failed to build. Otherwise, it returns the descriptor with a bad status. If the problem
     * descriptor successfully builds, it returns normally.
     *
     * @param convParams A struct containing the pooling parameters.
     * @return ResampleDesc_v8 Returns a built resampling (pooling) descriptor.
     */
    ResampleDesc_v8
    buildResample(PoolingParams &poolingParams) {
        int64_t nSpatialDims = poolingParams.nbSpatialDims;
        return ResampleDescBuilder_v8()
            .setComputeType(mathPrec_)
            .setSpatialDim(nSpatialDims, poolingParams.windowDim)
            .setNanPropagation(poolingParams.nanOpt)
            .setResampleMode(poolingParams.mode)
            .setPaddingMode(poolingParams.padding_mode)
            .setSpatialStride(nSpatialDims, poolingParams.stride)
            .setPrePadding(nSpatialDims, poolingParams.prePadding)
            .setPostPadding(nSpatialDims, poolingParams.postPadding)
            .build();
    };

    /**
     * @brief Builds a pointwise descriptor. If NV_CUDNN_DISABLE_EXCEPTION is not defined, it throws an exception if the
     * problem descriptor failed to build. Otherwise, it returns the descriptor with a bad status. If the problem
     * descriptor successfully builds, it returns normally.
     *
     * @return PointWiseDesc A built pointwise descriptor.
     */
    PointWiseDesc
    buildPointwiseDesc() {
#ifndef NV_CUDNN_DISABLE_EXCEPTION
        try {
#endif

            return PointWiseDescBuilder().setMode(pwMode_).setMathPrecision(mathPrec_).build();
#ifndef NV_CUDNN_DISABLE_EXCEPTION
        } catch (cudnn_frontend::cudnnException &e) {
            std::string error_message =
                std::string("Error in creating bias problem descriptor. Error message: ") + e.what();
            throw cudnnException(error_message.c_str(), e.getCudnnStatus());
        }
#endif
    };

   private:
    // Math precision defaults to float
    cudnnDataType_t mathPrec_ = CUDNN_DATA_FLOAT;
    // Pointwise mode defaults to ADD
    cudnnPointwiseMode_t pwMode_ = CUDNN_POINTWISE_ADD;
};

/**
 * @brief A struct for a node in a subgraph. A node consists of:
 * - op_name: The name of the operation.
 * - type: A backened descriptor type
 * - problem_name: The name of the problem for problem descriptor
 * - edges: A vector of strings containing names of tensors that represent the edges of the node.
 */
struct Node {
    std::string op_name;
    cudnnBackendDescriptorType_t type;
    std::string problem_name;
    std::vector<std::string> edges;
};

// A subgraph is just a vector of nodes
using SubGraph = std::vector<Node>;

/**
 * @brief A parent class for building blocks. A block is a collection of operations/operation graphs that can be
 * executed with a single function call. The IBlock class contain methods to build the operation graph, create execution
 * plans, get workspace size, and execute the block. It also contains maps to store operation graphs, tensors, variant
 * packs, etc.
 *
 */
class IBlock {
   public:
    enum Direction { FORWARD, BACKWARD, BACKWARD_MIXED_PRECISION, FORWARD_INFERENCE };

   protected:
    // List of Tensors in the Layer
    std::unordered_map<std::string, std::shared_ptr<cudnn_frontend::Tensor>> tensor_map;

    // Built List of execution plans
    std::unordered_map<std::string, ExecutionPlan_v8> execution_plans;

    // Built List of variant packs for execution plans
    std::unordered_map<std::string, VariantPack_v8> variant_packs;
    std::unordered_map<std::string, SubGraph> sub_graphs;
    std::queue<std::string> sub_graph_name_queue;
    std::unordered_map<std::string, OperationGraph> op_graphs;

    // Name of the block
    std::string blockName = "";

    // Status for the block
    cudnnStatus_t status_ = CUDNN_STATUS_SUCCESS;
    // Error message for any CUDNN failures. Defaults to "SUCCESS" to check against.
    std::string error_message = "SUCCESS";

    /////////////////////////////////////////////////////////////////////////
    // Workspace
    /////////////////////////////////////////////////////////////////////////
    // Each plan inside the block as its own workspac.
    // The max value of workspace required by each plan is requested from the use.
    size_t plan_execution_workspace_size   = 0;
    void *plan_execution_workspace_pointer = nullptr;

    // Additionally, intermediate tensors also need to be stored in the workspace.
    // This is because they are shared between plans but need to be passed as tensors to the kernel.
    // The maximum size of intermidiate tensor required by each plan is requested from the user.
    size_t intermediate_tensor_workspace_size   = 0;
    void *intermediate_tensor_workspace_pointer = nullptr;

    /**
     * @brief Virtual method to build the operation graph. Classes that inherit from this class need to
     * implement this method
     *
     * @param handle a cudnn handle for the block
     * @param sub_graph Subgraphs to create the operation graph with
     * @return cudnnStatus_t Status of the operation graph build
     */
    virtual cudnnStatus_t
    buildOpGraph(cudnnHandle_t &handle, SubGraph const &sub_graph, const std::string &graph_name) {
        (void)handle;
        (void)sub_graph;
        (void)graph_name;
        return CUDNN_STATUS_SUCCESS;
    };

    /**
     * @brief Virtual implemented method to get the max workspace size for the block. Stores the result in
     * the workspace_size member variable.
     */
    virtual cudnnStatus_t
    calculatePlanExecutionWorkspaceSize() {
        getLogger() << "[cudnn_frontend] " << "INFO: Calculating Plan execution workspace size" << std::endl;
        int64_t max_workspace_size = -1;

        for (auto const &execution_plan : execution_plans) {
            max_workspace_size = std::max(max_workspace_size, execution_plan.second.getWorkspaceSize());
        }
        if (max_workspace_size == -1) {
            logErrorMessage(CUDNN_STATUS_BAD_PARAM, blockName, "workspace ptr", "workspace ptr", error_message);
            return CUDNN_STATUS_BAD_PARAM;
        }

        getLogger() << "[cudnn_frontend] " << "INFO: Plan execution Workspace size is " << max_workspace_size
                    << std::endl;
        plan_execution_workspace_size = max_workspace_size;
        return CUDNN_STATUS_SUCCESS;
    };

    /**
     * @brief Builds the execution plan from the operation graph using the backened heuristics.
     *
     * @param graph_name The name of the operation graph to get the execution plan from
     * @param opGraph Operation graph to build the execution plan with
     * @param handle A cudnn handle
     * @return ExecutionPlan_v8 Returns a build execution plan, or an error if not supported.
     */
    ExecutionPlan_v8
    getPlanFromHeuristics(const std::string &graph_name, OperationGraph_v8 &opGraph, cudnnHandle_t &handle) {
        getLogger() << "[cudnn_frontend] " << "INFO: Getting plan from heuristics" << std::endl;

        cudnn_frontend::EngineConfigList filtered_configs;
        auto statuses = cudnn_frontend::get_heuristics_list<2>(
            {"heuristics_instant", "heuristics_fallback"}, opGraph, AllowAll, filtered_configs, true);

        getLogger() << "[cudnn_frontend] " << "get_heuristics_list Statuses: ";
        for (size_t i = 0; i < statuses.size(); i++) {
            getLogger() << "[cudnn_frontend] " << cudnn_frontend::to_string(statuses[i]) << " ";
        }

        getLogger() << "[cudnn_frontend] " << std::endl;
        getLogger() << "[cudnn_frontend] " << "Filter config list has " << filtered_configs.size() << " configurations "
                    << std::endl;

        for (size_t i = 0; i < filtered_configs.size(); i++) {
            getLogger() << "[cudnn_frontend] " << "Trying config: " << i << std::endl;

#ifndef NV_CUDNN_DISABLE_EXCEPTION
            try {
#endif

                auto plan = cudnn_frontend::ExecutionPlanBuilder()
                                .setHandle(handle)
                                .setEngineConfig(filtered_configs[i], opGraph.getTag())
                                .build();

                // The last config didn't work (E.g. all configs didn't work)
                if (i == filtered_configs.size() - 1 && plan.get_status() != CUDNN_STATUS_SUCCESS) {
                    logErrorMessage(CUDNN_STATUS_EXECUTION_FAILED,
                                    blockName,
                                    "execution plan descriptor",
                                    graph_name,
                                    error_message,
                                    plan.get_error());
                    return plan;

                } else if (plan.get_status() != CUDNN_STATUS_SUCCESS) {
                    continue;
                }
                getLogger() << "[cudnn_frontend] " << "Config " << i << " succeeded! Plan has built!" << std::endl;
                getLogger() << "[cudnn_frontend] " << plan.describe() << std::endl;
                getLogger() << "[cudnn_frontend] "
                            << "========================================================" << std::endl;
                return plan;

#ifndef NV_CUDNN_DISABLE_EXCEPTION
            } catch (cudnn_frontend::cudnnException &e) {
                // The last config didn't work (E.g. all configs didn't work)
                if (i == filtered_configs.size() - 1) {
                    logErrorMessage(e.getCudnnStatus(),
                                    blockName,
                                    "execution plan descriptor",
                                    graph_name,
                                    error_message,
                                    e.what());
                    throw cudnnException(error_message.c_str(), e.getCudnnStatus());
                }
                continue;
            }
#endif
        }

        // Dummy plan
        return cudnn_frontend::ExecutionPlanBuilder()
            .setHandle(handle)
            .setEngineConfig(filtered_configs[1], opGraph.getTag())
            .build();
    }

    /**
     * @brief Adds a tensor to the tensor map
     *
     * @param tensor_name A name for the tensor
     * @param tensor The actual cudnn tensor object
     */
    void
    addTensor(const std::string &tensor_name, std::shared_ptr<Tensor> const tensor) {
        tensor_map.emplace(tensor_name, tensor);
    }

    Direction direction;

   public:
    /**
     * @brief Destroy the IBlock object and free any resources allocated
     *
     */
    virtual ~IBlock() = default;
    IBlock(Direction d) : direction(d){};

    Direction
    getDirection() {
        return direction;
    }
    /**
     * @brief Virtual createVariantPacks method. Create the necessary variant packs for the block pass. Done internally
     * for the user. All the user needs to pass in is A ptr to a ResidualBlockDevPtrStore object which contains the
     * necessary pointers to the device memory. See resnet_sample.cpp and resnet_test_list.cpp for examples.
     *
     * @param devPtrStore A ptr to A ptr to a ResidualBlockDevPtrStore object which contains the necessary pointers to
     * the device memory (see cudnn_frontend_residual_block_dev_ptr_store.h for details).
     * @return cudnnStatus_t Returns CUDNN_STATUS_SUCCESS if all tensors were successfully created, otherwise, if
     * NV_CUDNN_DISABLE_EXCEPTION is not defined, it returns a bad status with an error message. Otherwise, it throws an
     * error for which tensor failed to build. Use getErrorMessage() to see error message.
     */
    virtual cudnnStatus_t
    createVariantPacks(ResidualBlockDevPtrStore *devPtrStore) {
        (void)devPtrStore;
        return CUDNN_STATUS_SUCCESS;
    }

    // /**
    //  * @brief Virtual method. Overload for the classifier block. Create the necessary variant packs for the block
    //  pass. Done internally for the user. All the user needs to pass in is A  ptr to a ClassifierBlockDevPtrStore
    //  object which contains the necessary pointers to the device memory. See resnet_sample.cpp and
    //  resnet_test_list.cpp for examples.
    //  *
    //  * @param devPtrStore A ptr to A  ptr to a ClassifierBlockDevPtrStore object which contains the necessary
    //  pointers to the device memory (see cudnn_frontend_classifier_block_dev_ptr_store.h for details).
    //  * @return cudnnStatus_t Returns CUDNN_STATUS_SUCCESS if all tensors were successfully created, otherwise, if
    //  NV_CUDNN_DISABLE_EXCEPTION is not defined, it returns a bad status with an error message. Otherwise, it throws
    //  an error for which tensor failed to build. Use getErrorMessage() to see error message.
    //  */
    // virtual
    // cudnnStatus_t createVariantPacks(ClassifierBlockDevPtrStore* devPtrStore) {return CUDNN_STATUS_SUCCESS;};

    // /**
    //  * @brief Virtual method. Overload for the GPT block. Create the necessary variant packs for the block pass. Done
    //  internally for the user. All the user needs to pass in is a GPTDevPtrStore object which contains the necessary
    //  pointers to the device memory. See resnet_sample.cpp and resnet_test_list.cpp for examples.
    //  *
    //  * @param devPtrStore A ptr to a GPTDevPtrStore object which contains the necessary pointers to the device memory
    //  (see cudnn_frontend_gpt_dev_ptr_store.h for details).
    //  * @return cudnnStatus_t Returns CUDNN_STATUS_SUCCESS if all tensors were successfully created, otherwise, if
    //  NV_CUDNN_DISABLE_EXCEPTION is not defined, it returns a bad status with an error message. Otherwise, it throws
    //  an error for which tensor failed to build. Use getErrorMessage() to see error message.
    //  */
    // virtual
    // cudnnStatus_t createVariantPacks(GPTDevPtrStore* devPtrStore) {return CUDNN_STATUS_SUCCESS;};

    /**
     * @brief Returns the error message of the block
     *
     * @return Error message of block
     */
    const std::string &
    getErrorMessage() const {
        return error_message;
    }

    /**
     * @brief Returns the current status of the block
     *
     * @return Current status of block
     */
    const cudnnStatus_t &
    getStatus() const {
        return status_;
    }

    /**
     * @brief Sets the workspace ptr for the block based on the workspace size
     *
     * @return cudnnStatus_t A cudnnStatus_t value indicating whether or not the allocation was successful. If it built,
     * it returns CUDNN_STATUS_SUCCESS. Otherwise, it may return CUDNN_STATUS_BAD_PARAM, e.getCudnnStatus(), etc.
     */
    cudnnStatus_t
    setWorkspace(void *ptr) {
        plan_execution_workspace_pointer      = ptr;
        intermediate_tensor_workspace_pointer = (char *)ptr + plan_execution_workspace_size;
        return CUDNN_STATUS_SUCCESS;
    }

    /**
     * @brief Adds a subgraph to the subgraph map
     *
     * @param subgraph_name A name for the subgraph
     * @param sub_graph The subgraph to add
     */
    void
    addSubOpGraph(const std::string &subgraph_name, SubGraph const &sub_graph) {
        sub_graph_name_queue.push(subgraph_name);
        sub_graphs.emplace(subgraph_name, sub_graph);
    }

    /**
     * @brief A method to clone a block
     *
     */
    void
    clone() {}

    /**
     * @brief Builds the operation graph and execution plan for the block.
     *
     * @param handle A cudnn handle
     */
    cudnnStatus_t
    build(cudnnHandle_t handle) {
        getLogger() << "[cudnn_frontend]" << "INFO: Building " << sub_graphs.size() << " subgraphs. " << std::endl;

        for (auto &sub_graph : sub_graphs) {
            auto opGraphStatus = buildOpGraph(handle, sub_graph.second, sub_graph.first);
            if (opGraphStatus != CUDNN_STATUS_SUCCESS) {
                getLogger() << "[cudnn_frontend] " << "[ERROR] building operation graph for graph " << sub_graph.first
                            << ". Error message: " << error_message << std::endl;
                return opGraphStatus;
            }

            auto plan = getPlanFromHeuristics(sub_graph.first, op_graphs.at(sub_graph.first), handle);
            if (plan.get_status() != CUDNN_STATUS_SUCCESS) {
                getLogger() << "[cudnn_frontend] " << "[ERROR] building execution plan for graph " << sub_graph.first
                            << ". Error message: " << error_message << std::endl;
                return plan.get_status();
            }

            execution_plans.emplace(sub_graph.first, std::move(plan));
        }

        // Calculate the maximum workspace size after all the execution plans are built
        status_ = calculatePlanExecutionWorkspaceSize();
        if (status_ != CUDNN_STATUS_SUCCESS) {
            getLogger() << "[cudnn_frontend] " << error_message << std::endl;
            return status_;
        }

        return CUDNN_STATUS_SUCCESS;
    }

    /**
     * @brief Updates the variant pack for the plan, or creates a new variant pack for an execution plan if
     * the plan does not yet contain a variant pack
     *
     * @param name The name of the execution plan/operation graph
     * @param variant_pack The variant pack for the plan
     */
    void
    updateVariantPackforPlan(const std::string &name, VariantPack_v8 &variant_pack) {
        auto it = variant_packs.find(name);
        if (it != variant_packs.end()) {
            it->second = std::move(variant_pack);
        } else {
            getLogger() << "[cudnn_frontend] " << "INFO: Could not find plan " << name
                        << " in variant pack map. Creating new variant pack for plan." << std::endl;
            variant_packs.emplace(name, std::move(variant_pack));
        }
    }

    /**
     * A getter for the workspace size
     *
     * @return size_t the workspace size
     */
    size_t
    getWorkspaceSize() const {
        return plan_execution_workspace_size + intermediate_tensor_workspace_size;
    }

    /**
     * @brief Executes the block. Goes through each execution plan and executes it with the corresponding
     * variant pack.
     *
     * @param handle The cudnn handle
     * @return cudnnStatus_t A cudnnStatus_t value indicating whether or not the executions plan executed successfully.
     * If so, it returns a CUDNN_STATUS_SUCCESS. Otherwise, if NV_CUDNN_DISABLE_EXCEPTION is enabled, it returns a bad
     * status. If not, it throws an exception.
     */
    virtual cudnnStatus_t
    execute(cudnnHandle_t &handle) {
        auto copy_sub_graph_queue = sub_graph_name_queue;
        if (copy_sub_graph_queue.size() == 0) {
            return CUDNN_STATUS_BAD_PARAM;
        }
        while (!copy_sub_graph_queue.empty()) {
            // Get the string name of the plan
            std::string plan_name = copy_sub_graph_queue.front();
            copy_sub_graph_queue.pop();
            cudnnStatus_t status = executePlan(handle, plan_name);
            if (status != CUDNN_STATUS_SUCCESS) {
                getLogger() << "[cudnn_frontend] " << error_message << std::endl;
                return status;
            }
        }
        return CUDNN_STATUS_SUCCESS;
    }

    /**
     * @brief Executes a specific plan given a specific plan name.
     * variant pack.
     *
     * @param handle The cudnn handle
     * @param plan The name of the plan to execute
     * @return cudnnStatus_t A cudnnStatus_t value indicating whether or not the executions plan executed successfully.
     * If so, it returns a CUDNN_STATUS_SUCCESS. Otherwise, if NV_CUDNN_DISABLE_EXCEPTION is enabled, it returns a bad
     * status. If not, it throws an exception.
     */
    cudnnStatus_t
    executePlan(cudnnHandle_t &handle, const std::string &plan) {
        getLogger() << "[cudnn_frontend] " << "INFO: Executing plan " << plan << std::endl;

        if (execution_plans.find(plan) == execution_plans.end()) {
            error_message = "[ERROR]: Could not find plan " + plan;
            return CUDNN_STATUS_EXECUTION_FAILED;
        }

        // Get the plan
        ExecutionPlan_v8 execution_plan = execution_plans.at(plan);

        // Check if there is a variant pack for the plan
        if (variant_packs.find(plan) == variant_packs.end()) {
            error_message = "[ERROR]: No variant pack found for plan: " + plan;
            return CUDNN_STATUS_EXECUTION_FAILED;
        }

        // Get the variant pack
        VariantPack_v8 variant_pack = std::move(variant_packs.at(plan));

        // Calls cudnnBackenedExecute() on the execution plan and the variant pack and log the time
        cudnnStatus_t status = executePlanOnce(handle, execution_plan, variant_pack, plan);

        if (status != CUDNN_STATUS_SUCCESS) {
            logErrorMessage(status, blockName, "execution plan", plan, error_message);
#ifndef NV_CUDNN_DISABLE_EXCEPTION
            throw cudnnException(error_message.c_str(), status);
#endif
            return status;
        }
        getLogger() << "[cudnn_frontend] " << "INFO: Plan " << plan << " executed successfully" << std::endl;

        return CUDNN_STATUS_SUCCESS;
    }

    /**
     * @brief Executes a cudnn execution plan and logs the time it took to execute
     *
     * @param handle a cudnn handle for the block
     * @param plan An execution plan to execute
     * @param variantPack A variant pack for the execution plan
     * @param planName The name of the execution plan for logging
     * @return cudnnStatus_t A cudnnStatus_t value indicating whether or not the the plan succeeded in executing
     */
    cudnnStatus_t
    executePlanAndLogTime(cudnnHandle_t handle,
                          ExecutionPlan_v8 &plan,
                          const VariantPack_v8 &variantPack,
                          const std::string &planName) {
        (void)planName;
        int maxIterCount = 100;  // Run till stable

        const float threshhold = 0.95f;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaDeviceSynchronize();

        cudaStream_t stream = nullptr;
        ::cudnnGetStream(handle, &stream);

        float time_ms       = 0.0f;
        float final_time_ms = 0.0f;
        float min_time_ms   = std::numeric_limits<float>::max();

        // Warm-up run
        auto status = cudnnBackendExecute(handle, plan.get_raw_desc(), variantPack.get_raw_desc());
        if (status != CUDNN_STATUS_SUCCESS) {
            logErrorMessage(status, blockName, "execution plan", planName, error_message);
#ifndef NV_CUDNN_DISABLE_EXCEPTION
            throw cudnnException(error_message.c_str(), status);
#endif
            return status;
        }
        cudaDeviceSynchronize();

        for (int i = 0; i < maxIterCount; i++) {
            cudaEventRecord(start, stream);

            cudnnBackendExecute(handle, plan.get_raw_desc(), variantPack.get_raw_desc());

            cudaEventRecord(stop, stream);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&time_ms, start, stop);

            if (maxIterCount > 3) {
                final_time_ms = std::min(min_time_ms, time_ms);
                if (time_ms / min_time_ms < threshhold) {
                    min_time_ms = final_time_ms;
                } else {
                    break;
                }
            } else {
                final_time_ms = i == (maxIterCount / 2) ? time_ms : final_time_ms;
            }
        }
        getLogger() << "[cudnn_frontend] Plan " << planName << " took " << std::setw(10) << final_time_ms << " "
                    << "milliseconds" << std::endl;
        getLogger() << "[cudnn_frontend] Plan tag " << plan.getTag() << std::endl;
        return CUDNN_STATUS_SUCCESS;
    }

    cudnnStatus_t
    executePlanOnce(cudnnHandle_t handle,
                    ExecutionPlan_v8 &plan,
                    const VariantPack_v8 &variantPack,
                    const std::string &planName) {
        (void)planName;
        cudnnBackendExecute(handle, plan.get_raw_desc(), variantPack.get_raw_desc());
        getLogger() << "[cudnn_frontend] Plan tag " << plan.getTag() << std::endl;

        return CUDNN_STATUS_SUCCESS;
    }

    /* Future */
    void
    flatten() {}
};

}  // namespace cudnn_frontend