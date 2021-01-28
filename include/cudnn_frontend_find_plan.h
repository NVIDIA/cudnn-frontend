/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#pragma once

#include <cudnn_frontend_EngineConfigGenerator.h>
#include <map>

namespace cudnn_frontend {

/// Sorts the execution plans by their run time.
/// The run time of plan may not trivial and hence we
/// run it multiple times till we get a stable value.
/// We have an additional dry-run which helps stabilize the
/// time further.
template <CudnnFindSamplingTechnique samplingTechnique>
auto
time_sorted_plan(cudnnHandle_t handle, executionPlans_t plans, VariantPack &variantPack) -> executionOptions_t {
    executionOptions_t time_sorted_plans;
    std::map<float, ExecutionPlan &> timed_execution_plans;

    const int maxIterCount =
        (samplingTechnique == CudnnFindSamplingTechnique::CUDNN_FIND_SAMPLE_ONCE)
            ? 1
            : (samplingTechnique == CudnnFindSamplingTechnique::CUDNN_FIND_SAMPLE_MEDIAN_OF_THREE) ? 3 : 100;
    const float threshhold = 0.95f;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaDeviceSynchronize();

    for (auto &plan : plans) {
        float time_ms       = 0.0f;
        float final_time_ms = 0.0f;
        float min_time_ms   = std::numeric_limits<float>::max();

        // Warm-up run
        ::cudnnBackendExecute(handle, plan.get_raw_desc(), variantPack.get_raw_desc());
        cudaDeviceSynchronize();

        for (int i = 0; i < maxIterCount; i++) {
            cudaEventRecord(start);

            ::cudnnBackendExecute(handle, plan.get_raw_desc(), variantPack.get_raw_desc());

            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&time_ms, start, stop);

            if (samplingTechnique == CudnnFindSamplingTechnique::CUDNN_FIND_SAMPLE_TILL_STABLE) {
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
        timed_execution_plans.insert({final_time_ms, plan});
    }
    std::transform(
        timed_execution_plans.begin(),
        timed_execution_plans.end(),
        std::back_inserter(time_sorted_plans),
        [](const std::map<float, cudnn_frontend::ExecutionPlan &>::value_type &pair) -> struct executionOption {
            return {std::move(pair.second), pair.first};
        });

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return time_sorted_plans;
}

template <CudnnFindSamplingTechnique samplingTechnique>
auto
EngineConfigGenerator::cudnnFindPlan(cudnnHandle_t handle,
                                     cudnn_frontend::OperationGraph &&opGraph,
                                     cudnn_frontend::VariantPack &variantPack,
                                     Predicate pred) -> executionOptions_t {
    /// Creating a set of execution plans that are supported.
    executionPlans_t plans;
    for (auto &engine_config : generate_engine_config(opGraph)) {
#ifndef NV_CUDNN_DISABLE_EXCEPTION
        try {
#endif
            plans.push_back(
                cudnn_frontend::ExecutionPlanBuilder().setHandle(handle).setEngineConfig(engine_config).build());
#ifndef NV_CUDNN_DISABLE_EXCEPTION
        } catch (cudnnException e) {
            continue;
        }
#endif
    }
    return time_sorted_plan<samplingTechnique>(handle, filter(pred, plans), variantPack);
}
}
