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

namespace cudnn_frontend {

auto
EngineConfigGenerator::cudnnGetPlan(cudnnHandle_t handle, OperationGraph & opGraph)
    -> executionPlans_t {
    // Creating a set of execution plans that are supported.
    executionPlans_t plans;
    for (auto& engine_config : generate_engine_config(opGraph)) {
#ifndef NV_CUDNN_DISABLE_EXCEPTION
        try {
#endif
            plans.push_back(ExecutionPlanBuilder().setHandle(handle).setEngineConfig(engine_config, opGraph.getTag()).build());
#ifndef NV_CUDNN_DISABLE_EXCEPTION
        } catch (cudnnException &e) {
            continue;
        }
#endif
    }
    return plans;
}

auto
EngineConfigGenerator::cudnnGetPlan(cudnnHandle_t handle, OperationGraph & opGraph, Predicate pred)
    -> executionPlans_t {
    // Creating a set of execution plans that are supported.
    executionPlans_t plans = cudnnGetPlan(handle, opGraph);
    return filter(pred, plans);
}
}
