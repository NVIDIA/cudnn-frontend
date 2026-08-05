/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "cudnn_frontend_EngineConfigGenerator.h"

namespace cudnn_frontend {

auto inline EngineConfigGenerator::cudnnGetPlan(cudnnHandle_t handle, OperationGraph& opGraph, size_t max_plans)
    -> executionPlans_t {
    // Creating a set of execution plans that are supported.
    executionPlans_t plans;
    for (auto& engine_config : generate_engine_config(opGraph)) {
#ifndef NV_CUDNN_DISABLE_EXCEPTION
        try {
#endif
            plans.push_back(
                ExecutionPlanBuilder().setHandle(handle).setEngineConfig(engine_config, opGraph.getTag()).build());
            CUDNN_FE_LOG_LABEL_ENDL("Added plan " << plans.back().getTag() << " "
                                                  << to_string(plans.back().get_status()));
#ifndef NV_CUDNN_DISABLE_EXCEPTION
        } catch (cudnnException& e) {
            CUDNN_FRONTEND_UNUSED(e);
            continue;
        }
#endif
        if (plans.size() >= max_plans) {
            break;
        }
    }
    return plans;
}

auto inline EngineConfigGenerator::cudnnGetPlan(cudnnHandle_t handle,
                                                OperationGraph& opGraph,
                                                Predicate pred,
                                                size_t max_plans) -> executionPlans_t {
    // Creating a set of execution plans that are supported.
    executionPlans_t plans = cudnnGetPlan(handle, opGraph, max_plans);
    return filter(pred, plans);
}
}  // namespace cudnn_frontend
