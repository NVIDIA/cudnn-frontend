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

#include "cudnn_frontend_ConvDesc.h"
#include "cudnn_frontend_Heuristics.h"
#include "cudnn_frontend_Engine.h"
#include "cudnn_frontend_EngineConfig.h"
#include "cudnn_frontend_EngineFallbackList.h"
#include "cudnn_frontend_ExecutionPlan.h"
#include "cudnn_frontend_Filters.h"
#include "cudnn_frontend_Operation.h"
#include "cudnn_frontend_OperationGraph.h"
#include "cudnn_frontend_Tensor.h"
#include "cudnn_frontend_VariantPack.h"
#include "cudnn_frontend_PointWiseDesc.h"

namespace cudnn_frontend {
using Tensor                    = Tensor_v8;
using TensorBuilder             = TensorBuilder_v8;
using ConvDesc                  = ConvDesc_v8;
using ConvDescBuilder           = ConvDescBuilder_v8;
using PointWiseDescBuilder      = PointWiseDescBuilder_v8;
using PointWiseDesc             = PointWiseDesc_v8;
using Operation                 = Operation_v8;
using OperationBuilder          = OperationBuilder_v8;
using OperationGraph            = OperationGraph_v8;
using OperationGraphBuilder     = OperationGraphBuilder_v8;
using EngineHeuristicsBuilder   = EngineHeuristicsBuilder_v8;
using EngineHeuristics          = EngineHeuristics_v8;
using EngineBuilder             = EngineBuilder_v8;
using Engine                    = Engine_v8;
using EngineConfig              = EngineConfig_v8;
using EngineConfigBuilder       = EngineConfigBuilder_v8;
using ExecutionPlan             = ExecutionPlan_v8;
using ExecutionPlanBuilder      = ExecutionPlanBuilder_v8;
using VariantPack               = VariantPack_v8;
using VariantPackBuilder        = VariantPackBuilder_v8;
using EngineFallbackList        = EngineFallbackList_v8;
using EngineFallbackListBuilder = EngineFallbackListBuilder_v8;
}
