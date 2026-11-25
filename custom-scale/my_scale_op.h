/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <inttypes.h>
#include <cudnn.h>

void
run_my_scale_op(int64_t* tensorDim, cudnnDataType_t dataType, void* devPtrX, void* devPtrScale, void* devPtrY);
