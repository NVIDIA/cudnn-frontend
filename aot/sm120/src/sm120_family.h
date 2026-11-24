// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

namespace cudnn_frontend::aot_sm120 {

constexpr bool
supports_device(int cc) {
    return cc == 120 || cc == 121;
}

}  // namespace cudnn_frontend::aot_sm120
