// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// The exported host wrappers report a failed TMA descriptor encode through cuda_dialect_assert_success, which the
// CuTeDSL runtime implements as abort(). While a guard is installed, the in-tree runtime shim instead longjmps back to
// it with the error. Only DSL-generated frames (no destructors) lie between the guard and the longjmp.
#pragma once

#include <csetjmp>
#include <cstdint>

struct AotAssertGuard {
    std::jmp_buf env;
    int err;
};

// Defined in aot_attention.cpp; nullptr outside a guarded call.
extern thread_local AotAssertGuard* g_aot_assert_guard;
