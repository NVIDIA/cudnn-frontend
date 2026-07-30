# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Intentionally empty. Keeping the package initializer free of imports makes the
# dependency layer's import order explicit and guarantees that importing a
# per-framework helper (e.g. ``cudnn._deps.torch_dep``) cannot pull in unrelated
# framework probes or create a cycle back into top-level ``cudnn``.
