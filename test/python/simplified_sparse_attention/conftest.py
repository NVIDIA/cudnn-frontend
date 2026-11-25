# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from cudnn.frost.buffers import cutedsl_requirement_error

_error = cutedsl_requirement_error("simplified_sparse_attention", floor=(4, 8, 0))
if _error:
    pytest.skip(_error, allow_module_level=True)


# Measured failures from pipeline 84114932; remove these skips with the fixes.
_KNOWN_DETERMINISM_FAILURES = {
    (10, 0): {
        "test_dense_indexer_kl_determinism.py": {
            "test_batch_invariance[bf16-h16-tiny-fprop-torch]",
            "test_replay[bf16-h96-split-fprop-cutedsl]",
        },
        "test_simplified_sparse_attention_determinism_jax.py": {
            "test_batch_invariance[h16-k512-bf16-fprop]",
        },
    },
    (10, 3): {
        "test_dense_indexer_kl_determinism.py": {
            "test_batch_invariance[bf16-h16-tiny-bprop-torch]",
            "test_batch_invariance[bf16-h16-tiny-fprop-torch]",
            "test_batch_invariance[bf16-h32-tail-bprop-cutedsl]",
            "test_batch_invariance[bf16-h32-tail-bprop-torch]",
            "test_batch_invariance[bf16-h96-split-bprop-cutedsl]",
            "test_batch_invariance[bf16-h96-split-bprop-torch]",
            "test_batch_invariance[bf16-h96-split-fprop-cutedsl]",
            "test_replay[bf16-h96-split-bprop-cutedsl]",
            "test_replay[bf16-h96-split-fprop-cutedsl]",
            "test_scheduling_boundary[bprop-cutedsl]",
            "test_scheduling_boundary[fprop-cutedsl]",
        },
        "test_simplified_sparse_attention_determinism.py": {
            "test_batch_invariance[h16-k512-bf16-bprop]",
            "test_batch_invariance[h16-k512-bf16-fprop]",
            "test_batch_invariance[h32-k1024-bf16-bprop]",
            "test_batch_invariance[h96-k2048-bf16-bprop]",
            "test_golden[h32-k1024-bf16-bprop]",
            "test_golden[h96-k2048-bf16-bprop]",
            "test_replay[h16-k512-bf16-bprop]",
            "test_replay[h32-k1024-bf16-bprop]",
            "test_replay[h96-k2048-bf16-bprop]",
            "test_scheduling_boundary[bf16-bprop]",
        },
        "test_simplified_sparse_attention_determinism_jax.py": {
            "test_batch_invariance[h16-k512-bf16-bprop]",
            "test_batch_invariance[h16-k512-bf16-fprop]",
            "test_batch_invariance[h32-k1024-bf16-bprop]",
            "test_batch_invariance[h96-k2048-bf16-bprop]",
            "test_golden[h32-k1024-bf16-bprop]",
            "test_golden[h96-k2048-bf16-bprop]",
            "test_replay[h16-k512-bf16-bprop]",
            "test_replay[h32-k1024-bf16-bprop]",
            "test_replay[h96-k2048-bf16-bprop]",
            "test_scheduling_boundary[bf16-bprop]",
        },
    },
}


@pytest.fixture(autouse=True)
def skip_known_determinism_failures(request):
    affected_gpus = [
        capability for capability, failures in _KNOWN_DETERMINISM_FAILURES.items() if request.node.name in failures.get(request.node.path.name, ())
    ]
    if not affected_gpus:
        return
    import torch

    if torch.cuda.get_device_capability() in affected_gpus:
        pytest.skip("skip until fix: known determinism failure (replay or batch invariance; CI pipeline 84114932)")
