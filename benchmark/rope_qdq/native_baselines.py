# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Numerical baseline: unchanged FlashInfer RoPE sources with FP32 FTZ disabled.

The kernel remains FlashInfer's implementation and attribution. This authored
wrapper constructs a separate JIT module; it does not patch the default provider.
Default-provider results must remain visible alongside this numerical control.
"""

import hashlib
from pathlib import Path


def prepare():
    from flashinfer.jit.core import gen_jit_spec
    from flashinfer.jit.rope import gen_rope_module

    default = gen_rope_module()
    controlled = gen_jit_spec("rope_qdq_noftz_baseline", default.sources, extra_cuda_cflags=["--ftz=false"])
    assert controlled.extra_cuda_cflags == default.extra_cuda_cflags + ["--ftz=false"]
    assert list(controlled.sources) == list(default.sources)
    record = dict(
        attribution="FlashInfer kernel; our wrapper only appends --ftz=false to the JIT flags",
        original_cuda_flags=default.extra_cuda_cflags,
        controlled_cuda_flags=controlled.extra_cuda_cflags,
        sources={str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in controlled.sources},
    )
    module = controlled.build_and_load()

    def rotate(positions, query, empty_key, cache):
        q = query.view(query.shape[0], -1, 64)
        k = empty_key.view(empty_key.shape[0], -1, 64)
        module.apply_rope_pos_ids_cos_sin_cache(q, k, q, k, cache, positions, True)

    return rotate, record
