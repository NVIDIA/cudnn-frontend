# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SM80 RoPE declarations decline before compiling any staging recipe."""

from dataclasses import replace

import pytest
import torch

from test_sdpa_native_admission_sm80 import _api, _descs, declared_sm80  # noqa: F401

pytestmark = pytest.mark.L0


@pytest.mark.parametrize("direction", ["fwd", "bwd"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("wide_role", [None, "q", "o"])
def test_rope_declines_before_kernel_or_copy_compilation(direction, dtype, wide_role, monkeypatch, declared_sm80):
    from cudnn.sdpa.bwd import api_dsl as bwd
    from cudnn.sdpa.fwd import api_dsl as fwd

    descs = _descs(128, 128)
    for role in descs:
        if role != "stats":
            descs[role] = replace(descs[role], dtype=dtype)
    if wide_role is not None:
        descs[wide_role] = replace(descs[wide_role], stride=(2**32 + 16, *descs[wide_role].stride[1:]))

    def forbidden(*args, **kwargs):
        pytest.fail("unsupported RoPE declaration reached kernel compilation")

    monkeypatch.setattr(fwd, "_sm80_load_kernel_module", forbidden)
    monkeypatch.setattr(bwd, "_load_sm80_bwd_module", forbidden)
    kwargs = dict(rope_max_s=32)
    if direction == "bwd":
        kwargs["has_rope"] = True
    api = _api(direction, descs, **kwargs)
    with pytest.raises(NotImplementedError, match="RoPE"):
        api.compile()
    assert not api._is_supported
