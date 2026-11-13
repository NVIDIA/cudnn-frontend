# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SM80 graph compilation when a different architecture is visible first."""

import pytest
import torch

from frost_test_utils import requires_dsl

pytestmark = [pytest.mark.L0, requires_dsl]


def test_graph_thd_targets_nonzero_ordinal_a100(monkeypatch):
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("needs a nonzero-ordinal A100 and another visible GPU")
    if torch.cuda.get_device_capability(0) == (8, 0):
        pytest.skip("needs a non-SM80 GPU at ordinal zero")
    target = next((i for i in range(1, torch.cuda.device_count()) if torch.cuda.get_device_capability(i) == (8, 0)), None)
    if target is None:
        pytest.skip("needs an A100 at a nonzero visible ordinal")

    import cutlass.cute as cute
    from cudnn.sdpa.bwd.kernels.thd_helpers import thd_meta_host
    from cudnn.sdpa.fwd.kernels.sm80 import prepared_host
    from sdpa.frost.test_sdpa_fwd_thd_sm80 import _check_graph_thd_scale_and_replay

    # Force both compilation steps even if another test has warmed the caches.
    monkeypatch.setenv("CUDNN_FRONTEND_DISABLE_COMPILED_CACHE", "1")
    monkeypatch.setenv("CUDNN_FRONTEND_COMPILED_CACHE_INPROCESS_MEMO", "0")
    prepared_host.compile_thd_host.cache_clear()
    compile_kernel = cute.compile
    seen = set()

    def compile_spy(fn, *args, **kwargs):
        if fn is thd_meta_host or fn is prepared_host.thd_host:
            assert "--gpu-arch=sm_80" in kwargs.get("options", "")
            seen.add("thd_meta_host" if fn is thd_meta_host else "thd_host")
        return compile_kernel(fn, *args, **kwargs)

    monkeypatch.setattr(cute, "compile", compile_spy)
    try:
        with torch.cuda.device(target):
            _check_graph_thd_scale_and_replay(0.125, 128, 128)
        assert seen == {"thd_meta_host", "thd_host"}
    finally:
        prepared_host.compile_thd_host.cache_clear()
