# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""build_device() scopes a frost build to a chosen GPU, so every device-derived
kernel constant follows the handle's device rather than whatever CUDA device is
current. Proven here by scoping to a *different real* GPU and checking the
constants report that GPU's facts -- the multi-GPU behaviour a single-GPU run
cannot otherwise exercise. Uses only device queries (no kernel launch), so the
second GPU need not be a Blackwell."""

import pytest

from cudnn.frost import device as D

pytestmark = pytest.mark.L0


def _distinct_arch_devices():
    seen = {}
    for o in range(D.device_count()):
        try:
            seen.setdefault(D.compute_capability(o), o)
        except Exception:  # noqa: BLE001 — a device we cannot query is just skipped
            pass
    ccs = list(seen.items())
    if len(ccs) < 2:
        pytest.skip("need two GPUs of different compute capability to prove the redirect")
    return ccs[0], ccs[1]


def test_build_device_scopes_current_device_and_restores():
    if D.device_count() == 0:
        pytest.skip("no CUDA device")
    live = D.current_device()
    other = next((o for o in range(D.device_count()) if o != live), None)
    if other is None:
        pytest.skip("need a second device")
    with D.build_device(other):
        assert D.current_device() == other
        assert D.resolve_device(None) == other
    assert D.current_device() == live  # restored on exit


def test_build_device_redirects_every_frost_build_constant():
    from cudnn.gemm.frost import compiler as C
    from cudnn.gemm.frost import tile_config as TC

    (cc_a, a), (cc_b, b) = _distinct_arch_devices()
    # Scope to each device in turn; every device-derived build constant must report
    # THAT device's facts, not the ambient one -- i.e. a build follows the scope.
    for cc, dev in ((cc_a, a), (cc_b, b)):
        with D.build_device(dev):
            assert D.current_device() == dev
            assert C._current_arch() == cc[0] * 10 + cc[1]
            assert C._plan_device() == dev
            # _sm_count was re-routed off torch.cuda.current_device -> honours the scope
            assert TC._sm_count() == D.multiprocessor_count(dev)


def test_build_device_none_is_no_override():
    if D.device_count() == 0:
        pytest.skip("no CUDA device")
    live = D.current_device()
    with D.build_device(None):
        assert D.current_device() == live  # None = classic current-device behaviour
