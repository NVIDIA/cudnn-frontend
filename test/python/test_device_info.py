# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""cudnn._device.DeviceInfo is the FE's single owner of a GPU's facts: each fact
is queried from the driver once and cached ON the instance, and there is one
instance per CUDA ordinal. Handle.device is that object; frost reads the same one
through thin shims (no parallel introspection stack)."""

import pytest

import cudnn
from cudnn._device import DeviceInfo, device_info, is_available

pytestmark = pytest.mark.L0


def test_device_info_one_per_ordinal_and_caches_on_the_instance():
    if not is_available():
        pytest.skip("no CUDA device")
    device_info.cache_clear()
    d0 = device_info(0)
    assert isinstance(d0, DeviceInfo)
    assert device_info(0) is d0  # one shared instance per ordinal

    # A fact is cached on the INSTANCE (cached_property), not in a free-function cache.
    assert "compute_capability" not in d0.__dict__  # not queried yet
    cc = d0.compute_capability
    assert d0.__dict__["compute_capability"] == cc  # now stored on the instance
    assert d0.sm_version == cc[0] * 10 + cc[1]  # derived, cannot drift from the tuple


def test_handle_device_is_the_shared_device_info():
    if not is_available():
        pytest.skip("no CUDA device")
    h = cudnn.create_handle()
    try:
        assert isinstance(h.device, DeviceInfo)
        assert h.device is device_info(h.device.ordinal)  # the handle reads the shared object
    finally:
        cudnn.destroy_handle(h)


def test_frost_facts_are_shims_onto_the_common_device_info():
    if not is_available():
        pytest.skip("no CUDA device")
    from cudnn.frost import device as F

    d = device_info(0)
    assert F.compute_capability(0) == d.compute_capability
    assert F.multiprocessor_count(0) == d.sm_count
    assert F.shared_memory_per_block_optin(0) == d.shared_memory_per_block_optin
    assert F.l2_cache_bytes(0) == d.l2_cache_bytes
    assert F.device_name(0) == d.device_name
