# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rendering records retain both runtime-selected fold specializations."""

import hashlib

import pytest

from frost_test_utils import ptx_kernel_md5s

pytestmark = pytest.mark.L0


def _entry(name, body="ret;"):
    return f".visible .entry cudnn_kernel_{name}()\n{{\n{body}\n}}\n"


def test_shared_fold_specializations_are_both_hashed():
    name = "dkv_reduce_thd_origins_kernel"
    first, second = _entry(name + "_straight"), _entry(name + "_strided", "bra loop;")
    own = _entry("attention_kernel")
    hashes = ptx_kernel_md5s(own + first + second)
    assert hashes["attention_kernel"] == hashlib.md5(own.encode()).hexdigest()
    assert set(hashes) == {"attention_kernel", name}
    assert hashes[name] == ptx_kernel_md5s(own + second + first)[name]
    assert hashes[name] != ptx_kernel_md5s(own + first.replace("ret;", "exit;") + second)[name]
    assert hashes[name] != ptx_kernel_md5s(own + first + second.replace("bra loop;", "exit;"))[name]


@pytest.mark.parametrize(
    "text",
    [
        _entry("attention_kernel_a") + _entry("attention_kernel_b"),
        _entry("dkv_reduce_thd_origins_kernel_a") * 2,
        "".join(_entry("dkv_reduce_thd_origins_kernel_" + suffix) for suffix in ("a", "b", "c")),
    ],
)
def test_unexpected_duplicate_entries_remain_errors(text):
    with pytest.raises(AssertionError):
        ptx_kernel_md5s(text)
