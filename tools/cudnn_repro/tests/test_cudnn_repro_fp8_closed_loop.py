import os

import pytest

from .test_cudnn_repro_closed_loop import _assert_reproducer_json_matches_target


def _target_test_id(target):
    return target.split("::", 1)[-1]


def _targets():
    raw = os.environ.get("CUDNN_REPRO_FP8_TARGETS")
    if raw:
        return [item.strip() for item in raw.split(",") if item.strip()]
    return [
        "test/python/test_mhas_v2.py::test_sdpa_fp8_fwd_L0[test1]",
        "test/python/test_mhas_v2.py::test_sdpa_fp8_fwd_paged_L0[test1]",
        "test/python/test_mhas_v2.py::test_sdpa_fp8_bwd_L0[test1]",
    ]


@pytest.mark.parametrize("target", _targets(), ids=_target_test_id)
def test_fp8_reproducer_json_matches(tmp_path, target):
    _assert_reproducer_json_matches_target(tmp_path, target)
