import pytest
from looseversion import LooseVersion

from .test_cudnn_repro_closed_loop import _assert_reproducer_json_matches_target


@pytest.mark.parametrize(
    "target",
    [
        "test/python/test_mhas_v2.py::test_sdpa_mxfp8_fwd_L0[test1]",
        "test/python/test_mhas_v2.py::test_sdpa_mxfp8_bwd_L0[test1]",
    ],
)
def test_mxfp8_reproducer_json_matches(tmp_path, target):
    cudnn = pytest.importorskip("cudnn")
    torch = pytest.importorskip("torch")

    if LooseVersion(cudnn.backend_version_string()) < "9.21.0":
        pytest.skip("MXFP8 repro requires cuDNN 9.21.0 or higher")
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("MXFP8 repro requires Blackwell or higher")

    _assert_reproducer_json_matches_target(tmp_path, target)
