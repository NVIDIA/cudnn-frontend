# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from cudnn.collect_env import collect_env_info, format_report


@pytest.fixture(scope="module")
def report():
    # The core contract: collection never raises, regardless of environment
    # (no GPU, missing optional packages, ...). Collect once for all tests.
    return collect_env_info()


@pytest.mark.L0
def test_sections_present(report):
    for section in (
        "cuDNN Frontend",
        "Python / Platform",
        "GPU / Driver",
        "CUDA Toolkit",
        "PyTorch",
        "GPU Libraries: loaded vs on disk",
        "Relevant Packages",
        "Environment Variables",
    ):
        assert section in report
        assert isinstance(report[section], dict)


@pytest.mark.L0
def test_report_has_frontend_version(report):
    import cudnn

    assert report["cuDNN Frontend"]["cudnn-frontend"] == cudnn.__version__


@pytest.mark.L0
def test_report_has_loaded_backend(report):
    import cudnn

    assert str(cudnn.backend_version()) in report["cuDNN Frontend"]["cudnn backend (loaded)"]


@pytest.mark.L0
def test_format_report(report):
    text = format_report(report)
    assert "cuDNN frontend environment report" in text
    assert "==== Relevant Packages ====" in text


@pytest.mark.L0
def test_json_serializable(report):
    json.dumps(report)
