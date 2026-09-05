# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch


@pytest.fixture(scope="session")
def env_info():
    assert torch.cuda.is_available(), "no CUDA device"

    gpu_type = torch.cuda.get_device_capability()
    gpu_name = torch.cuda.get_device_name()
    device = torch.device("cuda:0")
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count

    gpu_arch = f"SM_{gpu_type[0]}{gpu_type[1]}"
    gpu_info = f"{sm_count} SM-s, {gpu_name}"
    cudnn_ver = str(torch.backends.cudnn.version())

    return {"gpu_arch": gpu_arch, "gpu_info": gpu_info, "cudnn_ver": cudnn_ver}
