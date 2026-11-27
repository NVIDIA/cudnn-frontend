# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch


@pytest.fixture
def quant_device():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3), (10, 7)):
        pytest.skip("batch-invariant MX quantization requires SM100, SM103 or SM107")
    return torch.device("cuda", torch.cuda.current_device())
