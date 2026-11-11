# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from enum import Enum, auto
from hashlib import sha256
from typing import Optional, Tuple

import torch

EPSILON = 1e-5
MOMENTUM = 0.2
FP8_E4M3 = getattr(torch, "float8_e4m3fn", None)
FP8_E5M2 = getattr(torch, "float8_e5m2", None)


class BatchNormPattern(Enum):
    BN = auto()
    BN_RELU = auto()
    BN_ADD_RELU = auto()
    DBN = auto()
    DRELU_DBN = auto()
    DRELU_DADD_DBN = auto()

    @property
    def is_forward(self) -> bool:
        return self in (BatchNormPattern.BN, BatchNormPattern.BN_RELU, BatchNormPattern.BN_ADD_RELU)

    @property
    def is_backward(self) -> bool:
        return self in (BatchNormPattern.DBN, BatchNormPattern.DRELU_DBN, BatchNormPattern.DRELU_DADD_DBN)

    @property
    def has_add(self) -> bool:
        return self in (BatchNormPattern.BN_ADD_RELU, BatchNormPattern.DRELU_DADD_DBN)

    @property
    def has_relu(self) -> bool:
        return self in (BatchNormPattern.BN_RELU, BatchNormPattern.BN_ADD_RELU)

    @property
    def has_backward_mask(self) -> bool:
        return self in (BatchNormPattern.DRELU_DBN, BatchNormPattern.DRELU_DADD_DBN)


@dataclass(frozen=True)
class BatchNormCase:
    case_id: str
    seed: int
    pattern: BatchNormPattern
    shape: Tuple[int, int, int, int]
    input_dtype: Optional[torch.dtype]
    output_dtype: Optional[torch.dtype]
    level: str
    rtol: float
    atol: float
    min_compute_capability: Optional[Tuple[int, int]] = None
    max_mismatch_rate: float = 0.0
    grad_dtype: Optional[torch.dtype] = None
    uses_fp8: bool = False

    @property
    def parameter_shape(self) -> Tuple[int, int, int, int]:
        return (1, self.shape[1], 1, 1)


BATCHNORM_CASES = (
    BatchNormCase(
        case_id="sgbn_train_n4_c512_h32_w32_fp16",
        seed=3001,
        pattern=BatchNormPattern.BN,
        shape=(4, 512, 32, 32),
        input_dtype=torch.float16,
        output_dtype=torch.float16,
        level="L0",
        rtol=1e-3,
        atol=1e-3,
    ),
    BatchNormCase(
        case_id="bn_relu_n4_c32_h11_w11_bf16",
        seed=3101,
        pattern=BatchNormPattern.BN_RELU,
        shape=(4, 32, 11, 11),
        input_dtype=torch.bfloat16,
        output_dtype=torch.bfloat16,
        level="L0",
        rtol=1e-3,
        atol=1e-3,
        max_mismatch_rate=5e-4,
    ),
    BatchNormCase(
        case_id="bn_add_relu_n4_c32_h11_w11_fp32",
        seed=3102,
        pattern=BatchNormPattern.BN_ADD_RELU,
        shape=(4, 32, 11, 11),
        input_dtype=torch.float32,
        output_dtype=torch.float32,
        level="L1",
        rtol=2.5e-3,
        atol=2.5e-3,
        max_mismatch_rate=5e-4,
    ),
    BatchNormCase(
        case_id="bn_add_relu_n400_c256_h56_w56_e4m3",
        seed=3103,
        pattern=BatchNormPattern.BN_ADD_RELU,
        shape=(400, 256, 56, 56),
        input_dtype=FP8_E4M3,
        output_dtype=FP8_E4M3,
        level="L1",
        min_compute_capability=(8, 9),
        rtol=0.2,
        atol=0.2,
        max_mismatch_rate=0.002,
        uses_fp8=True,
    ),
    BatchNormCase(
        case_id="dbn_n416_c512_h7_w7_fp16",
        seed=3301,
        pattern=BatchNormPattern.DBN,
        shape=(416, 512, 7, 7),
        input_dtype=torch.float16,
        output_dtype=torch.float16,
        level="L1",
        rtol=1e-3,
        atol=1e-3,
        grad_dtype=torch.float16,
    ),
    BatchNormCase(
        case_id="drelu_dbn_n4_c32_h11_w11_bf16",
        seed=3302,
        pattern=BatchNormPattern.DRELU_DBN,
        shape=(4, 32, 11, 11),
        input_dtype=torch.bfloat16,
        output_dtype=torch.bfloat16,
        level="L1",
        rtol=1e-3,
        atol=1e-3,
        grad_dtype=torch.bfloat16,
    ),
    BatchNormCase(
        case_id="drelu_dadd_dbn_n4_c32_h11_w11_fp32",
        seed=3303,
        pattern=BatchNormPattern.DRELU_DADD_DBN,
        shape=(4, 32, 11, 11),
        input_dtype=torch.float32,
        output_dtype=torch.float32,
        level="L1",
        rtol=1e-3,
        atol=1e-3,
        grad_dtype=torch.float32,
    ),
    BatchNormCase(
        case_id="drelu_dadd_dbn_n400_c256_h28_w28_e4m3_e5m2",
        seed=3201,
        pattern=BatchNormPattern.DRELU_DADD_DBN,
        shape=(400, 256, 28, 28),
        input_dtype=FP8_E4M3,
        output_dtype=FP8_E5M2,
        level="L1",
        min_compute_capability=(9, 0),
        rtol=0.25,
        atol=0.25,
        grad_dtype=FP8_E5M2,
        uses_fp8=True,
    ),
)


def _fp8_expansion_cases():
    shapes = (
        (400, 128, 56, 56),
        (400, 256, 14, 14),
        (400, 512, 28, 28),
        (400, 1024, 14, 14),
        (400, 64, 56, 56),
        (400, 128, 28, 28),
        (400, 512, 14, 14),
        (400, 512, 7, 7),
        (400, 2048, 7, 7),
        (400, 256, 56, 56),
        (400, 256, 28, 28),
    )
    forward_types = (("e4m3", FP8_E4M3), ("e5m2", FP8_E5M2))
    backward_types = (
        ("e4m3_e4m3", FP8_E4M3, FP8_E4M3),
        ("e5m2_e5m2", FP8_E5M2, FP8_E5M2),
        ("e4m3_e5m2", FP8_E4M3, FP8_E5M2),
    )
    existing_ids = {case.case_id for case in BATCHNORM_CASES}
    for pattern in BatchNormPattern:
        types = tuple((name, dtype, None) for name, dtype in forward_types) if pattern.is_forward else backward_types
        for shape in shapes:
            n, c, h, w = shape
            for dtype_name, input_dtype, grad_dtype in types:
                case_id = f"{pattern.name.lower()}_n{n}_c{c}_h{h}_w{w}_{dtype_name}"
                # Retain the original parameters and seeds for already covered problems.
                # Symbolic IDs also remain distinct when PyTorch lacks an FP8 dtype.
                if case_id in existing_ids:
                    continue
                if pattern == BatchNormPattern.BN:
                    rtol, atol = (0.2, 0.5) if dtype_name == "e4m3" else (0.25, 0.25)
                    mismatch_rate = 0.0
                elif pattern.is_forward:
                    rtol, atol = 0.2, 0.2
                    mismatch_rate = 0.002 if pattern.has_add else 0.003
                else:
                    rtol, atol = 0.25, 0.25
                    mismatch_rate = 0.0
                yield BatchNormCase(
                    case_id=case_id,
                    # Identity-based seeds do not shift when cases are inserted or reordered.
                    seed=int.from_bytes(sha256(case_id.encode("ascii")).digest()[:4], "big"),
                    pattern=pattern,
                    shape=shape,
                    input_dtype=input_dtype,
                    output_dtype=input_dtype if pattern.is_forward else grad_dtype,
                    level="L1",
                    rtol=rtol,
                    atol=atol,
                    min_compute_capability=(9, 0) if pattern.has_backward_mask else (8, 9),
                    max_mismatch_rate=mismatch_rate,
                    grad_dtype=grad_dtype,
                    uses_fp8=True,
                )


BATCHNORM_CASES += tuple(_fp8_expansion_cases())
