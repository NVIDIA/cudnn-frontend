# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

FROM nvcr.io/nvidia/jax:26.07-py3@sha256:fa7ce0f892bb1b6ea410b81cba9f1f50468074b1db78f33f62ed070d079eb4fb

ENV XLA_PYTHON_CLIENT_PREALLOCATE=false \
    CMAKE_BUILD_PARALLEL_LEVEL=4 \
    CUDNN_PATH=/usr

RUN python -m pip install --no-cache-dir "nvidia-cutlass-dsl[cu13]==4.7.0" pytest pytest-xdist

WORKDIR /opt/cudnn-frontend
COPY . .
RUN python -m pip install --no-cache-dir .

CMD ["python", "-m", "pytest", "test/jax", "-q", "--require-sm100"]
