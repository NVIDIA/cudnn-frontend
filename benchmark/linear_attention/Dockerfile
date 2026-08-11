# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

FROM nvcr.io/nvidia/pytorch:26.07-py3

# Set working directory
WORKDIR /workspace

# Update libcudnn9-cuda-13
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/sbsa/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    apt-get remove -y *cudnn9* && \
    apt-get update && \
    apt-get -y install cudnn && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Clone cudnn_frontend and install latest cudnn
RUN git clone https://github.com/NVIDIA/cudnn-frontend.git
RUN pip install -v /workspace/cudnn-frontend

# Install the Cutlass DSL runtime (cuDNN FROST engines) and FLA.
RUN pip install nvidia-cutlass-dsl[cu13]==4.7.0 apache-tvm-ffi flash-linear-attention

# Install FlashQLA from source.
RUN git clone https://github.com/QwenLM/FlashQLA.git
RUN pip install -v /workspace/FlashQLA

# Install FlashKDA from source.
RUN git clone https://github.com/MoonshotAI/FlashKDA.git
RUN pip install -v /workspace/FlashKDA

# Install the chart dependencies for plot_results.py
RUN pip install pandas matplotlib seaborn
