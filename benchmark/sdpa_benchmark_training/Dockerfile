FROM nvcr.io/nvidia/pytorch:25.06-py3

RUN pip install --upgrade pip && \
    pip install seaborn

RUN apt-get update && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    apt-get update && \
    apt-get -y install cudnn

RUN pip uninstall -y cudnn

COPY benchmark_bf16_sdpa.py .

COPY benchmark_fp8_sdpa.py .

COPY benchmark_single_sdpa.py .

ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/:$LD_LIBRARY_PATH

WORKDIR /workspace