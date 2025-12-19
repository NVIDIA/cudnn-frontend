FROM nvcr.io/nvidia/pytorch:25.11-py3

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

RUN pip install nvidia-cutlass-dsl

RUN pip uninstall -y flash_attn

ENV CUDA_HOME=/usr/local/cuda
ENV MAX_JOBS=32

RUN git clone https://github.com/Dao-AILab/flash-attention.git --recursive

WORKDIR /workspace/flash-attention

ENV FLASH_ATTENTION_DISABLE_BACKWARD=TRUE
ENV FLASH_ATTENTION_DISABLE_APPENDKV=TRUE
ENV FLASH_ATTENTION_DISABLE_SOFTCAP=TRUE
ENV FLASH_ATTENTION_DISABLE_FP8=TRUE
ENV FLASH_ATTENTION_DISABLE_CLUSTER=TRUE
ENV FLASH_ATTENTION_DISABLE_HDIM96=TRUE
ENV FLASH_ATTENTION_DISABLE_HDIM192=TRUE
ENV FLASH_ATTENTION_DISABLE_SM80=TRUE

RUN python3 setup.py install

WORKDIR /workspace