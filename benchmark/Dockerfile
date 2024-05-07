FROM nvcr.io/nvidia/pytorch:24.03-py3

RUN apt-get update && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    apt-get update && \
    apt-get -y install cudnn


RUN pip uninstall -y cudnn

RUN CMAKE_BUILD_PARALLEL_LEVEL=16 pip install git+https://github.com/NVIDIA/cudnn-frontend.git -v

COPY benchmark_flash_attention.py .

ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/:$LD_LIBRARY_PATH

CMD ["python", "/workspace/benchmark_flash_attention.py"]

WORKDIR /workspace
