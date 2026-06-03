FROM nvcr.io/nvidia/pytorch:25.12-py3

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

# Install additional dependencies for benchmarking
RUN pip install seaborn quack-kernels
