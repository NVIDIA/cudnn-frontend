# Verify the environment

Below is a Python script to confirm the cuDNN Frontend is installed correctly. It confirms which frontend is imported, which backend is loaded, which CUDA runtime PyTorch reports, and whether the toolchain and GPU are visible.

## Run the probe

Complete Python code:

<Code
  src="../samples/python/verify_environment.py"
  language="python"
  maxLines={0}
/>

Run from the repository root:

```bash
python docs/samples/python/verify_environment.py
```

## Interpret the result

Example output:

```text
cudnn_file=/path/to/python3.13/site-packages/cudnn/__init__.py
frontend=1.32.0
backend=9.15.1
torch=2.10.0+cu130
torch_cuda=13.0
cuda_available=True
gpu=NVIDIA Graphics Device
handle_stream=0
nvcc=nvcc: NVIDIA (R) Cuda compiler driver
cmake=cmake version 3.31.4
environment probe ok
```

- `cudnn_file` must point to the intended wheel or checkout.
- `frontend` identifies the Python/source package version.
- `backend` identifies the loaded cuDNN shared library and can differ from the frontend version.
- `torch_cuda` is the CUDA version used to build PyTorch; it is not necessarily the `nvcc` toolkit version.
- `cuda_available` and `gpu` confirm device visibility.
- A created handle and queried stream confirm basic frontend/backend initialization.
- `nvcc` and CMake are required for C++ interface, not used in Python interface.

If this works, you can run the examples from the [Quick Start Guide](/getting-started/quickstart), through [the canonical shell driver](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/samples/shell/run_quickstarts.sh) at the repository root.
