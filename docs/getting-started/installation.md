# Installation

Using cuDNN Frontend requires a compatible CUDA, cuDNN, driver, and GPU stack. Depends on whether you want to use Python or C++ interface, you can setup the environment differently.

## Requirements

- Python 3.9 or newer for the Python package.
- CMake 3.23 or newer and a C++17 compiler for repository CMake builds.
- An NVIDIA driver, CUDA toolkit/runtime, and cuDNN release that support the selected feature.
- Linux or Windows for the core package. Optional CUTLASS DSL packages can have narrower platform support.

<Tabs>
<Tab title="Python" language="python">


## Python

You can install the released core package and the optional kernel dependencies:

```bash
# Install the core package
pip install nvidia-cudnn-frontend

# Optional: Install the CUTLASS DSL kernels and their pinned dependencies
pip install nvidia-cudnn-frontend[cutedsl]

# Optional: Install the dependent Python packages explicitly
pip install nvidia-cuda-runtime nvidia-cudnn-cu13 nvidia-cutlass-dsl-libs-cu13
```

You may also install from source:

```bash
# optional: Set the paths if CUDA or cuDNN are not in default system locations
export CUDAToolkit_ROOT=/path/to/cuda
export CUDA_PATH=/path/to/cuda
export CUDNN_PATH=/path/to/cudnn

# Install from source
pip install git+https://github.com/NVIDIA/cudnn-frontend.git
```

The source build uses CMake and pybind11 to build `cudnn._compiled_module`. Set `CUDAToolkit_ROOT` or `CUDA_PATH` when CMake cannot find the intended toolkit. Set `CUDNN_PATH` when cuDNN headers or libraries are outside normal discovery paths.

After activating the Python environment you want to test, run the probe from
the repository root:

```bash
python docs/samples/python/verify_environment.py
```

The path printed as `cudnn_file` must resolve under this checkout when verifying unreleased source behavior. A site-packages path tests the installed wheel instead.

## Verify the installation

After activating the Python environment you want to test, run the probe from
the repository root:

```bash
python docs/samples/python/verify_environment.py
```

The path printed as `cudnn_file` shows the installed path of the Python package, usually a site-packages path if installed from wheel.

</Tab>
<Tab title="C++" language="cpp">


## C++

The C++ frontend is header-only. The header files are located at `include/` directory. The main header file is `cudnn_frontend.h`. Applications still need to link cuDNN and the CUDA libraries they call. You can find instructions on how to install cuDNN and CUDA libraries from the following links:

- CUDA: [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)
- cuDNN: [https://developer.nvidia.com/cudnn-downloads](https://developer.nvidia.com/cudnn-downloads)

You can add this repository as a CMake subdirectory and link `cudnn_frontend` with appropriate libraries. Below is an example of the `CMakeLists.txt` to use the cuDNN Frontend:

<Code
  src="../samples/cmake/CMakeLists.txt"
  language="cmake"
  maxLines={0}
/>

Configure a source-tree build with samples and tests disabled when only the package target is needed:

```bash
cmake -S . -B build/install \
  -DCUDNN_FRONTEND_BUILD_SAMPLES=OFF \
  -DCUDNN_FRONTEND_BUILD_TESTS=OFF \
  -DCMAKE_INSTALL_PREFIX=/path/to/prefix
cmake --build build/install
cmake --install build/install
```

Then set `CMAKE_PREFIX_PATH=/path/to/prefix` for the consuming project. On Windows, use the appropriate MSVC generator and ensure cuDNN/CUDA DLL directories are on `PATH`. On Linux, ensure shared libraries are discoverable through the dynamic loader or package installation.

</Tab>
</Tabs>
## Optional: Select the CUDA runtime

When several `libcudart.so` major versions are visible, in your system, you may need to set the exact runtime required by the environment before running any cuDNN application. This applies to both Python and C++ interfaces:

```bash
export CUDNN_FRONTEND_CUDART_LIB_NAME=libcudart.so.13
```

Do not set that value automatically on a CUDA 12 deployment. Record the chosen library with test results.

## Next steps

Run the [Python and C++ quick starts](/getting-started/quickstart) to learn how to use the cuDNN Frontend and verify your installation.

## Verification limits

The repository README states cuDNN 8.5 as a general frontend minimum, while the
Python `cudnn.Graph` wrapper requires backend 9.12. Treat minimum versions as
interface-specific. Windows installation is source-supported but was not
executed in the recorded verification environment.
