---
layout: guide
title: "cuDNN Frontend Installation"
sidebar_title: "Frontend Install"
description: "How to install cuDNN Frontend for C++ and Python."
order: 1
permalink: /guides/frontend-installation/
---

So you want to use cuDNN but don't want to wrestle with the raw backend API? Good news — that's exactly what **cuDNN Frontend** is for. It's a higher-level, developer-friendly interface that sits on top of the cuDNN backend and makes your life significantly easier.

Let's get you set up.

## What is cuDNN Frontend?

cuDNN Frontend is a C++ header-only library (with Python bindings!) that provides a graph-based API for building and executing cuDNN operations. Instead of manually managing tensors, descriptors, and workspace allocations, you describe your computation as a graph and let the frontend figure out the optimal execution plan.

Think of it this way: the **backend** is the engine, and the **frontend** is the steering wheel. You *can* hotwire the engine directly, but why would you when there's a perfectly good steering wheel right there?

## Installing for Python Users

This is the easy path. One line and you're done:

```bash
pip install nvidia-cudnn-frontend
```

That's it. Seriously. You now have access to the full cuDNN Frontend graph API from Python.

Here's a quick sanity check to make sure it's working:

```python
import cudnn
print(cudnn.__version__)
```

If that prints a version number without exploding, you're golden.

### Requirements

- Python 3.8+
- A CUDA-capable GPU (obviously)
- cuDNN backend installed (see the [backend installation guide]({{ site.baseurl }}/guides/backend-installation/))

## Installing for C++ Users

cuDNN Frontend is a **header-only** library, so there's no separate build step. You just need to get the headers into your project.

### Option 1: Git Submodule (Recommended)

The cleanest approach — pin to a version and update when you're ready:

```bash
cd your-project/
git submodule add https://github.com/NVIDIA/cudnn-frontend.git third_party/cudnn-frontend
```

Then in your `CMakeLists.txt`:

```cmake
add_subdirectory(third_party/cudnn-frontend)
target_link_libraries(your_target PRIVATE cudnn_frontend)
```

### Option 2: CMake FetchContent

If you prefer not to use submodules, CMake can pull it in automatically:

```cmake
include(FetchContent)
FetchContent_Declare(
  cudnn_frontend
  GIT_REPOSITORY https://github.com/NVIDIA/cudnn-frontend.git
  GIT_TAG        main
)
FetchContent_MakeAvailable(cudnn_frontend)

target_link_libraries(your_target PRIVATE cudnn_frontend)
```

### Option 3: Just Copy the Headers

Feeling old-school? Clone the repo and copy the `include/` directory into your project. No judgment.

```bash
git clone https://github.com/NVIDIA/cudnn-frontend.git
cp -r cudnn-frontend/include/ your-project/third_party/cudnn_frontend/
```

Then add that path to your compiler's include directories.

## A Quick Taste of the API

Here's a minimal example that builds a forward convolution graph:

```python
import cudnn
import torch

# Create a graph
graph = cudnn.pygraph()

# Define input tensors
X = graph.tensor(name="X", dim=[1, 32, 224, 224],
                 stride=[32*224*224, 224*224, 224, 1],
                 data_type=cudnn.data_type.HALF)
W = graph.tensor(name="W", dim=[64, 32, 3, 3],
                 stride=[32*3*3, 3*3, 3, 1],
                 data_type=cudnn.data_type.HALF)

# Build a convolution operation
Y = graph.conv_fprop(image=X, weight=W,
                     padding=[1,1], stride=[1,1], dilation=[1,1])

# That's the graph defined — now build and execute it!
```

The C++ API follows the same graph-based pattern. Check out the [samples on GitHub](https://github.com/NVIDIA/cudnn-frontend/tree/main/samples) for full working examples.
