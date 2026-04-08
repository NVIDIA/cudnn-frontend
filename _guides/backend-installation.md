---
layout: guide
title: "cuDNN Backend Installation"
sidebar_title: "Backend Install"
description: "How to install cuDNN backend via pip, conda, and apt-get."
order: 2
permalink: /guides/backend-installation/
---

Before you can use cuDNN Frontend (or any cuDNN-based framework, really), you need the **cuDNN backend** installed on your system. This is the core library — the actual GPU kernels that do the heavy lifting for convolutions, attention, normalization, and all the other operations that make deep learning fast.

Let's get it installed. Pick your flavor.

## Installing with pip

The fastest way to get cuDNN if you're working in a Python environment:

```bash
pip install nvidia-cudnn-cu13
```

This pulls in the cuDNN runtime libraries that match CUDA 13.x. If you're on an older CUDA toolkit, adjust the package name accordingly (e.g., `nvidia-cudnn-cu12` for CUDA 12.x).

Quick verification:

```python
import nvidia.cudnn as cudnn
print(cudnn.__version__)
```

### When to use pip

- You're working in a virtualenv or conda env and want a self-contained setup
- You don't have root access to the system
- You want the quickest path from zero to cuDNN

## Installing with conda

If conda is your package manager of choice:

```bash
conda install -c nvidia cudnn
```

Conda is nice because it'll resolve the CUDA toolkit dependency for you automatically. If you need a specific version:

```bash
conda install -c nvidia cudnn=9.x
```

### When to use conda

- Your project already lives in a conda environment
- You want automatic CUDA toolkit dependency resolution
- You're managing multiple CUDA versions across projects

## Installing with apt-get (Ubuntu/Debian)

For system-wide installation on Ubuntu or Debian-based systems:

```bash
# Add the NVIDIA package repository (if not already added)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update

# Install cuDNN
sudo apt-get install -y libcudnn9-cuda-13
```

For the development headers (needed if you're compiling C++ code against cuDNN):

```bash
sudo apt-get install -y libcudnn9-dev-cuda-13
```

### When to use apt-get

- You're setting up a dedicated GPU server or VM
- You want a system-wide installation available to all users
- You're building Docker images or setting up CI/CD pipelines
- You need the development headers for C++ compilation

## Verifying Your Installation

No matter which method you used, let's make sure cuDNN is actually working:

```bash
# Check the library is findable
ldconfig -p | grep cudnn
```

You should see something like `libcudnn.so.9 => /usr/lib/x86_64-linux-gnu/libcudnn.so.9`.

From Python:

```python
import torch
print(torch.backends.cudnn.version())
print(torch.backends.cudnn.is_available())
```

If `is_available()` returns `True`, you're good to go.

## Full Installation Guide

We've covered the quick-start paths here, but cuDNN supports a lot of configurations — different CUDA versions, different Linux distros, Windows, cross-compilation, and more. For the complete, authoritative installation guide:

**[NVIDIA cuDNN Installation Guide →](https://docs.nvidia.com/cudnn)**
