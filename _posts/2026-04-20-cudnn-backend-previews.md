---
layout: post
title: "cuDNN Backend Now Has Preview Releases"
sidebar_title: "Backend Preview Builds"
date: 2026-04-20
description: "Try upcoming cuDNN backend features early with pip install --pre. Stable releases remain unchanged."
---

Good news for folks who like living on the edge — **cuDNN backend packages now have preview releases** available on PyPI. This means you can try upcoming features and fixes before they hit the stable channel.

## Installing a Preview

```bash
pip install nvidia-cudnn-cu13 --pre --extra-index-url https://pypi.nvidia.com
```

The `--pre` flag tells pip to include pre-release versions, and the extra index URL points to NVIDIA's PyPI repository where preview builds are published.

## Stable Releases

Nothing changes for stable installs. If you want the latest tested, production-ready build:

```bash
pip install nvidia-cudnn-cu13
```

No extra flags, no extra index — same as always.

## Why Previews?

Preview releases give you early access to:

- **New kernel support** for upcoming GPU architectures
- **Performance improvements** before they're fully validated across all configurations
- **Bug fixes** that haven't gone through the full release qualification cycle yet

This is especially useful if you're developing against a new hardware target or want to validate that an upcoming release doesn't regress your workload.

## A Note on Stability

Preview builds go through CI and basic validation, but they haven't completed the full release qualification matrix. Use them for development and testing — not for production training runs where reproducibility matters. If you hit an issue, file it on [GitHub](https://github.com/NVIDIA/cudnn-frontend/issues) and we'll prioritize it for the stable release.

## CUDA 12 Users

Same pattern, just swap the package name:

```bash
# Preview
pip install nvidia-cudnn-cu12 --pre --extra-index-url https://pypi.nvidia.com

# Stable
pip install nvidia-cudnn-cu12
```
