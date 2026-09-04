# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""cuDNN SDPA reproducer tool."""


def main() -> None:
    from .__main__ import main as run

    run()


__all__ = ["main"]
