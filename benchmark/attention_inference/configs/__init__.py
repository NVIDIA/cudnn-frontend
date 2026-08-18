# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
import pkgutil


def list_configs():
    return sorted(m.name for m in pkgutil.iter_modules(__path__) if not m.name.startswith("_"))


def load_config(name: str):
    try:
        module = importlib.import_module(f".{name}", __package__)
    except ModuleNotFoundError as e:
        # only translate "no such config module" — a missing dependency raised
        # from inside a valid config module must propagate as itself
        if e.name == f"{__package__}.{name}":
            raise ValueError(f"Unknown config '{name}'. Available: {', '.join(list_configs())}") from None
        raise
    return module.CONFIG
