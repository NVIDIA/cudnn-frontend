# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Preallocated CUDA Graph timing shared by the SE comparisons."""

import random
import statistics
import time
import warnings

import torch


def measure(functions, repeats=100, rounds=21, cold=False):
    """Measure preallocated CUDA graphs in randomized order, optionally evicting cache before samples."""
    eviction = torch.empty(512 * 1024 * 1024, device="cuda", dtype=torch.uint8) if cold else None
    if cold:
        repeats = 1
    graphs = {}
    for name, fn in functions.items():
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with warnings.catch_warnings():
            warnings.filterwarnings("error", message="The CUDA Graph is empty.*")
            with torch.cuda.graph(graph):
                for _ in range(repeats):
                    fn()
        graphs[name] = graph
    start = time.monotonic()
    while time.monotonic() - start < 0.3:
        for graph in graphs.values():
            graph.replay()
        torch.cuda.synchronize()
    rng = random.Random(20261022)
    samples = {name: [] for name in graphs}
    positions = {name: [] for name in graphs}
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(rounds):
        order = list(graphs)
        rng.shuffle(order)
        for position, name in enumerate(order):
            if eviction is not None:
                eviction.zero_()
            begin.record()
            graphs[name].replay()
            end.record()
            end.synchronize()
            samples[name].append(begin.elapsed_time(end) * 1000 / repeats)
            positions[name].append(position)
    return {name: {"median_us": statistics.median(vals), "samples_us": vals, "round_positions": positions[name]} for name, vals in samples.items()}
