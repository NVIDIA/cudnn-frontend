# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scratch for the conv-history last-reader election in the fused GDN steps."""


class ReaderTickets:
    """Cache zero-ending counters per eager stream or per stream within a capture. Separate graphs can
    replay concurrently even when they were captured on the same stream."""

    def __init__(self):
        self.eager = {}
        self.last_capture = (None, {})

    def get(self, size, device):
        import torch

        cache = self.eager
        stream = None
        if device.type == "cuda":
            from cudnn.frost.capture import capture_state

            stream = torch.cuda.current_stream(device).cuda_stream
            capturing, capture_id = capture_state(stream)
            if capturing:
                previous_id, cache = self.last_capture
                if previous_id != capture_id:
                    # Keep one capture's cache. The graph pool owns the storage for replay even
                    # after a later capture displaces these references.
                    cache = {}
                    self.last_capture = (capture_id, cache)
        key = (size, stream)
        if key not in cache:
            cache[key] = torch.zeros(size, device=device, dtype=torch.int32)
        return cache[key]
