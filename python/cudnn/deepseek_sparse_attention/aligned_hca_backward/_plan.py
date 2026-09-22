# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Precompiled native HCA pipeline: six Triton kernels and three cuDNN GEMMs."""

import math

import torch
from triton.runtime.jit import MockTensor
from triton.tools.tensor_descriptor import TensorDescriptor

from ._kernels import normalize, reduce_local_keys, reduce_local_rows, reduce_sink, pack_rank_major_keys, reduce_rank_major_keys, separate_tma_scores
from ._workspace import workspace_layout


class _HcaPlan:
    """Fixed aligned S65536/CP16 BF16 HCA, physical KV shape (4752, 512)."""

    def __init__(self, rank, softmax_scale, device):
        self.layout = workspace_layout(rank)
        self._gemms = None
        if type(softmax_scale) not in (int, float) or not math.isfinite(softmax_scale):
            raise ValueError("softmax_scale must be a finite host scalar")
        self.scale = float(softmax_scale)
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("A CUDA device is required")
        self.device = torch.device("cuda", torch.cuda.current_device() if device.index is None else device.index)
        if torch.cuda.get_device_capability(self.device) != (10, 3):
            raise ValueError("This plan is validated only for GB300")
        self.sms = torch.cuda.get_device_properties(self.device).multi_processor_count
        self._launches = None

    @property
    def workspace_size(self):
        if self._gemms is None:
            raise RuntimeError("Compile cuDNN plans before querying workspace")
        gemm_bytes = self._gemms.workspace_size if self._gemms is not None else 0
        return self.layout.nbytes + ((gemm_bytes + 255) // 256) * 256

    def _schedule(self):
        w = self.layout
        n, h, d = 4096 * 128, 128, 512
        g, k, gt, nc = w.groups, w.keys, w.group_tokens, w.compressed_keys
        preparation = (normalize, (n // 16, 1, 1), ("out", "dout", "lse", "sink", "normalization", "sink_partial"), (n, h, d, 16), dict(num_warps=4))
        scores = (
            separate_tma_scores,
            (self.sms, 1, 1),
            ("q_desc", "do_desc", "keys_desc", "normalization", "p_desc", "ds_desc"),
            (n, k, h, d, w.rank * 4096, nc, self.scale, gt, 128, 128, 64, self.sms),
            dict(num_warps=8, num_stages=4 if w.rank in (0, 3, 4, 9, 11, 13, 14, 15) else 3),
        )
        if gt == 32:
            local_reduction = (reduce_local_rows, (4224 * d // 2048, 1, 1), ("dk", "dv", "dkv"), (k, g, d, gt, 2048), dict(num_warps=4))
        else:
            local_reduction = (reduce_local_keys, (4224 * 4, 1, 1), ("dk", "dv", "dkv"), (k, g, d, gt, 128, False), dict(num_warps=4))
        return (
            preparation,
            (pack_rank_major_keys, (k * d // 1024, g, 1), ("kv", "packed_keys"), (k, nc, gt, 1024, w.rank * 4096), {}),
            scores,
            local_reduction,
            (reduce_rank_major_keys, (528 * 4, 1, 1), ("dk", "dv", "dkv"), (nc, k, g, gt, 128), dict(num_warps=8)),
            (reduce_sink, (32, 1, 1), ("sink_partial", "d_sink"), (4096, h), dict(num_warps=8)),
        )

    def _descriptors(self, tensors):
        w = self.layout
        n = 4096 * 128
        tensors.update(
            q_desc=TensorDescriptor(tensors["q"], [n, 512], [512, 1], [128, 64]),
            do_desc=TensorDescriptor(tensors["dout"], [n, 512], [512, 1], [128, 64]),
            keys_desc=TensorDescriptor(tensors["packed_keys"], [w.groups * w.keys, 512], [512, 1], [128, 64]),
            p_desc=TensorDescriptor(tensors["p"], [n, w.keys], [w.keys, 1], [128, 128]),
            ds_desc=TensorDescriptor(tensors["ds"], [n, w.keys], [w.keys, 1], [128, 128]),
        )

    def compile(self):
        if self._launches is not None:
            return self
        tensors = {name: MockTensor(torch.bfloat16) for name in ("q", "kv", "out", "dout", "dq", "dkv")}
        tensors.update({name: MockTensor(torch.float32) for name in ("lse", "sink", "d_sink")})
        for spec in self.layout.buffers:
            tensors[spec.name] = MockTensor(getattr(torch, spec.dtype), spec.shape)
        self._descriptors(tensors)
        launches = []
        with torch.cuda.device(self.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Compile the HCA plan before capture")
            for kernel, grid, names, constants, options in self._schedule():
                compiled = kernel.warmup(*(tensors[name] for name in names), *constants, grid=grid, **options)
                launches.append((compiled[grid], names, constants))
            from ._gemms import _HcaGemms

            self._gemms = _HcaGemms(self.layout)
        self._launches = tuple(launches)
        return self

    def _validate(self, tensors, workspace, stream):
        if self._launches is None:
            raise RuntimeError("Compile the HCA plan before execute")
        if not isinstance(stream, torch.cuda.Stream) or stream.device != self.device:
            raise ValueError("An explicit Torch CUDA stream on the plan device is required")
        shapes = dict(
            q=(4096, 128, 512),
            out=(4096, 128, 512),
            dout=(4096, 128, 512),
            dq=(4096, 128, 512),
            kv=(4752, 512),
            dkv=(4752, 512),
            lse=(4096, 128),
            sink=(128,),
            d_sink=(128,),
        )
        for name, tensor in tensors.items():
            dtype = torch.float32 if name in ("lse", "sink", "d_sink") else torch.bfloat16
            if (
                tuple(tensor.shape) != shapes[name]
                or tensor.dtype != dtype
                or tensor.device != self.device
                or not tensor.is_contiguous()
                or tensor.data_ptr() % 16
            ):
                raise ValueError(f"Unsupported {name} metadata or pointer alignment")
        if (
            workspace.dtype != torch.uint8
            or workspace.ndim != 1
            or not workspace.is_contiguous()
            or workspace.device != self.device
            or workspace.numel() < self.workspace_size
            or workspace.data_ptr() % 256
        ):
            raise ValueError("Workspace must be sufficiently large, contiguous CUDA uint8 and 256-byte aligned")
        inputs = {tensors[name].untyped_storage().data_ptr() for name in ("q", "kv", "out", "dout", "lse", "sink")}
        outputs = [tensors[name].untyped_storage().data_ptr() for name in ("dq", "dkv", "d_sink")]
        outputs.append(workspace.untyped_storage().data_ptr())
        if len(set(outputs)) != len(outputs) or inputs.intersection(outputs):
            raise ValueError("Outputs and workspace must have disjoint storage from each other and inputs")

    def execute(self, q, kv, out, dout, lse, sink, dq, dkv, d_sink, workspace, *, stream):
        tensors = dict(q=q, kv=kv, out=out, dout=dout, lse=lse, sink=sink, dq=dq, dkv=dkv, d_sink=d_sink)
        self._validate(tensors, workspace, stream)
        w = self.layout
        with torch.cuda.device(self.device), torch.cuda.stream(stream):
            for spec in w.buffers:
                tensors[spec.name] = workspace.narrow(0, spec.offset, spec.nbytes).view(getattr(torch, spec.dtype)).view(spec.shape)
            self._descriptors(tensors)

            def launch(index):
                run, names, constants = self._launches[index]
                run(*(tensors[name] for name in names), *constants, stream=stream.cuda_stream)

            launch(0)
            launch(1)
            launch(2)
            gemm_workspace = workspace.narrow(0, w.nbytes, self._gemms.workspace_size)
            self._gemms.execute(tensors, gemm_workspace, stream)
            launch(3)
            launch(4)
            launch(5)
        return {"dq": dq, "dkv": dkv, "d_sink": d_sink}
