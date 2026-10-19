# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One native binder for the standalone SM80 packed host ABI."""

from functools import lru_cache
from types import SimpleNamespace


@lru_cache(maxsize=128)
def build_launch(module, device_index, h, h_kv, n_seq, swa_window, prefix_dtypes, sink_dtype):
    """Cache only the artifact and immutable host contract, never caller storage."""
    from cudnn import _pybind_module
    from .kernels.sm80.prepared_host import compile_thd_host

    artifact, fn = compile_thd_host(module, h, h_kv, n_seq, swa_window, prefix_dtypes, sink_dtype)
    params = module.PARAMS
    spec = SimpleNamespace(
        artifact=artifact,
        fn=fn,
        device_index=device_index,
        heads=(h, h_kv),
        dimensions=(params.d_qk, params.d_v),
        n_seq=n_seq,
        dtype="bfloat16" if params.io_bf16 else "float16",
        prefix_dtypes=tuple(t.removeprefix("torch.") for t in prefix_dtypes),
        sink_dtype=sink_dtype.removeprefix("torch."),
        has_sink=params.has_sink,
    )
    return _pybind_module._SdpaSm80ThdBinder(spec)


def execute(launch, buffers, max_s_q, scale, right_bound, stream):
    """Observe the current carriers once and invoke the sole native host binder."""
    from cudnn import _pybind_module
    from .prepared import _set_native_fact, facts_of_tensor

    pack, unread = _pybind_module._read_buffer_sequence(buffers)
    for index in unread:
        _set_native_fact(pack, index, facts_of_tensor(buffers[index]))
    launch.execute(pack, max_s_q, scale, right_bound, stream)
