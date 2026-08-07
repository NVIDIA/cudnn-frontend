# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``cudnn.frost.buffers.probe`` reads a buffer; it must not touch streams.

probe() is the boundary where a caller's framework buffer becomes
(pointer, shape, strides, dtype, device) for a kernel launch. It reads metadata
and nothing else, so it has no business synchronising -- and a producer that
does bookkeeping on its behalf breaks CUDA graph capture.

That is not hypothetical: the DLPack fallback called ``__dlpack__()`` with the
default stream argument, which makes torch call ``record_stream``, which is
illegal during capture. It took out every GDN and KDA capture test (286 of
them) while reporting only "operation failed due to a previous error during
capture" -- an error that names neither this function nor this file.
"""

import pytest

torch = pytest.importorskip("torch")

from cudnn.frost import buffers

pytestmark = pytest.mark.L0

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
# bfloat16 needs SM80+. Reported per dtype so float16/float32 still run on older
# cards -- they take the CAI path, which is the control for the DLPack one.
_BF16_OK = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8

# float16/float32 resolve through __cuda_array_interface__; bfloat16 has no
# numpy typestr (torch reports raw "<V2"), so it is the dtype that reaches the
# DLPack path -- the one that has to stay capture-safe. fp8 is absent because
# probe() does not map it at all ("unsupported buffer dtype (code=10, bits=8)"),
# which is a separate gap and not this fix's business.
_DTYPES = [
    torch.float16,
    torch.float32,
    pytest.param(torch.bfloat16, marks=pytest.mark.skipif(not _BF16_OK, reason="bfloat16 needs SM80+")),
]


@requires_cuda
@pytest.mark.parametrize("dtype", _DTYPES, ids=lambda d: str(d).split(".")[-1])
def test_probe_reports_the_buffer(dtype):
    t = torch.zeros((4, 8), dtype=dtype, device="cuda")
    ptr, shape, strides, name, device = buffers.probe(t)

    assert ptr == t.data_ptr()
    assert shape == (4, 8)
    assert strides in (None, (8, 1))  # None == densely packed
    assert name == str(dtype).split(".")[-1]
    assert device == t.device.index


@requires_cuda
@pytest.mark.parametrize("dtype", _DTYPES, ids=lambda d: str(d).split(".")[-1])
def test_probe_is_capture_safe(dtype):
    """Probing inside a capture must not abort it.

    Guarding here rather than trusting the linear-attention suites to notice:
    they do, but only after a hundred lines of unrelated setup and with an
    error that points at torch's stream bookkeeping.
    """
    t = torch.zeros(64, dtype=dtype, device="cuda")
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())  # order the zeros before the write
    with torch.cuda.stream(side):  # warm up allocation off the capture stream
        t.add_(1)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=torch.cuda.Stream()):
        buffers.probe(t)
        t.add_(1)
    graph.replay()
    torch.cuda.synchronize()
