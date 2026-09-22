# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Boundary contracts of the explicit f16/bf16 SDPA host entry."""

import math
from pathlib import Path

import pytest
import torch

from frost_test_utils import launch_f16, requires_blackwell, requires_dsl, requires_pre_rubin_blackwell

pytestmark = [pytest.mark.L0, requires_dsl]


def _load(arch, template, d_qk, d_v, **overrides):
    from cudnn.frost.template_loader import load_template
    from cudnn.sdpa.fwd import api_dsl
    from cudnn.sdpa.fwd.config_sm100 import TemplateParams

    values = dict(dtype_qkv=3, seq_kv_lens_present=True, cta_mma=1 if template.startswith("decode") else 2)
    if template == "decode_d256_f16":
        values["decode_q_tile"] = 16
    values.update(overrides)
    path = Path(api_dsl.__file__).parent / "kernels" / arch / f"{template}.py"
    return load_template(str(path), TemplateParams(**values), tag=f"pointer_abi_{arch}_{template}_{d_qk}_{d_v}")


@requires_blackwell
@pytest.mark.parametrize(
    "arch,template,d_qk,d_v",
    [
        ("sm100", "decode_d128_f16", 128, 128),
        ("sm100", "decode_d256_f16", 256, 256),
        ("sm100", "prefill_d128_f16", 128, 128),
        ("sm100", "prefill_d192_d128_f16", 192, 128),
        ("sm100", "prefill_d256_f16", 256, 256),
        ("sm100", "prefill_d512_f16", 512, 512),
        ("sm107", "prefill_d128_f16", 128, 128),
        ("sm107", "prefill_d192_d128_f16", 192, 128),
        ("sm107", "prefill_d512_f16", 512, 512),
    ],
)
def test_page_table_singleton_batch_stride_is_int64(arch, template, d_qk, d_v):
    """An unused singleton stride above int32 is legal without a huge allocation.

    Exercise the compiled FFI boundary and check both O and LSE, so merely widening
    the annotation while leaving an int32 compile placeholder remains RED. Templates
    without paged KV check their unused ABI slot on a dense launch, not paged support.
    """
    major, minor = torch.cuda.get_device_capability()
    if (arch == "sm107") != ((major, minor) == (10, 7)):
        pytest.skip("run each template on its native architecture")

    import cuda.bindings.driver as cuda_driver

    paged = arch == "sm100" and template != "prefill_d512_f16"
    mod = _load(arch, template, d_qk, d_v, **(dict(paged_kv=True, page_size=128) if paged else {}))
    fn = mod.compile(d_qk=d_qk, d_v=d_v, has_lse=True, paged_hnd=False)
    gen = torch.Generator(device="cuda").manual_seed(17)
    q = torch.randn((1, 4, 2, d_qk), device="cuda", dtype=torch.float16, generator=gen)
    k = torch.randn((2, 128, 2, d_qk), device="cuda", dtype=torch.float16, generator=gen)
    v = torch.randn((2, 128, 2, d_v), device="cuda", dtype=torch.float16, generator=gen)
    o = torch.full((1, 4, 2, d_v), float("nan"), device="cuda", dtype=torch.float16)
    lse = torch.full((1, 2, 4), float("nan"), device="cuda", dtype=torch.float32)
    k_table = torch.tensor([[1, 0]], device="cuda", dtype=torch.int32).as_strided((1, 2), (2**31 + 32, 1))
    v_table = torch.tensor([[0, 1]], device="cuda", dtype=torch.int32).as_strided((1, 2), (2**31 + 32, 1))
    lens = torch.tensor([193], device="cuda", dtype=torch.int32)
    scale = 1.0 / math.sqrt(d_qk)

    def launch_dense_with_unused_table_strides(**kwargs):
        kwargs["table_strides"] = (2**31 + 32, 1)
        fn(**kwargs)

    launch_f16(
        fn if paged else launch_dense_with_unused_table_strides,
        q,
        k if paged else k.view(1, 256, 2, d_qk),
        v if paged else v.view(1, 256, 2, d_v),
        o,
        lse,
        torch.zeros(2, device="cuda", dtype=torch.float32),
        lens,
        torch.zeros(1, device="cuda", dtype=torch.int64),
        (1, 2, 2, 4, 256, 0),
        scale * math.log2(math.e),
        0,
        0,
        block_table_tensor=k_table if paged else None,
        block_table_v_tensor=v_table if paged else None,
        page_size=128,
        stream=cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream),
        host=mod._host,
    )
    keys = (k[[1, 0]] if paged else k).reshape(256, 2, d_qk)[:193].double()
    values = v.reshape(256, 2, d_v)[:193].double()
    scores = torch.einsum("qhd,khd->hqk", q[0].double(), keys) * scale
    expected_o = torch.einsum("hqk,khd->qhd", scores.softmax(-1), values)
    torch.testing.assert_close(o[0].float(), expected_o.float(), atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(lse[0], scores.logsumexp(-1).float(), atol=2e-3, rtol=2e-3)


@requires_pre_rubin_blackwell
@pytest.mark.parametrize("d", [128, 256])
def test_direct_decode_launch_rejects_wrong_packed_head_ratio(d):
    """Reject an invalid private-call contract before any pointer is bound or launched."""
    mod = _load("sm100", f"decode_d{d}_f16", d, d, pack_gqa=True, qh_per_kh=4)
    with pytest.raises(ValueError, match="PACK_GQA requires"):
        launch_f16(None, *([None] * 8), (1, 4, 2, 1, 128, 0), 1.0, 0, 0, host=mod._host)


@requires_pre_rubin_blackwell
def test_direct_d256_decode_launch_rejects_rows_outside_tile():
    mod = _load("sm100", "decode_d256_f16", 256, 256)
    with pytest.raises(ValueError, match="decode Q rows exceed"):
        launch_f16(None, *([None] * 8), (1, 2, 2, 17, 128, 0), 1.0, 0, 0, host=mod._host)


@requires_pre_rubin_blackwell
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("splits", [1, 2])
def test_d256_paged_host_rebinds_table_column_stride(batch, splits):
    """One compiled host handles unit, nonunit, then unit strides, including wide unused slots."""
    import cuda.bindings.driver as cuda_driver

    h, d, page_size = 8, 256, 16
    mod = _load("sm100", "decode_d256_f16", d, d, paged_kv=True, page_size=page_size, pack_gqa=True, qh_per_kh=h, split_kv=splits)
    fn = mod.compile(d_qk=d, d_v=d, has_lse=True, paged_hnd=False)
    gen = torch.Generator(device="cuda").manual_seed(912)
    q = torch.randn((batch, 1, h, d), device="cuda", dtype=torch.float16, generator=gen)
    sinks = torch.zeros(h, device="cuda", dtype=torch.float32)
    desc = torch.zeros(1, device="cuda", dtype=torch.int64)
    scale = 1.0 / math.sqrt(d)
    stream = cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)

    for pages, column_stride in ((16, 1), (16, 3), (16, 1), (1, 2**31 + 32)):
        num_pages = batch * pages + 2
        k = torch.randn((num_pages, page_size, 1, d), device="cuda", dtype=torch.float16, generator=gen)
        v = torch.randn((num_pages, page_size, 1, d), device="cuda", dtype=torch.float16, generator=gen)
        # Padding holds a valid, deliberately different page: reading it exposes
        # a mistaken unit-stride path without an out-of-bounds page access.
        v[0].fill_(42.0)
        batch_stride = 2**31 + 32 if batch == 1 else (pages - 1) * column_stride + 3
        span = (batch - 1) * batch_stride + (pages - 1) * column_stride + 1
        tables = []
        for reverse in (False, True):
            table = torch.zeros(span, device="cuda", dtype=torch.int32).as_strided((batch, pages), (batch_stride, column_stride))
            ids = torch.arange(1, batch * pages + 1, device="cuda", dtype=torch.int32).view(batch, pages)
            table.copy_(ids.flip(1) if reverse else ids)
            tables.append(table)
        k_table, v_table = tables
        lens = torch.full((batch,), pages * page_size - 3, device="cuda", dtype=torch.int32)
        o = torch.full((batch * splits, 1, h, d), float("nan"), device="cuda", dtype=torch.float32 if splits > 1 else q.dtype)
        lse = torch.full((batch * splits, h, 1), float("nan"), device="cuda", dtype=torch.float32)
        launch_f16(
            fn,
            q,
            k,
            v,
            o,
            lse,
            sinks,
            lens,
            desc,
            (batch, h, 1, 1, pages * page_size, 0),
            scale * math.log2(math.e),
            0,
            0,
            o_partial_f32=o if splits > 1 else None,
            block_table_tensor=k_table,
            block_table_v_tensor=v_table,
            page_size=page_size,
            stream=stream,
            host=mod._host,
        )
        length = pages * page_size - 3
        tiles = (length + 127) // 128
        for b in range(batch):
            keys = k[k_table[b].long()].reshape(-1, d).double()
            values = v[v_table[b].long()].reshape(-1, d).double()
            for split in range(splits):
                begin = (split * (tiles // splits) + min(split, tiles % splits)) * 128
                end = min(begin + (tiles // splits + (split < tiles % splits)) * 128, length)
                out_index = b + split * batch
                if begin >= end:
                    assert torch.count_nonzero(o[out_index]) == 0
                    assert torch.isneginf(lse[out_index]).all()
                else:
                    scores = (q[b, 0].double() @ keys[begin:end].T) * scale
                    expected = scores.softmax(-1) @ values[begin:end]
                    torch.testing.assert_close(o[out_index, 0].float(), expected.float(), atol=2e-3, rtol=2e-3)
                    torch.testing.assert_close(lse[out_index, :, 0], scores.logsumexp(-1).float(), atol=2e-3, rtol=2e-3)
