"""
Utilities for SDPA forward FE API tests.
"""

import logging
import math
from typing import Dict, Optional, Tuple

import pytest
import torch

SDPA_FWD_PARAM_MARKS = [
    pytest.mark.parametrize("layout", ["bhsd", "thd"]),
    pytest.mark.parametrize("dtype", [torch.float16]),
    pytest.mark.parametrize("is_causal", [True, False]),
    pytest.mark.parametrize("window_size", [(-1, -1), (64, 0)]),
]


def with_sdpa_fwd_params(func):
    for mark in reversed(SDPA_FWD_PARAM_MARKS):
        func = mark(func)
    return func


def sdpa_fwd_init(
    request,
    layout: str = "bhsd",
    dtype: torch.dtype = torch.float16,
    is_causal: bool = False,
    mma_tiler_mn: Tuple[int, int] = (128, 128),
    window_size: Optional[Tuple[int, int]] = None,
) -> Dict:
    major, _ = torch.cuda.get_device_capability()
    if major < 10:
        pytest.skip(f"Environment not supported: requires compute capability >= 10, found {major}")

    b = request.config.getoption("--b") if request.config.getoption("--b") is not None else 2
    s_q = request.config.getoption("--s_q") if request.config.getoption("--s_q") is not None else 512
    s_kv = request.config.getoption("--s_kv") if request.config.getoption("--s_kv") is not None else 512
    d = 256
    h_q = request.config.getoption("--h_q") if request.config.getoption("--h_q") is not None else 24
    h_k = request.config.getoption("--h_k") if request.config.getoption("--h_k") is not None else 4
    h_v = request.config.getoption("--h_v") if request.config.getoption("--h_v") is not None else 4
    if h_k != h_v:
        pytest.skip(f"Forward d=256 kernel expects K/V head counts to match, got h_k={h_k}, h_v={h_v}")

    actual_s_q = torch.tensor([s_q] * b, dtype=torch.int32, device="cuda") if layout == "thd" else None
    actual_s_kv = torch.tensor([s_kv] * b, dtype=torch.int32, device="cuda") if layout == "thd" else None
    skip_ref = request.config.getoption("--skip-ref", default=False)

    window_specified = window_size is not None
    if window_size is None:
        window_size = (-1, -1)

    if is_causal:
        desired_window = (-1, 0)
        if window_specified and window_size != desired_window:
            logging.info("window_size=%s ignored for causal mode; forcing %s", window_size, desired_window)
        window_size = desired_window

    return {
        "layout": layout,
        "dtype": dtype,
        "is_causal": is_causal,
        "mma_tiler_mn": mma_tiler_mn,
        "window_size": window_size,
        "b": b,
        "s_q": s_q,
        "s_kv": s_kv,
        "d": d,
        "h_q": h_q,
        "h_k": h_k,
        "h_v": h_v,
        "actual_s_q": actual_s_q,
        "actual_s_kv": actual_s_kv,
        "skip_ref": skip_ref,
    }


def allocate_sdpa_fwd_input_tensors(cfg: Dict) -> Dict:
    b = cfg["b"]
    s_q = cfg["s_q"]
    s_kv = cfg["s_kv"]
    d = cfg["d"]
    h_q = cfg["h_q"]
    h_k = cfg["h_k"]
    actual_s_q = cfg["actual_s_q"]
    actual_s_kv = cfg["actual_s_kv"]
    dtype = cfg["dtype"]
    layout = cfg["layout"]

    if layout == "bhsd":
        q = torch.empty(b, s_q, h_q, d, dtype=torch.float32).random_(-2, 2).transpose(1, 2).to(dtype).cuda()
        k = torch.empty(b, s_kv, h_k, d, dtype=torch.float32).random_(-2, 2).transpose(1, 2).to(dtype).cuda()
        v = torch.empty(b, s_kv, h_k, d, dtype=torch.float32).random_(-2, 2).transpose(1, 2).to(dtype).cuda()
        cum_seqlen_q = None
        cum_seqlen_kv = None
    elif layout == "thd":
        cum_seqlen_q = torch.cat([torch.tensor([0], device="cuda"), torch.cumsum(actual_s_q, dim=0)]).to(torch.int32)
        cum_seqlen_kv = torch.cat([torch.tensor([0], device="cuda"), torch.cumsum(actual_s_kv, dim=0)]).to(torch.int32)
        total_seq_len_q = actual_s_q.sum().item()
        total_seq_len_kv = actual_s_kv.sum().item()
        q = torch.empty((total_seq_len_q, h_q, d), dtype=torch.float32).random_(-2, 2).to(dtype).cuda()
        k = torch.empty((total_seq_len_kv, h_k, d), dtype=torch.float32).random_(-2, 2).to(dtype).cuda()
        v = torch.empty((total_seq_len_kv, h_k, d), dtype=torch.float32).random_(-2, 2).to(dtype).cuda()
    else:
        raise ValueError(f"Unsupported layout: {layout}")

    return {
        "q": q,
        "k": k,
        "v": v,
        "cum_seqlen_q": cum_seqlen_q,
        "cum_seqlen_kv": cum_seqlen_kv,
    }


def allocate_sdpa_fwd_output_tensors(cfg: Dict, inputs: Dict) -> Dict:
    q = inputs["q"]
    cum_seqlen_q = inputs["cum_seqlen_q"]
    o = torch.empty_like(q)
    if cum_seqlen_q is None:
        lse = torch.empty((q.shape[0], q.shape[1], q.shape[2]), dtype=torch.float32, device=q.device)
    else:
        lse = torch.empty((q.shape[0], q.shape[1]), dtype=torch.float32, device=q.device)
    return {"o": o, "lse": lse}


def _to_reference_layout(cfg: Dict, inputs: Dict):
    layout = cfg["layout"]
    h_q = cfg["h_q"]
    h_k = cfg["h_k"]
    h_r = h_q // h_k

    if layout == "bhsd":
        q = inputs["q"].transpose(1, 2).reshape(cfg["b"], cfg["s_q"], h_k, h_r, cfg["d"])
        k = inputs["k"].transpose(1, 2).reshape(cfg["b"], cfg["s_kv"], h_k, 1, cfg["d"])
        v = inputs["v"].transpose(1, 2).reshape(cfg["b"], cfg["s_kv"], h_k, 1, cfg["d"])
    else:
        q = inputs["q"].reshape(1, inputs["q"].shape[0], h_k, h_r, cfg["d"])
        k = inputs["k"].reshape(1, inputs["k"].shape[0], h_k, 1, cfg["d"])
        v = inputs["v"].reshape(1, inputs["v"].shape[0], h_k, 1, cfg["d"])
    return q, k, v


def sdpa_fwd_reference(
    problem_size: Tuple[int, int, int, Tuple[Tuple[int, int], int]],
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cumulative_s_q: torch.Tensor | None,
    cumulative_s_k: torch.Tensor | None,
    is_causal: bool,
    window_size: Tuple[int, int],
    upcast: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    s_q_max, s_k_max, _, hb = problem_size
    (h_r, h_k), orig_b = hb

    if upcast:
        q = q.to(dtype=torch.float32)
        k = k.to(dtype=torch.float32)
        v = v.to(dtype=torch.float32)

    softmax_scale = 1.0 / math.sqrt(problem_size[2])
    o = torch.zeros_like(q)
    lse = torch.zeros((1 if cumulative_s_q is not None else orig_b, h_k, h_r, q.shape[1]), dtype=torch.float32, device=q.device)

    for b in range(orig_b):
        q_offset = int(cumulative_s_q[b].item()) if cumulative_s_q is not None else 0
        k_offset = int(cumulative_s_k[b].item()) if cumulative_s_k is not None else 0
        s_q = int((cumulative_s_q[b + 1] - cumulative_s_q[b]).item()) if cumulative_s_q is not None else s_q_max
        s_k = int((cumulative_s_k[b + 1] - cumulative_s_k[b]).item()) if cumulative_s_k is not None else s_k_max

        for h_k_idx in range(h_k):
            b_idx = 0 if cumulative_s_k is not None else b
            cur_k = k[b_idx, k_offset : k_offset + s_k, h_k_idx, 0, :]
            cur_v = v[b_idx, k_offset : k_offset + s_k, h_k_idx, 0, :]

            for h_r_idx in range(h_r):
                cur_q = q[b_idx, q_offset : q_offset + s_q, h_k_idx, h_r_idx, :]
                cur_s = torch.einsum("qd,kd->qk", cur_q, cur_k) * softmax_scale

                window_size_left, window_size_right = window_size
                if is_causal:
                    window_size_right = 0
                if window_size_left >= 0 or window_size_right >= 0:
                    q_coords = torch.arange(0, s_q, device=cur_s.device).view(-1, 1)
                    k_coords = torch.arange(0, s_k, device=cur_s.device).view(1, -1)
                    if window_size_left < 0:
                        mask = k_coords > q_coords + s_k - s_q + window_size_right
                    else:
                        mask = (k_coords > q_coords + s_k - s_q + window_size_right) | (k_coords < q_coords + s_k - s_q - window_size_left)
                    cur_s = cur_s.masked_fill(mask, -torch.inf)

                cur_lse = torch.logsumexp(cur_s, dim=-1)
                cur_p = torch.softmax(cur_s, dim=-1)
                cur_o = torch.einsum("qk,kd->qd", cur_p, cur_v)

                o[b_idx, q_offset : q_offset + s_q, h_k_idx, h_r_idx, :] = cur_o
                lse[b_idx, h_k_idx, h_r_idx, q_offset : q_offset + s_q] = cur_lse

    return o.to(dtype=torch.float32), lse.to(dtype=torch.float32)


def check_ref_sdpa_fwd(cfg: Dict, inputs: Dict, outputs: Dict, skip_ref: bool = False):
    if skip_ref:
        return

    q_ref, k_ref, v_ref = _to_reference_layout(cfg, inputs)
    h_r = cfg["h_q"] // cfg["h_k"]
    problem_size = (cfg["s_q"], cfg["s_kv"], cfg["d"], ((h_r, cfg["h_k"]), cfg["b"]))

    o_ref, lse_ref = sdpa_fwd_reference(
        problem_size,
        q_ref,
        k_ref,
        v_ref,
        inputs["cum_seqlen_q"],
        inputs["cum_seqlen_kv"],
        cfg["is_causal"],
        cfg["window_size"],
    )

    if cfg["layout"] == "bhsd":
        o_ref = o_ref.reshape(cfg["b"], cfg["s_q"], cfg["h_q"], cfg["d"]).transpose(1, 2).contiguous()
        lse_ref = lse_ref.reshape(cfg["b"], cfg["h_q"], cfg["s_q"]).contiguous()
    else:
        o_ref = o_ref.reshape(inputs["q"].shape[0], cfg["h_q"], cfg["d"]).contiguous()
        lse_ref = lse_ref.reshape(inputs["q"].shape[0], cfg["h_q"]).contiguous()

    torch.testing.assert_close(outputs["o"].float(), o_ref.float(), atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(outputs["lse"].float(), lse_ref.float(), atol=5e-2, rtol=5e-2)
