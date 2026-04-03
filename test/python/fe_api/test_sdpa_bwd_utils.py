"""
Utilities for SDPA backward FE API tests.
"""

import logging
import math
from typing import Dict, Optional, Tuple
import pytest
import torch

SDPA_BWD_PARAM_MARKS = [
    pytest.mark.parametrize("layout", ["bhsd", "thd"]),
    pytest.mark.parametrize("dtype", [torch.float16]),
    pytest.mark.parametrize("is_causal", [True, False]),
    pytest.mark.parametrize("window_size", [(-1, -1), (64, 0)]),
]


def with_sdpa_bwd_params(func):
    for mark in reversed(SDPA_BWD_PARAM_MARKS):
        func = mark(func)
    return func


def sdpa_bwd_init(
    request,
    layout: str = "bhsd",
    dtype: torch.dtype = torch.float16,
    is_causal: bool = False,
    mma_tiler_mn: Tuple[int, int] = (128, 128),
    dkdv_mma_tiler_mn: Tuple[int, int] = (128, 64),
    window_size: Optional[Tuple[int, int]] = None,
) -> Dict:
    """Build the SDPA backward test configuration.

    Causal mode enforces `window_size = (-1, 0)`; any caller-provided
    `window_size` is ignored in causal mode (with a warning) and defaults to
    `(-1, -1)` when unspecified.
    """
    major, _ = torch.cuda.get_device_capability()
    if major < 10:
        pytest.skip(f"Environment not supported: requires compute capability >= 10, found {major}")

    b = request.config.getoption("--b") if request.config.getoption("--b") is not None else 2
    s_q = request.config.getoption("--s_q") if request.config.getoption("--s_q") is not None else 512
    s_kv = request.config.getoption("--s_kv") if request.config.getoption("--s_kv") is not None else 512
    d = 256  # d must be 256
    h_q = request.config.getoption("--h_q") if request.config.getoption("--h_q") is not None else 24
    h_k = request.config.getoption("--h_k") if request.config.getoption("--h_k") is not None else 4
    h_v = request.config.getoption("--h_v") if request.config.getoption("--h_v") is not None else 4

    actual_s_q = torch.tensor([s_q] * b, dtype=torch.int32).cuda() if layout == "thd" else None
    actual_s_kv = torch.tensor([s_kv] * b, dtype=torch.int32).cuda() if layout == "thd" else None
    skip_ref = request.config.getoption("--skip-ref", default=False)

    window_specified = window_size is not None
    if window_size is None:
        window_size = (-1, -1)

    if is_causal:
        desired_window = (-1, 0)
        if window_specified and window_size != desired_window:
            logging.info(f"window_size={window_size} ignored for causal mode; forcing {desired_window}")
        window_size = desired_window

    return {
        "layout": layout,
        "dtype": dtype,
        "is_causal": is_causal,
        "mma_tiler_mn": mma_tiler_mn,
        "dkdv_mma_tiler_mn": dkdv_mma_tiler_mn,
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


def allocate_sdpa_bwd_input_tensors(cfg: Dict) -> Dict:
    b = cfg["b"]
    s_q = cfg["s_q"]
    s_kv = cfg["s_kv"]
    d = cfg["d"]
    h_q = cfg["h_q"]
    h_k = cfg["h_k"]
    h_v = cfg["h_v"]
    actual_s_q = cfg["actual_s_q"]
    actual_s_kv = cfg["actual_s_kv"]

    dtype = cfg["dtype"]
    layout = cfg["layout"]

    # ref_tensor = (
    #             torch.empty(*shape, dtype=torch.float32)
    #             .random_(min_val, max_val)
    #             .permute(permute_order)
    #         )
    if layout == "bhsd":
        q = torch.empty(b, s_q, h_q, d, dtype=torch.float32).random_(-2, 2).transpose(1, 2).to(dtype).cuda()
        k = torch.empty(b, s_kv, h_k, d, dtype=torch.float32).random_(-2, 2).transpose(1, 2).to(dtype).cuda()
        v = torch.empty(b, s_kv, h_v, d, dtype=torch.float32).random_(-2, 2).transpose(1, 2).to(dtype).cuda()
        o = torch.empty(b, s_q, h_q, d, dtype=torch.float32).random_(-2, 2).transpose(1, 2).to(dtype).cuda()
        do = torch.empty(b, s_q, h_q, d, dtype=torch.float32).random_(-2, 2).transpose(1, 2).to(dtype).cuda()
        lse = torch.empty(b, h_q, s_q, dtype=torch.float32).random_(10, 11).cuda()
        cum_seqlen_q = None
        cum_seqlen_kv = None
    elif layout == "thd":
        cum_seqlen_q = torch.cat([torch.tensor([0]).cuda(), torch.cumsum(actual_s_q, dim=0)]).to(torch.int32).cuda()
        cum_seqlen_kv = torch.cat([torch.tensor([0]).cuda(), torch.cumsum(actual_s_kv, dim=0)]).to(torch.int32).cuda()

        total_seq_len_q = actual_s_q.sum().item()
        total_seq_len_kv = actual_s_kv.sum().item()
        q = torch.empty((total_seq_len_q, h_q, d), dtype=torch.float32).random_(-2, 2).to(dtype).cuda()
        k = torch.empty((total_seq_len_kv, h_k, d), dtype=torch.float32).random_(-2, 2).to(dtype).cuda()
        v = torch.empty((total_seq_len_kv, h_v, d), dtype=torch.float32).random_(-2, 2).to(dtype).cuda()
        o = torch.empty((total_seq_len_q, h_q, d), dtype=torch.float32).random_(-2, 2).to(dtype).cuda()
        do = torch.empty((total_seq_len_q, h_q, d), dtype=torch.float32).random_(-2, 2).to(dtype).cuda()
        lse = torch.empty((1, h_q, total_seq_len_q), dtype=torch.float32).random_(10, 11).transpose(0, 2).cuda()  # TODO @mingyangw

    return {
        "q": q,
        "k": k,
        "v": v,
        "o": o,
        "do": do,
        "lse": lse,
        "cum_seqlen_q": cum_seqlen_q,
        "cum_seqlen_kv": cum_seqlen_kv,
    }


def allocate_sdpa_bwd_output_tensors(cfg: Dict) -> Dict:
    b = cfg["b"]
    s_q = cfg["s_q"]
    s_kv = cfg["s_kv"]
    d = cfg["d"]
    h_q = cfg["h_q"]
    h_k = cfg["h_k"]
    h_v = cfg["h_v"]
    actual_s_q = cfg["actual_s_q"]
    actual_s_kv = cfg["actual_s_kv"]

    dtype = cfg["dtype"]
    layout = cfg["layout"]

    if layout == "bhsd":
        dq = torch.empty(b, s_q, h_q, d, dtype=dtype).transpose(1, 2).cuda()
        dk = torch.empty(b, s_kv, h_k, d, dtype=dtype).transpose(1, 2).cuda()
        dv = torch.empty(b, s_kv, h_v, d, dtype=dtype).transpose(1, 2).cuda()
    elif layout == "thd":
        total_seq_len_q = actual_s_q.sum().item()
        total_seq_len_k = actual_s_kv.sum().item()
        dq = torch.empty(total_seq_len_q, h_q, d, dtype=dtype).cuda()
        dk = torch.empty(total_seq_len_k, h_k, d, dtype=dtype).cuda()
        dv = torch.empty(total_seq_len_k, h_v, d, dtype=dtype).cuda()
    return {
        "dq": dq,
        "dk": dk,
        "dv": dv,
    }


def sdpa_bwd_reference(
    problem_shape: Tuple[int, int, int, Tuple[Tuple[int, int], int]],
    q: torch.Tensor,  # [B, S_q, H_k, H_r, D]
    k: torch.Tensor,  # [B, S_k, H_k, 1,   D]
    v: torch.Tensor,  # [B, S_k, H_k, 1,   D]
    do: torch.Tensor,  # [B, S_q, H_k, H_r, D]
    o: torch.Tensor,  # [B, S_q, H_k, H_r, D]
    lse: torch.Tensor,  # [B, H_k, H_r, S_q]
    cumulative_s_q: torch.Tensor | None,
    cumulative_s_k: torch.Tensor | None,
    is_causal: bool,
    window_size: Tuple[int, int],
    upcast: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Torch reference for SDPA backward.

    This is ported from the original kernel source and intentionally kept
    close to the kernel math for comparison in FE API tests.
    """

    s_q_max, s_k_max, _, hb = problem_shape
    (h_r, h_k), orig_b = hb

    if upcast:
        q = q.to(dtype=torch.float32)
        k = k.to(dtype=torch.float32)
        v = v.to(dtype=torch.float32)
        do = do.to(dtype=torch.float32)
        o = o.to(dtype=torch.float32)
        lse = lse.to(dtype=torch.float32)

    softmax_scale = 1.0 / math.sqrt(problem_shape[2])
    dv = torch.zeros_like(v)
    dk = torch.zeros_like(k)
    dq = torch.zeros_like(q)

    for b in range(orig_b):
        q_offset = int(cumulative_s_q[b].item()) if cumulative_s_q is not None else 0
        k_offset = int(cumulative_s_k[b].item()) if cumulative_s_k is not None else 0
        s_q = int((cumulative_s_q[b + 1] - cumulative_s_q[b]).item()) if cumulative_s_q is not None else s_q_max
        s_k = int((cumulative_s_k[b + 1] - cumulative_s_k[b]).item()) if cumulative_s_k is not None else s_k_max

        for h_k_idx in range(h_k):
            # Packed-varlen representation stores all sequences in physical batch 1.
            b_idx = 0 if cumulative_s_k is not None else b
            cur_k = k[b_idx, k_offset : k_offset + s_k, h_k_idx, 0, :]
            cur_v = v[b_idx, k_offset : k_offset + s_k, h_k_idx, 0, :]

            for h_r_idx in range(h_r):
                cur_q = q[b_idx, q_offset : q_offset + s_q, h_k_idx, h_r_idx, :]
                cur_do = do[b_idx, q_offset : q_offset + s_q, h_k_idx, h_r_idx, :]
                cur_o = o[b_idx, q_offset : q_offset + s_q, h_k_idx, h_r_idx, :]
                cur_lse = lse[b_idx, h_k_idx, h_r_idx, q_offset : q_offset + s_q]

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

                cur_p = torch.exp(cur_s - cur_lse.reshape(cur_lse.shape[0], 1))
                cur_pt = cur_p.transpose(1, 0).to(dtype=q.dtype)
                cur_dv = torch.einsum("kq,qd->kd", cur_pt, cur_do)

                cur_dp = torch.einsum("qd,kd->qk", cur_do, cur_v)
                cur_d = torch.einsum("qd,qd->q", cur_o, cur_do).reshape(-1, 1)
                cur_ds = cur_p * (cur_dp - cur_d) * softmax_scale
                cur_ds = cur_ds.to(dtype=q.dtype)

                cur_dst = cur_ds.transpose(1, 0)
                cur_dk = torch.einsum("kq,qd->kd", cur_dst, cur_q)
                cur_dq = torch.einsum("qk,kd->qd", cur_ds, cur_k)

                dq[b_idx, q_offset : q_offset + s_q, h_k_idx, h_r_idx, :] = cur_dq
                dv[b_idx, k_offset : k_offset + s_k, h_k_idx, 0, :] += cur_dv
                dk[b_idx, k_offset : k_offset + s_k, h_k_idx, 0, :] += cur_dk

    return dv.to(dtype=torch.float32), dk.to(dtype=torch.float32), dq.to(dtype=torch.float32)


def _to_reference_layout(cfg: Dict, inputs: Dict):
    layout = cfg["layout"]
    h_q = cfg["h_q"]
    h_k = cfg["h_k"]
    h_r = h_q // h_k

    if layout == "bhsd":
        # (B, H_q, S, D) -> (B, S, H_k, H_r, D)
        def q_like(x):
            b, h_q, s, d = x.shape
            return x.view(b, h_k, h_r, s, d).permute(0, 3, 1, 2, 4).contiguous()

        # (B, H_k, S, D) -> (B, S, H_k, 1, D)
        def kv_like(x):
            return x.permute(0, 2, 1, 3).unsqueeze(3).contiguous()

        q = q_like(inputs["q"])
        k = kv_like(inputs["k"])
        v = kv_like(inputs["v"])
        o = q_like(inputs["o"])
        do = q_like(inputs["do"])

        # (B, H_q, S) -> (B, H_k, H_r, S)
        lse = inputs["lse"].view(inputs["lse"].shape[0], h_k, h_r, inputs["lse"].shape[2]).contiguous()
        return q, k, v, do, o, lse

    if layout == "thd":
        # (T, H_q, D) -> (1, T, H_k, H_r, D)
        def q_like(x):
            t, h_q, d = x.shape
            return x.view(t, h_k, h_r, d).unsqueeze(0).contiguous()

        # (T, H_k, D) -> (1, T, H_k, 1, D)
        def kv_like(x):
            return x.unsqueeze(0).unsqueeze(3).contiguous()

        q = q_like(inputs["q"])
        k = kv_like(inputs["k"])
        v = kv_like(inputs["v"])
        o = q_like(inputs["o"])
        do = q_like(inputs["do"])

        # (T, H_q, 1) or (T, H_q) -> (1, H_k, H_r, T)
        lse_thd = inputs["lse"].squeeze(-1) if inputs["lse"].ndim == 3 else inputs["lse"]
        lse = lse_thd.transpose(0, 1).contiguous().view(h_k, h_r, lse_thd.shape[0]).unsqueeze(0).contiguous()
        return q, k, v, do, o, lse

    raise ValueError(f"Unsupported layout={layout}")


def check_ref_sdpa_bwd(
    cfg: Dict,
    inputs: Dict,
    outputs: Dict,
    *,
    skip_ref: bool = False,
):
    if skip_ref:
        print("Skipping reference check")
        return

    if cfg["layout"] == "thd":
        problem_shape = (max(cfg["actual_s_q"]).item(), max(cfg["actual_s_kv"]).item(), cfg["d"], ((cfg["h_q"] // cfg["h_k"], cfg["h_k"]), cfg["b"]))
    else:
        problem_shape = (cfg["s_q"], cfg["s_kv"], cfg["d"], ((cfg["h_q"] // cfg["h_k"], cfg["h_k"]), cfg["b"]))
    q_ref, k_ref, v_ref, do_ref, o_ref, lse_ref = _to_reference_layout(cfg, inputs)

    dv_ref, dk_ref, dq_ref = sdpa_bwd_reference(
        problem_shape,
        q_ref.to(dtype=cfg["dtype"]),
        k_ref.to(dtype=cfg["dtype"]),
        v_ref.to(dtype=cfg["dtype"]),
        do_ref.to(dtype=cfg["dtype"]),
        o_ref.to(dtype=cfg["dtype"]),
        lse_ref,
        inputs["cum_seqlen_q"],
        inputs["cum_seqlen_kv"],
        cfg["is_causal"],
        cfg["window_size"],
    )

    if cfg["layout"] == "bhsd":
        dq_ref = dq_ref.permute(0, 2, 3, 1, 4).reshape_as(outputs["dq"])
        dk_ref = dk_ref.squeeze(3).permute(0, 2, 1, 3).reshape_as(outputs["dk"])
        dv_ref = dv_ref.squeeze(3).permute(0, 2, 1, 3).reshape_as(outputs["dv"])
    else:
        dq_ref = dq_ref.squeeze(0).reshape_as(outputs["dq"])
        dk_ref = dk_ref.squeeze(0).squeeze(2).reshape_as(outputs["dk"])
        dv_ref = dv_ref.squeeze(0).squeeze(2).reshape_as(outputs["dv"])

    torch.testing.assert_close(outputs["dq"].to(dtype=torch.float32), dq_ref.to(dtype=torch.float32), atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(outputs["dk"].to(dtype=torch.float32), dk_ref.to(dtype=torch.float32), atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(outputs["dv"].to(dtype=torch.float32), dv_ref.to(dtype=torch.float32), atol=1e-3, rtol=1e-3)

    print("Reference check passed")
