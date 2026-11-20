# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent FP64 checks and repeated gradient-overwrite checks for BF16 K4."""

import torch


def reference_check(out, x, w, b, dy, lengths):
    ids = torch.linspace(0, x.shape[1] - 1, 8, device=x.device).long()
    xx, ww, bb, gy = x[:, ids].double().cpu(), w[ids].double().cpu(), b[ids].double().cpu(), dy[:, ids].double().cpu()
    dx = torch.zeros_like(xx)
    dw = torch.zeros_like(ww)
    db = torch.zeros_like(bb)
    start = 0
    for length in lengths:
        inp = xx[start : start + length]
        padded = torch.cat((torch.zeros((3, 8), dtype=torch.float64), inp))
        z = bb.expand(length, 8).clone()
        for k in range(4):
            z += padded[k : k + length] * ww[:, k]
        sig = z.sigmoid()
        g = gy[start : start + length] * sig * (1 + z * (1 - sig))
        for k in range(4):
            dw[:, k] += (g * padded[k : k + length]).sum(0)
            shift = 3 - k
            if length > shift:
                dx[start : start + length - shift] += g[shift:] * ww[:, k]
        db += g.sum(0)
        start += length
    errors = {}
    for key, ref in [("dx", dx), ("dw", dw), ("db", db)]:
        actual = (out[key][:, ids] if key == "dx" else out[key][ids]).cpu()
        torch.testing.assert_close(actual, ref.to(actual.dtype), atol=0.03, rtol=0.02)
        errors[key] = float((actual.double() - ref).norm() / ref.norm().clamp_min(1e-30))
    return errors


def replay_check(fn, out, route):
    fn()
    torch.cuda.synchronize()
    expected = {k: v.clone() for k, v in out.items()}
    for i in range(100):
        for v in out.values():
            v.fill_(float("nan"))
        fn()
        for key, v in out.items():
            assert torch.isfinite(v).all(), (i, key)
            if route != "dense-tiled" or key == "dx":
                assert torch.equal(v.view(torch.int16), expected[key].view(torch.int16)), (i, key)
            else:
                torch.testing.assert_close(v, expected[key], atol=0.03, rtol=0.02)
    return {"poisoned_replays": 100, "status": "passed"}
