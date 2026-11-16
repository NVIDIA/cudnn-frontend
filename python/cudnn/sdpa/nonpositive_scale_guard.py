# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native half-SDPA engines with incorrect nonpositive-scale results (#1428).

Hopper forward engines 2/9 and Blackwell forward engines 3/10 return NaNs
or incorrect finite O for zero/negative scales, including without Stats.
Blackwell backward engine 5 fails independently with correct O/Stats inputs.
Measured on 9.16 and 9.28; no safe earlier release or fixed version is known.
Legacy forward 2/3 remain important on older libraries that offer them.

Some full-tile forward configurations happen to work. The common guard excludes
an unsafe engine as a whole: its other knob choices and masked tails still fail.
Other native candidates are still checked (forward 1/8, backward 7). Tensor scales
can change after planning, so their sign cannot be used to admit these engines.
No tensor value is read or synchronized during preparation or execution.
"""

from __future__ import annotations

# Per-engine measurements, not guessed release ceilings. The numerical currency
# detector in test_sdpa_nonpositive_scale_backend.py runs with this guard lifted.
# Update an entry only after that engine's full trigger matrix is correct.
FORWARD_FIXED_IN = {2: None, 3: None, 9: None, 10: None}
BACKWARD_FIXED_IN = {5: None}
NONDETERMINISTIC_BACKWARD_FIXED_IN = {6: None, 7: None}
# For aligned KV, H100 engine 6 is correct on 9.16 through 9.25.1.
# Its noncausal regression from 9.26 also affects deterministic Sq=1;
# determinism is not a correctness exemption. Engine 7's affected atomic
# configurations already fail on 9.16. Partial KV tiles have a separate bound.
NONDETERMINISTIC_BACKWARD_BROKEN_SINCE = {6: 92600, 7: 0}
BACKWARD_KV_TAIL_FIXED_IN = {6: None, 7: None}


def _kv_tail_engines(facts, version):
    """Unmasked partial KV tiles fail even with deterministic plans.

    Engine 7 TILE_N=1 (legacy KERNEL_CFG=26) is correct at KV64/128 but
    returns NaNs at KV127/129/257. Engine 6 has the same defect on Hopper,
    including 9.16 and 9.24. Explicit padding masks use a different, correct
    path, including variable device KV lengths and packed THD on Hopper.
    """
    if not facts.is_backward or facts.causal or facts.padded:
        return ()
    if facts.s_kv > 0 and facts.s_kv % 64 == 0:
        return ()
    if facts.device_cc == (9, 0):
        engines = (6, 7)
    elif facts.device_cc in ((8, 0), (8, 9), (10, 0), (10, 3)):
        engines = (7,)
    else:
        return ()
    return tuple(engine for engine in engines if BACKWARD_KV_TAIL_FIXED_IN[engine] is None or version < BACKWARD_KV_TAIL_FIXED_IN[engine])


def backend_engine_guard(facts):
    import cudnn

    if facts is None or facts.invalid is not None or facts.is_fp8 or facts.is_mxfp8:
        return {}
    if facts.dtype not in (cudnn.data_type.HALF, cudnn.data_type.BFLOAT16):
        return {}
    if not facts.dynamic_scale and (facts.scale is None or not facts.scale <= 0):
        return {}
    if facts.device_cc == (9, 0):
        engines = (6,) if facts.is_backward and not facts.causal and not facts.padded else (() if facts.is_backward else (2, 9))
    elif facts.device_cc in ((10, 0), (10, 3)):
        engines = (5,) if facts.is_backward else (3, 10)
    elif facts.is_backward and facts.device_cc in ((8, 0), (8, 9)):
        engines = ()
    else:
        return {}
    bounds = BACKWARD_FIXED_IN if facts.is_backward else FORWARD_FIXED_IN
    if facts.is_backward and facts.device_cc == (9, 0):
        bounds = NONDETERMINISTIC_BACKWARD_FIXED_IN
    version = cudnn.backend_version()
    scale = "a tensor attn_scale whose runtime value may be nonpositive" if facts.dynamic_scale else "attn_scale <= 0"
    excluded = {
        engine: f"cuDNN {cudnn.backend_version_string()} engine {engine} returns incorrect SDPA "
        f"{'gradients' if facts.is_backward else 'O/Stats'} for {scale} (NVIDIA/cudnn-frontend#1428); other backend candidates are still checked"
        for engine in engines
        if (bounds[engine] is None or version < bounds[engine])
        and (not facts.is_backward or engine not in NONDETERMINISTIC_BACKWARD_BROKEN_SINCE or version >= NONDETERMINISTIC_BACKWARD_BROKEN_SINCE[engine])
    }
    for engine in _kv_tail_engines(facts, version):
        excluded.setdefault(
            engine,
            f"cuDNN {cudnn.backend_version_string()} engine {engine} returns incorrect SDPA gradients for {scale} "
            "on unmasked backward with a partial KV tile (NVIDIA/cudnn-frontend#1428); "
            "a correct native fallback is not guaranteed",
        )
    return excluded


def backend_plan_guard(graph, facts):
    """Keep the broken engine-7 atomic dQ configs out while retaining its correct fallback.

    Measured on 9.16 (KERNEL_CFG=28 vs safe 26) and 9.28 (KERNEL_CFG=2,
    TILE_N=2 vs safe TILE_N=1 with aligned KV). These noncausal atomic configs
    corrupt dQ even with zero-initialized outputs. Unmasked partial KV tiles
    have an additional whole-engine exclusion, including deterministic graphs.
    Causal and explicit-padding controls remain correct. This is independent of engine 5's defect.
    """
    import cudnn

    if facts is None or facts.invalid is not None or not facts.is_backward or facts.deterministic or facts.causal or facts.padded:
        return None
    if facts.is_fp8 or facts.is_mxfp8 or facts.dtype not in (cudnn.data_type.HALF, cudnn.data_type.BFLOAT16):
        return None
    if not facts.dynamic_scale and (facts.scale is None or not facts.scale <= 0):
        return None
    fixed = NONDETERMINISTIC_BACKWARD_FIXED_IN[7]
    if fixed is not None and cudnn.backend_version() >= fixed:
        return None

    def check(engine_id, knobs):
        if engine_id != 7:
            return None
        cfg, tile_n = knobs.get(cudnn.knob_type.KERNEL_CFG), knobs.get(cudnn.knob_type.TILE_N)
        if cfg in (None, 28) or (cfg == 2 and tile_n in (None, 2)):
            return (
                f"cuDNN {cudnn.backend_version_string()} engine 7 atomic dQ config {knobs} returns incorrect SDPA gradients "
                "for nonpositive or runtime tensor attn_scale (NVIDIA/cudnn-frontend#1428); its other configs remain available"
            )
        return None

    return check
