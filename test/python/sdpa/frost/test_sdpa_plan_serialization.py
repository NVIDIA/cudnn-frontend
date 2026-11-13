# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A FROST SDPA plan serialized in one process deserializes in another (own, empty compiled cache)
and writes the same O. The blob's kernels are installed only under an identical environment
manifest; otherwise the plan compiles, and ``enforce_precompiled`` refuses."""

import json
import os
import subprocess
import sys

import pytest

from frost_test_utils import requires_blackwell, requires_dsl

pytestmark = [pytest.mark.L0, requires_blackwell, requires_dsl]

_PRELUDE = r"""
import hashlib, json, math, os, sys, torch, cudnn
from cudnn.frost import compiled_cache as cc
sys.path.insert(0, %(tests)r)
from frost_test_utils import select_engine, _SM
from cudnn.sdpa.fwd.engines import engine_name
b, h, s, d = 2, 8, 256, 128
torch.manual_seed(0)
# Both layouts bind in place; a split plan carries attention and combine artifacts.
BHSD = os.environ.get("PLAN_TEST_LAYOUT") == "bhsd"
alloc = (lambda f: f(b, h, s, d)) if BHSD else (lambda f: f(b, s, h, d).transpose(1, 2))
mk = lambda: alloc(lambda *shape: torch.randn(*shape, device="cuda", dtype=torch.bfloat16))
q_gpu, k_gpu, v_gpu = mk(), mk(), mk()
o_gpu = alloc(lambda *shape: torch.full(shape, float("nan"), device="cuda", dtype=torch.bfloat16))
UIDS = dict(q=1, k=2, v=3, o=4)

def run(g):
    ws = torch.empty(max(g.get_workspace_size(), 1), device="cuda", dtype=torch.uint8)
    g.execute({UIDS["q"]: q_gpu, UIDS["k"]: k_gpu, UIDS["v"]: v_gpu, UIDS["o"]: o_gpu}, ws); torch.cuda.synchronize()
    return hashlib.sha256(o_gpu.contiguous().view(torch.int16).cpu().numpy().tobytes()).hexdigest()
"""

_PRODUCE = r"""
g = cudnn.pygraph(io_data_type=cudnn.data_type.BFLOAT16, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
q, k, v = (g.tensor_like(t).set_uid(UIDS[n]) for n, t in (("q", q_gpu), ("k", k_gpu), ("v", v_gpu)))
o, _ = g.sdpa(name="sdpa", q=q, k=k, v=v, generate_stats=False, attn_scale=1.0 / math.sqrt(d), use_causal_mask=True)
o.set_output(True).set_dim(q_gpu.shape).set_stride(q_gpu.stride()).set_uid(UIDS["o"])
g.validate(); g.build_operation_graph(); g.create_execution_plans([cudnn.heur_mode.A])
select_engine(g, engine_name(arch="sm107" if _SM == 107 else "sm100"))
if BHSD:
    engine, knobs = g.get_engine_and_knobs_at_index(g._plan_index)
    knobs[cudnn.knob_type.SPLIT_KV] = 2
    g.create_execution_plan(engine, knobs)
    g.select_plan(len(g.plans) - 1)
g.check_support(); g.build_plans()
digest = run(g)
open(sys.argv[1], "wb").write(g.serialize(as_bytes=True))
open(sys.argv[1] + ".bare", "wb").write(g.serialize(as_bytes=True, include_kernels=False))
assert g.serialize() == list(open(sys.argv[1], "rb").read())
print(json.dumps({"digest": digest, "plan": g.get_plan_name_at_index(g._plan_index)}))
"""

_CONSUME = r"""
g = cudnn.pygraph()
try:
    g.deserialize(open(sys.argv[1], "rb").read(), enforce_precompiled=sys.argv[2] == "strict")
except Exception as exc:
    error = f"{type(exc).__name__}: {exc}"
    if sys.argv[3:] != ["retry"]:
        print(json.dumps({"error": error, "stats": cc.stats()})); sys.exit(0)
    g.deserialize(open(sys.argv[1], "rb").read())  # a strict refusal leaves the graph empty: retry loose on it
    print(json.dumps({"error": error, "digest": run(g), "stats": cc.stats()})); sys.exit(0)
print(json.dumps({"digest": run(g), "plan": g.get_plan_name_at_index(g._plan_index), "stats": cc.stats()}))
"""


def _child(source, cache_dir, *argv, layout="bshd"):
    env = dict(os.environ, CUDNN_FRONTEND_COMPILED_CACHE=str(cache_dir), PLAN_TEST_LAYOUT=layout)
    env.pop("CUDNN_FRONTEND_DISABLE_COMPILED_CACHE", None)
    code = (_PRELUDE % {"tests": os.path.dirname(os.path.abspath(__file__))}) + source
    out = subprocess.run([sys.executable, "-c", code, *map(str, argv)], env=env, capture_output=True, text=True, timeout=900)
    assert out.returncode == 0, out.stderr[-3000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def produced(tmp_path_factory):
    root = tmp_path_factory.mktemp("plan_serdes")
    blob = root / "plan.bin"
    return blob, _child(_PRODUCE, root / "producer_cache", blob)


@pytest.fixture(scope="module")
def produced_bhsd(tmp_path_factory):
    root = tmp_path_factory.mktemp("plan_serdes_bhsd")
    blob = root / "plan.bin"
    return blob, _child(_PRODUCE, root / "producer_cache", blob, layout="bhsd")


def _rewrite_header(blob_path, out_path, edit):
    import cudnn._plan_serdes as ps

    data = blob_path.read_bytes()
    start = len(ps.MAGIC) + ps._LEN.size
    (length,) = ps._LEN.unpack_from(data, len(ps.MAGIC))
    header = json.loads(data[start : start + length])
    assert header["kernels"], "the producer's plan carried no kernels"
    edit(header)
    text = json.dumps(header).encode()
    out_path.write_bytes(ps.MAGIC + ps._LEN.pack(len(text)) + text + data[start + length :])


def _with_foreign_environment(blob_path, out_path):
    """The same blob, its kernels claimed by another driver version."""

    def edit(header):
        for kernel in header["kernels"]:
            kernel["environment"]["cuda_driver"] = "1"

    _rewrite_header(blob_path, out_path, edit)


def test_frost_plan_round_trips_without_compiling(produced, tmp_path):
    blob, first = produced
    second = _child(_CONSUME, tmp_path / "consumer_cache", blob, "strict")
    assert "error" not in second, second
    assert second["plan"] == first["plan"]
    assert second["stats"]["misses"] == 0 and second["stats"]["seeded"] >= 1 and second["stats"]["hits"] >= 1, second
    assert second["digest"] == first["digest"], "the deserialized plan must write exactly what the serialized one did"


def test_kernels_from_another_environment_are_never_loaded(produced, tmp_path):
    blob, first = produced
    foreign = tmp_path / "foreign.bin"
    _with_foreign_environment(blob, foreign)
    loose = _child(_CONSUME, tmp_path / "loose_cache", foreign, "loose")
    assert "error" not in loose, loose
    assert loose["stats"]["seeded"] == 0 and loose["stats"]["seed_rejected"] >= 1 and loose["stats"]["misses"] >= 1, loose
    assert loose["digest"] == first["digest"]
    strict = _child(_CONSUME, tmp_path / "strict_cache", foreign, "strict", "retry")
    assert "enforce_precompiled" in strict.get("error", ""), strict
    assert strict["digest"] == first["digest"], strict


def test_a_blob_without_kernels_compiles_the_plan(produced, tmp_path):
    blob, first = produced
    bare = blob.with_name(blob.name + ".bare")
    assert bare.stat().st_size < blob.stat().st_size
    loose = _child(_CONSUME, tmp_path / "loose_cache", bare, "loose")
    assert "error" not in loose, loose
    assert loose["stats"]["misses"] >= 1 and loose["digest"] == first["digest"], loose
    strict = _child(_CONSUME, tmp_path / "strict_cache", bare, "strict")
    assert "enforce_precompiled" in strict.get("error", ""), strict
    warm = _child(_CONSUME, tmp_path / "loose_cache", bare, "strict")  # the loose run left the kernels on disk
    assert "error" not in warm and warm["stats"]["misses"] == 0 and warm["digest"] == first["digest"], warm


def test_a_blob_missing_one_kernel_is_refused_strictly_before_anything_compiles(produced_bhsd, tmp_path):
    blob, first = produced_bhsd
    partial = tmp_path / "partial.bin"

    def drop_one(header):
        assert len(header["kernels"]) >= 2, header["kernels"]
        header["kernels"].pop()

    _rewrite_header(blob, partial, drop_one)
    strict = _child(_CONSUME, tmp_path / "strict_cache", partial, "strict", "retry", layout="bhsd")
    assert "neither in the blob nor in the compiled cache" in strict.get("error", ""), strict
    assert strict["stats"]["misses"] == 1, strict  # only the loose retry compiled
    assert strict["digest"] == first["digest"], strict
