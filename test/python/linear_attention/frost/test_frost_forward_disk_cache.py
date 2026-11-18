# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A fresh process runs the FROST forwards from the on-disk compiled cache: the cache directory the cold process filled is
read, not rewritten, by the warm one, and both produce the same outputs."""

import json
import os
from pathlib import Path
import subprocess
import sys

import cudnn
import pytest
import torch

from linear_attention.test_la import frost_la_unavailable

pytestmark = pytest.mark.L0

_CHILD = r"""
import json, sys, time
from pathlib import Path
import torch, cudnn
from linear_attention.test_la import Backend, FWD_TOL, STATE_TOL, make_case, random_state, reference, pinned_op, op_args, assert_rms_close
variant, phase, result_path, expected_module = sys.argv[1:]
assert Path(cudnn.__file__).resolve() == Path(expected_module).resolve(), cudnn.__file__
dtype = torch.bfloat16
outputs = []
started = time.perf_counter()
for length in (17, 128, 512):
    case = make_case(variant, dtype, seq_lens=[length, length // 2], H=4, seed=173 + length)
    initial = random_state(case)
    kwargs = dict(initial_state=initial, output_final_state=True)
    op = pinned_op(Backend('frost'), variant)
    args = op_args(case)
    expected = reference(case, initial_state=initial)
    output, state = op(*args, **kwargs)
    assert_rms_close('eager output', output, expected[0], FWD_TOL[dtype])
    assert_rms_close('eager state', state, expected[1], STATE_TOL[dtype])
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            captured_output, captured_state = op(*args, **kwargs)
        captured_output.fill_(float('nan')); captured_state.fill_(float('nan'))
        graph.replay(); torch.cuda.synchronize()
        assert_rms_close('replay output', captured_output, expected[0], FWD_TOL[dtype])
        assert_rms_close('replay state', captured_state, expected[1], STATE_TOL[dtype])
    finally:
        graph.reset()
    outputs += [output.double().sum().item(), state.double().sum().item()]
Path(result_path).write_text(json.dumps({'outputs': outputs, 'source': cudnn.__file__, 'case_seconds': time.perf_counter() - started}))
"""


def _cache_listing(root):
    """(path, size, mtime) of every file under the cache directory."""
    return sorted((str(p.relative_to(root)), p.stat().st_size, p.stat().st_mtime_ns) for p in Path(root).rglob("*") if p.is_file())


@pytest.mark.parametrize("variant", ["gdn", "gdn2", "gdp", "kda"])
def test_frost_forward_reloads_in_a_fresh_process(tmp_path, variant):
    reason = frost_la_unavailable()
    if reason:
        pytest.skip(reason)
    source = Path(__file__).resolve().parents[4]
    env = dict(os.environ, CUDNN_FRONTEND_COMPILED_CACHE=str(tmp_path / "cache"), CUDNN_FRONTEND_COMPILED_CACHE_INPROCESS_MEMO="0")
    env.pop("CUDNN_FRONTEND_DISABLE_COMPILED_CACHE", None)
    # CI installs the built wheel; the source tree has no extension module.
    # Pin the child to the same package as this process, whether installed or
    # imported from a development checkout, and add only the test helpers.
    module_path = Path(cudnn.__file__).resolve()
    env["PYTHONPATH"] = os.pathsep.join([str(module_path.parent.parent), str(source / "test/python"), env.get("PYTHONPATH", "")])
    results, listings = [], []
    for phase in ("cold", "warm"):
        path = tmp_path / f"{phase}.json"
        process = subprocess.run(
            [sys.executable, "-c", _CHILD, variant, phase, str(path), str(module_path)], env=env, cwd=tmp_path, capture_output=True, text=True, timeout=600
        )
        assert process.returncode == 0, process.stdout[-4000:] + process.stderr[-6000:]
        results.append(json.loads(path.read_text()))
        listings.append(_cache_listing(tmp_path / "cache"))
    print(json.dumps({"variant": variant, "cold": results[0], "warm": results[1], "cache_files": len(listings[0])}))
    names = {name for name, _, _ in listings[0]}
    objects = [name for name in names if name.endswith("kernel.o")]
    assert objects, "the cold process stored no compiled kernel object (kernel.o) in the cache directory"
    assert all(name[: -len("kernel.o")] + "entry.json" in names for name in objects), "a stored kernel object lacks its entry.json commit marker"
    assert listings[1] == listings[0], "the warm process rewrote or added cache artifacts instead of reusing them"
    assert results[1]["outputs"] == results[0]["outputs"], "the warm process computed different outputs from the cached artifacts"
