# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``tile_dsl.pointwise.warp_abs_max_f32`` -- ``max |x|`` over the 32 lanes of a warp in ONE ``redux.sync.max.abs.f32``
(sm_100a+, PTX 8.6) -- and its shuffle-tree twin ``warp_abs_max_f32_shfl`` (five ``shfl.sync.bfly`` + ``max.f32``).

The along-kv dS scale of the MXFP8 d=256 backward (one E8M0 per 32 kv rows of a q column) needs a per-column amax over 32 LANES; the
redux is the cheap form, the shuffle tree the reference and the fallback.  Both return the same value to every lane.

* tier 1 (any box whose cutlass-dsl knows ``sm_107a`` -- an assembly check that needs no Rubin device): the sm_107a
  trace-compile of a 1-warp probe is asserted; its PTX carries exactly one ``redux.sync.max.abs.f32``
  and five ``shfl.sync.bfly``; with a decoding nvdisasm the SASS carries the redux (``CREDUX.MAXABS.F32 UR, R`` on sm_107a --
  the result lands in a UNIFORM register, i.e. the hardware states it is warp-uniform) and five ``SHFL``.
* tier 2 (``requires_rubin``): on random and edge inputs (0, -0, fp32 denormals alone and mixed with normals, +-inf mixed
  with finite, huge finite) the redux is BITWISE the shuffle tree, and both equal torch's ``|x|.amax`` per warp.  NaN is
  out of contract (softmax gradients carry none) and is not fed.
"""

import glob
import importlib.util
import os
import subprocess
import sys
import textwrap

import pytest
import torch

from frost_test_utils import arch_known_to_the_dsl, nvdisasm_candidates, requires_dsl, requires_rubin

pytestmark = [pytest.mark.L0, requires_dsl]

_PROBE_SRC = textwrap.dedent('''
    """Probe: one warp per block; lane value -> (redux abs-max, shuffle-tree abs-max) stored per lane."""
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute

    from cudnn.frost.tile_dsl.pointwise import warp_abs_max_f32, warp_abs_max_f32_shfl
    from cudnn.frost.tile_dsl.tma import ld_global, st_global


    @cute.kernel
    def probe(mSrc: cute.Tensor, mRedux: cute.Tensor, mShfl: cute.Tensor):
        tidx = cutlass.Int32(cute.arch.thread_idx()[0])
        bidx = cutlass.Int32(cute.arch.block_idx()[0])
        lane64 = (bidx * cutlass.Int32(32) + tidx).to(cutlass.Int64) * cutlass.Int64(4)
        x = ld_global(mSrc.iterator.toint() + lane64, cutlass.Float32)
        st_global(mRedux.iterator.toint() + lane64, warp_abs_max_f32(x), cutlass.Float32)
        st_global(mShfl.iterator.toint() + lane64, warp_abs_max_f32_shfl(x), cutlass.Float32)


    probe.set_name_prefix("cudnn", remove_cutlass_symbol=True)


    @cute.jit
    def launch(src: cute.Tensor, redux: cute.Tensor, shfl: cute.Tensor, n_warps: cutlass.Int32, stream: cuda.CUstream):
        probe(src, redux, shfl).launch(grid=(n_warps, 1, 1), block=(32, 1, 1), stream=stream)


    if __name__ == "__main__":
        # argv: <arch> <dump dir> [nvdisasm candidates...]; CUTE_DSL_DUMP_DIR set by the caller (keeps ptx + cubin).
        import glob, os, subprocess, sys
        from cutlass.cute.runtime import make_fake_stream, make_fake_tensor

        arch, dump, cands = sys.argv[1], sys.argv[2], sys.argv[3:]

        def fake():
            return make_fake_tensor(dtype=cutlass.Float32, shape=(cute.sym_int(),), stride=(1,), assumed_align=16)

        cute.compile(launch, fake(), fake(), fake(), cutlass.Int32(1), make_fake_stream(use_tvm_ffi_env_stream=False), options=f"--enable-tvm-ffi --gpu-arch {arch} --keep-cubin --keep-ptx")
        print("COMPILED", arch)
        ptxs = glob.glob(os.path.join(dump, f"*.{arch}.ptx"))
        if not ptxs:
            print("FAIL no ptx landed in", dump, os.listdir(dump)); sys.exit(3)
        ptx = open(ptxs[0]).read()
        print("PTX REDUX_MAX_ABS", ptx.count("redux.sync.max.abs.f32"))
        print("PTX REDUX_ANY", ptx.count("redux.sync"))
        print("PTX SHFL_BFLY", ptx.count("shfl.sync.bfly"))
        cubins = glob.glob(os.path.join(dump, f"*.{arch}.cubin"))
        if not cubins:
            print("FAIL no cubin landed in", dump); sys.exit(3)
        sass = None
        for nvd in cands:
            try:
                proc = subprocess.run([nvd, "-c", cubins[0]], capture_output=True, text=True, timeout=120)
            except (OSError, subprocess.SubprocessError) as exc:
                print("REJECT", nvd, "->", repr(exc)); continue
            if proc.returncode == 0 and proc.stdout.strip():
                sass = proc.stdout.splitlines(); print("NVDISASM", nvd); break
            print("REJECT", nvd, "->", (proc.stderr.strip().splitlines() or [str(proc.returncode)])[-1])
        if sass is None:
            print("SKIP no nvdisasm candidate decodes", arch); sys.exit(0)
        print("SASS REDUX", sum(1 for ln in sass if "REDUX" in ln))  # sm_107a spells it CREDUX.MAXABS.F32 (a UNIFORM-register result)
        print("SASS SHFL", sum(1 for ln in sass if " SHFL" in ln))
        print("SASS FMNMX", sum(1 for ln in sass if " FMNMX" in ln))
        print("SASS SPILL", sum(1 for ln in sass if "STL" in ln or "LDL" in ln))
        print("SASS LINES", len(sass))
    ''')


def _write_probe(tmp_path):
    probe_py = tmp_path / "warp_abs_max_probe.py"
    probe_py.write_text(_PROBE_SRC)
    return probe_py


def test_sm107a_redux_form(tmp_path):
    """PTX: one ``redux.sync.max.abs.f32`` (the wrapper's ``abs=True`` spelling) and five ``shfl.sync.bfly`` (the tree);
    SASS (when an nvdisasm decodes sm_107a): one ``CREDUX.MAXABS.F32`` (counted as a line containing ``REDUX``), 5 ``SHFL``,
    no spill."""
    if not arch_known_to_the_dsl("sm_107a"):
        pytest.skip("this cutlass-dsl has no sm_107a (needs >= 4.8.0)")
    dump = tmp_path / "dump_sm107a"
    dump.mkdir()
    env = dict(os.environ, CUTE_DSL_DUMP_DIR=str(dump), CUTE_DSL_KEEP="ptx,cubin")
    proc = subprocess.run(
        [sys.executable, str(_write_probe(tmp_path)), "sm_107a", str(dump), *nvdisasm_candidates()], capture_output=True, text=True, timeout=900, env=env
    )
    assert proc.returncode == 0, f"sm_107a trace-compile failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
    lines = proc.stdout.splitlines()
    assert "COMPILED sm_107a" in lines, proc.stdout[-2000:]
    ptx = {ln.split()[1]: int(ln.split()[2]) for ln in lines if ln.startswith("PTX ")}
    assert ptx["REDUX_MAX_ABS"] == 1 and ptx["REDUX_ANY"] == 1, f"expected exactly one redux.sync.max.abs.f32 -- {ptx}"
    assert ptx["SHFL_BFLY"] == 5, f"the shuffle tree is five butterfly stages -- {ptx}"
    if not nvdisasm_candidates():
        pytest.skip("compiled + PTX pinned; no nvdisasm executable to try (CUDA_PATH unset and none on PATH)")
    if any(ln.startswith("SKIP") for ln in lines):
        pytest.skip(f"compiled + PTX pinned; SASS half skipped: {[ln for ln in lines if ln.startswith(('SKIP', 'REJECT'))]}")
    sass = {ln.split()[1]: int(ln.split()[2]) for ln in lines if ln.startswith("SASS ")}
    print(f"\nwarp_abs_max sm_107a PTX {ptx} SASS {sass}")
    assert sass["REDUX"] >= 1, f"no REDUX in the sm_107a SASS -- {sass}"
    assert sass["SHFL"] == 5, sass
    assert sass["SPILL"] == 0, sass


def _warp_inputs(n_warps: int, seed: int = 7) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(n_warps, 32, generator=g) * torch.exp2(torch.rand(n_warps, 32, generator=g) * 80.0 - 40.0)
    den = torch.tensor([1e-40, -1e-40, 7.35e-40, -7.35e-40, 1e-39, 3e-39, -2e-39, 1e-45], dtype=torch.float32)
    edge = [
        torch.zeros(32),
        torch.full((32,), -0.0),
        torch.tensor([0.0, -0.0] * 16),
        den.repeat(4),  # denormals ONLY: the max is itself a denormal
        torch.cat([den, torch.tensor([1.0, -2.0, 0.5, -0.25] * 6)]),  # denormals mixed with normals
        torch.cat([torch.tensor([float("inf")]), torch.randn(31, generator=g)]),
        torch.cat([torch.randn(31, generator=g), torch.tensor([-float("inf")])]),
        torch.cat([torch.tensor([float("inf"), -float("inf")]), torch.randn(30, generator=g)]),
        torch.tensor([3e38, -3.4e38, 1e38] + [1.0] * 29),
        torch.cat([torch.zeros(31), torch.tensor([-1e-38])]),  # a lone normal-range tiny value among zeros
        torch.cat([torch.full((31,), -0.0), torch.tensor([2.0**-126])]),  # the min normal among -0
        torch.tensor([-(2.0**k) for k in range(-20, 12)]),  # all negative
    ]
    for i, row in enumerate(edge):
        x[i] = row.float()
    return x.contiguous()


@requires_rubin
def test_redux_is_bitwise_the_shuffle_tree_and_torch_on_device(tmp_path):
    import cuda.bindings.driver as _cuda_driver
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack

    spec = importlib.util.spec_from_file_location("_warp_abs_max_probe_device", str(_write_probe(tmp_path)))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    dev = torch.device("cuda")
    n_warps = 512
    x = _warp_inputs(n_warps).to(dev)
    redux = torch.full_like(x, float("nan"))
    shfl = torch.full_like(x, float("nan"))
    stream = _cuda_driver.CUstream(torch.cuda.current_stream(dev).cuda_stream)
    args = (from_dlpack(x, assumed_align=16), from_dlpack(redux, assumed_align=16), from_dlpack(shfl, assumed_align=16), cutlass.Int32(n_warps))
    cute.compile(mod.launch, *args, stream)(*args, stream)
    torch.cuda.synchronize()
    assert not torch.isnan(redux).any() and not torch.isnan(shfl).any(), "unwritten or NaN lanes"
    ref = x.abs().amax(dim=1, keepdim=True).expand_as(x)
    diff = (redux.view(torch.int32) != shfl.view(torch.int32)).any(dim=1)
    assert (
        int(diff.sum().item()) == 0
    ), f"redux != shuffle tree on {int(diff.sum().item())} warps; first warp {int(diff.nonzero()[0])}: x={x[diff.nonzero()[0]].tolist()} redux={redux[diff.nonzero()[0], 0].item()} shfl={shfl[diff.nonzero()[0], 0].item()}"
    diff_t = (redux.view(torch.int32) != ref.view(torch.int32)).any(dim=1)
    assert (
        int(diff_t.sum().item()) == 0
    ), f"redux != torch |x|.amax on {int(diff_t.sum().item())} warps; first warp {int(diff_t.nonzero()[0])}: x={x[diff_t.nonzero()[0]].tolist()} redux={redux[diff_t.nonzero()[0], 0].item()} ref={ref[diff_t.nonzero()[0], 0].item()}"
    # every lane of a warp holds the same value (the redux and the butterfly tree both broadcast)
    assert torch.equal(redux, redux[:, :1].expand_as(redux)) and torch.equal(shfl, shfl[:, :1].expand_as(shfl))
