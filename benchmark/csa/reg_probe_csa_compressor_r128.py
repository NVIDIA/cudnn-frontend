# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ptxas register/spill probe for every shipped ratio=128 CSA compressor kernel.

JIT-compiles the full shipped dispatch envelope — every (config, schedule) pair the
``nb_total`` bucket tables can select, forward and backward, over coff {1, 2} x
head_dim {128, 512} (16 kernels) — then runs ``ptxas -v`` on each kernel's PTX and
prints a table of registers / spill bytes / stack bytes / ex2.approx count. This
reproduces the register table published in docs/fe-oss-apis/csa.md.

Exits nonzero if any kernel spills, uses stack, or fails ptxas.

Requires a CC 10.0 GPU (the JIT needs a device), ``ptxas`` on PATH, and the
``cudnn[cutedsl]`` install. Not collected by pytest. Run, e.g.::

    CUDA_VISIBLE_DEVICES=0 python benchmark/csa/reg_probe_csa_compressor_r128.py
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile

os.environ.setdefault("CUTE_DSL_KEEP", "ptx")  # keep PTX artifacts on the compiled handles

import torch  # noqa: E402


def ptxas_one(tag, ptx_text, arch, out_dir):
    path = os.path.join(out_dir, f"{tag}.ptx")
    with open(path, "w") as f:
        f.write(ptx_text)
    r = subprocess.run(["ptxas", "-v", f"-arch={arch}", "-o", os.devnull, path], capture_output=True, text=True)
    out = r.stderr + r.stdout
    regs = re.search(r"Used (\d+) registers", out)
    spill = re.search(r"(\d+) bytes spill stores, (\d+) bytes spill loads", out)
    stack = re.search(r"(\d+) bytes stack frame", out)
    n_ex2 = ptx_text.count("ex2.approx")
    print(
        f"{tag:36} regs={regs.group(1) if regs else '?':>3} "
        f"spill={(spill.group(1) + '/' + spill.group(2)) if spill else '?'} "
        f"stack={stack.group(1) if stack else '?'} ex2.approx={n_ex2:3d} rc={r.returncode}",
        flush=True,
    )
    clean = r.returncode == 0 and spill is not None and stack is not None and spill.group(1) == spill.group(2) == "0" and stack.group(1) == "0"
    return clean


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", default="sm_100a", help="ptxas target architecture (default: sm_100a)")
    ap.add_argument("--keep-ptx", default=None, help="directory to keep the per-kernel .ptx files in (default: temporary)")
    args = ap.parse_args()
    # CUTE_DSL_KEEP=ptx makes the DSL drop each kernel's PTX into the current
    # directory; run the compiles from the (possibly temporary) output directory so
    # the repository tree stays clean.
    out_dir = os.path.abspath(args.keep_ptx) if args.keep_ptx else tempfile.mkdtemp(prefix="csa_r128_ptx_")
    os.makedirs(out_dir, exist_ok=True)
    os.chdir(out_dir)
    from cudnn.csa.compressor import compressor_sm100_r128 as M

    assert torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 0), "requires a CC 10.0 GPU"
    dev = torch.device("cuda", torch.cuda.current_device())
    # Compile every schedule bucket each shipped config can select at runtime
    # (precompile with nb_total=None walks the small/default/large tables).
    for coff, d in [(1, 128), (2, 128), (1, 512), (2, 512)]:
        M.precompile_fwd_r128(128, d, coff, dev)
        M.precompile_bwd_r128(128, d, coff, dev)

    all_clean = True
    n = 0
    for key, fn in sorted(M._COMPILED.items(), key=str):
        kind, _ratio, d, coff, sched, _dev = key
        if kind == "r128fwd":
            vec, tchunks, threads_x, twophase, fastexp = sched
            tag = f"fwd_c{coff}d{d}_v{vec}t{tchunks}x{threads_x}" + ("_2ph" if twophase else "") + ("_fexp" if fastexp else "")
        else:
            vec, tchunks, threads_x, fastexp = sched
            tag = f"bwd_c{coff}d{d}_v{vec}t{tchunks}x{threads_x}" + ("_fexp" if fastexp else "")
        all_clean = ptxas_one(tag, fn.artifacts.PTX, args.arch, out_dir) and all_clean
        n += 1
    print(f"{n} kernels probed ({args.arch}); {'ALL 0 spill / 0 stack' if all_clean else 'SPILL/STACK OR PTXAS FAILURE DETECTED'}", flush=True)
    if args.keep_ptx:
        print(f"PTX kept in {out_dir}", flush=True)
    else:
        os.chdir(tempfile.gettempdir())
        shutil.rmtree(out_dir, ignore_errors=True)
    sys.exit(0 if all_clean else 1)


if __name__ == "__main__":
    main()
