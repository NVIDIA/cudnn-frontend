"""Compare cuDNN native norm vs cuDNN frost norm (sm_100) for LN/RMS fprop+bprop.

For each problem shape (from benchmark/norms/configs), spawns two worker
processes -- one using the real cuDNN frontend (against the custom cuDNN build)
and one using the frost kernels -- times fwd + bwd, and reports ms, GB/s, and the
frost/cuDNN speedup. Two processes so `import cudnn` resolves cleanly (frontend
vs the repo's frost package) in each.

    # custom cuDNN build + CUDA TK for the cudnn worker are set automatically
    python benchmark_cudnn_vs_frost.py                 # all config shapes, bf16
    python benchmark_cudnn_vs_frost.py --dtype float16 --iters 50
    python benchmark_cudnn_vs_frost.py --models llama3-8b,gpt3-175b --csv out.csv
"""
import argparse
import os
import re
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
_WORKER = os.path.join(_HERE, "_norm_bench_worker.py")

# Custom cuDNN build + CUDA TK the cuDNN worker links against (override via env).
_CUDNN_LIB = os.environ.get("CUSTOM_CUDNN_LIB", "")
_CUDA_TK_LIB = os.environ.get("CUSTOM_CUDA_LIB", "")

_DTYPE_BYTES = {"bfloat16": 2, "float16": 2, "float32": 4}


def bytes_moved(norm_type, N, C, has_bias, eb, mode):
    """Bytes read+written (matches benchmark_single_norm.compute_bandwidth_bytes)."""
    if mode == "fwd":
        if norm_type == "rms_norm":
            r = (N * C + C) * eb
            w = N * C * eb + N * 4
        else:
            r = (N * C + C + (C if has_bias else 0)) * eb
            w = N * C * eb + N * 4 + N * 4
    else:
        if norm_type == "rms_norm":
            r = (N * C + N * C + C) * eb + N * 4
            w = (N * C + C) * eb + (C * eb if has_bias else 0)
        else:
            r = (N * C + N * C + C) * eb + N * 4 + N * 4
            w = (N * C + C + C) * eb
    return r + w


def gbps(nbytes, ms):
    return nbytes / (ms * 1e-3) / 1e9 if ms > 0 else 0.0


def run_worker(backend, p, dtype, iters, warmup, env):
    cmd = [sys.executable, _WORKER, "--backend", backend, "--norm_type", p.norm_type,
           "--N", str(p.N), "--C", str(p.C), "--has_bias", str(int(p.has_bias)),
           "--dtype", dtype, "--epsilon", str(p.epsilon), "--iters", str(iters), "--warmup", str(warmup)]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    m = re.search(r"fwd_ms=(\S+) bwd_ms=(\S+) fwd_maxabs=(\S+) bwd_maxrel=(\S+)", r.stdout)
    if not m:
        tail = "\n".join((r.stderr or r.stdout).strip().splitlines()[-3:])
        print(f"    [{backend} FAILED] {tail}")
        return None
    return dict(fwd_ms=float(m[1]), bwd_ms=float(m[2]), fwd_maxabs=float(m[3]), bwd_maxrel=float(m[4]))


def load_presets(models):
    sys.path.insert(0, _REPO)
    from benchmark.norms.configs.all_models import CONFIG
    presets = CONFIG.norms
    if models:
        want = set(models.split(","))
        presets = [p for p in presets if p.name in want]
    return presets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--models", default=None, help="comma-separated preset names (default: all)")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    presets = load_presets(args.models)
    eb = _DTYPE_BYTES[args.dtype]

    env_cudnn = dict(os.environ)
    env_cudnn["LD_LIBRARY_PATH"] = f"{_CUDNN_LIB}:{_CUDA_TK_LIB}:" + env_cudnn.get("LD_LIBRARY_PATH", "")
    env_frost = dict(os.environ)

    print(f"\ncuDNN (custom build) vs frost  |  dtype={args.dtype}  iters={args.iters}\n")
    hdr = (f"{'model':<22} {'type':<10} {'N':>7} {'C':>6} | "
           f"{'cuDNN fwd':>18} {'frost fwd':>18} {'spd':>5} | "
           f"{'cuDNN bwd':>18} {'frost bwd':>18} {'spd':>5}")
    print(hdr)
    print("-" * len(hdr))
    rows = []
    for p in presets:
        c = run_worker("cudnn", p, args.dtype, args.iters, args.warmup, env_cudnn)
        f = run_worker("frost", p, args.dtype, args.iters, args.warmup, env_frost)
        if c is None or f is None:
            continue
        fb = bytes_moved(p.norm_type, p.N, p.C, p.has_bias, eb, "fwd")
        bb = bytes_moved(p.norm_type, p.N, p.C, p.has_bias, eb, "bwd")

        def cell(ms, nbytes):
            return f"{ms:7.3f}ms {gbps(nbytes, ms):6.0f}GB/s"

        fwd_spd = c["fwd_ms"] / f["fwd_ms"] if f["fwd_ms"] else 0
        bwd_spd = c["bwd_ms"] / f["bwd_ms"] if f["bwd_ms"] else 0
        print(f"{p.name:<22} {p.norm_type:<10} {p.N:>7} {p.C:>6} | "
              f"{cell(c['fwd_ms'], fb):>18} {cell(f['fwd_ms'], fb):>18} {fwd_spd:>4.2f}x | "
              f"{cell(c['bwd_ms'], bb):>18} {cell(f['bwd_ms'], bb):>18} {bwd_spd:>4.2f}x")
        rows.append((p, c, f, fb, bb, fwd_spd, bwd_spd))

    if rows:
        gm = lambda xs: (1.0 if not xs else __import__("math").exp(sum(__import__("math").log(x) for x in xs) / len(xs)))
        print("-" * len(hdr))
        print(f"geomean frost/cuDNN speedup:  fwd {gm([r[5] for r in rows]):.2f}x   bwd {gm([r[6] for r in rows]):.2f}x")
        print("(speedup = cuDNN_time / frost_time; >1 means frost is faster)")
        maxfa = max(r[2]["fwd_maxabs"] for r in rows)
        maxbr = max(r[2]["bwd_maxrel"] for r in rows)
        print(f"frost correctness vs torch:  fwd maxabs<={maxfa:.1e}  bwd maxrel<={maxbr:.1e}")

    if args.csv and rows:
        with open(args.csv, "w") as fh:
            fh.write("model,norm_type,N,C,has_bias,dtype,cudnn_fwd_ms,frost_fwd_ms,cudnn_fwd_gbps,frost_fwd_gbps,"
                     "cudnn_bwd_ms,frost_bwd_ms,cudnn_bwd_gbps,frost_bwd_gbps,fwd_speedup,bwd_speedup\n")
            for p, c, f, fb, bb, fs, bs in rows:
                fh.write(f"{p.name},{p.norm_type},{p.N},{p.C},{int(p.has_bias)},{args.dtype},"
                         f"{c['fwd_ms']:.5f},{f['fwd_ms']:.5f},{gbps(fb,c['fwd_ms']):.1f},{gbps(fb,f['fwd_ms']):.1f},"
                         f"{c['bwd_ms']:.5f},{f['bwd_ms']:.5f},{gbps(bb,c['bwd_ms']):.1f},{gbps(bb,f['bwd_ms']):.1f},"
                         f"{fs:.3f},{bs:.3f}\n")
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
