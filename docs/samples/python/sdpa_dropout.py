"""Complete standalone Python sample.
API page: docs/api/python/sdpa.md
API: sdpa
Use case: Dropout
Source: adapted from the dropout setup in `test/python/test_mhas.py`.
"""

import sys

try:
    import cudnn
    import torch
except Exception as exc:
    print(f"SKIP: could not import cudnn or torch: {exc}")
    raise SystemExit(0)


def run_sample(sample):
    if not torch.cuda.is_available():
        print("SKIP: CUDA is not available to PyTorch")
        return
    try:
        sample()
    except Exception as exc:
        if "--strict" in sys.argv:
            raise
        print(f"SKIP: sample could not run on this installation: {exc}")


def sample():
    B, Hq, Hkv, S, D = 1, 2, 1, 16, 64
    q_shape = (B, Hq, S, D)
    q_stride = (Hq * S * D, D, Hq * D, 1)
    kv_shape = (B, Hkv, S, D)
    kv_stride = (Hkv * S * D, D, Hkv * D, 1)
    q = torch.randn(q_shape, device="cuda", dtype=torch.bfloat16).as_strided(q_shape, q_stride)
    k = torch.randn(kv_shape, device="cuda", dtype=torch.bfloat16).as_strided(kv_shape, kv_stride)
    v = torch.randn(kv_shape, device="cuda", dtype=torch.bfloat16).as_strided(kv_shape, kv_stride)
    seed = torch.full((1, 1, 1, 1), 123456, device="cuda", dtype=torch.int64)
    offset = torch.full((1, 1, 1, 1), 789, device="cuda", dtype=torch.int64)
    rng_dump = torch.zeros((B, Hq, S, S), device="cuda", dtype=torch.float32)
    with cudnn.Graph(
        io_data_type=torch.bfloat16,
        intermediate_data_type=torch.float32,
        compute_data_type=torch.float32,
        inputs=["q", "k", "v", "seed", "offset", "rng_dump"],
        outputs=["out", "stats"],
        handle="auto",
    ) as graph:
        Q = graph.tensor(name="q", dim=q_shape, stride=q_stride)
        K = graph.tensor(name="k", dim=kv_shape, stride=kv_stride)
        V = graph.tensor(name="v", dim=kv_shape, stride=kv_stride)
        Seed = graph.tensor(
            name="seed",
            dim=seed.shape,
            stride=seed.stride(),
            data_type=cudnn.data_type.INT64,
        )
        Offset = graph.tensor(
            name="offset",
            dim=offset.shape,
            stride=offset.stride(),
            data_type=cudnn.data_type.INT64,
        )
        RngDump = graph.tensor(
            name="rng_dump",
            dim=rng_dump.shape,
            stride=rng_dump.stride(),
            data_type=cudnn.data_type.FLOAT,
        )
        out, stats = graph.sdpa(
            q=Q,
            k=K,
            v=V,
            generate_stats=True,
            attn_scale=float(D) ** -0.5,
            dropout=(0.1, Seed, Offset),
            rng_dump=RngDump,
            name="sdpa",
        )
        out.set_output(True).set_name("out").set_dim(q_shape).set_stride(q_stride)
        if stats is not None:
            stats.set_output(True).set_name("stats").set_data_type(cudnn.data_type.FLOAT)
    result = graph(q, k, v, seed, offset, rng_dump)
    print(
        "sdpa output",
        tuple(result[0].shape) if isinstance(result, (list, tuple)) else tuple(result.shape),
    )


if __name__ == "__main__":
    run_sample(sample)
