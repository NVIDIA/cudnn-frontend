"""Export one FROST SDPA forward graph, for bench_sdpa_cpu_costs.cpp to execute.

The C++ side builds the SAME problem on the cuDNN backend engine and times both
call paths against each other. This half only has to produce the artifact, so it
is the ordinary lifecycle plus set_name() and export_to_disk().

    python bench_sdpa_export.py [sdpa.cudnn]
"""

import json
import sys

import torch

import cudnn

B, H, S, D = 2, 8, 1024, 128
DIMS = (B, H, S, D)
# BHSD logical view over BSHD storage -- what an attention stack actually hands
# a kernel, and not row-major contiguous.
STRIDES = (S * H * D, D, H * D, 1)
NAME = "sdpa_fwd_causal_f16"


def build(name=NAME):
    dt = cudnn.data_type.HALF
    graph = cudnn.pygraph(
        io_data_type=dt,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    ports = {p: graph.tensor(dim=DIMS, stride=STRIDES, data_type=dt, name=p) for p in ("q", "k", "v")}
    o, _stats = graph.sdpa(
        name="sdpa",
        q=ports["q"],
        k=ports["k"],
        v=ports["v"],
        attn_scale=1.0 / (D**0.5),
        is_inference=True,
        use_causal_mask=True,
    )
    o.set_output(True).set_dim(DIMS).set_stride(STRIDES).set_data_type(dt)
    ports["o"] = o

    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans()
    graph.build_plans()
    graph.set_name(name)
    return graph, ports


def main():
    if not torch.cuda.is_available():
        raise SystemExit("needs a CUDA device")
    path = sys.argv[1] if len(sys.argv) > 1 else "sdpa.cudnn"

    graph, ports = build()
    engine = getattr(graph.selected_engine, "name", graph.selected_engine)
    cudnn.export_to_disk([graph], path=path)

    # Shapes and uid roles for the C++ side. Not part of the contract -- the
    # container already carries the uid ORDER, which is what execute() needs.
    with open(path + ".bench.json", "w") as f:
        json.dump(
            {
                "kernel": NAME,
                "engine": engine,
                "dims": list(DIMS),
                "strides": list(STRIDES),
                "uids": {p: t.get_uid() for p, t in ports.items()},
                "workspace": graph.get_workspace_size(),
            },
            f,
            indent=2,
        )

    print(f"wrote {path}")
    print(f"  {NAME}  B={B} H={H} S={S} D={D}  engine={engine}")
    print(f"  workspace {graph.get_workspace_size()} bytes")


if __name__ == "__main__":
    main()
