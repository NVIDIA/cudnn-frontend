# Ahead-of-time plans

Compile a Python-engine plan once, then run it where there is no Python engine and no JIT.

| File | What it does |
|---|---|
| `export_plan.py` | Builds a bf16 SDPA forward graph, lets the FROST engine plan it, and writes `graph.serialize()` plus the tensors' bytes. It then reloads the blob and checks it bit for bit. |
| `run_plan.cpp` | Loads that blob with `cudnn_frontend::graph::Graph::deserialize` and executes it from C++. No Python, no cuDNN handle. |

```bash
python samples/aot/export_plan.py /tmp/aot_plan

g++ -std=c++17 -DNV_CUDNN_FRONTEND_USE_DYNAMIC_LOADING \
    -I include -I $CUDA_HOME/include -I $CUDNN_PATH/include \
    samples/aot/run_plan.cpp -L $CUDA_HOME/lib64 -lcudart -ldl -o /tmp/run_plan

LD_LIBRARY_PATH=<dir of libtvm_ffi.so>:<dir of libcute_dsl_runtime.so>:$LD_LIBRARY_PATH \
    /tmp/run_plan /tmp/aot_plan/plan.bin /tmp/aot_plan
cmp /tmp/aot_plan/<o uid>.bin /tmp/aot_plan/expected/<o uid>.bin
```

The two runtime libraries come from the `apache-tvm-ffi` and `nvidia-cutlass-dsl` wheels. `python -c "import tvm_ffi; print(tvm_ffi.libinfo.find_libtvm_ffi())"` finds the first. The second is under `nvidia_cutlass_dsl/cu13/lib` (or `cu12/lib`).

See [docs/utilities/ahead-of-time-python-engine-plans.md](../../docs/utilities/ahead-of-time-python-engine-plans.md) for what can be exported and the contract a deserialized plan keeps.
