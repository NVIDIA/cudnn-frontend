# cuDNN Python Frontend

This folder exposes the Python Frontend Graph APIs and the high-level Graph wrapper, along with several frontend-only, ready-to-use APIs.

- **Graph API**: Low-level primitives for building, compiling, and executing cuDNN operation graphs in Python.
- **Graph Wrapper (`Graph`)**: A convenience layer that reduces boilerplate, manages workspace and tensor mapping, and makes execution ergonomic.
- **Frontend-only APIs**: Individual turnkey kernels with Python-first APIs

## Directory Structure

A simplified view of package structure:

```
pyproject.toml                       # Project metadata and dependencies. Optional dependencies for frontend-only APIs are registered here.
python/cudnn/
├── __init__.py                     # Top-level exports (Graph, graph, jit, wrappers, kernels)
├── graph.py                        # Low-level graph helpers (graph, jit, graph_cache)
├── wrapper.py                      # High-level Graph wrapper class
├── datatypes.py                    # Data type conversions and helpers
├── api_base.py                     # Abstract API base class for frontend-only APIs
├── {frontend-only-api-name}/
│   ├── __init__.py                 # Frontend-only API class
│   └── api.py                      # High-level API implementation
│   └── {kernel_name}.py            # Kernel implementation, i.e CuteDSL
test/python/                        # Test files
└── fe_api/                         # Test files for frontend-only APIs
```

## 

## Adding new frontend-only APIs

To add a new frontend-only API, follow these steps:
1. Create a new directory in the `python/cudnn` directory with the name of the API.
2. Add your kernel implementation and implement the high level API implementation in `api.py`, extending the `APIBase` class in `api_base.py`.
3. Expose the API import in `python/cudnn/__init__.py` and register the folder in `pyproject.toml`. Register any optional dependences if required.
4. Add a sample usage/test file in `test/python/fe_api/`.

**Currently implemented frontend-only APIs**:
- `GEMM + Amax`
- `GEMM + SwiGLU`
