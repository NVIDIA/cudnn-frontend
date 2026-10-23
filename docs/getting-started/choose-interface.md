# Choose an interface

There are multiple interfaces to choose from to integrate cuDNN into your application.
Choose the interface that exposes the control your application actually needs.

| Interface | Lifecycle ownership | Runtime buffers | Use it when |
| --- | --- | --- | --- |
| `cudnn.Graph` (Python) | Automatic at context exit | PyTorch inputs; wrapper allocates outputs/workspace | Building Python or PyTorch applications quickly |
| `cudnn.pygraph` (Python) | Explicit or `build()` convenience | Caller binds tensors/workspace | Inspecting IR, routing engines, filtering plans, or serializing |
| `graph::Graph` (C++) | Explicit | Caller owns handle, pointers, workspace, synchronization | Integrating into C++ services or runtimes |
| `cudnn.graph` / `cudnn.jit` (Python) | Automatic and cached | PyTorch | Shape-specialized decorated Python functions |
| `cudnn.ops` (Python) | PyTorch dispatcher owns schema/autograd | PyTorch | Using a supplied convenience operator such as causal Conv1d |
| OSS packages (Python) | Package-specific | Usually PyTorch/CUTLASS DSL tensors | Using an exact architecture-specific fused kernel |
| `cudnn.experimental.ops` (Python) | PyTorch convenience wrapper | PyTorch | Evaluating explicitly experimental features |
| Legacy C++ builders | Explicit descriptors | Caller owned | Maintaining existing code while migrating |

Start with `cudnn.Graph` for Python learning. Move to `cudnn.pygraph` when you can name the lifecycle or routing control you need. Start C++ directly when the host application is C++ or must own every resource.

Python IR and engine routing are unreleased in this checkout. See the
[routing availability and invariants](/concepts/python-ir-routing#availability)
before depending on that interface.

Optional wrappers can be faster or more specialized, but their layouts, dependencies, architecture range, stability, and fallback behavior are narrower. Evaluate them through [Select OSS engines and kernels](/optional-kernels/oss-engines).
