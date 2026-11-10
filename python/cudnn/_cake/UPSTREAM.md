# Cake build support

`loader.py` is adapted from
[radixark/miles-ext-kernels](https://github.com/radixark/miles-ext-kernels/blob/f55fb99674be5edb0e1f6a0e4d6a911335da86ce/miles_ext_kernels/cake/_runtime/loader.py),
commit `f55fb99674be5edb0e1f6a0e4d6a911335da86ce`, under Apache-2.0.

The adapter removes Miles configuration and registry dependencies, uses a
Frontend cache namespace, and makes the no-build guard local to each context.
Operation PRs add their own pinned kernel packages and source records under
`_vendor`. Keep imported sources unchanged and retain their license headers.

The loader requires Linux, nvcc 13.4 or newer, and the installed TVM-FFI headers.
It supports SM100a, SM103a, and SM107a. It imports neither Torch nor TVM-FFI
until an operation requests them. It does not choose an execution plan or own
GPU memory.

Call `install(package.cake_build)` before requesting an upstream JIT spec.
Build and load all selected libraries at plan construction. Execute must use
the stored launch functions and caller-owned buffers on the caller's stream.
`no_build()` is a test guard against an unprepared launch compiling a library.
The guard is nestable and does not affect other threads.

Cache keys include the manifest closure, flags, compiler version, TVM-FFI
version, and host architecture. Builds use a file lock and atomic rename.
Libraries in upstream `aot/` retain the upstream build-info checks.
