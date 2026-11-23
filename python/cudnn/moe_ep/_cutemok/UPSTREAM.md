# CuTe MoK sources

This package and the public module `cudnn/moe_ep/cutemok.py` are derived from
[Mixture-of-Kittens](https://github.com/cursor/mixture-of-kittens), commit
`22fc95ae6e331a738c4a58a227a8b03cac586e12`, under Apache-2.0. The CuTe DSL kernels
and hosts in `cutedsl/` are ports of its CUDA megakernels and scheduler;
`functional.py`, `ops.py` and the public module derive from its Python package.
NVIDIA ported them to CuTe DSL and modified them. The port contains no code from
Mixture-of-Kittens' ThunderKittens dependency.

`cutedsl/_compat/` holds helpers adapted from
[quack-kernels](https://github.com/Dao-AILab/quack) 0.5.0, under Apache-2.0, and
maintained locally.

Every file keeps its license header: files derived from Mixture-of-Kittens carry
the Cursor Research notice followed by NVIDIA's copyright, the `_compat/` files
their original QuACK notices, and files NVIDIA added NVIDIA's copyright only.
`THIRD_PARTY_LICENSES.txt` and `ACKNOWLEDGEMENTS.md` list both sources. The files
are not byte-for-byte copies of either source, so there is no hash record.
