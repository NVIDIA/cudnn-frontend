# Acknowledgements

## CuTe DSL OSS kernels

Several open-source kernels in this repository originated in NVIDIA's CuTe DSL kernel
library. We gratefully acknowledge the contributors who helped develop, bring up,
and test these kernels.

Contributors include:

- Rachit Garg <rachitg@nvidia.com>
- Jack Yang <jackyang@nvidia.com>
- Hao Sheng <hsheng@nvidia.com>
- Alex Li <alel@nvidia.com>
- Aragorn Guan <aragorng@nvidia.com>
- Bangyu Shen <bangyus@nvidia.com>
- Caleb Du <cadu@nvidia.com>
- Xiao Song <xiaos@nvidia.com>
- Siddhartha Raman <sraman@nvidia.com>

This acknowledgement covers the CuTe DSL kernel work now represented by these
modules in `python/cudnn/`:

- `gemm/cutedsl/dense/amax`
- `gemm/cutedsl/dense/dsrelu`
- `gemm/cutedsl/dense/srelu`
- `gemm/cutedsl/dense/swiglu`
- `gemm/cutedsl/grouped/dglu`
- `gemm/cutedsl/grouped/dsrelu`
- `gemm/cutedsl/grouped/dswiglu`
- `gemm/cutedsl/grouped/glu`
- `gemm/cutedsl/grouped/glu_hadamard`
- `gemm/cutedsl/grouped/quant`
- `gemm/cutedsl/grouped/srelu`
- `gemm/cutedsl/grouped/swiglu`
- `gemm/cutedsl/grouped/wgrad`
- `gemm/cutedsl/discrete_grouped/dswiglu`
- `gemm/cutedsl/discrete_grouped/swiglu`
- `rmsnorm_rht_amax`

Thank you also to the broader CUTLASS/CuTe DSL and infrastructure teams who
supported the original kernel development.

## Frost MoE pipeline tuning

The paired BF16 FC2 six-stage option follows Kernel Factory campaign2132,
candidate `e459dffedc6afc798f46073db14a159b49cff24d696024d801a45191d7e76521`.
It builds on NVIDIA Frost, KF624/2f5c compact-resource work, and the paired-MMA
and canonical rank-5 layout guidance from Yanqin Zhai's PR1090 and CUTLASS
example113. FlashInfer integration and independent full-power validation
establish the workload-specific scope of the measured benefit.
