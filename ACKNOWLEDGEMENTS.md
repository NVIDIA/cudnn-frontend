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

- `gemm_amax`
- `gemm_dsrelu`
- `gemm_srelu`
- `gemm_swiglu`
- `grouped_gemm/grouped_gemm_dglu`
- `grouped_gemm/grouped_gemm_dsrelu`
- `grouped_gemm/grouped_gemm_dswiglu`
- `grouped_gemm/grouped_gemm_glu`
- `grouped_gemm/grouped_gemm_glu_hadamard`
- `grouped_gemm/grouped_gemm_quant`
- `grouped_gemm/grouped_gemm_srelu`
- `grouped_gemm/grouped_gemm_swiglu`
- `grouped_gemm/grouped_gemm_wgrad`
- `discrete_grouped_gemm/discrete_grouped_gemm_dswiglu`
- `discrete_grouped_gemm/discrete_grouped_gemm_swiglu`
- `rmsnorm_rht_amax`

Thank you also to the broader CUTLASS/CuTe DSL and infrastructure teams who
supported the original kernel development.