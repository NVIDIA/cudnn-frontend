# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepared canonical dense MXFP8 calls with caller-owned execution lifetimes."""

from dataclasses import dataclass, field
import inspect
import os

from cudnn.api_base import TupleDict
from .backend_utils import wrapper_operand_meta

# Operands whose leading extent is the routed row count M. Prepared plans accept any M.
M_OPERANDS = ("a_tensor", "prob_tensor", "row_scale_tensor")

# (kind, stream, M-free contract, scalars, environment) -> compiled API. Plans on one
# stream execute in order, so they can share the compiled workspace.
compiled_apis = {}


def output_spec(tensor):
    return None if tensor is None else (tensor.shape, tensor.stride(), tensor.dtype, tensor.device)


def m_free_meta(name, tensor):
    """Operand metadata without the M extent; SFA is a dense buffer sized by M."""
    meta = wrapper_operand_meta(tensor)
    if meta is None or not hasattr(tensor, "shape"):
        return meta
    shape, stride, *rest = meta
    if name == "sfa_tensor":
        return (len(shape), tensor.is_contiguous(), *rest)
    if name in M_OPERANDS:
        return (shape[1:], stride, *rest)
    return meta


@dataclass
class PreparedGroupedGemm:
    """A fixed-metadata call for any routed row count M.

    ``check=False`` is for integrations that guarantee the preparation contract:
    identical tensor metadata apart from M and scalar configuration, fresh runtime
    tensor values, and ordered execution on the preparation stream. Row-output
    reuse is explicit: a subsequent call with the same M overwrites D/SFD_row
    after previous stream work completes. C and column outputs always receive
    fresh storage for backward consumers.
    """

    api: object
    kind: str
    defaults: dict
    tensor_contract: dict
    scalar_contract: dict
    runtime_names: tuple
    stream: object
    make_outputs: object
    reuse_rows: bool
    sfa_elements_per_128_rows: int
    m: int = -1
    output_specs: dict = field(default_factory=dict)
    row_outputs: dict = field(default_factory=dict)

    def __post_init__(self):
        implementation = self.api._implementation if self.kind == "glu" else self.api
        self.execute = implementation.execute
        self.stream_handle = int(self.stream)
        self.fixed_arguments = {name: self.defaults.get(name) for name in self.runtime_names if name not in self.tensor_contract}
        if self.kind == "quant":
            self.fixed_arguments["norm_const_tensor"] = None
        self.fixed_arguments["current_stream"] = self.stream
        self.tensor_names = tuple(name for name in self.runtime_names if name not in self.fixed_arguments)

    def set_m(self, m):
        """Derive output metadata for M; row outputs are reusable only within one M."""
        samples = dict(self.make_outputs(m).items())
        self.m = m
        self.output_specs = {name: output_spec(tensor) for name, tensor in samples.items()}
        self.row_outputs = {name: samples[name] for name in ("d_tensor", "sfd_row_tensor")} if self.reuse_rows else {}
        return samples

    def check_call(self, kwargs):
        unknown = kwargs.keys() - self.defaults.keys() - {"current_stream"}
        if unknown:
            raise ValueError(f"Unknown prepared arguments: {sorted(unknown)}")
        for name, expected in self.tensor_contract.items():
            if m_free_meta(name, kwargs.get(name, self.defaults.get(name))) != expected:
                raise ValueError(f"Prepared {name} metadata changed; prepare a new plan")
        rows = -(-kwargs["a_tensor"].shape[0] // 128)
        if kwargs["sfa_tensor"].numel() != rows * self.sfa_elements_per_128_rows:
            raise ValueError("Prepared sfa_tensor size does not match A rows")
        for name, expected in self.scalar_contract.items():
            if kwargs.get(name, self.defaults.get(name)) != expected:
                raise ValueError(f"Prepared {name} changed; prepare a new plan")

    def allocate_outputs(self, provided=None, check=True, fresh=None):
        import torch

        supplied = {} if provided is None else provided
        unknown = supplied.keys() - self.output_specs.keys() if check else ()
        if unknown:
            raise ValueError(f"Unknown prepared outputs: {sorted(unknown)}")
        outputs = {}
        for name, spec in self.output_specs.items():
            tensor = supplied.get(name, self.row_outputs.get(name))
            if tensor is not None:
                if check and name in supplied and output_spec(tensor) != spec:
                    raise ValueError(f"Prepared output {name} metadata does not match")
            elif fresh is not None:
                tensor = fresh[name]
            elif spec is not None:
                shape, stride, dtype, device = spec
                tensor = torch.empty_strided(shape, stride, dtype=dtype, device=device)
            outputs[name] = tensor
        return TupleDict(**outputs)

    def run(self, *, check=True, outputs=None, **kwargs):
        if check:
            import torch

            self.check_call(kwargs)
            device = kwargs["a_tensor"].device
            if torch.cuda.current_stream(device).cuda_stream != self.stream_handle:
                raise ValueError("Prepared output allocation requires the preparation stream to be current")
        stream = kwargs.get("current_stream")
        if stream is not None and int(stream) != self.stream_handle:
            raise ValueError("Prepared plans require ordered execution on their preparation stream")
        m = kwargs["a_tensor"].shape[0]
        fresh = self.set_m(m) if m != self.m else None
        if self.kind == "quant" and kwargs.get("d_tensor") is not None:
            outputs = dict(outputs or {}, d_tensor=kwargs["d_tensor"])
        result = self.allocate_outputs(outputs, check, fresh)
        arguments = self.fixed_arguments.copy()
        for name in self.tensor_names:
            arguments[name] = kwargs.get(name)
        arguments.update(result.items())
        self.execute(**arguments)
        return result


def prepare_grouped_gemm(kind, *, reuse_row_outputs=False, **kwargs):
    """Prepare a canonical dense MXFP8 GLU or BF16/FP16 quant call for any routed M.

    Does not execute a GEMM. Tensor values and pointers are never retained except
    explicitly reusable row outputs. Scalars/environment settings freeze here.
    Prepare separate plans for concurrent streams or configurations; plans with the
    same configuration on one stream share one compiled kernel.
    """
    import torch
    from cuda.bindings import driver as cuda
    from functools import partial
    from .glu.api import GroupedGemmGluSm100, grouped_gemm_glu_wrapper_sm100, glu_block_scaled_outputs
    from .quant.api import GroupedGemmQuantSm100, grouped_gemm_quant_wrapper_sm100

    if kind not in ("glu", "quant"):
        raise ValueError("kind must be 'glu' or 'quant'")
    wrapper = grouped_gemm_glu_wrapper_sm100 if kind == "glu" else grouped_gemm_quant_wrapper_sm100
    bound = inspect.signature(wrapper).bind(**kwargs)
    bound.apply_defaults()
    values = bound.arguments
    a, b, sfa, sfb = (values[name] for name in ("a_tensor", "b_tensor", "sfa_tensor", "sfb_tensor"))
    if b is None or values["b_ptrs"] is not None or a.ndim != 2 or b.ndim != 3:
        raise ValueError("Prepared calls require canonical dense A/B")
    for tensor in (a, b, sfa, sfb):
        if tensor is None or not tensor.is_cuda or not tensor.is_contiguous() or tensor.device != a.device:
            raise ValueError("Prepared A/B/scales must be contiguous CUDA tensors on one device")
    if a.dtype != torch.float8_e4m3fn or b.dtype != a.dtype or sfa.dtype != torch.float8_e8m0fnu or sfb.dtype != sfa.dtype:
        raise ValueError("Prepared calls currently require MXFP8 E4M3 inputs")
    if values["cd_major"] != "n":
        raise ValueError("Prepared calls require n-major output")
    m, k = a.shape
    experts, n, b_k = b.shape
    if k != b_k or m == 0:
        raise ValueError("Prepared input dimensions must match and M must be nonzero")
    stream = values["current_stream"]
    if stream is None:
        stream = cuda.CUstream(torch.cuda.current_stream(a.device).cuda_stream)
    if torch.cuda.current_stream(a.device).cuda_stream != int(stream):
        raise ValueError("Prepare with the execution CUDA stream current")
    values["current_stream"] = stream
    if kind == "glu":
        c_dtype = values["c_dtype"] or torch.bfloat16
        d_dtype = values["d_dtype"] or torch.bfloat16
        if c_dtype not in (torch.bfloat16, torch.float16) or d_dtype != torch.float8_e4m3fn:
            raise ValueError("Prepared GLU requires FP16/BF16 C and E4M3 D")
        make_outputs = partial(
            glu_block_scaled_outputs,
            n_full=n,
            n_out=n // 2,
            l=experts,
            c_dtype=c_dtype,
            d_dtype=d_dtype,
            sf_dtype=sfa.dtype,
            sf_vec_size=values["sf_vec_size"],
            device=a.device,
            canonical=True,
        )
        api_class = GroupedGemmGluSm100
    else:
        d_dtype = values["d_dtype"] or torch.bfloat16
        if d_dtype not in (torch.bfloat16, torch.float16) or values["generate_amax"]:
            raise ValueError("Prepared quant requires FP16/BF16 D and generate_amax=False")

        def make_outputs(rows, device=a.device):
            return TupleDict(
                d_tensor=torch.empty((rows, n), dtype=d_dtype, device=device),
                d_col_tensor=None,
                amax_tensor=None,
                sfd_row_tensor=None,
                sfd_col_tensor=None,
            )

        api_class = GroupedGemmQuantSm100
    samples = make_outputs(m)
    sample_args = {"sample_" + name.removesuffix("_tensor"): tensor for name, tensor in samples.items()}
    for name in (
        "a_tensor",
        "b_tensor",
        "sfa_tensor",
        "sfb_tensor",
        "alpha_tensor",
        "bias_tensor",
        "prob_tensor",
        "row_scale_tensor",
        "scheduler_counter_tensor",
    ):
        if name in values:
            sample_args["sample_" + name.removesuffix("_tensor")] = values[name]
    sample_args["sample_padded_offsets"] = values["padded_offsets"]
    sample_args["sample_norm_const"] = values["norm_const_tensor"] if kind == "glu" else None
    constructor = inspect.signature(api_class).parameters
    sample_args.update({name: value for name, value in values.items() if name in constructor and not name.startswith("sample_")})
    sample_args = {name: value for name, value in sample_args.items() if name in constructor}
    api = api_class(**sample_args)
    if not api.check_support():
        raise ValueError("Unsupported prepared configuration")
    tensor_names = tuple(name for name in values if name.endswith("_tensor") or name in ("padded_offsets", "b_ptrs", "sfb_ptrs"))
    contract = {name: m_free_meta(name, values[name]) for name in tensor_names if name != "d_tensor"}
    defaults = {name: value for name, value in values.items() if name not in tensor_names and name != "current_stream"}
    scalar_contract = dict(defaults)
    defaults.update({name: None for name in tensor_names})
    compile_key = (
        kind,
        int(stream),
        tuple(contract.items()),
        tuple(scalar_contract.items()),
        os.getenv("CUDNNFE_CLUSTER_OVERLAP_MARGIN", "0"),
        os.getenv("CUDNN_FE_GROUPED_GEMM_DYNAMIC_MNKL", "1"),
    )
    api = compiled_apis.setdefault(compile_key, api)
    api.compile()
    implementation = api._implementation if kind == "glu" else api
    runtime_names = tuple(name for name in inspect.signature(implementation.execute).parameters if name not in samples and name != "current_stream")
    plan = PreparedGroupedGemm(
        api, kind, defaults, contract, scalar_contract, runtime_names, stream, make_outputs, reuse_row_outputs and kind == "glu", sfa.numel() // -(-m // 128)
    )
    plan.set_m(m)
    if values.get("d_tensor") is not None and output_spec(values["d_tensor"]) != plan.output_specs["d_tensor"]:
        raise ValueError("Prepared output D metadata does not match")
    return plan
