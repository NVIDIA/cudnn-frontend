# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compile native MegaMoE into positional TVM-FFI launch sequences."""

from dataclasses import dataclass
import threading

import jax
import jax.numpy as jnp

from cudnn._cake.jax import TensorRef, make_call
from cudnn._device import device_info, ensure_current_context

from ._config import Mxfp8KernelConfig
from ._jax_resources import SymmetricResources
from ._native_kernel import make_backward_kernel, make_forward_kernel


@dataclass(frozen=True)
class PreparedPhase:
    config: Mxfp8KernelConfig
    kernel: object


def output_specs(prepared, tokens):
    """XLA carriers preserve native producer bytes, including blocked scales."""
    config, kernel = prepared.config, prepared.kernel
    specs = [((tokens, config.hidden), "bfloat16"), ((2,), "int32")]
    if config.dfc2_recompute:
        aux = kernel.get_aux_output_shapes()
        specs += [
            ((tokens, config.top_k), "float32"),
            (aux["fc1_col_output"], "uint8"),
            (aux["fc1_col_output_sf"], "uint8"),
            (aux["fc1_recompute"], "uint8"),
            (aux["fc1_recompute_sf"], "uint8"),
            ((config.hidden, aux["grad_y2"][0]), "uint8"),
            (aux["grad_y2_sf"], "uint8"),
        ]
    elif config.generate_c:
        aux = kernel.get_aux_output_shapes()
        specs += [
            (aux["fc1_c"], "bfloat16"),
            (aux["col_quant_data"], "uint8"),
            (aux["col_quant_sf"], "uint8"),
            ((config.num_experts,), "int32"),
            ((config.num_experts,), "int32"),
        ]
    return tuple(jax.ShapeDtypeStruct(tuple(shape), jnp.dtype(dtype)) for shape, dtype in specs)


def weight_specs(config, *, backward):
    experts, hidden, intermediate = config.num_experts, config.hidden, config.intermediate
    if backward:
        return (
            (experts, hidden, intermediate),
            (experts, intermediate * (hidden // 32)),
            (experts, 2 * intermediate, hidden),
            (experts, hidden * (2 * intermediate // 32)),
        )
    return (
        (experts, 2 * intermediate, hidden),
        (experts, 2 * intermediate * (hidden // 32)),
        (experts, hidden, intermediate),
        (experts, hidden * (intermediate // 32)),
    )


class NativeBindings:
    """One prepared communication world with device-specific compiled adapters."""

    def __init__(self, resolved, *, device, launch_clusters):
        self.device = device
        self.phases = {}
        for phase, factory in (
            ("inference", Mxfp8KernelConfig.for_inference),
            ("training_forward", Mxfp8KernelConfig.for_training_forward),
            ("training_backward", Mxfp8KernelConfig.for_training_backward),
        ):
            config = factory(resolved, launch_cluster_count=launch_clusters)
            construct = make_backward_kernel if phase == "training_backward" else make_forward_kernel
            kernel = construct(config, target_arch="sm_107a", sm_count=device_info(device).sm_count)
            self.phases[phase] = PreparedPhase(config, kernel)
        topology = resolved.topology
        self.resources = SymmetricResources(self.phases, device=device, rank=topology.ep_rank, world_size=topology.ep_size)
        self.rank = topology.ep_rank
        self.world_size = topology.ep_size
        self._compiled = {}
        self._calls = {}
        self._prepare_lock = threading.RLock()

    def close(self):
        with self._prepare_lock:
            self.resources.close()

    def call(self, phase, source, indices, scores, weights, preact=None, dependency=None):
        with self._prepare_lock:
            if self.resources._closed:
                raise RuntimeError("the JAX MoeEP plan is closed")
            key = (phase, source.dtype.name, source.shape[0], dependency is not None)
            invoke = self._calls.get(key)
            if invoke is None:
                invoke = self._prepare_call(phase, source.dtype.name, source.shape[0], dependency is not None)
                self._calls[key] = invoke
        inputs = (source, indices, scores, *weights)
        if preact is not None:
            inputs += (preact,)
        if dependency is not None:
            inputs += (dependency,)
        return invoke(*inputs)

    def _prepare_call(self, phase, source_dtype, tokens, has_dependency):
        prepared = self.phases[phase]
        config = prepared.config
        backward = phase == "training_backward"
        input_specs = [((tokens, config.hidden), source_dtype), ((tokens, config.top_k), "int32"), ((tokens, config.top_k), "float32")]
        input_specs += [(shape, "uint8") for shape in weight_specs(config, backward=backward)]
        if backward:
            input_specs.append((prepared.kernel.get_fc1_preact_shape(), "bfloat16"))
        kernel_input_count = len(input_specs)
        if has_dependency:
            input_specs.append(((2,), "int32"))
        input_count = len(input_specs)
        inputs = [TensorRef(i, tuple(shape), dtype) for i, (shape, dtype) in enumerate(input_specs)]
        outputs = output_specs(prepared, tokens)
        result_refs = [TensorRef(input_count + i, spec.shape, spec.dtype.name) for i, spec in enumerate(outputs)]
        resources = self.resources
        external = [resources.ref(name) for name in ("activation", "activation_sf", "indices", "scores", "output", "overflow")]
        external += [resources.ref(phase + ".local"), resources.ref(phase + ".shared")]
        if backward:
            external += [resources.ref("dprob"), resources.ref("beta")]
        compile_key = (phase, source_dtype)
        fn = self._compiled.get(compile_key)
        if fn is None:
            ensure_current_context(device=self.device)
            fn = self._compile(phase, source_dtype, external)
            self._compiled[compile_key] = fn
        zeros = [resources.ref("overflow"), *result_refs]
        if backward:
            zeros.append(resources.ref("dprob"))
        if config.fc2_in_kernel_topk_reduce:
            zeros.append(resources.ref("output"))
        return make_call(
            ((fn, (*inputs[:kernel_input_count], *external, *result_refs)),),
            input_count,
            outputs,
            self.device,
            zero_buffers=zeros,
            context=resources.context,
            has_side_effect=True,
        )

    def _compile(self, phase, source_dtype, external):
        import cutlass
        import cutlass.cute as cute

        prepared = self.phases[phase]
        config = prepared.config
        backward = phase == "training_backward"
        token_count = cute.sym_int32()
        specs = [((token_count, config.hidden), source_dtype), ((token_count, config.top_k), "int32"), ((token_count, config.top_k), "float32")]
        specs += [(shape, "uint8") for shape in weight_specs(config, backward=backward)]
        if backward:
            specs.append((prepared.kernel.get_fc1_preact_shape(), "bfloat16"))
        specs += [(ref.shape, ref.dtype) for ref in external]
        outputs = output_specs(prepared, 1)
        specs += [(((token_count, spec.shape[1]) if i == 0 or (backward and i == 2) else spec.shape), spec.dtype.name) for i, spec in enumerate(outputs)]
        types = {"uint8": cutlass.Uint8, "int32": cutlass.Int32, "float32": cutlass.Float32, "bfloat16": cutlass.BFloat16}
        fake = [
            cute.runtime.make_fake_compact_tensor(types[dtype], tuple(shape), stride_order=tuple(reversed(range(len(shape)))), assumed_align=16)
            for shape, dtype in specs
        ]
        launcher = self._launcher(phase)
        return cute.compile(launcher, *fake, cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True), options="--enable-tvm-ffi --gpu-arch=sm_107a")

    def _launcher(self, phase):
        import cutlass
        import cutlass.cute as cute

        from ..cutedsl_src.communication.nvlink_domain.symmetric_buffer import SymmetricBufferHost
        from ._jax_kernel import copy_dprob, copy_output, expert_metadata, fill_beta, stage_input, typed

        prepared = self.phases[phase]
        config, kernel = prepared.config, prepared.kernel
        experts, hidden, intermediate = config.num_experts, config.hidden, config.intermediate
        mapper = SymmetricBufferHost(self.resources.address, self.resources.offsets, cutlass.Int32(self.rank), self.world_size)

        @cute.jit
        def forward(
            source,
            idx,
            scores,
            w1,
            s1,
            w2,
            s2,
            a,
            sf,
            routing,
            routing_scores,
            output,
            overflow,
            local,
            shared,
            result,
            status,
            preact,
            col_data,
            col_sf,
            counts,
            offsets,
            stream,
        ):
            stage_input(source, idx, scores, a, sf, routing, routing_scores, status, experts * config.world_size, stream)
            kernel(
                typed(a, cutlass.Float8E4M3FN, a.shape),
                typed(sf, cutlass.Float8E8M0FNU, sf.shape),
                routing,
                routing_scores,
                typed(w1, cutlass.Float8E4M3FN, (experts, hidden, 2 * intermediate), (hidden * 2 * intermediate, 1, hidden)),
                typed(s1, cutlass.Float8E8M0FNU, s1.shape),
                typed(w2, cutlass.Float8E4M3FN, (experts, intermediate, hidden), (intermediate * hidden, 1, intermediate)),
                typed(s2, cutlass.Float8E8M0FNU, s2.shape),
                output,
                preact,
                None if cutlass.const_expr(col_data is None) else typed(col_data, cutlass.Float8E4M3FN, col_data.shape),
                col_sf,
                overflow,
                local.iterator,
                shared.iterator,
                mapper,
                stream,
            )
            copy_output(output, result, overflow, status, stream)
            if cutlass.const_expr(config.generate_c):
                byte_offset = kernel._mega_device_workspace.offset(kernel.col_quant_sizes_region)
                expert_metadata(local, counts, offsets, byte_offset, config.token_padding_block).launch(grid=[1, 1, 1], block=[1, 1, 1], stream=stream)

        @cute.jit
        def inference(source, idx, scores, w1, s1, w2, s2, a, sf, routing, routing_scores, output, overflow, local, shared, result, status, stream):
            forward(
                source,
                idx,
                scores,
                w1,
                s1,
                w2,
                s2,
                a,
                sf,
                routing,
                routing_scores,
                output,
                overflow,
                local,
                shared,
                result,
                status,
                None,
                None,
                None,
                None,
                None,
                stream,
            )

        @cute.jit
        def backward(
            source,
            idx,
            scores,
            w2t,
            s2t,
            w1t,
            s1t,
            preact,
            a,
            sf,
            routing,
            routing_scores,
            output,
            overflow,
            local,
            shared,
            dprob,
            beta,
            result,
            status,
            grad_prob,
            fc1_b,
            fc1_sfb,
            fc2_a,
            fc2_sfa,
            fc2_b,
            fc2_sfb,
            stream,
        ):
            stage_input(source, idx, scores, a, sf, routing, routing_scores, status, experts * config.world_size, stream)
            fill_beta(beta).launch(grid=[cute.ceil_div(experts, 256), 1, 1], block=[256, 1, 1], stream=stream)
            kernel(
                typed(a, cutlass.Float8E4M3FN, a.shape),
                typed(sf, cutlass.Float8E8M0FNU, sf.shape),
                routing,
                routing_scores,
                typed(w2t, cutlass.Float8E4M3FN, w2t.shape),
                typed(s2t, cutlass.Float8E8M0FNU, s2t.shape),
                typed(w1t, cutlass.Float8E4M3FN, w1t.shape),
                typed(s1t, cutlass.Float8E8M0FNU, s1t.shape),
                beta,
                preact,
                output,
                overflow,
                dprob,
                typed(fc2_a, cutlass.Float8E4M3FN, fc2_a.shape),
                typed(fc2_sfa, cutlass.Float8E8M0FNU, fc2_sfa.shape),
                typed(fc1_b, cutlass.Float8E4M3FN, fc1_b.shape),
                typed(fc1_sfb, cutlass.Float8E8M0FNU, fc1_sfb.shape),
                typed(fc2_b, cutlass.Float8E4M3FN, (fc2_b.shape[1], hidden), (1, fc2_b.shape[1])),
                fc2_sfb,
                local.iterator,
                shared.iterator,
                mapper,
                stream,
            )
            copy_output(output, result, overflow, status, stream)
            if cute.size(grad_prob) > 0:
                copy_dprob(dprob, grad_prob).launch(grid=[cute.ceil_div(cute.size(grad_prob), 256), 1, 1], block=[256, 1, 1], stream=stream)

        return backward if phase == "training_backward" else forward if phase == "training_forward" else inference
