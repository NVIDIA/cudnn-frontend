# MoeEP receive-capacity API direction

> Status: design proposal for TE integration. This document does not describe
> a released API change.

## Capacity terms

- `R` — raw route count derived from topology and input limits, normally
  `ep_size * max_tokens_per_rank * top_k`.
- `L` — logical receive-route limit. Routing overflow and upstream kernel
  presets operate on this value.
- `P` — padded physical receive-pool rows. The kernel derives it from `L`,
  local expert count, and padding granularity. Workspace and WGrad operand
  shapes use this value.

`L` and `P` are intentionally different: per-expert token segments are padded
independently, so a logical capacity of `L` routes can require more than `L`
physical rows.

## Current behavior

The public `MoeEpParallelConfig.max_recv_size_per_rank` means physical rows
`P`, while the vendored kernel parameter with the same name means logical
limit `L`.

The regular backend path therefore:

1. accepts `P` from the caller, or computes a default `P`;
2. duplicates the kernel padding rule to reverse-map `P` to `L`;
3. passes `L` through the kernel parameter named
   `max_recv_size_per_rank`;
4. lets the kernel calculate `P` again and verifies that it matches the
   requested physical pool.

Some upstream optimization presets constrain `L` directly. Supporting them
requires additional preset-specific handling because the public API currently
expresses `P`.

Callers do not allocate private MoeEP workspace. After `prepare_training()`,
they allocate caller-owned forward state and WGrad operands from the returned
`(shape, stride, dtype, alignment)` contracts.

## Proposed behavior

Expose logical capacity explicitly, for example as
`logical_recv_route_limit`, and make the prepared kernel the source of truth
for physical capacity:

1. the caller provides `L`, or lets MoeEP derive its uncapped default from
   the topology;
2. MoeEP passes `L` unchanged to the kernel;
3. the kernel derives `P` once;
4. `prepare_training()` returns the complete allocation contracts derived
   from the prepared kernel;
5. the caller allocates output bundles from those contracts and then executes
   forward followed by backward.

MoeEP may additionally expose a read-only
`resolved_physical_pool_rows` value for logging and memory planning. Callers
should still allocate from the complete contracts rather than deriving tensor
shapes from `P` alone, because scale layouts, strides, dtypes, and alignment
requirements are also part of the ABI.

For compatibility, the existing `max_recv_size_per_rank` can temporarily
retain its physical-`P` meaning during a deprecation period. Setting both the
legacy physical field and the new logical field should be rejected.

## Example

For an upstream preset requiring:

```text
L = 131072
local experts = 8
padding block = 128
```

the kernel derives:

```text
P = 131968
```

Today, the TE caller expresses the physical value `131968`, and MoeEP must
recover the logical value `131072`. With the proposed API, the caller expresses
`L=131072`; `prepare_training()` then reports WGrad operand shapes whose pool
dimension is `P=131968`.

## Expected benefits

- Aligns the public capacity knob with routing overflow semantics and upstream
  optimization presets.
- Removes the normal-path `P -> L -> P` round trip and its duplicated padding
  logic.
- Reduces preset-specific adapters and local vendored-code overlays.
- Gives TE exact, kernel-derived allocation contracts while keeping private
  workspace management inside MoeEP.
- Makes compile keys, diagnostics, and capacity errors use unambiguous units.
- Allows future kernel padding/layout changes without requiring TE to
  reimplement capacity formulas.

The intended ownership boundary is therefore: TE chooses logical workload
capacity `L`; the kernel owns physical layout `P`; MoeEP publishes the exact
allocation ABI connecting them.
