# SM100 forward implementations

Each migrated main implementation has a local owner directory. Half D256,
exact half D512/D576 and per-tensor FP8 D192/D256/D512 graph plans use explicit engines;
other SM100 implementations remain at their existing paths.

```text
sm100/
  d256_f16_lookahead/   # single-Q lookahead, 128 query rows / two CTAs
    contract.py        # lightweight identity, fixed geometry, local support
    config.py          # this implementation's compile-time layout/config
    kernel.py          # main device pipeline and compile entry
    host.py            # fixed lowering arguments; no performance crossover
    __init__.py        # no eager kernel or DSL imports
  d256_f16_swap_ab/     # swap-AB, 16 or 32 query rows / one CTA
    contract.py
    config.py
    kernel.py
    host.py
    __init__.py
  d256_f16_contract.py  # immutable selection record / common exact-D256 domain
  d512_f16_role_split/  # D512, 128 query rows / four-CTA cluster
    contract.py
    config.py
    kernel.py
    host.py
    __init__.py
  d512_f16_2x2/         # D512/D576, 64 query rows / four- or two-CTA cluster
    contract.py
    config.py
    kernel.py
    host.py
    __init__.py
  d512_f16_contract.py  # immutable selection; full cluster vs fixed MMA pair
  d192_fp8_multi_q/    # per-tensor FP8 D192/V128, one/two CTA variants
    contract.py
    config.py
    kernel.py
    host.py
    __init__.py
  d256_fp8_single_q/    # per-tensor FP8 D256, one/two softmax WG variants
    contract.py
    config.py
    kernel.py
    host.py
    __init__.py
  d512_fp8_role_split/  # per-tensor FP8 D512, separate from half / block scale
    contract.py
    config.py
    kernel.py
    host.py
    __init__.py
```

The graph parser, variant-pack binder, compilation cache and launch lifecycle
remain shared. Performance selection belongs to `fwd/heuristics.py`; a selected
engine cannot silently switch to another main pipeline. Importing an owner's
contract/config does not load a kernel body or the tensor/DSL runtimes.

This is implementation isolation, not duplication of shared utilities.
`fwd/config_common.py` owns common value types and layout arithmetic;
`fwd/config_swap_ab.py` owns geometry shared with the independently maintained
SM107 swap-AB body. The existing SM107 shared lookahead route, gated-attention
wrapper and restricted D128 swap-AB route explicitly import/load these owners.
Their tests remain part of the change surface. No old-path forwarding modules
or cross-import of sibling kernel bodies are needed.

Wide half configuration is independent of the per-tensor FP8 record.
`fwd/config_d512_2x2_common.py` owns only the immutable 2x2 record and geometry
helpers shared with SM107; the two architectures retain their own factories.
The explicit D512 rows expose `TILE_CGA_M=4`, while D576 exposes `2`. Both
lower to a fixed two-CTA MMA group. This translation is checked at compile time
and does not change the legacy split-cost model's input geometry.

The per-tensor D192 owner has a standalone configuration record; the shared
root D192 factory remains for half/MXFP8 consumers. Neither implementation
record inherits mutable defaults from the other.

The per-tensor D256 owner keeps its own configuration record and factory.
The legacy D256 record remains with its MXFP8 consumers, including wide MXFP8;
neither path imports the other implementation's factory or validator.

Kernel diagnostic identities retain their `prefill_d256_f16`, `decode_d256_f16`,
`prefill_d192_d128_fp8`, `prefill_d256_fp8`, `prefill_d512_f16`, `prefill_d512_f16_2x2` and `prefill_d512_fp8` spellings; file basenames are not a
capability API. Runtime capabilities are queried from
the final selected plan's behavior notes.
