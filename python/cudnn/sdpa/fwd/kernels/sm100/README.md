# SM100 forward implementations

The first explicit-engine pilot gives each D256 half main implementation a local
owner directory. Other SM100 implementations remain at their existing paths.

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
```

The graph parser, variant-pack binder, compilation cache and launch lifecycle
remain shared. Performance selection belongs to `fwd/heuristics.py`; a selected
engine cannot silently switch to the other main pipeline. Importing either
contract/config does not load a kernel body or the tensor/DSL runtimes.

This is implementation isolation, not duplication of shared utilities.
`fwd/config_common.py` owns common value types and layout arithmetic;
`fwd/config_swap_ab.py` owns geometry shared with the independently maintained
SM107 swap-AB body. The existing SM107 shared lookahead route, gated-attention
wrapper and restricted D128 swap-AB route explicitly import/load these owners.
Their tests remain part of the change surface. No old-path forwarding modules
or cross-import of sibling kernel bodies are needed.

Kernel diagnostic identities remain `prefill_d256_f16` and `decode_d256_f16`;
file basenames are not a capability API. Runtime capabilities are queried from
the final selected plan's behavior notes.
