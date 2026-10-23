# Migrate from the legacy C++ builder API

Move descriptor-by-descriptor code to the modern Graph API while preserving layouts, operation attributes, engine policy, and runtime bindings.

## Before you begin

Inventory one legacy graph and capture its dimensions, strides, UIDs, data
types, attributes, selected-plan policy, workspace, output values, and error
behavior. Keep that result as the migration baseline.

## Map the construction model

| Legacy builder | Modern Graph API |
| --- | --- |
| Tensor, convolution, pointwise, reduction descriptors | `Tensor_attributes` and operation-specific attribute values |
| Operation builders and `OperationGraph` | Calls on `cudnn_frontend::graph::Graph` |
| Engine heuristics/config generators | Graph heuristic and plan-creation methods |
| `ExecutionPlan` | Built Graph plan or at-index plan |
| `VariantPack` | Tensor-attribute-to-pointer map plus workspace |
| Exception/status builder checks | `error_t` checks at each Graph lifecycle stage |

## Migrate one graph at a time

Record the old graph's tensor dimensions, strides, UIDs, types, operation attributes, notes, workspace, and numerical results. Rebuild the same graph with modern attributes, compare output and resource behavior, then replace plan-selection code. Do not mix legacy and modern descriptor ownership unless a documented interoperability API permits it.

## Keep legacy reference scoped

Legacy headers remain installed for compatibility, but new examples and tutorials should use the Graph API. Mark behavior that exists only in a legacy descriptor rather than implying a modern equivalent.

## Expected result

The modern Graph produces the same numerical result and compatible resource
behavior as the recorded legacy graph. Migrate and verify one complete graph at
a time, then remove its legacy construction and execution path.
