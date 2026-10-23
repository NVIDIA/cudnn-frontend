# Compilation, reuse, caching, and deployment

Reuse is valid only while graph structure, metadata, plan policy, and target compatibility remain within the built support surface.

## Static graph reuse

A static graph can execute repeatedly with new pointers when dimensions, strides, data types, operation attributes, and plan assumptions remain compatible. Wrapper and decorator caches turn those metadata values into keys.

## Dynamic graphs

Dynamic-shape and override-shape modes allow specific runtime metadata changes. The engine still defines a bounded support surface. Kernel caches store specializations; they do not make arbitrary shapes legal.

## Serialization

Backend serialization preserves a cuDNN graph representation, not arbitrary Python engines or inspection dictionaries. Device properties, target SM, backend version, and serialization compatibility determine whether another process or system can load it.

## Warmup and AOT

Warmup materializes lazy work before service. Deviceless AOT supplies target properties explicitly so compilation does not inspect a local GPU. Both require a fallback or rebuild policy when deployment compatibility changes.

## Cache identity

Include frontend and backend versions, CUDA/driver where relevant, GPU architecture, graph key, tensor metadata, operation attributes, engine/knob policy, and dynamic bounds. Reject partial matches rather than executing a stale plan.
