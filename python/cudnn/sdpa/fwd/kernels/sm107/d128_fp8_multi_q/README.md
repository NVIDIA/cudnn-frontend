# SM107 D64/D128 per-tensor FP8

This owner contains the native multi-Q body and its independent configuration.
The shared host handles graph binding, compilation and execution. The central
policy chooses candidate geometry; explicit records fix it before lowering.
