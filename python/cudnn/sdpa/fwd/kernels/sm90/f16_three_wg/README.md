# SM90 implementation ownership

The three-warpgroup body retains the existing QK/softmax/PV pipeline. In this
first migration batch engine 20540 owns only exact D64/V64 paged graphs. The
other widths and envelope shapes still use the legacy row; they are not
claimed as migrated by this batch.

The immutable parameter ABI, pure layout helpers, prepared binder and launch
services remain shared. `host.py` fixes the body before lowering; candidate
ranking stays in the central heuristics. No runtime Stats-stride note is added.

Source baseline: `5913f355354bd35d5be54494b91b1227b01917bb`.
