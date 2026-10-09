# Third-party code and provenance

This directory vendors *unaltered algorithm bodies* from AReaL-DTA
[areal-project/AReaL](https://github.com/areal-project/AReaL/tree/a5b0b4811a3ef7bf58f0270abcd81d7154f03ce6/areal/experimental/dta)
at commit `a5b0b4811a3ef7bf58f0270abcd81d7154f03ce6` (branch `feat/dta`):

- `dp.py` from `areal/experimental/dta/dp.py`
- `token_trie.py` from `areal/experimental/dta/token_trie.py`
- `trie.py` from `areal/experimental/dta/trie.py`
- `tree_time_model.py` from `areal/experimental/dta/tree_time_model.py`

The **only source changes** are namespace-only imports from
`areal.experimental.dta.*` to `verl.models.mcore.tpr._vendor.areal_dta.*`.
Keep upstream files separate from the VERL adapter; never amend vendored
algorithms to work around a VERL dispatch mismatch.

Each AReaL source has SPDX-License-Identifier: Apache-2.0.
See `LICENSE-APACHE-2.0` in this directory and the upstream notices.

AReaL notes that these modules are adapted from
[Whisper-6/DynamicTreeAttn](https://github.com/Whisper-6/DynamicTreeAttn)
by Yuchen Yang (original MIT license; see `LICENSE-MIT-DynamicTreeAttn`).
Both attributions are preserved.

Important behavioral constraints:
- `TokenTrie` leafization combines repeated and strictly contained
  trajectories, so the upstream leaf partition may contain fewer than
  `dp_size` groups; empty DP replicas are **not** supported by current VERL.
- `LB_by_DFS_and_TM` permits unequal numbers of original samples per
  DP replica; VERL's existing `RayPPOTrainer._balance_batch` equal-cardinality
  dispatch cannot safely consume such plans without further work.
- The upstream `TreeTimeModel` uses numpy/scipy; the local wrapper provides
  an interface-compatible tree-token predictor without adding those deps.
