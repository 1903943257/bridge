# Native full recomputation: compatibility gate

## Status (2026-09-16)

**Not enabled for TPR. The requested configuration-only integration is blocked
by the current external-KV autograd contract.** No checkpoint implementation,
attention kernel, or Push/Visit/Pop lifecycle was changed. No long-context
performance results have been obtained.

The local Python environment has no PyTorch or NPU runtime. The contract tests
were syntax-checked only; they are not a passed correctness gate. They are
expected to fail for the inspected native implementation and are deliberately
not marked xfail. They test prerequisites, not full-model equivalence or Ring
communication replay.

## Sources inspected

- Local MindSpeed checkout: `376e9cc3`; `setup.py` declares `0.16.0`.
- Local Megatron-LM checkout: `59b72fa57`. This checkout uses
  `megatron/core/recompute.py`; it has **not** been verified to be the exact
  Megatron package imported by the server's MindSpeed core_r0.16.0 environment.
- VERL source: `D:/project/uni-agent/verl/verl/workers/config/engine.py` and
  `workers/engine/megatron/transformer_impl.py`.
- TPR overlay: `verl/models/mcore/tpr/{attention,context,segment_executor}.py`.

Before interpreting server results, record `torch.__version__`, imported
`megatron.core.__file__`, `mindspeed.__file__`, their installed versions/revisions,
and the actual checkpoint callable's source path after `repatch`.

## Existing configuration path

`MegatronEngineConfig.override_transformer_config` already exists. The engine
passes it through `bridge.set_extra_args(**override_transformer_config)` for
the vanilla bridge, or merges it into Megatron-Bridge provider overrides.
No additional top-level configuration mechanism is needed.

The inspected TransformerConfig exposes:

- `recompute_granularity`: `None`, `selective`, `full` (default `None`).
- `recompute_method`: `None`, `uniform`, `block`.
- `recompute_num_layers`: optional integer.
- `recompute_modules`: optional list for selective recomputation.

The native parser's legacy `recompute_activations` flag maps to **selective**,
not full. The intended full configuration, placed under the applicable
Megatron engine configuration, is:

```yaml
override_transformer_config:
  recompute_granularity: full
  recompute_method: uniform
  recompute_num_layers: 1
```

This is a configuration example, **not a validated TPR launch configuration**.
The direct profile model builder bypasses the engine config, so eventually it
must pass the same fields to `hf_to_mcore_config_dense` before construction.
Reference and TPR in that profile share one model; the fields must never be
changed between the two paths.

## Why passthrough alone is insufficient

1. Native BF16 full recomputation uses `tensor_parallel.checkpoint`.
   `CheckpointFunction.forward` executes its closure under `torch.no_grad()`.
   Transformer checkpoint closures return hidden states (and optional ordinary
   Transformer context), not TPR's layer-internal K/V tensors.
2. `TPRSelfAttention._tree_forward` exports K/V by side effect through
   `TPRAttentionContext.set_new_kv`. During the first checkpointed forward these
   tensors have no autograd graph. `SegmentExecutor.pop` later includes those
   tensors in `torch.autograd.backward(roots, grad_tensors=...)`. Full checkpoint
   does not make side-effect tensors differentiable checkpoint outputs.
3. `SegmentExecutor._forward` exits `use_tpr_attention_context` before calling
   backward. Native checkpoint stores its function and explicit inputs, not
   this Python ContextVar. Replay therefore lacks the TPR context and can take
   the ordinary attention path instead of external-prefix/Ring attention.
4. Simply retaining the context is insufficient: `set_new_kv` rejects a second
   write for a layer, and retaining context does not fix the missing Pop roots.

The inspected MindSpeed TransformerBlock checkpoint path also delegates BF16
execution to `tensor_parallel.checkpoint`; it does not expose a TPR side-output
or ContextVar restoration interface. Selective core-attention checkpointing
is explicitly rejected by the current TPR attention implementation. No new
selective policy was introduced as a workaround.

These are graph/interface incompatibilities, not a choice of uniform chunk
size. Resolving them would require work beyond configuration passthrough or an
existing native integration interface in the actual server version that is
absent from the inspected sources. Do not silently drop prefix gradients.

## Native contract probes

Run CP=1 first in the server environment:

```bash
TPR_RUN_NATIVE_RECOMPUTE_CONTRACT=1 torchrun --nproc_per_node=1 \
  --master_port=29561 -m pytest -s -v --tb=short \
  tests/models/mcore/tpr/correctness/test_tpr_native_recompute_contract_npu.py
```

The tests call the actual framework `tensor_parallel.checkpoint`; they do not
implement or wrap a replacement checkpoint mechanism. One tests differentiable
KV side outputs, the other tests context restoration during native replay.
Run with `--nproc_per_node=2` only after the CP=1 prerequisites pass.

Full-model loss/logprob/all-parameter-gradient/prefix-gradient OFF/ON equivalence,
multi-iteration stability and Ring replay deadlock checks remain outstanding.

## Deferred performance cases

After all correctness gates pass, use Qwen3-4B BF16, CP=4 and the same native
`full/uniform/1` config for both paths. Set profile cases in the file:

```python
_PROFILE_CASES = (
    _ProfileCase(32768, 24576, 8),  # 57344 tokens per trajectory
    _ProfileCase(32768, 32768, 8),  # 65536 tokens, if memory permits
)
```

Log the effective TransformerConfig before measuring: activation recompute
enabled, granularity, method and num_layers. Preserve the existing latency,
allocated/reserved memory, Ring communication, FA and merge reports. Full-layer
recomputation does not eliminate full-vocabulary logits/loss memory. The
existing forward-only adapter trace and module timing hooks also need validation
under replay; physical model phases and checkpoint layer replays must be counted
separately. These performance changes are deliberately not enabled before the
compatibility gates pass.
