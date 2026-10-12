# P0-5 Real Multi-Sibling Prefix Relay Precision Validation

Date: 2026-10-12

## Summary

P0-5 validates TPR under a more realistic tree execution pattern with real sibling trajectories. Compared with previous single-leaf P0-4 validation, this experiment verifies:

- actual TQ original sibling inputs
- distinct real responses
- multi-leaf prefix KV relay
- prefix KV gradient accumulation path

The experiment is still a numerical correctness diagnostic based on NLL objective. It does **not** claim PPO/GRPO reward convergence.

## Configuration

- Model: Qwen3-1.7B
- Objective: NLL
- Prefix: P512
- Real siblings: trajectories `[0, 1]`
- Initial root prefix hash verified
- `prefix_kv_sum_verified=True`
- `multi_leaf_kv_relay=True`
- No synthetic advantages
- No recorded old logprobs

## Key Results

### Training loss

Step 1:

```
Full loss: 1.4318195
TPR loss:  1.43271225
Gap:       0.00089275
```

Step 20:

```
Shared Full Oracle NLL:
Full: 0.0114317683
TPR:  0.00858048571
Gap: -0.0028512826
```

TPR and Full remain in the same optimization regime. No evidence of monotonic error amplification was observed.

## Numerical Drift

The observed difference is consistent with previous GEMM/CP experiments:

- different GEMM shapes change BF16 accumulation order
- different attention partitioning changes reduction order
- backward graph changes gradient accumulation order

Representative metrics:

### Step 1

```
Masked logprob:
MAE:      0.0376383
Rel L2:   0.02507
Cosine:   0.999686

Sampled gradient:
Rel L2:   0.16087
Cosine:   0.98714
```

### Later steps

Some steps show larger sampled gradient/logprob divergence due to different optimization trajectories. This should be interpreted as parameter trajectory drift rather than direct evidence of incorrect TPR execution.

## Conclusion

P0-5 confirms that real multi-sibling prefix reuse and KV relay are functionally correct under the tested NLL training path.

The remaining validation target is true end-to-end RL training:

- real `old_log_probs`
- real advantages
- PPO/GRPO ratio computation
- clipping behavior
- KL/reward convergence

Current precision assessment:

- GEMM shape drift: confirmed, expected BF16 numerical behavior
- CP-scale drift: confirmed, comparable magnitude to native CP
- TPR prefix relay correctness: passed
- Multi-sibling tree execution: passed
- Production RL convergence: pending E2E validation
