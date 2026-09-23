# TPR Ring: streaming KV 生命周期

## Native 对照与改造

参考本仓库 MindSpeed 的 `ring_context_parallel.py::AttentionWithCp`、
`context_parallel_kv_cache.py::ContextParallelKVCache` 和 `utils.py::RingP2P`。
Native forward 是 send-next → 当前 FA/merge → wait → buffer swap；
backward 使用 `is_backward=True` 反向 Ring 和滚动 dKV accumulator。
注意 native 存在 full/half KV cache policy，不能笼统说它一定无缓存。
本次对齐的是无缓存的 streaming 行为。
`ordinary_ring_cp_attention` 现在对齐 MindSpeed 默认配置，显式设置
`cache_policy=None`，即不启用 KV cache；Ref 容量测试因此使用 native no-cache Ring。

TPR 的区别：native 无缓存路径保存 forward 最后一块 KV（source=rank+1）；
这里按需求只保存 local KV。因此 backward 用一次反向 hop 恢复 rank+1
起点，之后按 rank+1,…,rank-1,rank 计算；最后直接用 local KV。
合计 CP-1 次 KV hop 和 CP-1 次 dKV hop，全部复用 native RingP2P，
不增加 collective，不修改 MindSpeed。第一版 dKV send/wait 同步，
不额外实现多流 dKV overlap。

每个 segment：

- forward：两个 packed KV ping-pong buffers，收到 source 后立即 FA/merge；
  不再建立 by_source。两个 Q chunk 均在这个 source 的 buffer 有效期内计算。
- ctx：只保存 Q、各 segment 的 local K/V、最终 output 和 max/sum。
  没有 CP 份 remote KV，也没有历史 source attention output。
- backward：两个 KV buffer + 两个 dKV accumulator；每步计算 dQ/dKV，
  dKV 沿反向 Ring 累加，最后在 owner 完成。没有 segment×source 梯度表。
- Prefix FULL、Current FULL/CAUSAL/SKIP、GQA、physical padding mask、
  Prefix KV/Q coalescing 的条件和 FA 算法不变。source 累加顺序从原先
  materialized source-index 顺序改为 native Ring step 顺序；容差不变。

对相同 local P/S，KV 通信 scratch 和 ctx 保存 KV 的大小不再乘 CP。
local KV、输出/softmax、所有 segment 的**local**最终梯度仍必须保留。
allocator reserved peak、模型整体 peak 不一定完全相同；mask、FA workspace、
HCCL 内部 buffer 和 activation reload 也会影响峰值，不能预先承诺减少 7 GiB。

## 本地验证

```bash
python -B tests/models/mcore/tpr/unit/test_ring_coalescing_schedule.py
python -B tests/models/mcore/tpr/unit/test_ring_streaming_transport.py
```

前者运行实际调度代码和 shape-only FA stub，检查 CP2/CP4 所有 rank 的
coalescing FA 次数、返回梯度 shape 和 ctx 中 KV 与 local 输入的对象对应。
后者用线程模拟 RingP2P 运行实际 transport 循环，验证 CP2/CP4/CP8 的
source 顺序、owner 归约、local 输入不被覆盖，以及 forward 仅2个、backward
仅4个新 packed buffer。它们不证明 HCCL/NPU kernel correctness。

## 服务器数值验收（原容差）

在 bridge 根目录，使用原来已验证的 MindSpeed/Megatron/NPU 环境：

```bash
export TPR_RING_COALESCE_PREFIX_FULL=1
export TPR_RING_COALESCE_PREFIX_QUERY=1
for cp in 2 4; do
  timeout 45m torchrun --standalone --nproc_per_node=$cp -m pytest -x -s -v \
    tests/models/mcore/tpr/parallel/test_ring_cp_attention_npu.py
  TPR_RUN_RING_COALESCING_TREE=1 timeout 45m \
    torchrun --standalone --nproc_per_node=$cp -m pytest -x -s -v \
    tests/models/mcore/tpr/parallel/test_ring_coalescing_tree_npu.py
done
```

算子测试还检查 saved KV tensor count = 2×segment count（不是再乘 CP），
保存 storage bytes 与 local KV 相等，forward/backward scratch storage count
分别为2/4。tree 测试覆盖 Prefix dKV、parameter gradients；随后重跑 Phase B
offload correctness，不能用 OFF/ON 相近替代 dense/reference 数值验收。

## 等 local-shape 显存对照

```bash
export TPR_RUN_OFFLOAD_B=1 TPR_OFFLOAD_B_CAPACITY=1
export TPR_QWEN_PROFILE_SIZE=1.7B
export TPR_QWEN_1_7B_PATH=/workspace/hf_models/Qwen3-1.7B
export TPR_SWAP_MODULES=self_attention,mlp
export TPR_RING_STORAGE_AUDIT=1
for spec in '2 32768 32768' '4 65536 65536'; do
  read -r cp p s <<< "$spec"
  for off in 0 1; do
    TPR_PREFIX=$p TPR_SUFFIX=$s TPR_OFFLOAD=$off timeout 60m \
      torchrun --standalone --nproc_per_node=$cp -m pytest -x -s -v \
      tests/models/mcore/tpr/profiling/test_activation_offload_phase_b_npu.py -k capacity
  done
done
```

每格独立进程；先对照 OFF 消除 reload 干扰，再看 ON。检查各 rank 输出的
`ring_storage`：forward_buffers / backward_buffers 的 storage_count=2/4，
相同 local P/S 的 storage_bytes 应一致；saved_local_kv 的 tensor_count 和
storage_bytes 不乘 CP。这里统计的是明确管理的 buffers/ctx KV，不是完整
attention 峰值，也不包含 FA workspace。结合 allocated/incremental max-rank
peak 和 NPU memory timeline 判断实际收益；reserved 单独报告。

Phase B event profiler 的 `ring_stream` 是包含 FA 的 traversal 时间，不能再
标成纯 ring_comm 或与 FA 相加。Qwen Ring profile 的 comm_forward/backward
已改为 native RingP2P launch/wait 的 exposed current-stream intervals，
不是通信流的 wire time；计数口径也从整次 traversal 变为 launch/wait。

本地无 torch/pytest/NPU，完整 NPU 数值测试及容量/性能结果待服务器运行。
本轮本地通过：4 个 coalescing 调度测试、2 个 threaded transport 测试、
22 个 activation-offload adapter 测试。额外运行的旧 `test_offload_b_gate.py`
有5个既有接口错误：仍引用现有 helper 已删除的 parameter_gate/NOISE_FLOORS；
本轮未修改 gate 或容差，未顺带修复该不相关测试。
