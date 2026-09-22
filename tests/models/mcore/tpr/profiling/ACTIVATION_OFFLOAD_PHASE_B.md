# Phase B — Full Attention / Ring activation offload

## 实现与边界

沿用 Phase A 的 MindSpeed `SwapPrefetch` / `SwapTensor`：native tensor
筛选、pinned buffer、stream、layer-level release/reload 和 Visit/Pop 生命周期。
没有新建 Ring OffloadManager，没有修改 Ring 数学、P2P schedule、online
softmax、dKV reduce/accumulate 或 padding。

- `activation_offload.py`：仅允许 CP1 或 Ring CP2/CP4，核对实际 backend
  parallel_size、model config 和 launcher args；保持 TP/PP/EP=1 和原有互斥检查。
- `segment_executor.py`：将实际 backend 连同 CP size 传入验证及卸载 scope。
- layer release 前读取**当前** KVStack 的 persistent Prefix KV storage，并与
  新导出的 KV 一起保护。Pop 已弹出的 storage 不会提前加入保护集合。
- 不枚举 transient Prefix anchors 为保护对象；它们和 Ring 保存的 Q/K/V、
  output、softmax stats 继续由 native leaf/view/grad_fn/size 筛选决定。
  若 anchor 与仍存活的 persistent KV 共用 storage，必须继承其保护。
- 不按 logical valid length 裁剪卸载张量，完整 physical tensor 仍由原有
  Ring alignment padding、shard 和 loss ownership 逻辑管理。
- Push 仍 no_grad；Pop 仍 recompute。没有实现 Phase C/D。

QKV merge 与 swap 正交。此验收入口固定以下两个开关为 1（OFF/ON 都相同），
生产启动也请显式设置；本提交不改变其他 Ring 调用方的历史默认值：

```bash
export TPR_RING_COALESCE_PREFIX_FULL=1
export TPR_RING_COALESCE_PREFIX_QUERY=1
```

## 服务器 correctness

在 bridge 根目录执行，沿用 Phase A 已验证的 MindSpeed/Megatron/NPU 环境。
模型/spec helpers 在 MindSpeed repatch 后才导入，兼容 Core-only Megatron
缺少 `megatron.training` 的启动方式。不需要安装 Transformer Engine。

```bash
export TPR_RUN_OFFLOAD_B=1
export TPR_QWEN_PROFILE_SIZE=1.7B
export TPR_QWEN_1_7B_PATH=/workspace/hf_models/Qwen3-1.7B
export TPR_SWAP_MODULES=self_attention,mlp
TEST=tests/models/mcore/tpr/profiling/test_activation_offload_phase_b_npu.py
for cp in 2 4; do
  timeout 45m torchrun --standalone --nproc_per_node=$cp -m pytest -x -s -v \
    "$TEST" -k correctness
done
```

每个 CP 覆盖 divisible、non-divisible、non-divisible + sparse loss；均为
两个 sibling。sparse 只保留 query 0 的 loss，确保部分 rank 无 local loss。
`TPR_OFFLOAD_B_SHARDS` 输出各段 logical length、physical local length、local
loss 数，检查 padding 和 loss ownership 确实命中。默认 P/S=2048，padding
case 为 P2047/S2045，可用 `TPR_OFFLOAD_B_CHECK_LENGTH` 调整。

依次跑 OFF、OFF repeat、ON、ON repeat、OFF after；对比 loss、本 rank 的
全部 owned logprobs、CP finalize 后所有参数梯度、Pop 前本 rank Prefix dKV。
沿用 Phase A rtol=2e-3/atol=2e-4，不通过放宽容差掩盖 OFF repeat 的波动。
各 rank 完成通信后统一汇总失败；CPU 保留 reference gradients，会需要额外
主机内存。Push 必须零 transfer；每次 ON Visit/Pop 必须 release/H2D > 0，
并检查 native storage 已释放及 host buffer pinned。零命中是失败而不是通过。

## 容量与性能矩阵

真实 Qwen3-1.7B、N=2、native fused CE，OFF/ON 每格使用新进程。仅运行
capacity 测试，不夹带 correctness 的 CPU gradient/logprob 快照。

```bash
export TPR_OFFLOAD_B_CAPACITY=1
export TPR_OFFLOAD_B_WARMUP=1
export TPR_OFFLOAD_B_REPEATS=3
mkdir -p /tmp/tpr-offload-b
for spec in '2 16384 16384' '2 32768 32768' \
            '4 32768 32768' '4 32768 65536' \
            '4 65536 32768' '4 65536 65536'; do
  read -r cp p s <<< "$spec"
  for off in 0 1; do
    TPR_PREFIX=$p TPR_SUFFIX=$s TPR_OFFLOAD=$off \
      timeout 60m torchrun --standalone --nproc_per_node=$cp -m pytest -x -s -v \
      "$TEST" -k capacity > /tmp/tpr-offload-b/cp${cp}_p${p}_s${s}_off${off}.log 2>&1
    echo "cp=$cp p=$p s=$s off=$off exit=$?"
  done
done
```

上面命令继承 correctness 章节的环境变量及 TEST。不要启用 shell `set -e`，
否则预期的 OFF OOM 会中断整个矩阵。timeout/进程异常同样视为失败，不能作为
capacity 通过。OOM 的 rank 会输出 `TPR_OFFLOAD_B_FAILURE`，记录阶段；其后
不做 all-gather，交给 torchrun 终止其他进程，保留所有 rank 的原始日志。

每次迭代输出 `TPR_OFFLOAD_B_RANK` 和 `TPR_OFFLOAD_B_MAX`：

- allocated/reserved peak、相对清梯度/empty_cache 后基线的 incremental peak；
- 端到端 latency（含 Push、两个 Visit、Pop 和参数梯度 CP finalize）；
- D2H/released/H2D bytes、分阶段/分 native layer 命中量；
- H2D launch 后采样的 allocated restore peak；它是采样值，不是严格恢复
  窗口最大值，也不能单凭此值归因 Ring backward；
- Linux process peak RSS（含模型加载历史峰值，不是 pinned buffer 专用计数）；
- 每个指标分别取 max-rank，不能只报告 rank 0。

先用默认不加事件的运行比较 latency/容量。需要 Ring/FA/merge breakdown 时，
用相同矩阵另跑 `TPR_OFFLOAD_B_TIMING=1`。其 current-stream event intervals
覆盖 Ring circulation/reduction、FA forward/backward、online-softmax merge，
并非互斥分解，也不是 HCCL 专用 stream 的净执行时间，不能直接相加成端到端时间。
确认瓶颈须结合 NPU profiler trace；D2H/H2D transfer time、exposed wait
目前输出 null，不为统计改造 native runtime。详细计数仅存在于测试探针。

## 验证状态与下一步

本地适配器 unit tests 通过，NPU 测试仅完成静态检查；CP2/CP4 correctness、
显存节省、P/S 容量矩阵、无 HCCL hang/OOB/stale storage **待服务器验收**。
不能将 Phase A CP1 P16K/S16K 的既有结果作为 Phase B 通过证据。

已知风险：Ring backward 取 `ctx.saved_tensors` 可能集中触发 layer restore；
合并的 Q/K/V 或 view 可能被 native 筛选排除；常驻 Prefix KV/dKV 和 Ring
临时 buffer 不会因 activation swap 消失；多 rank CPU pinned memory 压力
可能成为容量约束。先以命中记录、max-rank peak 和 profiler 定位。

只有 correctness 全通过且 trace 明确显示集中 H2D restore 是峰值/性能瓶颈，
才评估 source-wise restore / source-ahead prefetch / double buffering；本版不实现。
