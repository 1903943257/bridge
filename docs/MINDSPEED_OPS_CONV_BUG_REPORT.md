# MindSpeed-Ops stateful causal_conv1d backward 单算子故障报告

## 问题摘要

Ascend910B2C / arch32 上，直接调用 MindSpeed-Ops 的公开 `causal_conv1d`
API，在 BF16、SiLU、bias、请求但不使用 final state 的配置下：

| Case | initial_state | 用户服务器复现结果 |
|---|---|---|
| A | None | PASS |
| B | 非 None，requires_grad=False | backward FAIL |
| C | 非 None，requires_grad=True | backward FAIL |

错误为 `507035`、`VEC instruction error: the ub address out of bounds`。
Case C 完整日志确认 forward 成功，实际进入 arch32 backward。
详细边界诊断中，forward、状态更新及 backward 内前向重算均完成同步，
随后在 backward kernel 执行阶段出现异常。

以上是用户服务器结果；准备文档的 Windows 环境没有 PyTorch/NPU，未另行实测。
报告定位到 stateful Conv backward 路径，尚未确定具体故障指令或最终根因。

## 最小复现

附件提供 `repro_causal_conv1d_initial_state_bwd_oob.py`，可单独放到任意目录，
使用安装了对应 MindSpeed-Ops 的 Python 环境运行。不要求安装 Bridge。
不加载模型、VERL、Megatron、TPR、GDR，不初始化 distributed/HCCL/CP/TP。
只需一张 NPU。脚本使用原生公开 API 和 autograd，没有替换 kernel。
Python backward launcher 的临时包装仅记录参数及实际调用线程的局部变量。

```bash
python -u repro_causal_conv1d_initial_state_bwd_oob.py --case none --length 64 --activation silu --bias --final unused > case_a.log 2>&1
python -u repro_causal_conv1d_initial_state_bwd_oob.py --case state_no_grad --length 64 --activation silu --bias --final unused > case_b.log 2>&1
python -u repro_causal_conv1d_initial_state_bwd_oob.py --case state_grad --length 64 --activation silu --bias --final unused > case_c.log 2>&1
```

也可以省略 `--case`，父进程会依次启动三个独立子进程，避免失败后的设备
context 被下一 case 复用。参数必须包含上述 length/activation/bias/final；
脚本旧默认配置的 PASS 不能替代本配置的测试结果。

输入：B=1，T=64，D=3072，W=4，BF16，连续张量。

| Tensor | Shape | requires_grad |
|---|---|---|
| x | [1,64,3072] | True |
| weight | [4,3072] | True |
| bias | [3072] | True |
| initial_state（B/C） | [1,3072,4] | False / True |

`activation='silu'`，`output_final_state=True`，`residual=None`，
`cu_seqlens=None`，loss 为 `output.float().sum()`，不包含 final_state。
Case B 仍通过 x/weight/bias 触发 backward。
边界诊断观察到未使用的 final_state 对应的 dht 是非 None 的全零张量。
因此不能把 unused final_state 等同于 dht=None。

## 环境与代码身份

服务器日志记录：

```text
Device: Ascend910B2C, device index 0
is_arch35: False
Python: 3.11.15
torch: 2.9.0+cpu
torch_npu: 2.9.0
triton: 3.2.0
triton-ascend distribution: 3.2.1
MindSpeed-Ops HEAD: babee85cb00ade056b1c1627e64b10a524447025
Working tree: modified (must include diff; HEAD alone is insufficient)
API: /workspace/MindSpeed-Ops/mindspeed_ops/api/triton/convolution.py
Kernel: /workspace/MindSpeed-Ops/mindspeed_ops/arch32/triton/convolution.py
Kernel SHA256: 699531907f2b3e58c67b396cee59a6562957cb183774ddf630026cd2d1f59a6b
```

triton 模块版本与 triton-ascend 分发包版本按日志分别记录，不将其自行统一。
还需附件补充服务器实际 CANN toolkit/runtime、驱动/固件版本及 OS/架构。

## 调用链及门禁说明

```text
causal_conv1d (mindspeed_ops.api.triton.convolution)
 -> CausalConv1dFunction.forward
 -> arch32.triton.convolution.causal_conv1d_fwd_impl
 -> output.float().sum().backward()
 -> CausalConv1dFunction.backward
 -> arch32.triton.convolution.causal_conv1d_bwd_impl
 -> causal_conv1d_bwd_kernel
```

完整 Hybrid/TPR 实验同时调用 Conv 和 GDR。GDR 的 initial_state 门禁不移除时，
完整流程可能在 GDR forward 就停止，无法继续到 Conv backward。
这与直接调用 Conv 的单算子脚本不同：后者不执行 GDR，不需要移除这两个门禁。
因此提交源码 diff 时应保留并说明门禁改动，但不将其列为 Conv demo 必需步骤。

## 本地改动披露

用户确认以下是该服务器 MindSpeed-Ops 仓库的全部源码修改范围：

1. `api/triton/chunk_gated_delta_rule.py`：关闭 GDR API initial_state 门禁。
2. `arch32/triton/gdn/chunk_gated_delta_rule_fwd_h.py`：关闭 GDR launcher 门禁。
3. `arch32/triton/convolution.py`：此前尝试过 dh0 时间索引修改；用户贴出的
   内容显示相关修正已注释、活动读取仍用 `i_t * BT + i_t2`。
   必须附原始 diff 和实际文件，不能仅凭聊天中格式受损的 patch 认定与上游逐字等同。
4. `tests/unit_tests/triton/test_causal_conv1d/test_backward.py`：增加梯度诊断及测试。
   当前 standalone 脚本不导入该测试文件，不执行其中的 core-count monkeypatch。
5. 未跟踪的 stateful GDR 实验测试及本地构建 `.so`：记录其存在和构建来源；
   GDR 测试不由本 demo 执行。包初始化可能加载 `.so`，不能笼统声称它未被加载。

源码片段显示本次故障版本仍是 `if USE_INITIAL_STATE:`，没有新增的时间 tile
store guard。不要在收集故障附件之前改变 kernel，也不要把此版本称为 clean upstream。
此清单仅描述 MindSpeed-Ops，不代表独立 MindSpeed 仓库也没有其他修改。

## dh0 bounds 发现与结论边界

实际 backward locals：BT=2，NT=32，eff_NT=32，dh0.shape=[2,1,3072,4]。
分配 24,576 个 BF16 元素。当前已审计 store 使用
`dh0 + i_t * B * D * W + i_n * D * W + o_d * W + i_w`；若全部 32 个
time tile 均执行且仅有 channel mask，地址范围达到 393,216 个元素，
i_t=2..31 超过 dh0 分配。该全局 buffer 寻址缺陷是明确的待处理问题。

但设备异常报告 UB 地址越界，不能仅凭上述全局内存索引推导出其具体原因。
尚未通过仅加 store guard 的对照确认二者因果关系，也未排除其他读取越界、
编译或内部临时缓冲问题。请求团队检查该配置下的生成代码及故障指令。

## 提交附件清单

必需：

- 本文档。
- standalone Python 脚本（确认 --help 包含 --length/--activation/--bias/--final）。
- 同一环境运行的 case_a.log、case_b.log、case_c.log 完整原始输出。
- MindSpeed-Ops commit、git status、原始源码 diff、故障时实际 convolution.py。
- 环境信息：pip freeze、npu-smi info、CANN/驱动/固件版本、OS/CPU 架构。
- 对应失败时间/PID 的 Ascend plog/device 日志（保留完整原文件）。

补充附件：

- `run_stateful_ops_ab.py` 的故障完整日志，用于佐证各 kernel 同步边界。
  主 standalone demo 不依赖这套多文件诊断工具。
- 本地构建 .so 的 SHA256 和构建来源；团队要求时提供二进制或构建步骤。
- 若需展示完整模型背景，附 Stage45 命令即可，无需模型权重和训练数据。
- store-guard patch 可标为“候选修复，未确认消除该异常”另附，不能当作复现前置条件。

在服务器 MindSpeed-Ops 根目录收集源码身份（不修改源码）：

```bash
mkdir -p /workspace/conv_repro_report
git rev-parse HEAD > /workspace/conv_repro_report/mindspeed_ops_commit.txt
git status --short > /workspace/conv_repro_report/mindspeed_ops_status.txt
git diff --binary HEAD --output=/workspace/conv_repro_report/mindspeed_ops_worktree.patch
cp mindspeed_ops/arch32/triton/convolution.py /workspace/conv_repro_report/convolution.py
sha256sum mindspeed_ops/arch32/triton/convolution.py > /workspace/conv_repro_report/kernel_sha256.txt
python -m pip freeze > /workspace/conv_repro_report/pip_freeze.txt
npu-smi info > /workspace/conv_repro_report/npu_smi.txt 2>&1
uname -a > /workspace/conv_repro_report/os.txt
```

git diff 不包含未跟踪文件；如需提交未跟踪测试或构建信息需另行复制。
CANN/驱动/固件安装版本文件的位置以服务器安装布局为准，另附其原始输出。
不要从聊天文本还原 patch（符号、缩进已被富文本转换）；直接提交 git 导出的文件。

建议 issue 标题：
`[Ascend910B2C][arch32] Stateful causal_conv1d backward fails with 507035 / UB address out of bounds`

请求团队确认：上述参数的 stateful training 是否受支持；故障 kernel 指令及
dh0 store 边界问题；建议的修复与可验证的版本组合。
