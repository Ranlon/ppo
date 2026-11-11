# Qwen3-0.6B 全量 SFT/PPO 对比入口

入口为 `run_full.py`，实现隔离在 `full_runtime/`，不修改原来的4B LoRA入口或已有运行目录。
使用原 `llm200:torch2.6-cu124` 环境。完整模型权重已位于外部 `models/Qwen3-0.6B-Base`。
默认运行结果、日志和缓存均在本项目 `outputs/` 内；不会自动启动训练。
多进程通信临时目录使用项目内短路径 `.tmp/`，避免长运行目录触发 `AF_UNIX path too long`。

| 阶段 | baseline | optimized |
|---|---|---|
| SFT | 全量，batch1，累积8，1轮/935次更新 | 全量，ODB预算1024、窗口64、累积1，按样本覆盖完成1轮 |
| SFT检查点 | 全层梯度检查点 | 同样保留全层梯度检查点 |
| SFT优化器 | FP32可训练权重，AdamW，BF16 autocast，lr=2e-5 | 相同；学习率按已消费样本推进 |
| PPO初始化 | 本版完成的完整SFT模型 | 本版完成的完整SFT模型 |
| PPO reference | 独立冻结的完整SFT模型 | 同样独立冻结 |
| PPO batch/更新 | batch3，每批4次，最多2491批 | 相同 |
| PPO学习率/算法 | AdamW lr=2e-7，全量保护控制器，原K3=50硬保护 | 相同 |
| PPO生成 | native SDPA/dynamic cache | StaticCache768、Inductor/CUDA Graph、staged RMSNorm |
| PPO训练 | 原生head，完整梯度检查点 | 分块FP32概率/反向路径，完整梯度检查点 |
| 显式GC | 16次/批 | 1次/批 |
| LoRA合并/备份 | 无LoRA | 无LoRA，不创建旧4B合并备份 |

这比较两套完整配置，包含ODB引起的组批、更新次数和学习率进度差异，不是保持优化轨迹不变的单项消融。
PPO actor与reference均保留已保存的FP32参数并使用BF16 autocast；显式恢复加载中可能发生的BF16舍入。
冻结reference本身约2.22GiB，比只存BF16 reference多约1.11GiB，以保证初始SFT权重一致。
PPO的adaptive KL、连续零奖励、自然EOS及非有限值保护没有放宽。0.6B模型可能触发质量保护，不能保证跑满。

## 在容器内由用户启动

```bash
cd /hpc_stor01/home/mengqi.li_sx/Project200/ppo
bash tools/run_full_compare.sh
```

脚本选择**可见且支持BF16的GPU中剩余显存最多的一张**，默认至少24GiB空闲；使用UUID避免物理序号重排。
它尊重原有 `CUDA_VISIBLE_DEVICES`，不会扩大容器或平台分配的GPU范围，也不申请独占资源。
选择结果保存在 `GPU_UUID.txt`，baseline和optimized顺序使用同一卡。

按此前提供的显存快照，4号卡空闲最多。可指定：

```bash
GPU=4 bash tools/run_full_compare.sh
```

如果当前shell的 `CUDA_VISIBLE_DEVICES` 排除了4号卡，入口会报错；应先在平台分配允许的范围内设置可见GPU。
显存空闲不等于计算独占；GPU运行前快照保存到 `gpu_before.csv`。

默认顺序：baseline prepare → baseline完整SFT → baseline完整PPO → optimized prepare → ODB完整SFT → optimized完整PPO → 同前缀性能表。
若PPO正常触发并记录质量保护，保留该停止点并继续另一版；其它错误立即停止。
不自动跑32题质量门槛或1319题评估，便于只比较训练性能。

```bash
# 可以先限制双方的PPO工作量；SFT仍然完整覆盖7473条样本。
PPO_BATCHES=800 GPU=4 bash tools/run_full_compare.sh

# 默认2491批；修改预算或学习率时，双方使用同一组公共学习率。
TOKEN_BUDGET=1024 SFT_LR=0.00002 PPO_LR=0.000001 bash tools/run_full_compare.sh
```

全量PPO默认使用更保守的`PPO_LR=2e-7`、初始KL系数0.5、horizon=500、old/new policy KL early stopping；K3=50仍是硬停止条件。
如需复用已经完成的基线SFT，指定完整模型目录（不是adapter）：

```bash
BASELINE_SFT=/hpc_stor01/home/mengqi.li_sx/ppo/outputs/full_06b_20261109_014924/baseline/sft/model \
CUDA_VISIBLE_DEVICES=4 GPU=4 \
  bash tools/run_full_compare.sh
```

新运行目录默认 `outputs/full_06b_YYYYMMDD_HHMMSS/`。
`COMPARE_DIR` 可指定新目录，已有结果一律拒绝覆盖。

启动器故障修复后，可以继续一个已验证的对比目录，复用完成的SFT：

```bash
CUDA_VISIBLE_DEVICES=4 GPU=4 CONTINUE=1 \
  COMPARE_DIR="$PWD/outputs/full_06b_20261109_014924" \
  bash tools/run_full_compare.sh
```

继续模式会核验运行目录，跳过成功完成的SFT和已有完成/保护停止的PPO；未开始的阶段正常启动。
它必须使用原`GPU_UUID.txt`中的同一GPU，且不会替换已有的半程PPO。半程PPO需要后述的显式完整状态恢复。
若源码改变，仍要求指纹一致；不会自动覆盖指纹或忽略校验。2026-11-09这次修复仅涉及全量启动器和顺序脚本，
已完成SFT的运行目录单独记录`LAUNCHER_FIX.json`，包括变更前后指纹与保留的SFT模型身份。

## 单阶段运行与恢复

```bash
python run_full.py prepare --profile baseline --run-dir "$PWD/outputs/full06_baseline" \
  --model /hpc_stor01/home/mengqi.li_sx/models/Qwen3-0.6B-Base \
  --train /hpc_stor01/home/mengqi.li_sx/datasets/gsm8k/train.jsonl \
  --test /hpc_stor01/home/mengqi.li_sx/datasets/gsm8k/test.jsonl
python run_full.py sft --run-dir "$PWD/outputs/full06_baseline"
python run_full.py ppo --run-dir "$PWD/outputs/full06_baseline" --name full
```

改进版使用同样的三条命令，将profile改为`optimized`并换新运行目录。
两版默认各自完成SFT，PPO不会复用另一版SFT。
若只测PPO组件，可以通过`ppo --sft /path/to/completed/full/model`显式使用同一SFT模型；这需标明另一实验口径。

完整PPO状态支持恢复到同一运行目录中的新名字（只加载自己生成的可信checkpoint）：

```bash
python run_full.py ppo --run-dir /path/to/run --name resumed \
  --resume-state /path/to/run/runs/full/state_checkpoints/step_000250/training_state.pt
```

必须保持同一源码、配置、SFT身份、strict设置。质量保护状态也会恢复；不能靠重启清除已触发的保护。
ODB SFT恢复尚未实现。SFT smoke使用`--sft-limit 2 --sft-name smoke`，不会把短跑模型自动绑定为正式SFT。
全量checkpoint默认第1批、每250批和结束时保存，Adam完整状态体积远大于LoRA；两版全程保留结果可能需要约200–250GiB磁盘。

## 结果与计时口径

- `baseline/sft/SUCCESS.json`、`optimized/sft/SUCCESS.json`：完整样本/token覆盖、实际更新数、吞吐和峰值显存。
- `*/runs/full/batches.jsonl`：生成、PPO step、GC、实际输入与回复token及每批4次更新计数。
- `*/runs/full/PROGRESS.json`、`SUCCESS.json`或`QUALITY_STOPPED.json`：状态、完整完成或保护停止。
- `comparison.md`和`comparison.json`：完整SFT＋前min(800,双方完成批数)的吞吐、时间和相对倍数。

脚本结束会自动生成表格，也可手动重算：

```bash
python tools/compare_full.py --baseline /path/to/pair/baseline \
  --optimized /path/to/pair/optimized --batches 800
```

SFT用Trainer `train_runtime`，PPO用共同前缀的`wall_seconds`之和，包含冷编译；分子实际token只计一次，不乘PPO epochs。
该合计不包含PPO批外初始化和独立完整状态写出；完整PPO程序耗时、保存计时另存在日志中。
ODB改变SFT的有效batch和更新次数，两个独立SFT的权重/回复可能不同，报告中会保留这些差异。

需要额外测试集评估时，用完整模型入口：

```bash
python run_full.py evaluate --run-dir /path/to/run \
  --model /path/to/run/runs/full/model --label PPO
```

## 验证范围

2026-11-09：在llm200中19项CPU测试全部通过，包括9项新增全量入口测试、3项ODB测试及7项原LoRA回归测试。
本地真实0.6B模型和7473/1319条数据的prepare/哈希检查通过，最大PPO提示220token；未训练真实0.6B模型。
提供`tests/test_full_tuning.py`的CPU小Qwen3集成测试，覆盖全参数梯度、独立reference、原生及改进PPO四次更新、完整模型读写、Adam完整状态恢复与下一次更新、norm异常恢复和GPU选择。
同时复用`tests/test_odb_sft.py`检查ODB样本覆盖、监督token归一化和样本进度scheduler。
这些测试不等于真实0.6B GPU显存、CUDA编译更新可见性或完整训练质量验收；正式GPU运行由用户执行。
