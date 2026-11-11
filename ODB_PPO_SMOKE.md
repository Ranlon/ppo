# 已完成 ODB SFT 的 PPO 成对短测

本入口面向 `obd_sft` 分支已有的 **4B LoRA SFT**，不是当前0.6B全量实验。
不重训SFT，默认使用同一次既有SFT对比的两个完成模型：

| 起点 | 目录 | SFT更新数 |
|---|---|---:|
| 普通SFT | `outputs/sft_odb_compare_20261105_175703/baseline/sft/model` | 935 |
| ODB SFT | `outputs/sft_odb_compare_20261105_175703/odb/sft/model` | 1787 |

双方均使用**同一native baseline PPO**，batch3、每批4次更新、seed42、同一题目顺序、数值奖励及原质量保护。
这是验证“ODB产出的SFT能否正常衔接PPO，以及短程数值/质量/速度表现”，不会同时打开编译等优化来混淆起点影响。
actor和冻结reference分别从各自SFT初始化，新Adam和新value head。

## 在llm200内由用户运行

```bash
cd /hpc_stor01/home/mengqi.li_sx/Project200/ppo
bash tools/run_odb_ppo_smoke.sh
```

默认顺序：普通SFT前测32题 → native PPO20批 → 后测同32题 → ODB SFT前测同32题 → native PPO20批 → 后测同32题 → 汇总表。
即每个起点60条PPO训练样本、80次优化器更新；所有质量探针避开这次PPO使用的前缀，但它们仍属于SFT训练集。
探针为native greedy、batch1、max512，固定题目全部执行，不因EOS差提前终止。测试集不参与短测或选模型。

GPU在可见、BF16、至少24GiB空闲且利用率<=10%的卡中选择显存最多的一张，两个起点顺序使用同一卡。
当前正在运行0.6B实验时，应让短测使用另一张低负载卡，避免影响双方速度。可显式指定：

```bash
GPU=1 bash tools/run_odb_ppo_smoke.sh
# 或 --gpu 1；该卡必须在当前 CUDA_VISIBLE_DEVICES 范围内。
```

更多批数和探针：

```bash
bash tools/run_odb_ppo_smoke.sh --batches 100 --probe-samples 64
```

**默认20批仍完全位于原75批warmup内。** 不缩短scheduler来人为加大短跑更新；20批通过只能排除部分明显接入/数值问题，不能证明长期无质量损失。
更大的探针仍不是独立测试集结论，32题每错一题对应3.125个百分点。

## 输出和隔离

结果均位于项目 `outputs/odb_ppo_smoke_时间戳/`，已有目录拒绝覆盖。支持 `--output` 指定项目内新目录。

- `RESULT.md` / `RESULT.json`：前后数值正确率、EOS、回复长度；PPO共同前缀的吞吐、时间、KL、奖励、loss、显存及保护停止。
- `PROBE_PLAN.json`：双方共用探针题号及排除的PPO前缀长度。
- `EXPERIMENT.json`：SFT来源SHA、GPU、公共配置和实验边界。
- `baseline/`、`odb/`：独立prepare目录、PPO日志/权重/完整状态和前后探针回答。
- `source/`：运行时源码快照；仅在快照中修复已有科学计数法类型和Unicode JSONL兼容问题，算法、PPO保护保持不变。

不修改原SFT权重，不修改原运行目录，不修改当前工作目录的runtime或configs，因此不影响正在运行的0.6B源码指纹。
临时文件使用项目 `.tmp/` 短路径，避免AF_UNIX路径长度问题。普通进程错误会停止；正式质量保护停止会保留并汇总，不绕过阈值。

只准备（CPU，不做任何GPU生成或PPO）：

```bash
bash tools/run_odb_ppo_smoke.sh --prepare-only
```

单独重算已有结果：

```bash
python tools/odb_ppo_summary.py --experiment /path/to/odb_ppo_smoke_run
```

解释结果时先看ODB SFT在PPO之前是否已经与普通SFT有差异，再比较两者各自前后变化。
PPO loss包含value/策略及KL形状奖励影响，不能以loss更小直接判定模型更好；K3也分别相对各自SFT reference计算。
