# SFT与RL各自改进对比在LoRA上运行结果

归档 2026-11-05 至 2026-11-06 的 Qwen3-4B-Base / GSM8K / LoRA 历史实验结果。

## SFT：基线与 ODB

[对比报告](sft_odb/comparison.md) · [完整对比指标](sft_odb/COMPARISON.json) · [实验参数](sft_odb/EXPERIMENT.json)

两组均完成 7473 条样本、1430635 输入 token。基线 train_runtime 2581.3235 秒，ODB 为 697.6124 秒，约 3.700 倍加速。两组更新次数和损失归一化不同，不能据此断言泛化质量相同。

| 结果 | 基线 | ODB |
|---|---|---|
| 最终完成状态与指标 | [SUCCESS.json](sft_odb/baseline/SUCCESS.json) | [SUCCESS.json](sft_odb/odb/SUCCESS.json) |
| Trainer 最终指标 | [train_results.json](sft_odb/baseline/trainer/train_results.json) | [train_results.json](sft_odb/odb/trainer/train_results.json) |
| Loss 曲线 | [training_loss.png](sft_odb/baseline/trainer/training_loss.png) | [training_loss.png](sft_odb/odb/trainer/training_loss.png) |

## RL：基线 PPO 与优化版 PPO

[性能对比](rl_ppo/comparison/comparison.md) · [CSV 指标](rl_ppo/comparison/comparison.csv) · [完整 JSON 指标](rl_ppo/comparison/comparison.json) · [改进归因与异常分析](rl_ppo/comparison/attribution_and_anomalies.md)

基线完成 2491 batch / 9964 次更新；优化版在第 862 batch 因 K3=51.9216 > 50 触发保护停止。优化版归档为停止时的最终可用结果，未标为完整训练成功。

主比较使用共同的 batch 101–800：基线 21.320 秒/batch，优化版 6.006 秒/batch，约 3.550 倍加速。完整报告还保留 1–800、1–862 窗口，不能将不同工作量的完整运行耗时直接相除。训练奖励不等于独立测试集准确率，原实验没有完成的 held-out GSM8K 评估。

| 结果 | 基线 PPO | 优化版 PPO |
|---|---|---|
| 最终状态 | [SUCCESS.json](rl_ppo/baseline/SUCCESS.json) | [QUALITY_STOPPED.json](rl_ppo/optimized/QUALITY_STOPPED.json) |
| 最终进度 | [PROGRESS.json](rl_ppo/baseline/PROGRESS.json) | [PROGRESS.json](rl_ppo/optimized/PROGRESS.json) |
| Loss 曲线 | [training_loss.png](rl_ppo/baseline/trainer/training_loss.png) | [training_loss.png](rl_ppo/optimized/trainer/training_loss.png) |
| Reward 曲线 | [training_reward.png](rl_ppo/baseline/trainer/training_reward.png) | [training_reward.png](rl_ppo/optimized/trainer/training_reward.png) |

RL 原对比报告中的 SFT 耗时 2560.1699 → 1915.9082 秒来自另一组 SFT 配置实验（包括梯度检查点设置差异），与上方 ODB 实验不同。其原始最终指标保存在 [sft_initialization](rl_ppo/sft_initialization/)，用于核对 RL 对比报告的来源。

## 文件来源

[来源与 SHA-256 清单](MANIFEST.json) 逐项记录复制文件的原始路径、大小和校验值。JSON 和原始分析报告保持原样，其中的本机绝对路径仅用于溯源；浏览本次归档请使用本文的相对链接。

本归档收录最终状态、Trainer 指标/历史、训练曲线、关键运行配置、SFT GPU 监控与既有对比分析。未收录模型权重、适配器权重、优化器/完整训练状态、tokenizer、数据集、编译缓存、逐条生成内容及原始 stdout 日志。未包含 2026-11-09 正在运行的实验或 K3 修复分支结果。
