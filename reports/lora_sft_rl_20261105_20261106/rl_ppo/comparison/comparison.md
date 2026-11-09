# A800 SFT/PPO 性能对比（2026-11-06）

Baseline PPO 完整完成 2491 batch / 9964 次更新；optimized 在第 862 batch 触发 K3 KL > 50 保护。没有将两者完整运行总时间直接相除。

## SFT（同样 7473 样本、935 更新、1430635 输入 token）

| 指标 | baseline | optimized |
|---|---:|---:|
| Trainer train_runtime 秒 | 2560.17 | 1915.91 |
| 输入 token/s（输入 token / train_runtime） | 558.80 | 746.71 |
| 训练程序总秒数 | 2580.05 | 1930.37 |
| 峰值 allocated GiB | 8.82 | 12.25 |
| 梯度检查点 | all | none |

SFT train_runtime 加速 1.336 倍，耗时减少 25.2%。两者 adapter SHA 不同。

## PPO 主比较窗口：batch 101–800（700 batch / 2100 个相同题目 / 2800 次更新）

| 指标 | baseline | optimized |
|---|---:|---:|
| 平均秒/batch | 21.320 | 6.006 |
| 输入+回复 token/s | 28.05 | 96.76 |
| 回复 token/s | 18.56 | 63.10 |
| generation 秒/batch | 13.903 | 2.837 |
| generation 回复 token/s | 28.47 | 133.59 |
| PPO step 秒/batch（每批4次更新） | 7.350 | 2.669 |
| Python GC 秒/batch（已包含在阶段计时中） | 2.911 | 0.429 |
| 平均回复长度 token/样本 | 131.93 | 126.33 |
| 峰值 allocated GiB | 19.04 | 25.44 |
| 训练采样数值正确率（非测试集） | 76.19% | 75.52% |
| 自然EOS比例 | 98.57% | 98.95% |

秒/batch 加速 3.550 倍；输入+回复 token/s 提升 3.450 倍；回复 token/s 提升 3.399 倍。

101–800 是统一排除前100批的窗口；不表示100批都在编译。另提供前1–800和1–862窗口，避免只报选定窗口。

## 初始化与保存

optimized 编译/生成 helper 初始化计时 194.88 秒，首批 wall 为 206.51 秒。该初始化是运行时计时字段，并非纯编译器CPU时间。

全程完整状态 checkpoint 写出累计 baseline 11.69 秒 / optimized 7.72 秒，保存次数不同，不能直接比较速度。此字段不包含所有 Hugging Face adapter/绘图等保存开销；这些开销没有完整独立分项计时。

Baseline 全程程序耗时 48631.10 秒；optimized 停止前 PROGRESS 耗时 5595.18 秒。二者工作量不同。

## 结论边界

- Independent SFT adapters; generated responses and lengths differ.
- One run per variant; no concurrent GPU load, clocks, or exclusive allocation history was recorded in the analyzed logs.
- Batch-wall timing includes measured loop work, but excludes separate complete-state checkpoint serialization and initialization outside the batch; generation/ppo/GC fields overlap and must not be added together.
- Optimized stopped at batch 862 with K3 KL 51.9216; this alone does not prove global quality collapse or identify its cause.
- No completed held-out GSM8K evaluation found.
- This baseline is the project baseline profile with shared K3/reward/runtime modifications, not unmodified upstream TRL.

数据来源：comparison.json 中 paths 指向原始逐批日志；完整指标见 comparison.csv。所有吞吐均为总 token / 总秒数，没有平均逐批 token/s；token 不乘 PPO epochs=4。
