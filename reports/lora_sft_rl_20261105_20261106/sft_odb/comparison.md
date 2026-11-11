# LoRA SFT：基线与 ODB 对比

来源：[COMPARISON.json](COMPARISON.json)、[实验配置](EXPERIMENT.json)。实验日期：2026-11-05。

| 指标 | 基线 | ODB |
|---|---:|---:|
| 完整样本数 | 7473 | 7473 |
| 输入 token | 1430635 | 1430635 |
| 有监督 token | 925107 | 925107 |
| 优化器更新次数 | 935 | 1787 |
| Trainer 训练耗时（秒） | 2581.3235 | 697.6124 |
| 输入 token/s | 554.23 | 2050.76 |
| 最终 train_loss | 0.24518649 | 0.24151503 |
| 峰值 allocated GiB | 8.817 | 9.718 |
| 数据完整完成 | 是 | 是 |

按 Trainer train_runtime 计算，加速 3.700 倍，耗时减少 72.97%。该耗时包含训练循环内 checkpoint 开销，排除模型加载与最终导出。

两组的有效 batch、更新次数和损失归一化不同；单次共享 GPU 实验的速度及训练 loss 不构成相同泛化质量的证明，未包含完整的独立测试集质量评估。

- [基线最终结果](baseline/SUCCESS.json) · [Trainer 指标](baseline/trainer/train_results.json) · [Loss 曲线](baseline/trainer/training_loss.png)
- [ODB 最终结果](odb/SUCCESS.json) · [Trainer 指标](odb/trainer/train_results.json) · [Loss 曲线](odb/trainer/training_loss.png)
