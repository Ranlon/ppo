# 来源与修改

随包 LLaMA-Factory 源自已冻结的 0.9.4 项目快照，保留上游 Apache-2.0 版权声明与 LICENSE。本项目为单卡 Qwen3-4B SFT/legacy PPO 提供兼容分支；包版本为 `0.9.4+ppo4.portable1`，并非未经修改的上游发行包。

在既有审计实现基础上整理了路径/命令行、外部输入准备、配置及 checkpoint 完整恢复；保留选定的 GC、编译、RMSNorm及训练head方案。加入面向大显存卡的全量临时合并/精确恢复实现、独立共享head处理和GPU/CPU备份选择。修改后的编译入口位于 `vendor/llamafactory/src/llamafactory/train/ppo/rollout_acceleration.py`，调用 `runtime/portable_rollout.py`。

原始复制来源的文件SHA在 `provenance/copied_source_hashes.json`，最终项目文件SHA以 `MANIFEST.sha256` 为准。未随包复制模型、tokenizer权重/词表、数据集或历史训练产物；这些第三方资产的许可由使用者取得时单独遵守。
