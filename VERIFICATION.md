# 打包验证记录（2026-11-04）

本文件记录源码包验证范围，不包含历史训练日志或数据内容。

v2 按用户要求仅保留两套方案，移除额外未合并对照及其专用代码，已重新通过全部7项CPU测试和compileall，并核验两份完整配置快照与实际入口相等。依赖、数据预处理及其先前实测核验沿用相同实现；本次没有启动GPU训练。

## 已执行

1. 在独立目录、`CUDA_VISIBLE_DEVICES=""` 下运行 7 项 CPU 测试，全部通过：配置/数据契约、mask、BF16/FP64 分块概率及反向、完整 head/value 分支和所有 LoRA 梯度、正常/异常 merge 恢复及共享头参数身份、最新 LoRA 读取、完整 checkpoint/RNG/下一次 Adam 更新一致性。
2. 使用现有外置 Qwen3-4B-Base tokenizer 和 GSM8K 训练集完成 prepare；7473 条提示 SHA 与 vendored LF 实际预处理逐项相等，最大提示 220 tokens。测试集 1319 条只在 prepare 中做结构/哈希和分离校验，没有用于挑选模型。
3. 全部项目及 vendored Python 通过 compileall；仅提供 baseline、optimized 两个 profile，可解析/展开。
4. 新建隔离 venv，在其中安装 NumPy1.26.4、TRL0.9.6 与随包本地 LF 元数据，其余包使用同机现有依赖。`pip check` 输出无破损依赖；完整 requirements 的 pip dry-run通过。没有修改原训练环境。
5. 导出并锁定实际 TRL0.9.6 算法参数，运行时逐项断言。
6. 最终交付包另做逐文件 SHA 清单和归档内容核验，排除权重、数据行、训练日志、checkpoint 和编译缓存。

## 仍需目标显卡执行

- 一台全新机器的干净 venv wheels 下载/安装；当前验证环境使用了一部分已有共享包。
- README 中真实 4B 的 `validate_cuda.py` 64/512-token 两档、最长3题×512回复×PPO4压力测试、实际中断/完整恢复对照、200批质量短跑。
- 大显存卡上 full merge + Inductor 的完整 SFT935/PPO2491/最终1319测试；SFT关闭检查点全程的容量与质量。
- 同一卡相同 SFT 下三次交错固定工作量性能比较；不能由旧24GiB结果推定本组合的大卡吞吐或最终正确率。

所有上述未执行项目均未填写虚构的通过状态。当前服务器有其他用户GPU负载，本轮只做CPU准备/测试，没有启动GPU训练或停止其他用户进程。
