# 本次 A800 + Qwen3-4B 改进归因与异常分析

分析范围：baseline bench 与 optimized full；性能主窗口为前800 batch，含首次编译。没有启动任何训练或修改训练配置。

## 可以量化的阶段差异

|阶段（秒/batch）|baseline|optimized|每批净节省|占总净节省|
|---|---:|---:|---:|---:|
|生成|13.6270|3.0982|10.5288|72.66%|
|PPO step：old/ref forward + 四次更新等|7.0900|2.7008|4.3892|30.29%|
|其余循环工作|0.0741|0.5013|-0.4272|-2.95%|

合计 20.7911 → 6.3003 秒/batch；前800批净节省 11592.65 秒。

GC 实测 2.8840 → 0.4334 秒/batch，减少 2.4505 秒，约相当于净节省的 16.91%。GC计时属于阶段内部，不能与上表相加；这也不是独立GC消融结果。empty_cache 次数均11次，时间 0.9536 → 0.2003 秒，不能称为减少empty_cache调用的收益。

## 每项改动能说什么

- SFT 的profile主要差异为关闭层梯度检查点：相同1430635输入token，train_runtime 2560.1699 → 1915.9082秒，吞吐+33.63%，耗时-25.16%，显存8.817 → 12.247GiB。两次独立运行，环境负载影响无法完全排除。
- PPO生成阶段联合使用全量LoRA临时合并、StaticCache、Inductor/CUDA Graph、staged RMSNorm：生成13.627 → 3.098秒/batch；实际回复token/生成秒数28.735 → 121.755（4.237倍），含初始化。没有各项独立消融，不能拆分其单项因果贡献或相乘。
- PPO head概率路径为全形状BF16 head GEMM加按32token FP32概率/反向分块，跳过纯非响应块；旧路径也有32token logprob/entropy保护。不是把全部head GEMM改成仅算response，也不是PPO关闭梯度检查点。
- PPO更新路径整体7.090 → 2.701秒/batch，但其中包含GC变化、head变化和变长输出影响。无法把全部收益归给head。
- 合并及精确恢复是新增成本：前800批累计合并60.48秒、恢复核验34.42秒，含一次额外warmup；它们已经包含在生成阶段，不能再次加到总耗时。备份+独立head额外7.492GiB。

## 异常与边界

- optimized第一批206.51秒；helper初始化194.88秒，首批generation202.77秒。第二批25.86秒（generation22.68秒），不能仅凭逐批日志确定第二批额外延迟是新编译、autotune还是环境干扰。
- baseline前800批median19.54秒、P95为35.11秒，max49.79秒（batch497，回复142/126/512）；许多慢批有长回复，不能当作脏数据删掉。optimized P95为9.01秒，除前两批的慢批也多包含512token长回复。
- batch862同题：baseline K3=2.118、value_loss=0.549；optimized K3=51.922、value_loss=5.527，触发>50保护。全程baseline最大K3=42.991（batch2440），也有稀有尖峰。单批越界不证明全局崩溃，但optimized没有完成完整训练稳定性验证。
- K3=x指数形式：expm1(logref-logactor)-(logref-logactor)，大正差有重尾；序列求和再batch平均的标量，不是逐token均值。只有batch级记录，没有逐token、逐样本K3，无法认定哪条回复/哪个token主导尖峰；它在PPO step前向计算，当前记录在四次更新结束后检查停止，因此不能说本批四次更新导致了本批这个KL值。
- optimized与baseline初始SFT SHA不同，从第一批回复就不同。前800批总回复313262 vs301772（-3.67%），平均最大回复188.18 vs178.76（-5.00%），存在变长工作量差异。
- 前800批有效response token / (3×每批最长response) 为69.36% vs70.34%；仅表示response decode槽位效率，不能直接换算为整个训练30%的浪费。
- 逐批记录未发现非有限数；OOM和allocator retries均0；optimized exact_restore记录全部通过。精确恢复不是合并后forward完全数值等价，不能因此排除BF16合并影响。
- PPO吞吐分母包括生成、old/reference、四次更新、GC、同步与批内保存等；分子只统计一次输入与有效回复token，不能乘PPO epochs=4。generation_response_tps才是纯生成阶段的有效回复token速率（仍包括合并、恢复等生成路径开销）。
- batch size=3，两版PPO均全36层梯度检查点；151936词表、hidden2560、intermediate9728、BF16基座且无量化。小batch decode、全词表概率及CPU调用开销都可能限制绝对token/s。没有CUDA profiler、GPU时钟/带宽、CPU quota或独占记录，无法把低利用率归因为模型或GPU某一因素。

该分析只依据本次日志和源码，没有复用历史4090结果。各改动的独立收益需要同一SFT、固定prompt/回复、逐项开关和重复测试才能确定。
