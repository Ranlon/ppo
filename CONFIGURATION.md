# 完整配置方案

机器可读的唯一运行参数来源是 `configs/common.yaml` + 所选 profile + `configs/trl_contract.json`。`python run.py config --profile ...` 展开三者；运行目录再补上实际路径、SHA 和 localhost 奖励地址。无模型/数据硬编码服务器路径。

## 两版共同算法

| 项目 | SFT | PPO |
|---|---|---|
| 模型 | Qwen3-4B-Base，36层，hidden2560，未量化 | 从本版完成的 SFT adapter 初始化 |
| 数据 | GSM8K main train 7473 | 同一训练集恰好一次，7473条 |
| 测试 | 可独立测试 SFT | 最终 native test 1319题 |
| 输入模板 | qwen3_nothink；原始题目 | 相同，禁止添加格式要求 |
| 长度 | cutoff2048；不 packing；不训 prompt | prompt cutoff1024；最多新生512；StaticCache768只用于优化生成 |
| LoRA | r8，alpha16，dropout0 | 相同；253个模块，506个 FP32 factor tensor |
| LoRA 目标 | q/k/v/o_proj、gate/up/down_proj、lm_head | 相同 |
| 精度 | 冻结 BF16 基座，FP32 LoRA | BF16 基座/autocast，FP32 LoRA/value head |
| 可训练参数 | 17,751,040 | 17,753,601，含 value head 2,561 |
| batch / GA | 1 / 8 | batch3、mini3、GA1、buffer1 |
| epoch / 更新 | 1 epoch，935次 optimizer更新 | 2491个 rollout batch，每批 PPO4；共9964次 optimizer更新 |
| 优化器 | AdamW Torch | 新 AdamW Torch；新 value head |
| lr / scheduler | 1e-4 / cosine935 | 1e-6 / cosine2491，每 rollout batch 调一次 scheduler |
| warmup | ratio0.03 →29步 | ratio0.03 →75批 |
| Adam | beta1=.9、beta2=.999、eps1e-8、weight_decay0 | 相同 |
| 梯度裁剪 | max_grad_norm1 | max_grad_norm1，每次使用完整的可训练参数 tuple |
| 随机种子 | seed/data_seed42 | seed/data_seed42；固定 randperm 顺序 |
| Attention | native SDPA | native SDPA；不用 FlashAttention2、量化、vLLM |
| GPU / 并行 | 单卡 | 单卡、torchrun单进程；不共享卡进行速度对比 |
| 保存 | first、250倍数、结束；完整HF状态 | first、250倍数、结束；adapter/value及完整PPO状态 |

SFT：dataloader workers4、preprocessing workers4、不 drop_last、不 streaming、val_size0、logging_steps10。PPO：预处理 workers4，实际审计 DataLoader workers0、shuffleFalse并应用显式固定顺序，logging_steps1。所有未列出的 LF 参数完整保存在 common.yaml；TRL 关键默认值在 trl_contract.json，训练启动逐项断言。

## PPO 数学、采样与保护

- 生成：do_sample=true，temperature1，top_p1，top_k0，repetition_penalty1，max_new_tokens512，正式运行无 min_new_tokens 强制长度。PPO EOS 仅 tokenizer eos（本数据路径为151645）；最终 evaluator 沿用 native EOS 列表。
- reward：仅训练答案的 Decimal 归一数值匹配，正确1、错误/缺失0；优先最终 ####，否则最终 boxed/数值；不加载奖励模型，不给格式奖励。
- reference：本版完成 SFT 的冻结 FP32 adapter，与 actor 共享冻结基座；参考调用前恢复全部合并状态，参考的参数不进入 optimizer。
- K3：`expm1(log_ref-log_actor) - (log_ref-log_actor)`，无 ratio 截断；仅允许 -1e-6 量级负舍入并 clamp 到0。TRL 配置中的 `kl_penalty="kl"` 入口被此自定义函数替换。
- adaptive KL：初始系数0.2、target6、horizon10000；gamma1、lambda0.95；policy/value clip各0.2、value loss coef0.1；early_stoppingFalse、ratio_threshold10；不做 score scaling/norm 或 reward whitening。
- 质量保护：20批连续全零奖励；K3>50；最近20批自然EOS不足50%；非有限值、base/参考身份改变、恢复不一致等立即失败。完整状态保留。

## profile 差异

| 项目 | baseline | optimized（全改进） |
|---|---|---|
| SFT checkpoint | 全层；reentrant=True | 关闭层checkpoint，保留输入梯度hook |
| PPO checkpoint | 全36层，nonreentrant | 相同 |
| 显式 Python GC | 每批原始16次 | 去重为每批1次 |
| empty_cache | 每批11次 | 保留11次；不屏蔽编译器清理 |
| 自动 Python GC | 开启 | 开启 |
| 生成 cache | dynamic | StaticCache768，BF16 |
| 编译 | 关闭 | Inductor fullgraph=True、dynamic=False、max-autotune |
| Inductor 精度 | 不适用 | freezing=False、emulate_precision_casts=True、force_same_precision=True |
| RMSNorm | 原生 | 生成期间 staged square/native ATen mean/post；训练不 patch |
| 生成 LoRA | 原生分支 | 全253模块临时 dense merge，生成结束全量精确恢复 |
| 共享输出头 | 原生独立 LoRA delta | 独立 BF16 工作头；原 embedding不写入 |
| 合并快照 | 无 | 默认GPU；可CPU；252模块BF16快照+独立head，约额外7.492GiB |
| PPO head | 全形状原生 | 全形状BF16 GEMM保留，FP32概率/反向按32token分块；跳过纯非响应块 |
| head梯度顺序 | 原生 | 保留Qwen3全范围hidden view和value分支相加顺序 |

baseline 已包含原 B4 必需的 frozen SFT reference、输出头 LoRA、K3、完整裁剪参数及原生32token logprob/entropy内存保护。它不是恢复到未修正的历史错误实现。

Inductor LoRA 小矩阵调度优化只在未合并路径有意义；全合并路径去掉这些分支，不能把两者加速比相乘。全改进允许合并的 BF16 计算近似，但不放宽恢复一致性；EOS logprob 0.1 不作为新增硬门槛。

## 阶段配置与恢复

`--limit` 只停止独立短跑，不缩短正式 scheduler。SFT smoke仍用935步/warm29 scheduler；PPO pilot仍用2491批/warm75 scheduler。PPO `--stress` 仅压测使用最长三题并强制512 tokens、四次更新。各次短跑独立从 SFT 开始；正式训练不接 pilot 权重。

PPO 完整恢复要求相同源码、输入身份、SFT SHA、算法参数、runtime profile 和 strict 设置；保存 optimizer、scheduler、KL controller、running moments、scaler（如有）、TRL counters、样本顺序、RNG、quality guard、actor/value 参数。重启使用新名字，禁止覆盖原结果。SFT使用原HF完整checkpoint恢复。

最终测试固定 native/greedy/batch1/max512；数值正确率为主要任务指标，####格式正确率另报。两版独立 fresh SFT 必须披露不同 SHA；共同组件实验可显式 `--sft` 指向同一完成 SFT。
