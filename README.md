# Qwen3-4B-Base：SFT + 四轮 PPO 可迁移项目（v2，仅两套方案）

本目录包含完整运行源码、冻结的 LLaMA-Factory 兼容分支、基线/全改进配置、依赖锁定表及验证工具。**不包含模型、tokenizer 文件、数据集、训练权重、历史日志或编译缓存。** 模型和原始数据由使用者放在外部目录；程序运行后才会创建输出。

仅提供两个入口配置：

| 配置 | 用途 | SFT | PPO 生成 | PPO 训练 head |
|---|---|---|---|---|
| `baseline` | 4B B4 基线 | 全层梯度检查点 | native SDPA / dynamic cache | 原生全形状 |
| `optimized` | 本次要求的大显存全改进候选 | 关闭 SFT 层检查点 | **全量 LoRA 临时合并 + 精确恢复**、独立输出头、StaticCache、Inductor、staged RMSNorm | 全形状 BF16 GEMM + 32-token FP32 概率分块 |

`optimized` 包含当前选定的优化，以及按要求加入、供大显存显卡测试的完整临时合并。生成阶段合并全部253个LoRA模块（含输出头），训练及冻结SFT参考使用恢复后的未合并权重。GPU备份和CPU备份仅是这一个方案的内存位置选项；没有第三套训练方案。

**验证范围：**打包前通过 CPU 小模型恢复/梯度/checkpoint 测试、实际 7473 条训练提示预处理一致性检查和隔离环境依赖检查。大显存卡上的“全量合并 + 编译 + 全程 SFT/PPO”组合仍需执行下述 CUDA 验证及短跑；没有把它标为已经完成全程验收，也不承诺固定加速比例。SFT 关闭检查点此前仅做过短窗验证。详见 [VERIFICATION.md](VERIFICATION.md)。

## 文件与完整配置

```text
run.py                         统一命令行入口
configs/common.yaml            完整 SFT/PPO 公共参数（JSON 是合法 YAML）
configs/baseline.yaml           B4 开关
configs/optimized.yaml          大显存全改进开关
configs/trl_contract.json       TRL 0.9.6 算法默认值及运行时断言
runtime/                       SFT/PPO、参考、奖励、保存恢复、编译/合并源码
vendor/llamafactory/            随包源码及独立兼容版本元数据、Apache-2.0 许可
tools/                         数据导出、CPU 预处理核验、目标卡 CUDA 核验
tests/                         不依赖真实模型/数据的 CPU 测试
requirements*.txt              环境依赖与构建工具
CONFIGURATION.md               两版完整参数说明
DEPENDENCIES.md                依赖说明和版本差异
provenance/                    源文件来源 SHA 和原环境版本，不含训练日志
MANIFEST.sha256                交付文件逐项校验（不包含清单自身）
```

查看展开的完整方案：

```bash
python run.py config --profile baseline
python run.py config --profile optimized
```

两版已经展开的完整配置快照也可直接查看：`configs/baseline.resolved.json`、`configs/optimized.resolved.json`；不需要手工拼接公共参数。

## 环境安装

支持目标是 **Linux x86_64、Python 3.11、NVIDIA BF16 显卡**。本包沿用 PyTorch 2.6.0 + CUDA 12.4 / Triton 3.2.0；适合在相应受支持的 Ampere/Ada/Hopper 显卡上先验证。全改进建议先用 48/80 GiB 卡验证，**这只是容量建议，未实测保证最低显存**。合并备份与独立 head 额外约 7.492 GiB，PPO 仍需激活、logits、KV、优化器和编译内存。CPU 备份模式还需要约 8 GiB 额外主存；CUDA 审计工具会额外保存一份模型到主存，建议主存至少 32 GiB。

原生 Windows Triton 路径没有验证，请使用 Linux 或带 NVIDIA 驱动支持的 WSL2。Blackwell/更新架构需要另行选择并验证新版 PyTorch/CUDA，不能直接声称这份 cu124 锁定环境兼容。

先安装支持 CUDA 12.4 的 NVIDIA 驱动，确保 `nvidia-smi` 正常。Inductor/Triton 还需要系统 C/C++ 编译器和 Python 3.11 开发头文件，例如 Ubuntu 的 `build-essential`、`python3.11-dev`、`python3.11-venv`。PyTorch wheel 自带所需 CUDA 用户态运行库；本项目不要求自行编译 FlashAttention。

```bash
cd qwen3_4b_sft_ppo_portable_20261104_v2
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-build.txt
python -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
python -m pip check

# 指定一张空闲 GPU；程序要求只看到这一张卡。
export CUDA_VISIBLE_DEVICES=0
python run.py doctor --gpu

# 仅使用 CPU 的随机小模型测试
CUDA_VISIBLE_DEVICES="" DISABLE_VERSION_CHECK=1 python -m unittest discover -s tests -v
```

必须从项目根目录安装 requirements，使用随包的 editable `llamafactory`，不要另装 PyPI 同名新版本覆盖。项目主动关闭上游面向新版 TRL 的通用版本检查，同时严格断言实际 legacy PPO 所需版本。NumPy 调整为 1.26.4，以解决原环境 TRL 0.9.6 与 NumPy 2.4.6 的声明冲突；这是明确披露的环境差异，跨版本/跨 GPU 不保证复现旧轨迹的逐位结果。

离线机器需要预先准备相应平台 wheels。包中没有安装包或模型镜像。官方安装与数值说明：[PyTorch 历史版本](https://pytorch.org/get-started/previous-versions/)、[PyTorch 2.6 数值精度](https://docs.pytorch.org/docs/2.6/notes/numerical_accuracy.html)、[TRL 0.9.6](https://huggingface.co/docs/trl/v0.9.6/trainer)。

## 外部模型与数据

使用 **Qwen/Qwen3-4B-Base，未经量化，36 层、hidden 2560、共享 embedding/head 基座**，不是 Instruct 模型，不加载旧 Alpaca 或 PPO adapter。外部 HF 模型目录应有完整 `config.json`、tokenizer/特殊 token 文件、safetensors（或 pytorch bin）及必要 shard index。tokenizer 不随包分发。

数据使用 GSM8K `main`：训练 7473 条、测试 1319 条。两个外部 JSONL 文件每行包含 `question` 和 `answer` 字符串，`answer` 保留原始推理及最终 `####` 数值。可选导出工具（联网时执行，输出目录必须不存在）：

```bash
python tools/export_gsm8k.py --output /datasets/gsm8k_jsonl
```

默认固定 revision 为 `3101c7d5072418e28b9008a6636bde82a006892c`；已有同版文件可直接使用。程序不向题目添加额外格式要求，SFT/PPO/奖励/测试都使用原始题目。

## 准备两版独立运行目录

```bash
python run.py prepare --profile baseline --run-dir /outputs/4b_baseline \
  --model /models/Qwen3-4B-Base \
  --train /datasets/gsm8k_jsonl/train.jsonl --test /datasets/gsm8k_jsonl/test.jsonl

python run.py prepare --profile optimized --run-dir /outputs/4b_optimized \
  --model /models/Qwen3-4B-Base \
  --train /datasets/gsm8k_jsonl/train.jsonl --test /datasets/gsm8k_jsonl/test.jsonl
```

`prepare` 不训练、不把模型搬到 GPU；会核验结构、数据数量、原始数值答案、训练/测试分离，生成 LF 数据格式及每条 PPO 提示哈希，并流式计算外部模型/数据和源码 SHA。模型大文件仍留在原处。实际题目最大提示 220 tokens，220+512≤768；自定义数据超过静态缓存容量会报错，不会静默缩短回复。

每个目录含 `PROFILE.json`、`SFT_CONFIG.json`、`REQUESTED_CONFIG.json`、`INPUT_IDENTITY.json`、`RUNTIME_BUILD.json` 等实际配置与哈希。启动每个阶段会复核源文件及输入，避免静默换源。原路径应在本次运行期间保持可用。调整配置应先修改项目内 profile，再准备**新目录**；不要修改已开始运行的源文件或配置。

可单独核验与 LF 真实预处理一致：

```bash
CUDA_VISIBLE_DEVICES="" python tools/verify_prepared_data.py --run-dir /outputs/4b_optimized
python run.py verify --run-dir /outputs/4b_optimized
```

## SFT、目标卡验证、短跑与正式训练

分别进行全新 SFT：每版从基座初始化，935 次更新。若只做 PPO 组件速度对比，可通过后面的 `--sft` 使用同一完成的 SFT，避免把初始化差异混入对比。

```bash
python run.py sft --run-dir /outputs/4b_baseline
python run.py quality --run-dir /outputs/4b_baseline --adapter /outputs/4b_baseline/sft/model --name sft_ready

python run.py sft --run-dir /outputs/4b_optimized
python run.py quality --run-dir /outputs/4b_optimized --adapter /outputs/4b_optimized/sft/model --name sft_ready
```

新显卡上先做全改进生成核验（正常/异常恢复的 CPU 测试已包含在 tests 中）。先 64 tokens，再 512 tokens，使用不同结果名：

```bash
python tools/validate_cuda.py --run-dir /outputs/4b_optimized --sft /outputs/4b_optimized/sft/model --tokens 64 --name cuda64
python tools/validate_cuda.py --run-dir /outputs/4b_optimized --sft /outputs/4b_optimized/sft/model --tokens 512 --name cuda512
```

此工具核验真实 4B 生成后所有参数恢复、共享头身份、冻结参考不变、更新后 compiled decode 读取新 LoRA；记录初始化/稳态、显存和 EOS logprob 误差。它不是完整 PPO 显存验收，仍需最长提示的实际四次 PPO 更新压力测试：

```bash
python run.py ppo --run-dir /outputs/4b_optimized --name pressure --limit 1 --stress
python run.py ppo --run-dir /outputs/4b_optimized --name pilot200 --limit 200
python run.py quality --run-dir /outputs/4b_optimized --adapter /outputs/4b_optimized/runs/pilot200/model --name pilot200_quality
```

`--stress` 只在该独立测试中使用最长三条提示并强制各 512 tokens，仍做四次更新；不修改正式训练参数。压力和 pilot 使用各自全新 Adam/value head，**不把 pilot200 的权重接到正式训练**。通过后正式从完成的 SFT 重新初始化：

```bash
python run.py ppo --run-dir /outputs/4b_baseline --name full
python run.py ppo --run-dir /outputs/4b_optimized --name full

python run.py quality --run-dir /outputs/4b_baseline --adapter /outputs/4b_baseline/runs/full/model --name ppo_final
python run.py quality --run-dir /outputs/4b_optimized --adapter /outputs/4b_optimized/runs/full/model --name ppo_final
python run.py evaluate --run-dir /outputs/4b_baseline --adapter /outputs/4b_baseline/runs/full/model --label PPO
python run.py evaluate --run-dir /outputs/4b_optimized --adapter /outputs/4b_optimized/runs/full/model --label PPO
```

最终测试均为相同 native evaluator、1319 题、greedy、batch1、max512；同时报告数值正确率和“数值正确且有 ####”比例。后者是格式诊断，不替代数值指标。测试集不参与 checkpoint 选择。需要 SFT 最终测试时以 `--label SFT` 另跑同一 evaluator。

环境和目标卡已验证后，可在**全新的已 prepare 目录**直接执行整个正式流程：

```bash
python run.py all --run-dir /outputs/another_4b_optimized
```

`all` 顺序为完整 SFT → 固定 32 题训练质量门槛 → 完整 PPO → 32 题门槛 → 最终 native 测试，不自动代替首次目标卡的 CUDA/压力/恢复测试；中途失败会停止并保留输出。

## 恢复验证和完整恢复

为了在目标卡确认真实恢复链路，使用同一已完成 SFT 做三次独立测试（不会修改 SFT）：

```bash
python run.py ppo --run-dir /outputs/4b_optimized --name resume_control --limit 2 --strict
python run.py ppo --run-dir /outputs/4b_optimized --name resume_part --limit 1 --strict
python run.py ppo --run-dir /outputs/4b_optimized --name resume_finish --limit 2 --strict \
  --resume-state /outputs/4b_optimized/runs/resume_part/state_checkpoints/step_000001/training_state.pt
python tools/compare_checkpoints.py \
  --control /outputs/4b_optimized/runs/resume_control/state_checkpoints/step_000002/training_state.pt \
  --resumed /outputs/4b_optimized/runs/resume_finish/state_checkpoints/step_000002/training_state.pt
```

常规 PPO 恢复同样指定旧 `training_state.pt` 和新的 `--name`；`--limit` 是**累计目标批数**。checkpoint 必须在这个运行目录的 `runs/` 下且通过 SHA 校验；只加载自己信任的 checkpoint。恢复内容含 actor/value head FP32 参数、Adam、scheduler、KL controller、running moments、RNG、样本顺序、TRL counters 和质量保护状态。保持相同 profile、SFT SHA、源码、输入及 strict 设置。改变优化模式不能作为同一训练的“完整恢复”。

SFT 使用 HF 完整 checkpoint，可恢复到新子目录，例如：

```bash
python run.py sft --run-dir /outputs/4b_optimized --sft-name sft_resumed \
  --resume-hf /outputs/4b_optimized/sft/model/checkpoint-500
```

跨机器/GPU/依赖版本的相同数值轨迹不保证；CPU 验证证明保存/加载机制，不证明目标卡的编译轨迹已逐位一致。

## 合并的语义、显存和可选 CPU 备份

合并时计算每个模块的当前 FP32 `B@A*scale`，分 1024 行加入 BF16 基座；252 个非 head 模块保存原始 BF16 快照，输出头另外构造 BF16 工作权重，**不把 head delta 写入共享 embedding**。离开生成作用域（包括异常）后拷回快照并恢复原 head Parameter，不使用减 delta。所有 LoRA、优化器参数和被冻结参考的参数对象保持不变；每次生成重新读取 PPO 更新后的 A/B。

GPU 备份是 `optimized` 默认选择。若希望用主存换显存/传输时间，在 prepare 时：

```bash
python run.py prepare --profile optimized --merge-backup cpu --run-dir /outputs/4b_optimized_cpu_backup \
  --model /models/Qwen3-4B-Base --train /datasets/gsm8k_jsonl/train.jsonl --test /datasets/gsm8k_jsonl/test.jsonl
```


CPU 备份不能直接按 GPU 备份的性能预估；恢复传输和精确核验开销会计入批次时间。

**精确恢复不等于合并期间计算等价。** BF16 合并改变舍入顺序；生成分布可能和恢复后原生 old-policy forward 有差异。本项目没有新增重要性校正来更改原 PPO 算法，也不以 EOS logprob 最大误差 0.1 作为硬拒绝阈值。全合并方案属于允许数值近似的实验配置，最终是否接受应看稳定性及完整测试数值正确率，不能仅凭某个单点误差小于旧 1.7B 就判定安全。

保留原硬保护：非有限值/恢复失败立即报错；连续 20 批全零奖励、K3 KL>50、最近 20 批自然 EOS<50% 停止并保存状态。固定 32 题质量门槛是自然 EOS≥29、数值正确≥16、空输出为零。这些条件没有为“跑满”而放宽。

## 输出与比较口径

每个 PPO 批次保存 loss、奖励、K3/原始 signed KL、clipfrac、entropy、生成/PPO/TRL 阶段时间、GC/cache 次数、显存及响应长度等。完整状态在第 1 批、每 250 批及停止点保存；SFT 同样保留 HF 完整 checkpoint。奖励服务仅绑定 localhost，主程序负责清理自己创建的子进程，不操作其他训练进程。

主要路径：`sft/model/`、`runs/<name>/model/`、`runs/<name>/state_checkpoints/`、`runs/<name>/batches.jsonl`、`runs/<name>/PROGRESS.json`、`runs/<name>/SUCCESS.json`、`quality/<name>/RESULT.json`、`evaluation/PPO/RESULT.json`。训练会产生新日志、数据副本与缓存，这些都是运行输出，不是随包历史文件。

全部 checkpoint 可能占用数十到百余 GiB，请为两版预留充足磁盘。程序不自动删除失败候选、训练结果或旧 checkpoint；同名结果存在时拒绝覆盖。

比较总流程吞吐时分别记录 SFT、初始化、冷编译、PPO 稳态、保存和测试成本；不要把相同时间里的重复 PPO 更新当作新增 token。做组件对比应在相同 SFT SHA、固定工作量、相同 GPU 条件下至少三次交错。两版各自 fresh SFT 的 SHA 不同时，在报告中写明是独立复跑。
