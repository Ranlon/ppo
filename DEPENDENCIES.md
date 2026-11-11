# 依赖与环境

`requirements.txt` 安装完整锁定表及随包的 LLaMA-Factory；`requirements.lock.txt` 列出直接和传递依赖的固定版本；`requirements-build.txt` 列出 pip/setuptools/wheel。无需再安装其他加速框架。

| 核心库 | 版本 | 用途 |
|---|---|---|
| Python | 3.11.x；原环境3.11.16 | 解释器 |
| PyTorch | 2.6.0+cu124 | BF16训练、CUDA、Inductor、自定义op |
| torchvision / torchaudio | 0.21.0+cu124 / 2.6.0+cu124 | 锁定配套版本 |
| Triton | 3.2.0 | 生成期staged RMSNorm与编译 |
| Transformers | 4.57.1 | Qwen3、StaticCache、Trainer、生成 |
| PEFT | 0.17.1 | FP32 LoRA和多adapter参考 |
| TRL | **0.9.6** | 旧版PPO API；不能随意升级 |
| Accelerate | 1.11.0 | 单卡训练包装 |
| Datasets / PyArrow | 4.0.0 / 25.0.1 | 数据加载与预处理 |
| NumPy | **1.26.4** | RNG、数值工具；满足TRL<2约束 |
| safetensors / tokenizers | 0.8.0 / 0.22.2 | adapter保存与tokenizer |
| PyYAML | 6.0.3 | 配置 |
| pandas / scipy / matplotlib | 2.3.3 / 1.17.1 / 3.11.2 | 统计与原生训练绘图 |
| requests | 2.34.2 | localhost规则奖励客户端 |
| LLaMA-Factory | 0.9.4+ppo4.portable1 | 本地冻结兼容分支，随包源码 |

完整列表以 requirements.lock.txt 为准，包括 gradio 等上游导入所需库及 NVIDIA CUDA 用户态依赖。未使用 flash-attn、bitsandbytes、DeepSpeed、vLLM、Liger、Unsloth 等可选后端，不应为这条流程额外启用。

原环境存在重复 dist-info，不能直接以全部 `pip freeze` 条目为安装依据。本包按实际生效版本去重。原 LLaMA-Factory 元数据要求新版 TRL，但冻结的 PPO 实现明确依赖 TRL0.8.6..0.9.6；因此本包使用独立版本号的兼容元数据并锁定0.9.6，而非关闭检查后任意混装。原 NumPy2.4.6 改为1.26.4以修复legacy TRL声明冲突，已在隔离环境完成CPU验证。包版本/来源见 provenance/source_environment.json。

验证环境借用原机器现有其他共享包，在新 venv 内单独覆盖 NumPy/TRL 和本地源码包；`pip check`及完整 requirements dry-run通过。**尚未在一台没有任何缓存/包的新机器完成所有 wheels 的干净下载安装，也没有验证新GPU。** 新机器按 README 创建不继承 system-site-packages 的干净 venv，并先执行 doctor、CPU测试和CUDA验证。

系统依赖：Linux x86_64、支持CUDA12.4的NVIDIA驱动、C/C++编译器、Python3.11开发头文件。CUDA用户态库由wheel安装；显卡架构需要该PyTorch发行版支持。不要把`nvidia-smi`显示的驱动最高CUDA版本等同于`torch.version.cuda`。
