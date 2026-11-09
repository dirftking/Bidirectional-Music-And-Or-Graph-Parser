# MOSS-Music 本地复现与音乐解译研究指南

本目录基于官方仓库提交 `ad107c7ddaa06de168a0dfbc18d3e1e6a40c0e5e`。这里的“复现”分为两层：

1. **已公开模型的推理复现**：使用官方架构、官方权重和官方测试音频，复现歌词、描述、结构、和声等输出。这是本项目可以完整执行和验证的部分。
2. **训练复现**：公开仓库目前提供模型结构、权重和数据流水线说明，但没有发布完整的预训练/SFT/推理数据、训练脚本和超参数。因此无法严格复现 9.1B 模型的训练结果。后续研究应把官方权重当作基线，再做提示词、评测、探针或参数高效微调实验。

## 1. 已检测的本机条件

- Windows，Python/Conda 可用
- NVIDIA GeForce RTX 4090，24GB 显存
- 64GB 系统内存
- Instruct 权重约 16.87GB（4 个 BF16 safetensors 分片）
- 官方推荐的 Python 3.12、CUDA 12.8 PyTorch 与 FFmpeg 7 已由 `moss-music-repro` 环境管理

完整 BF16 推理通常需要基本空闲的 24GB 显卡。运行前先执行 `nvidia-smi`；若其他程序占用大量显存，请等待其结束。`--device auto` 可以尝试 CPU 卸载，但会显著变慢，而且不是官方推荐的质量/吞吐路径。

## 2. 环境安装

当前机器已经创建环境。其他 Windows 机器可从仓库根目录执行：

```powershell
powershell -ExecutionPolicy Bypass -File repro/setup_windows.ps1
```

手工安装等价于：

```powershell
$env:PYTHONUTF8 = "1"
$env:PIP_CONFIG_FILE = "NUL"
$env:PIP_EXTRA_INDEX_URL = "https://pypi.org/simple"
conda create -n moss-music-repro --override-channels -c conda-forge python=3.12 "ffmpeg=7" pip -y
conda run -n moss-music-repro python -m pip --isolated install --index-url https://download.pytorch.org/whl/cu128 --no-deps "torch==2.9.1+cu128" "torchaudio==2.9.1+cu128"
conda run -n moss-music-repro python -m pip install --index-url https://pypi.org/simple -c repro/constraints.txt -e ".[torch-runtime]"
```

Windows 上不要先用 `defaults` 创建环境、再从 `conda-forge` 单独补 FFmpeg；混合渠道可能造成 FFmpeg DLL 的 ABI/入口点错误。这里将 Python 与 FFmpeg 一次性从同一渠道解析。

PyTorch CUDA wheel 单独从其官方索引安装；其余依赖只查询 PyPI，避免把 CUDA 索引作为 `extra-index` 时对每个普通包产生无意义的网络等待。

`repro/constraints.txt` 固定 Gradio 5.44.1，因为官方固定的 Transformers 4.57.1 需要 `huggingface-hub<1.0`，而较新的 Gradio 6.x 已转向 `huggingface-hub>=1.x`。不加约束会触发 pip 长时间逐版回溯。

诊断命令不会加载大模型：

```powershell
conda run -n moss-music-repro python repro/check_environment.py --json
```

## 3. 下载权重

先下载直接回答任务更稳定的 Instruct 版本：

```powershell
powershell -ExecutionPolicy Bypass -File repro/download_weights.ps1 -Variant Instruct
```

需要比较长文本推理时再下载 Thinking 版本：

```powershell
powershell -ExecutionPolicy Bypass -File repro/download_weights.ps1 -Variant Thinking
```

两个模型各约 17GB，权重目录已被 Git 忽略。

## 4. 最小官方基线

权重下载完成后，官方脚本中的默认相对路径即可工作：

```powershell
conda run -n moss-music-repro python infer.py
```

它使用 `test/tonghua.mp3` 和提示词“转录出这首歌的歌词”。这是判断模型、音频解码、CUDA 和生成链路是否全部打通的最小测试。

## 5. 可重复的研究 CLI

列出内置研究任务：

```powershell
conda run -n moss-music-repro python research_cli.py --list-tasks
```

对一段音频做曲式分析：

```powershell
conda run -n moss-music-repro python research_cli.py `
  --audio "path\to\music.mp3" `
  --task structure `
  --model "weights\MOSS-Music-8B-Instruct"
```

运行全部九个分析维度（含 MusicAOG 候选图）：

```powershell
conda run -n moss-music-repro python research_cli.py `
  --audio "path\to\music.mp3" `
  --task full `
  --model "weights\MOSS-Music-8B-Instruct"
```

默认使用 greedy decoding、固定 seed=42，便于重复实验。若要对照官方采样式生成，可添加 `--sample --temperature 1.0 --top-p 0.8 --top-k 50`。

每次运行都会在 `research_outputs/` 写入 JSON，记录：音频 SHA-256、完整提示词、模型路径、环境版本、生成参数、耗时和回答。即使多任务运行中途失败，已经完成的结果也会保留。

如果你的后续方向确实是 **MusicAOG**，请继续阅读 `research/MUSICAOG_BRIDGE_ZH.md`。`--task musicaog` 会要求模型输出一个结构化、可校验的 audio-derived candidate parse graph。它是从音频到符号 MusicAOG 的中间标注，不等同于论文中的完整能量模型。

## 6. 面向音乐解译的实验设计

建议把研究分成四个层次：

1. **感知事实**：歌词、BPM、调性、和弦、段落边界、乐器。尽量用人工标注或 MIR 工具建立参照。
2. **跨模态描述**：比较模型对风格、情绪、制作和结构的语言表达，避免只凭流畅度评分。
3. **解释链**：要求每个高层解释绑定时间点和可听证据，例如“副歌张力增强”必须对应配器、动态、和声或音域变化。
4. **不确定性**：记录置信度、替代解释和可证伪方法。`uncertainty` 任务就是为这一步设计的。

一个稳妥的评价表至少应包含：事实准确率、时间定位误差、证据覆盖率、解释一致性、替代解释质量和跨次运行稳定性。不要只用单个 LLM judge 总分替代人工音乐学判断。

## 7. 架构阅读入口

- `src/modeling_moss_music.py`：音频编码器、适配器、DeepStack 跨层注入和生成前向过程
- `src/processing_moss_music.py`：mel 特征、12.5Hz 音频 token 与每 2 秒时间标记
- `src/configuration_moss_music.py`：编码器与语言模型配置
- `research/prompts.json`：研究提示词，可作为实验自变量并纳入版本控制
- `research/musicaog_candidate.schema.json`：从音频推断的 MusicAOG 候选图 Schema
- `research/MUSICAOG_BRIDGE_ZH.md`：MOSS-Music 与 MusicAOG 的概念映射、边界和实验问题
- `research_cli.py`：批量运行与实验元数据落盘

Windows 原生环境先用 Transformers 做正确性验证。官方推荐的 SGLang 路径更适合 Linux/WSL2；它用于提升质量和吞吐，但不应成为首次复现的阻塞项。
