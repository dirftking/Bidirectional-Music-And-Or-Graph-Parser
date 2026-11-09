# Bidirectional Music And-Or Graph Parser

**Bidirectional Music And-Or Graph Parser** 同时作为本项目名称与算法名称。
系统从音乐音频构建可审计的 MusicAOG 候选图，将 MOSS-Music 语义观测与传统
MIR 声学证据进行融合，并通过 Bottom-up 证据和 Top-down 音乐约束交替推理，
得到最终 Parse Graph。

当前研究代码包含：

- MOSS-Music 结构、和声、调性和速度观测提取；
- MIR 节拍、结构边界、色度、调性和和弦候选提取；
- 带来源与置信度的多源候选融合；
- 包含序列转移约束的调性—和弦双向推理；
- 面向 POP909 的评测工具与音频交互网页。

为保持现有命令兼容，Python 包名暂时保留为 `audio2musicaog`；项目产物统一使用
`bidirectional-music-and-or-graph-parser` 作为版本前缀。

```bash
python -m pip install -e ".[research,web]"
python -m unittest discover -s test -p "test_*.py"
python -m audio2musicaog.cli --help
```

以上命令只验证源码包；启动模型或网页服务前，必须完成下方的完整部署流程。

研究定义与实验说明位于 [`research/`](./research/)；模型权重、上传音频、运行结果、
账号信息和本地运行文件不会进入版本库。

## 部署流程（推荐：Linux / AutoDL）

以下命令均应在新终端中按顺序执行。生产部署建议使用单张 NVIDIA GPU：BF16
推理实际至少需要约 24 GB 显存，推荐 32 GB 显存、64 GB 系统内存。一个
checkpoint 约 17 GB，部署一个模型建议至少预留 50 GB 磁盘空间；两个模型均下载
时需要更多空间。

### 1. 克隆项目

当前 GitHub 仓库是私有仓库，需要先完成 GitHub 身份验证，再克隆到 AutoDL
数据盘：

```bash
cd /root/autodl-tmp
git clone https://github.com/dirftking/Bidirectional-Music-And-Or-Graph-Parser.git
cd Bidirectional-Music-And-Or-Graph-Parser
export PROJECT_ROOT="$(pwd)"
```

HTTPS 提示输入凭据时，用户名使用 `dirftking`，密码位置填写具有该仓库读取权限的
GitHub Personal Access Token；GitHub 账号密码不能用于 Git 克隆。也可以先完成
GitHub CLI 登录，再执行
`gh repo clone dirftking/Bidirectional-Music-And-Or-Graph-Parser`。

不要把 GitHub Token、AutoDL 密码或 SSH 密钥写入克隆地址、脚本或提交文件。

### 2. 创建固定环境

基准环境固定为 Python 3.12、FFmpeg 7、PyTorch 2.9.1 和 CUDA 12.8 wheel；
仓库内的 SGLang 源码提供模型服务。

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -n bimusicaog --override-channels -c conda-forge \
  python=3.12 "ffmpeg=7" pip -y
conda activate bimusicaog

python -m pip install --upgrade pip setuptools wheel
python -m pip install --index-url https://download.pytorch.org/whl/cu128 \
  --no-deps "torch==2.9.1+cu128" "torchaudio==2.9.1+cu128"
python -m pip install -e "sglang/python"
python -m pip install "nvidia-cudnn-cu12==9.16.0.29"
python -m pip install --index-url https://pypi.org/simple \
  -c repro/constraints.txt -e ".[research,web]"
python -m pip check
```

Linux/SGLang 部署不要安装本项目的 `torch-runtime` extra：仓库内 SGLang 已固定
兼容的 `torchcodec` 版本。`torch-runtime` 只保留给独立 Transformers 推理流程。

下载模型前确认 GPU、CUDA 和 FFmpeg：

```bash
nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
ffmpeg -version | head -n 1
```

### 3. 下载 checkpoint 到固定路径

| 模型 | Hugging Face 路径 | 本地固定目录 | 用途 |
|---|---|---|---|
| Thinking（推荐） | [`OpenMOSS-Team/MOSS-Music-8B-Thinking`](https://huggingface.co/OpenMOSS-Team/MOSS-Music-8B-Thinking) | `weights/MOSS-Music-8B-Thinking` | 分析与推理能力更强 |
| Instruct | [`OpenMOSS-Team/MOSS-Music-8B-Instruct`](https://huggingface.co/OpenMOSS-Team/MOSS-Music-8B-Instruct) | `weights/MOSS-Music-8B-Instruct` | 直接指令跟随更快 |

推荐下载 Thinking 模型：

```bash
cd "$PROJECT_ROOT"
mkdir -p weights
hf download OpenMOSS-Team/MOSS-Music-8B-Thinking \
  --local-dir "$PROJECT_ROOT/weights/MOSS-Music-8B-Thinking"

python repro/check_environment.py \
  --model-path "$PROJECT_ROOT/weights/MOSS-Music-8B-Thinking"
```

如需 Instruct 模型，将仓库 ID 和本地目录中的 `Thinking` 同时替换为
`Instruct`。正确的 checkpoint 目录必须包含 `config.json`、
`model.safetensors.index.json`、`tokenizer_config.json` 和全部
`model-*-of-*.safetensors` 分片。不要把 `weights/` 提交到 Git。

### 4. 启动 SGLang 模型服务

在项目根目录执行，并确保当前已激活 `bimusicaog` 环境：

```bash
export MODEL_DIR="$PROJECT_ROOT/weights/MOSS-Music-8B-Thinking"

python -m sglang.launch_server \
  --model-path "$MODEL_DIR" \
  --trust-remote-code \
  --host 127.0.0.1 \
  --port 30000 \
  --mem-fraction-static 0.75 \
  --chunked-prefill-size 2048 \
  --max-prefill-tokens 16384 \
  --disable-piecewise-cuda-graph
```

等待日志明确显示服务就绪，再在另一个 SSH 终端检查：

```bash
curl --fail http://127.0.0.1:30000/health
```

该命令成功以前不要启动网页任务。

### 5. 启动网页服务

在第二个 SSH 终端执行：

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate bimusicaog
cd /root/autodl-tmp/Bidirectional-Music-And-Or-Graph-Parser

export MOSS_MUSIC_SGLANG_BASE_URL="http://127.0.0.1:30000"
export MUSIC_AOG_WEB_HOST="127.0.0.1"
export MUSIC_AOG_WEB_PORT="7861"
export MUSIC_AOG_JOBS_ROOT="$PWD/research_outputs/web_jobs"

python music_aog_web.py
```

检查网页后端以及它与 SGLang 的连接：

```bash
curl --fail http://127.0.0.1:7861/health
```

返回 JSON 必须同时包含 `"status":"ok"` 和 `"sglang_ready":true`。

### 6. 从本地电脑打开网页

AutoDL 上的两个服务都保持绑定 `127.0.0.1`。在本地电脑执行 SSH 隧道命令，
其中主机和端口以 AutoDL 控制台显示值为准：

```bash
export AUTODL_HOST="connect.example.com"
export AUTODL_SSH_PORT="12345"
ssh -N -L 7861:127.0.0.1:7861 \
  -p "$AUTODL_SSH_PORT" "root@$AUTODL_HOST"
```

保持该终端运行，然后打开 <http://127.0.0.1:7861/>。如果本地 7861 端口已被
占用，改用 `-L 17861:127.0.0.1:7861`，并访问
<http://127.0.0.1:17861/>。

### 7. 使用 tmux 保持后台运行（可选）

激活环境并设置 `PROJECT_ROOT` 后执行：

```bash
cd "$PROJECT_ROOT"
mkdir -p logs
PYTHON_BIN="$(command -v python)"

tmux new-session -d -s bmusic-model \
  "cd \"$PROJECT_ROOT\" && exec \"$PYTHON_BIN\" -m sglang.launch_server \
  --model-path \"$PROJECT_ROOT/weights/MOSS-Music-8B-Thinking\" \
  --trust-remote-code --host 127.0.0.1 --port 30000 \
  --mem-fraction-static 0.75 --chunked-prefill-size 2048 \
  --max-prefill-tokens 16384 --disable-piecewise-cuda-graph \
  > logs/model.log 2>&1"

tmux new-session -d -s bmusic-web \
  "cd \"$PROJECT_ROOT\" && export MOSS_MUSIC_SGLANG_BASE_URL=http://127.0.0.1:30000 \
  MUSIC_AOG_WEB_HOST=127.0.0.1 MUSIC_AOG_WEB_PORT=7861 \
  MUSIC_AOG_JOBS_ROOT=\"$PROJECT_ROOT/research_outputs/web_jobs\" && \
  exec \"$PYTHON_BIN\" music_aog_web.py > logs/web.log 2>&1"

tmux ls
tail -f logs/model.log
```

常用命令：`tmux attach -t bmusic-model`、`tmux attach -t bmusic-web`、
`tail -f logs/web.log` 和 `tmux kill-session -t <名称>`。

### 8. 完整验证

```bash
cd "$PROJECT_ROOT"
python -m unittest discover -s test -p "test_*.py"
python -m audio2musicaog.cli --help
curl --fail http://127.0.0.1:30000/health
curl --fail http://127.0.0.1:7861/health
```

当前仓库包含 41 项单元测试。网页支持 MP3、WAV 和 FLAC，单文件最大 200 MiB；
任务、日志和中间产物保存在 `research_outputs/web_jobs/`，且不会进入 Git。

### 部署环境变量

| 环境变量 | 默认值 | 作用 |
|---|---:|---|
| `MOSS_MUSIC_SGLANG_BASE_URL` | `http://127.0.0.1:30000` | 网页任务使用的 SGLang 地址 |
| `MUSIC_AOG_WEB_HOST` | `127.0.0.1` | 网页绑定地址；建议保持私有并使用 SSH 隧道 |
| `MUSIC_AOG_WEB_PORT` | `7861` | 网页端口 |
| `MUSIC_AOG_JOBS_ROOT` | `research_outputs/web_jobs` | 上传文件、中间产物、结果和日志目录 |
| `MUSIC_AOG_MAX_UPLOAD_BYTES` | `209715200` | 最大上传字节数 |
| `MUSIC_AOG_MOSS_MAX_NEW_TOKENS` | `2048` | 第一次 MOSS 生成预算 |
| `MUSIC_AOG_MOSS_REPAIR_MAX_NEW_TOKENS` | `4096` | 结构化输出修复预算 |
| `MUSIC_AOG_MOSS_REPAIR_ATTEMPTS` | `2` | 最大修复次数 |

### 常见部署错误

- **缺少 `ninja`**：在同一个 Conda 环境执行 `python -m pip install ninja`，
  然后重新启动模型服务。
- **CUDA 显存不足**：先停止其他 GPU 进程，用 `nvidia-smi` 确认显存，并保留
  `--mem-fraction-static 0.75`；不要同时运行两个模型服务。
- **网页能打开但显示模型不可用**：先检查 30000 端口，再确认
  `MOSS_MUSIC_SGLANG_BASE_URL`，并查看 `logs/model.log`。
- **本地打不开网页**：网页服务、SSH 隧道或本地端口至少有一项异常。不要为了
  绕过隧道而直接把服务器绑定到 `0.0.0.0`。
- **音频无法解码**：执行 `ffmpeg -version`，在同一 Conda 环境重新安装
  FFmpeg 7。
- **checkpoint 无法加载**：确认本地目录完全一致，重启前执行
  `python repro/check_environment.py --model-path <模型目录>`。

本项目基于上游 [OpenMOSS/MOSS-Music](https://github.com/OpenMOSS/MOSS-Music)
实现。为保留模型部署方法与来源说明，原始模型文档接续如下。

---

## MOSS-Music 上游基础

<p align="center">
  <img src="./assets/MOSS-Music.png" width="58%" alt="MOSS-Music logo" />
</p>

<div align="center">

<a href="https://huggingface.co/OpenMOSS-Team/MOSS-Music-8B-Instruct"><img src="https://img.shields.io/badge/Huggingface-Models-orange?logo=huggingface&amp"></a>
<!-- <a href="https://modelscope.cn/models/openmoss/MOSS-Music-8B-Instruct"><img src="https://img.shields.io/badge/ModelScope-Models-624AFF?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZmlsbD0id2hpdGUiIGQ9Ik0xMiAyQzYuNDggMiAyIDYuNDggMiAxMnM0LjQ4IDEwIDEwIDEwIDEwLTQuNDggMTAtMTBTMTcuNTIgMiAxMiAyeiIvPjwvc3ZnPg==&amp"></a> -->
<img src="https://img.shields.io/badge/Blog-Coming_Soon-blue?logo=internet-explorer&amp">
<img src="https://img.shields.io/badge/Arxiv-Coming_Soon-red?logo=Arxiv&amp">

<a href="https://x.com/Open_MOSS"><img src="https://img.shields.io/badge/Twitter-Follow-black?logo=x&amp"></a>
<a href="https://discord.gg/Xf3aXddCjc"><img src="https://img.shields.io/badge/Discord-Join-5865F2?logo=discord&amp"></a>

</div>

<p align="center">
  <a href="./README.md">English</a> | <a href="./README_zh.md">简体中文</a>
</p>

**MOSS-Music** 是由 [MOSI.AI](https://mosi.cn/#hero)、[OpenMOSS 团队](https://www.open-moss.com/)
与 [上海创智学院](https://www.sii.edu.cn/) 推出的开源 **音乐理解模型**。
它基于与 [MOSS-Audio](https://github.com/OpenMOSS/MOSS-Audio) 相同的音频 backbone，
在音乐上进行了专门的 **持续预训练** 和 **监督微调**，面向 **音乐描述、歌词 ASR、
结构分析、和弦 / 调式 / 节奏推理以及长时音乐问答** 等任务。本次发布共提供
**两个 8B 模型**：[**MOSS-Music-8B-Instruct**](https://huggingface.co/OpenMOSS-Team/MOSS-Music-8B-Instruct) 和 [**MOSS-Music-8B-Thinking**](https://huggingface.co/OpenMOSS-Team/MOSS-Music-8B-Thinking)。
其中 Instruct 版本更适合直接指令跟随，Thinking 版本则具备更强的音乐分析
链式思维推理能力。

## 新闻

* 2026.05.01：🎉🎉🎉 我们已发布 [MOSS-Music](https://huggingface.co/OpenMOSS-Team/MOSS-Music-8B-Instruct)。
* 2026.05.01：🎉🎉🎉 我们已发布用于大规模音乐数据标注与处理的 [MOSS-Music-Data-Pipeline](https://github.com/wx9songs/MOSS-Music-Data-Pipeline)。

## 目录

- [介绍](#介绍)
- [模型架构](#模型架构)
  - [DeepStack 跨层特征注入](#deepstack-跨层特征注入)
  - [时间感知表示](#时间感知表示)
- [已发布模型](#已发布模型)
- [音乐数据流水线](#音乐数据流水线)
- [评测](#评测)
- [快速开始](#快速开始)
  - [环境配置](#环境配置)
  - [SGLang 服务](#sglang-服务)
  - [本地推理 (Transformers)](#本地推理-transformers)
  - [Gradio 应用](#gradio-应用)
- [更多信息](#更多信息)
- [LICENSE](#license)
- [引用](#引用)

## 介绍

理解音乐并不只是「一段音频 + 一段文字」：它需要模型同时感知和声结构、
节奏、音色、乐器编排、演唱细节以及歌词语义，并在时间维度上进行联合推理。
**MOSS-Music** 的目标就是在单一模型中统一这些能力。

- **歌词 ASR 与时间戳对齐**：抗伴奏的歌唱 ASR，支持句级 / 词级时间戳。
- **音乐描述与标签**：用自然语言刻画情绪、风格、配器、制作风格以及情绪走向。
- **调式 / 节奏 / 和弦推理**：识别调式、节拍、下拍以及和弦进行，支持和弦转录与
  带时间戳和弦转录。
- **结构分析**：将歌曲切分为 intro / verse / chorus / bridge / outro，并对
  重复与对比段落进行推理。
- **乐器与声音识别**：识别主奏乐器、演唱声部（独唱 / 合唱、性别、音区）等。
- **音乐问答与长时分析**：针对一首完整作品进行开放式问答，Thinking 版本
  还支持链式思维推理。

<p align="center">
  <img src="./assets/moss-music_img.png" width="98%" alt="MOSS-Music overview" />
</p>

## 模型架构

MOSS-Music 继承了 MOSS-Audio 的模块化设计：音频编码器、模态适配器与大语言模型
三个部分。原始音频首先由 **MOSS-Audio-Encoder** 编码为 **12.5 Hz** 的连续时序
表征，然后通过适配器投影到语言模型的嵌入空间，最终由 LLM 完成自回归文本生成。

我们没有依赖现成的通用音频前端，而是从零训练专用编码器，以获得更鲁棒的
声学表征、更紧密的时间对齐能力，以及在音乐风格、歌唱与非语音内容上的
更好扩展性。

### DeepStack 跨层特征注入

如果仅使用编码器顶层特征，往往会丢失底层韵律、瞬态事件以及局部时频结构。
为了解决这一问题，我们在编码器与语言模型之间采用了受 **DeepStack** 启发的
跨层注入模块：除了编码器最终层输出外，还会选取更早期和中间层特征，分别
进行独立投影，并注入语言模型的前几层，从而保留从低层声学细节到高层语义
抽象的多粒度信息。

这一设计尤其适合音乐理解任务：它有助于保留节奏、音色、瞬态与乐器质感 ——
这些信息无法仅用一个高层表征承载，却对和弦识别、结构分析和细粒度音乐描述
至关重要。

### 时间感知表示

时间是音乐理解中的关键维度。为了增强模型对显式时间位置的感知能力，我们
在预训练阶段采用 **时间标记插入** 策略：按照固定时间间隔，在音频帧表征之间
插入显式时间 token 用于标记时间位置。该设计使模型能够在统一的文本生成
框架中学习「什么发生在什么时候」，从而自然支持带时间戳的歌词 ASR、
节拍 / 下拍定位、段落边界检测以及长歌回溯问答。

在 MOSS-Audio 骨干之上，MOSS-Music 做了：

- **持续预训练**：使用用于大规模音乐数据标注与处理的流水线
  [`MOSS-Music-Data-Pipeline`](https://github.com/wx9songs/MOSS-Music-Data-Pipeline)
  构建的大规模多样音乐语料，重点覆盖歌唱、歌词及完整歌曲；
- **音乐指令 SFT**：覆盖描述、歌词 ASR、和弦 / 调式 / 结构分析、长时音乐问答；
- **Thinking 版本的推理调优**。

## 已发布模型

| 模型 | 音频编码器 | LLM 骨干 | 总规模 | Hugging Face | ModelScope |
|---|---|---|---:|---|---|
| **MOSS‑Music‑8B‑Instruct** | MOSS-Audio-Encoder | Qwen3-8B | ~9.1B | [![Hugging Face](https://img.shields.io/badge/Huggingface-Model-orange?logo=huggingface)](https://huggingface.co/OpenMOSS-Team/MOSS-Music-8B-Instruct) | [![ModelScope](https://img.shields.io/badge/ModelScope-Model-624AFF)](https://modelscope.cn/models/openmoss/MOSS-Music-8B-Instruct) |
| **MOSS‑Music‑8B‑Thinking** | MOSS-Audio-Encoder | Qwen3-8B | ~9.1B | [![Hugging Face](https://img.shields.io/badge/Huggingface-Model-orange?logo=huggingface)](https://huggingface.co/OpenMOSS-Team/MOSS-Music-8B-Thinking) | [![ModelScope](https://img.shields.io/badge/ModelScope-Model-624AFF)](https://modelscope.cn/models/openmoss/MOSS-Music-8B-Thinking) |

> 更小规模（4B）及更多变体将后续放出，敬请期待。

## 音乐数据流水线

MOSS-Music 的训练数据由一条端到端的流水线生成：从原始音频直接产出
chat 格式训练样本。该流水线见仓库
[`MOSS-Music-Data-Pipeline`](https://github.com/wx9songs/MOSS-Music-Data-Pipeline)，
其中包括时长检测、MIR 特征抽取、歌曲结构切分、歌词 ASR、元数据清洗，
以及基于 ALM 的 caption / query 生成；ALM 侧可对接 Qwen3-Omni、MusicFlamingo
等音频语言模型。

<p align="center">
  <img src="./assets/music_pipeline.png" width="94%" />
</p>

## 评测

我们在一组公开音乐理解基准上评测 MOSS-Music，当前结果如下：

- **音乐 QA 与理解**：**MOSS-Music-8B-Instruct** 在 **8 个公开音乐 QA /
  理解基准**上取得 **80.38** 的平均准确率。
- **音乐描述（Music Captioning）**：在当前初步
  **GPT-5.4-as-a-Judge** 评测中，MOSS-Music 系列在两个 caption benchmark
  上均保持领先，其中 `MOSS-Music-8B-Thinking` 在 `MusicCaps` 上取得
  **4.53**，`MOSS-Music-8B-Instruct` 在 `SDD` 上取得 **4.58**。
- **歌词 ASR（歌声场景）**：**MOSS-Music-8B-Thinking** 在
  `MUSDB18`、`MIR-1K`、`Opencpop` 三个歌声数据集上取得 **15.88%** 的平均
  WER/CER，明显优于包括 `Gemini-3.1-Pro-Preview`、`MusicFlamingo` 与
  `Qwen3-Omni` 在内的所有对比 audio-language 模型。详细的歌声时间戳 ASR
  结果将在后续版本补充。
- **和弦转录**：MOSS-Music 支持和弦转录与带时间戳和弦转录，可用于和声分析、
  伴奏参考以及音乐教学等场景。详细 benchmark 结果将在后续版本补充。


<p align="center">
  <img src="./assets/music_bench.png" width="98%" />
</p>

### 音乐 QA 与理解（Accuracy↑）

| 模型 | MMAU-music | MMAU-mini-music | MMAU-Pro-music | MMAR-music | MuChoMusic | Music-AVQA | NSynth (instrument) | NSynth (source) | NSynth (pitch) | GTZAN | Medley-Solos-DB | Avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **MOSS‑Music‑8B‑Instruct** | **79.33** | **80.78** | 71.02 | 59.70 | **89.39** | **76.78** | **86.55** | 61.07 | **86.94** | **93.59** | 92.42 | **80.38** |
| Gemini‑3.1‑Pro | 71.69 | 77.18 | **73.06** | **71.64** | 79.53 | 61.51 | 13.38 | 38.90 | 6.47 | 86.39 | 80.34 | 75.17 |
| **MOSS‑Music‑8B‑Thinking** | 74.09 | 77.78 | 67.98 | 50.25 | 82.90 | 68.90 | 56.17 | 57.48 | 77.83 | 84.78 | 87.42 | 74.26 |
| MusicFlamingo | 76.83 | 76.35 | 65.60 | 48.66 | 74.58 | 73.60 | 80.76 | **75.89** | 0.00 | 84.45 | 90.86 | 73.87 |
| Audio‑Flamingo‑Next | 72.39 | 72.07 | 61.64 | 45.27 | 75.62 | 62.94 | 86.40 | 66.73 | 0.05 | 77.68 | 91.47 | 69.89 |
| MiMo‑Audio‑7B‑Instruct | 66.36 | 72.97 | 66.50 | 45.77 | 75.40 | 57.05 | 25.01 | 1.49 | 4.86 | 65.67 | **93.81** | 67.94 |
| Step‑Audio‑R1 | 66.46 | 75.08 | 62.34 | 50.75 | 72.62 | 57.98 | 13.75 | 15.87 | 2.39 | 73.67 | 82.45 | 67.67 |
| Qwen3‑Omni | 65.76 | 68.77 | 66.27 | 48.54 | 78.77 | 56.05 | 30.92 | 44.30 | 28.08 | 80.15 | 69.65 | 66.75 |
| Kimi‑Audio‑7B‑Instruct | 47.95 | 52.25 | 59.10 | 45.27 | 70.18 | 68.90 | 6.01 | 0.81 | 3.88 | 39.54 | 71.98 | 56.90 |

> `Avg` 由 8 个公开音乐 QA / 理解基准计算得到：
> `MMAU-music`、`MMAU-mini-music`、`MMAU-Pro-music`、`MMAR-music`、
> `MuChoMusic`、`Music-AVQA`、`GTZAN` 与 `Medley-Solos-DB`。
>
> 之所以不将 3 个 `NSynth` 子任务并入主平均分，是因为它们更强调短时单音上的
> 细粒度识别能力，包括乐器类别、声源属性（acoustic / electronic）以及精确
> 音高判别。部分对比模型并不是面向这种 note-level classification 设定设计的，
> 因此我们将 NSynth 结果单独保留在表中作为参考，而不混入 headline 平均分。

### 音乐描述（Music Captioning）

我们进一步在 `MusicCaps` 与 `Song Describer Dataset (SDD)` 上进行了
**GPT-5.4-as-a-Judge** 的初步 caption 评测。评分采用 1-5 分制，覆盖以下
9 个维度：`风格/流派`、`情绪/氛围`、`速度/节奏感`、`配器/音色`、`人声相关`、
`旋律/和声`、`结构与段落变化`、`制作与声学质感`、`场景/用途/语义联想`。

- **整体表现**：MOSS-Music 系列在两个 caption benchmark 上均保持领先，其中
  `MOSS-Music-8B-Thinking` 在 `MusicCaps` 上取得 **4.53**，而
  `MOSS-Music-8B-Instruct` 在 `SDD` 上取得 **4.58**。
- **结构理解优势明显**：在 `Structure / Form / Progression` 维度上，
  MOSS-Music 相比基线更强，尤其在 `SDD` 上优势更明显。
- **细粒度配器与场景联想**：`MusicFlamingo` 与 `Gemini-3.1-Pro` 在
  `Instrumentation / Timbre` 维度更有竞争力，其中 `Gemini-3.1-Pro`
  在 `Scene / Use Case` 维度表现最好。

#### MusicCaps

| 模型 | Genre | Mood | Tempo | Instr. | Vocals | Melody/Harmony | Structure | Production | Scene | Avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **MOSS‑Music‑8B‑Thinking** | 4.78 | **4.69** | **4.62** | 4.40 | **4.46** | **4.40** | **4.86** | 4.35 | 4.18 | **4.53** |
| Gemini‑3.1‑Pro | 4.70 | 4.60 | 4.48 | **4.68** | 4.18 | 4.18 | 3.86 | **4.40** | **4.72** | 4.42 |
| **MOSS‑Music‑8B‑Instruct** | 4.60 | 4.52 | 4.46 | 4.02 | 4.30 | 4.38 | 4.78 | 4.20 | 3.96 | 4.36 |
| MusicFlamingo | **4.80** | 4.36 | 4.50 | 4.64 | 3.94 | 4.08 | 3.58 | 4.30 | 3.72 | 4.21 |
| Audio‑Flamingo‑Next | 4.34 | 4.56 | 4.08 | 4.30 | 4.18 | 3.78 | 3.66 | 4.04 | 3.92 | 4.10 |
| MiMo‑Audio‑7B‑Instruct | 4.02 | 4.20 | 4.46 | 4.28 | 4.36 | 3.62 | 3.30 | 4.08 | 3.50 | 3.98 |
| Step‑Audio‑R1 | 4.22 | 4.02 | 4.20 | 3.96 | 3.84 | 4.02 | 3.24 | 4.10 | 3.54 | 3.90 |
| Qwen3‑Omni | 4.58 | 4.50 | 4.26 | 3.62 | 3.64 | 3.48 | 2.98 | 4.18 | 4.42 | 3.96 |
| Kimi‑Audio‑7B‑Instruct | 3.98 | 3.92 | 4.32 | 3.88 | 4.48 | 3.28 | 2.72 | 3.72 | 3.24 | 3.73 |

#### Song Describer Dataset (SDD)

| 模型 | Genre | Mood | Tempo | Instr. | Vocals | Melody/Harmony | Structure | Production | Scene | Avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **MOSS‑Music‑8B‑Instruct** | **4.84** | **4.76** | **4.68** | 4.24 | **4.52** | **4.56** | **4.92** | 4.42 | 4.24 | **4.58** |
| Gemini‑3.1‑Pro | 4.72 | 4.64 | 4.52 | **4.72** | 4.22 | 4.24 | 3.94 | **4.46** | **4.82** | 4.48 |
| **MOSS‑Music‑8B‑Thinking** | 4.66 | 4.58 | 4.50 | 4.36 | 4.36 | 4.44 | 4.84 | 4.26 | 4.02 | 4.45 |
| MusicFlamingo | 4.82 | 4.40 | 4.52 | 4.70 | 3.98 | 4.14 | 3.66 | 4.36 | 3.80 | 4.26 |
| Audio‑Flamingo‑Next | 4.40 | 4.62 | 4.14 | 4.36 | 4.22 | 3.84 | 3.74 | 4.10 | 4.00 | 4.16 |
| MiMo‑Audio‑7B‑Instruct | 4.08 | 4.26 | 4.52 | 4.34 | 4.42 | 3.70 | 3.38 | 4.16 | 3.58 | 4.05 |
| Step‑Audio‑R1 | 4.30 | 4.10 | 4.26 | 4.02 | 3.92 | 4.10 | 3.32 | 4.18 | 3.62 | 3.98 |
| Qwen3‑Omni | 4.62 | 4.54 | 4.30 | 3.68 | 3.70 | 3.56 | 3.06 | 4.24 | 4.50 | 4.02 |
| Kimi‑Audio‑7B‑Instruct | 4.04 | 3.98 | 4.38 | 3.96 | 4.54 | 3.36 | 2.80 | 3.80 | 3.32 | 3.80 |

### 歌词 ASR（WER / CER↓）

我们进一步在三个代表性的**歌声歌词 ASR** 基准上评测 MOSS-Music：

- `MUSDB18`：**带伴奏**的英文流行歌曲，以 **WER** 衡量；
- `MIR-1K`：**中文卡拉 OK** 片段，带伴奏，以 **CER** 衡量；
- `Opencpop`：**干净的普通话棚录歌声**，以 **CER** 衡量。

`Avg` 为三个数据集错误率的简单平均。

| 模型 | MUSDB18 WER | MIR-1K CER | Opencpop CER | Avg |
|-----|---:|---:|---:|---:|
| **MOSS‑Music‑8B‑Thinking** | 29.19% | **15.84%** | 2.60% | **15.88%** |
| **MOSS‑Music‑8B‑Instruct** | 32.99% | 23.96% | 4.62% | 20.52% |
| Gemini‑3.1‑Pro‑Preview | 26.25% | 36.37% | 6.00% | 22.87% |
| MusicFlamingo | **23.41%** | 38.98% | 18.73% | 27.04% |
| Qwen3‑Omni‑30B‑A3B‑Instruct | 62.67% | 20.48% | **2.26%** | 28.47% |
| MiMo‑Audio‑7B‑Instruct | 94.16% | 23.34% | 6.77% | 41.42% |
| Kimi‑Audio‑7B‑Instruct | 97.53% | 25.83% | 4.90% | 42.75% |
| Step‑Audio‑R1 | 81.67% | 48.03% | 4.15% | 44.62% |
| Audio‑Flamingo‑Next | 94.93% | 55.63% | 12.47% | 54.34% |

> **MOSS-Music-8B-Thinking** 在三个数据集上取得 **15.88%** 的最优平均错误率，
> 尤其在带伴奏的中文场景 `MIR-1K` 与干净普通话歌声 `Opencpop` 上有显著优势。
> MOSS-Music 还继承了 MOSS-Audio 的时间感知表示能力，歌声时间戳 ASR 的
> 详细结果将在后续版本补充。

### 和弦转录

MOSS-Music 支持和弦转录与带时间戳和弦转录，能够输出随时间变化的和弦进行，
可用于和声分析、伴奏参考、教学标注等任务。相关 benchmark 结果将在后续更新中
补充。


## 快速开始

### 环境配置

我们建议使用 Python 3.12 和 Conda 环境部署。

#### 推荐配置

```bash
git clone https://github.com/OpenMOSS/MOSS-Music.git
cd MOSS-Music

conda create -n moss-music python=3.12 -y
conda activate moss-music

conda install -c conda-forge "ffmpeg=7" -y
pip install --extra-index-url https://download.pytorch.org/whl/cu128 -e ".[torch-runtime]"
```

#### 可选：FlashAttention 2

如果你的 GPU 支持 FlashAttention 2，可以把最后一条安装命令替换为：

```bash
pip install --extra-index-url https://download.pytorch.org/whl/cu128 -e ".[torch-runtime,flash-attn]"
```

### SGLang 服务

> [!IMPORTANT]
> 为获得最佳生成质量和整体模型能力，我们**强烈推荐使用 SGLang Serving 进行推理**。

完整说明文档见 `moss_music_usage_guide.md`。

先下载模型：

```bash
hf download OpenMOSS-Team/MOSS-Music-8B-Instruct --local-dir ./weights/MOSS-Music-8B-Instruct
hf download OpenMOSS-Team/MOSS-Music-8B-Thinking --local-dir ./weights/MOSS-Music-8B-Thinking
```

最短的启动方式如下：

```bash
cd sglang
pip install -e "python[all]"
pip install nvidia-cudnn-cu12==9.16.0.29
cd ..

sglang serve \
  --model-path ./weights/MOSS-Music-8B-Instruct \
  --trust-remote-code
```

如果需要，也可以将 `./weights/MOSS-Music-8B-Instruct` 替换为
`./weights/MOSS-Music-8B-Thinking`。

如果你使用的是默认的 `torch==2.9.1+cu128` 运行时，建议在启动
`sglang serve` 之前先安装 `nvidia-cudnn-cu12==9.16.0.29`。

### 本地推理 (Transformers)

如果只是想本地快速跑通 demo，可以直接运行仓库自带的 `infer.py`：

```bash
python infer.py
```

修改 `infer.py` 顶部的 `MODEL_PATH`、`AUDIO_PATH`、prompt 和采样超参，即可
切换到自己的模型权重与音频。

> [!NOTE]
> 这条 Transformers 路径主要用于快速验证与调试，正式使用请优先选择
> [SGLang 服务](#sglang-服务) 以获得最佳生成质量和吞吐。

### Gradio 应用

使用以下命令启动 Gradio Demo：

```bash
python app.py
```

可通过 `MOSS_MUSIC_SERVER_NAME` / `MOSS_MUSIC_SERVER_PORT` 环境变量覆盖
监听地址与端口，并通过 `MOSS_MUSIC_MODEL_ID` 覆盖默认模型。

## 更多信息

- **MOSI.AI**：[https://mosi.cn](https://mosi.cn)
- **OpenMOSS**：[https://www.open-moss.com](https://www.open-moss.com)
- **MOSS-Audio（骨干）**：[https://github.com/OpenMOSS/MOSS-Audio](https://github.com/OpenMOSS/MOSS-Audio)
- **MOSS-Music 数据流水线**：[https://github.com/wx9songs/MOSS-Music-Data-Pipeline](https://github.com/wx9songs/MOSS-Music-Data-Pipeline)

## LICENSE

MOSS-Music 中的模型基于 Apache License 2.0 许可证发布，与 MOSS-Audio 保持一致。

## 引用

```bibtex
@misc{mossmusic2026,
      title={MOSS-Music Technical Report},
      author={OpenMOSS Team},
      year={2026},
      howpublished={\url{https://github.com/OpenMOSS/MOSS-Music}},
      note={GitHub repository}
}
```
