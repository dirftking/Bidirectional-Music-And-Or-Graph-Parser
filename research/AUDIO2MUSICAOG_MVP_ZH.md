# Audio2MusicAOG 最小可验证系统（MVP）

## 研究问题

本阶段只检验一个问题：MOSS-Music 能否从音频提出结构、全局音乐属性和和弦候选，并被确定性地组装为结构自洽、保留歧义的 MusicAOG candidate set。

这不是完整 MusicAOG 复现。当前图只覆盖：

```text
Song(And)
  -> Section(And)
       -> Section-label choice(Or)
            -> intro / verse / chorus / ... (Terminal)
       -> Chord events (Terminal)
```

后续版本再加入 period、phrase、metrical tree、note/radical、可学习能量函数和图推断。

## 为什么分三次推理

单次长提示同时要求结构、和声和全局属性，容易造成遗漏、无效 JSON 和任务间相互干扰。MVP 使用三个互相独立的观察任务：

1. `global`：BPM、拍号、主调和转调；
2. `structure`：段落边界、功能标签、重复关系和备选标签；
3. `harmony`：带时间区间的保守和弦序列。

原始回答、解析后对象、生成参数、模型信息和音频 SHA-256 都保存在实验产物中。

此外，系统会生成 `audit_warnings`。它检查拍号格式、段落覆盖、时间顺序，以及全局调性与和弦音级的一致性。警告不会被当作结构错误，也不会偷偷修改模型答案；这是因为转调、借用和弦和标注惯例都可能造成合理例外，必须保留给研究者复核。

## 运行

在与 SGLang 服务相同的 AutoDL 实例执行：

```bash
cd /root/autodl-tmp/mossmusic/MOSS-Music
source /root/autodl-tmp/mossmusic/venv/bin/activate

python -m audio2musicaog.cli analyze \
  --audio /root/autodl-tmp/mossmusic/test_5s.wav \
  --base-url http://127.0.0.1:30000 \
  --max-new-tokens 1024
```

默认情况下，超过 30 秒的音频会把和弦任务分块运行，再转换为全局时间并合并跨块的同名和弦。完整音频仍用于全局属性和段落结构。可用 `--harmony-chunk-seconds 0` 关闭，或用其他秒数改变块长。每个块的提示、原始回答、解析结果和错误都会保存在产物中。

验证已有产物：

```bash
python -m audio2musicaog.cli validate research_outputs/audio2musicaog/example.json
```

退出码含义：

- `0`：三个任务均成功解析，图语义校验通过；
- `1`：命令或图校验失败；
- `2`：产物已保存，但至少一个模型回答无法解析，或图存在语义错误。

## Schema 0.2 的关键约束

- 每个节点 ID、产生式 ID 和关系 ID 唯一；
- 所有引用必须指向已存在节点；
- 所有节点必须由根节点经产生式可达；
- 产生式图不得有环；
- And 节点恰有一个分解产生式；
- Or 节点的分支概率之和为 1，且恰有一个当前选中分支；
- 子节点时间范围必须包含在父节点内；
- `schema_errors=[]` 只代表图结构自洽，不代表音乐判断正确。
- `audit_warnings=[]` 才表示当前基础审计没有发现明显的跨任务矛盾；它仍不等价于音乐学 Ground Truth 正确。

完整 JSON Schema：`research/musicaog_candidate.schema.json`。

## 下一里程碑

1. 加入 POP909-CL 数据读取和 MIDI 渲染；
2. 将 Ground Truth 转换成同一图格式；
3. 实现 Tempo、Key、Chord、Section 和 Graph 指标；
4. 通过多次采样构造数据驱动的 Or 候选，而不只依赖模型自报置信度。

## POP909 Ground Truth 基准

基准组合两份互补标注：

- POP909-CL：人工校正的和弦，以及校正后的速度、调号和拍号 MIDI 元事件；
- hierarchical-structure-analysis：每首歌两套人工段落结构标签，例如 `i4A4B8...o3`。

安装可选的 MIDI 读取依赖：

```bash
pip install -r research/requirements-audio2musicaog.txt
```

把一首歌转换成统一 Ground Truth JSON：

```bash
python -m audio2musicaog.cli ground-truth \
  --midi /path/to/POP909_processed/001.mid \
  --structure-label /path/to/POP909/001/human_label1.txt \
  --source-id 001 \
  --output research_outputs/audio2musicaog/001-ground-truth.json
```

渲染评测音频时必须先移除和弦标签轨，否则会把答案直接混进模型输入。安装 `fluidsynth` 与 GM SoundFont 后执行：

```bash
python -m audio2musicaog.cli render-score \
  --midi /path/to/POP909_processed/001.mid \
  --output-wav research_outputs/audio2musicaog/001-score.wav \
  --manifest research_outputs/audio2musicaog/001-render.json
```

评测模型产物：

```bash
python -m audio2musicaog.cli evaluate \
  --prediction research_outputs/audio2musicaog/001-prediction.json \
  --ground-truth research_outputs/audio2musicaog/001-ground-truth.json \
  --output research_outputs/audio2musicaog/001-metrics.json
```

当前结构指标只评边界，不把 `A/B/...` 强行映射成 `verse/chorus/...`。功能标签的语义对齐必须单独设计标注协议，不能把字母段落标签伪装成同义 Ground Truth。

### 可恢复的十首 smoke benchmark

`001–010` 是固定、透明的工程验收子集，不应宣称为具有统计代表性的正式测试集。批处理会跳过已有产物，每完成一首都原子更新 `batch-progress.json`，因此实例或 SSH 中断后可直接续跑：

```bash
python -m audio2musicaog.cli batch-pop909 \
  --ids 001,002,003,004,005,006,007,008,009,010 \
  --dataset-root /root/autodl-tmp/mossmusic/datasets \
  --output-root /root/autodl-tmp/mossmusic/research_outputs/pop909 \
  --base-url http://127.0.0.1:30000 \
  --max-new-tokens 2048 \
  --harmony-chunk-seconds 30
```

最终 `summary.json` 报告成功/失败数量、完整图比例，以及 Tempo、拍号、调性、和弦和结构边界的宏平均；每项同时报告实际可评测歌曲数，避免用缺失结果抬高分数。

001–010 的已完成结果、误差归因和后续实验路线见 `research/POP909_001_010_BASELINE_ZH.md`。

### Beat-synchronous MIR 与融合

安装可选 MIR 前端：

```bash
pip install -r research/requirements-mir.txt
```

分析一首音频并与 Ground Truth 评测：

```bash
python -m audio2musicaog.cli mir-analyze \
  --audio /path/to/score.wav \
  --output /path/to/mir.json

python -m audio2musicaog.cli evaluate-mir \
  --mir /path/to/mir.json \
  --ground-truth /path/to/ground-truth.json \
  --output /path/to/mir-metrics.json
```

运行固定十首 MIR-only 条件：

```bash
python -m audio2musicaog.cli mir-batch-pop909 \
  --ids 001,002,003,004,005,006,007,008,009,010 \
  --output-root /root/autodl-tmp/mossmusic/research_outputs/pop909
```

融合 MOSS 语义图与 MIR 候选：

```bash
python -m audio2musicaog.cli fuse \
  --prediction /path/to/prediction.json \
  --mir /path/to/mir.json \
  --output /path/to/fusion.json
```

实现和十首对照结果见 `research/MIR_FUSION_001_010_ZH.md`。

### Bottom-up/Top-down 双向推理 v0.1

对保存的融合图运行单首推理：

```bash
python -m audio2musicaog.cli infer \
  --fusion /path/to/fusion-v02.json \
  --output /path/to/bidirectional-v01.json
```

对现有 POP909 结果运行固定十首实验：

```bash
python -m audio2musicaog.cli infer-batch-pop909 \
  --output-root /root/autodl-tmp/mossmusic/research_outputs/pop909
```

问题契约见 `research/BIDIRECTIONAL_PROBLEM_V01_ZH.md`，实验结果见 `research/BIDIRECTIONAL_001_010_ZH.md`，推理产物 Schema 见 `research/musicaog_inference.schema.json`。

加入相邻和弦 `succession` 势并执行前向—后向/Viterbi 链式推理：

```bash
python -m audio2musicaog.cli infer-batch-pop909 \
  --output-root /root/autodl-tmp/mossmusic/research_outputs/pop909 \
  --transition-weight 0.25
```

实现定义与开发集结果见 `research/BIDIRECTIONAL_SEQUENCE_V02_ZH.md`，Schema 见 `research/musicaog_inference_v02.schema.json`。
