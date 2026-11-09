# 从 MOSS-Music 到 MusicAOG 的研究桥接

MOSS-Music 与 MusicAOG 解决的是互补问题：

- **MOSS-Music** 从音频中提取或推理歌词、段落、速度、调性、和弦、配器、音色和长文本描述。
- **MusicAOG** 是符号音乐的 attributed And-Or Graph（A-AOG）及能量模型。它把一首作品实例表示为分层 parse graph，并以节点属性、产生式和横向关系组织音乐知识。

因此合理的研究链路不是把两者视作同一个模型，而是：

```text
音频
  -> MOSS-Music 多任务证据
  -> audio-derived candidate parse graph
  -> MIR / 乐谱 / 人工校正
  -> MusicAOG-compatible symbolic parse graph
  -> 解释、比较、能量特征学习或受控生成
```

## 论文表示与本项目字段的对应

MusicAOG 把音乐语法写成六元组 `<S, V, E, R, X, P>`。本项目目前只构造单首音频的候选 parse graph，不声称恢复完整语法或概率模型 `P`。

| MusicAOG 概念 | 候选 JSON | MOSS-Music 可提供的证据 |
|---|---|---|
| 根节点 `S` | `root` | 全曲时间范围与总体结构 |
| section / period | `nodes[type=section|period]` | 段落边界、重复、情绪/风格变化 |
| ensemble / phrase | `nodes[type=ensemble|phrase]` | 织体、动机、伴奏型、声部描述 |
| radical | `nodes[type=radical]` | 和弦事件、乐器事件、歌词/人声事件；精确音符需外部转录 |
| metrical tree | `nodes[level=metrical]` | BPM、拍号、强拍、和声时间区间 |
| production rules `R` | `productions` | 模型推断的父子分解 |
| succession `E_succ` | `relations[type=succession]` | 时间先后、连接和平滑度 |
| variation `E_var` | `relations[type=variation]` | 重复、变化、再现、模进 |
| diad `E_diad` | `relations[type=diad]` | 同时发声的音程/和弦关系；音频模型通常只能粗估 |
| synchronization `E_syn` | `relations[type=synchronization]` | 事件到节拍/小节位置的对齐 |
| attributes `X` | `attributes` + `confidence` + `evidence` | 时间、调性、情绪、风格、速度、和声、音色等 |

## 重要边界

MusicAOG 原论文面向**符号音乐**，radical 包含足以产生 MIDI/音频的音高、时值、音色和演奏法信息。MOSS-Music 的自然语言输出不能独自保证这种精度，所以本项目把输出明确命名为 `music_aog_audio_candidate`。

以下字段必须通过其他方法复核：

- 精确音高、复调声部分离、起止拍与时值：乐谱、MIDI 或 audio-to-MIDI 模型
- BPM、downbeat、和弦：独立 MIR 工具与人工听辨
- section/period/phrase 层级：多名标注者的一致性
- 情绪、叙事和哲学属性：听者研究或明确的分析编码手册

## 使用

```powershell
conda run -n moss-music-repro python research_cli.py `
  --audio "path\to\music.mp3" `
  --task musicaog `
  --model "weights\MOSS-Music-8B-Instruct"
```

CLI 会尝试从模型回答中提取 JSON，并检查节点 ID、引用和时间范围。`schema_errors` 为空只表示结构自洽，不表示音乐分析正确。完整 JSON Schema 位于 `research/musicaog_candidate.schema.json`。

## 推荐的第一组研究问题

1. MOSS-Music 的段落边界能否稳定映射到 section / period 层级？
2. 加入每 2 秒时间标记后，succession 与 synchronization 关系的定位误差是否下降？
3. Instruct 与 Thinking 模型在节点粒度、关系召回和无证据节点比例上有何差异？
4. “先自由描述再构图”与“直接生成候选图”哪种流程更准确？
5. 使用独立 MIR 结果约束节点属性后，MusicAOG 候选图的人工修改成本是否降低？

论文入口：[MusicAOG, arXiv:2401.02678](https://arxiv.org/abs/2401.02678)。
