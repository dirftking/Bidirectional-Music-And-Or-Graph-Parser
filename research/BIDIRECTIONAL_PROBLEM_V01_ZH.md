# Bidirectional Music And-Or Graph Parser：问题契约 v0.1

## 研究任务

给定且只给定一段音乐音频，Bottom-up 前端生成节拍同步的调性、和弦和结构候选；MusicAOG 将候选表示为 Or 分支；Top-down 音乐文法向下发送结构约束并重新评分；系统输出带时间、层级、候选、置信度、证据来源和推理轨迹的 Parse Graph。

形式化目标为：

```text
G* = argmax_G P(G | x) ∝ P(x | G) P(G)
```

- `x` 是音频，不包含 MIDI、标签轨、曲名或 Ground Truth。
- `P(x | G)` 是 Bottom-up 音频证据。
- `P(G)` 是 Top-down 音乐结构先验。
- Ground Truth 只用于最终评价，绝不参加候选生成、参数选择或推理。

## v0.1 的可执行范围

当前只实现第一个最小闭环：

```text
MIR/MOSS 调性与和弦候选
        ↓ Bottom-up (β)
MusicAOG 调性/和弦 Or 节点
        ↓ Top-down (γ)
全局调性预测和弦相容性
        ↓ Bottom-up (β)
已选和弦反向支持全局调性
        ↺ 直到分支选择稳定
```

Top-down 分数等于候选和弦音中属于当前调式音阶的比例。小调允许自然小调与和声小调的音级并集。无法解析的和弦类型记为中性，不伪装成负证据。

最终后验采用：

```text
score = audio_weight * log(source_reliability * bottom_up_probability)
      + key_weight   * log(key_chord_compatibility)
posterior = softmax(score)
```

默认权重是实现常量，不在 `001–010` 测试集上调参。正式结果发表前，只能在独立开发集上确定权重。

MIR 和 MOSS 的原始候选分数按来源分别保存。由于 MOSS 的文本自报置信度尚未校准，v0.1 给它固定的 `0.1` 来源可靠性；该值必须在冻结测试集之前由开发集校准，不能根据测试标签修改。

## 当前不声称解决的问题

- 尚未进行段落重复、节拍层级和终止式的 Top-down 推理；
- 尚未学习概率或能量函数；
- 尚未证明双向推理优于 MIR-only；
- 尚未处理主观情感、审美或创作意图解释；
- POP909 合成音频的结论尚不能直接外推到真实录音。

## 可证伪假设

主假设：在固定候选集合下，加入调性—和弦双向消息后，独立测试集上的和弦准确率、调性准确率或概率校准至少一项优于固定 MIR top-1，且其他指标不出现不可接受的退化。

必须报告：

- 调性准确率；
- 和弦 duration-weighted exact/root accuracy；
- Top-k candidate recall；
- 推理前后改变的 Or 分支数量；
- 收敛轮数与是否出现循环；
- MIR-only、一次融合和双向推理三组对照。

若双向推理没有提高测试指标，结果应作为反例保留，而不是利用测试标签调整规则。
