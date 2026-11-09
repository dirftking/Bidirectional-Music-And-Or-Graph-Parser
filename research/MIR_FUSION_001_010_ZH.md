# Beat-synchronous MIR 与 MOSS-Music 融合实验

实验日期：2026-09-03

## 研究问题

MOSS-Music 能生成流畅的音乐解释，但在调性、和弦和段落边界等严格符号任务上表现较弱。本实验检验一个更适合 MusicAOG 的分工：经典 MIR 前端恢复精确、节拍同步的音乐事实；MOSS-Music 提供段落语义和替代解释；MusicAOG 用 Or 节点显式保存多来源候选。

## MIR-only 前端

前端只读取无标签泄漏的 `score.wav`，不读取 POP909 Ground Truth。实现包括：

- `librosa.beat.beat_track` 动态规划 beat tracking；
- 谐波分量的 `chroma_cqt`；
- Krumhansl-Schmuckler 24 大小调模板匹配；
- 24 个大小三和弦余弦模板；
- 在相邻 beat 窗口上计算 chroma novelty，所有结构边界必须落在 beat 网格上。

拍号和 downbeat 暂未估计。配置中的四拍一小节只用于记录假设，不能作为预测参加评测。

## 十首结果

| 指标 | MOSS-only | MIR-only | 差值 |
|---|---:|---:|---:|
| Tempo Acc1 | 62.50%（8 首可用） | 100.00%（10 首） | +37.50 pp |
| 等音调性准确率 | 20.00% | 70.00% | +50.00 pp |
| 和弦完全匹配率 | 6.05% | 65.53% | +59.48 pp |
| 和弦根音准确率 | 10.98% | 83.70% | +72.72 pp |
| 和弦覆盖率 | 82.54% | 99.82% | +17.27 pp |
| 段落边界 F1（±3 秒） | 12.17% | 39.85% | +27.68 pp |

MIR-only 在十首上全部完成、零失败。MOSS-only 的 Tempo 数值只对八首有输出，因此 Tempo 行不是完全等样本比较；其余指标使用相同十首和同一 Ground Truth。

调性指标同时保存严格字符串准确率和等音准确率。例如 `F# major` 与 `Gb major` 在音乐上等价，主指标按相同音高类和调式判为正确，字符串指标仍可用于审计拼写。

## 融合 MusicAOG

融合器先保留 MOSS 的段落及功能语义，再加入：

- beat Terminal 节点；
- 调性 Or 节点及多个候选调；
- 每个 beat 区间的和弦 Or 节点；
- MIR 大小三和弦候选；
- 与该区间重叠的 MOSS 和弦候选；
- beat 与和弦各自的 succession 关系；
- beat 节点上的结构边界置信度。

为了避免在十首测试集上调参，当前选择策略预先固定为 MIR top-1。MOSS 候选保留在 Or 分支中，但其自报置信度不会覆盖当前选择。未来融合权重只能在独立开发集上学习。

十首融合图全部通过图验证。001 的完整示例包含 1,646 个节点、1,066 条产生式和 577 条关系，`graph_errors=[]`。

## 结论

结果支持“感知—解释分工”的研究假设：专用 MIR 前端显著改善时间精确的符号事实，MOSS-Music 则适合补充结构语义、证据描述和替代解释。当前融合图已经解决了表示问题，但还不能宣称融合模型优于 MIR-only，因为选择策略仍直接采用 MIR top-1。

下一步需要：

1. 在独立开发集上学习或校准 MIR/MOSS 候选概率；
2. 引入真正的 downbeat/meter 模型；
3. 加入 MIR-only、MOSS-only、无 Or 节点和完整融合的正式消融；
4. 扩大固定测试集并报告 bootstrap 置信区间；
5. 在真实录音数据上检验从 MIDI 合成音频得到的结论是否成立。

机器可读结果：`research_outputs/pop909/mir-summary.json`；001 融合图：`research_outputs/pop909/001/fusion.json`。
