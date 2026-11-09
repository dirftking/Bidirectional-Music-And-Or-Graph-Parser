# Bidirectional Music And-Or Graph Parser：链式推理 v0.2

实验日期：2026-09-03

## 新增问题

v0.1 在给定全局调性后逐个、独立地选择和弦，因此相邻 beat 之间可能产生缺乏整体依据的快速跳变。v0.2 将所有和弦 Or 节点连接成一条由 `succession` 关系定义的链，在同一目标函数中联合选择完整和弦路径。

对第 `t` 个和弦候选 `c_t`：

```text
unary(c_t) = audio_weight * log P_bottom_up(c_t)
           + key_weight   * log K(c_t, key)

pair(c_(t-1), c_t) = transition_weight * log T(c_(t-1), c_t)
```

- `K` 是 v0.1 的调性—和弦音相容性；
- `T` 是明确写死且可审计的一阶和弦转移相容性；
- sum-product 前向—后向算法计算每个 Or 节点的边缘后验；
- max-product/Viterbi 解码选择全局最优和弦路径；
- 选出的和弦路径再反向支持全局调性，直到固定点。

转移规则按照候选根音关系给分：同一和弦 `1.0`，四/五度进行 `0.95`，同根异性质 `0.85`，二度 `0.80`，三度 `0.75`，半音 `0.60`，三全音 `0.50`。`N` 与无法解析的标签按中性处理，不制造伪负证据。

## 固定配置

```text
audio_weight        = 1.0
key_weight          = 0.75
transition_weight   = 0.25
compatibility_floor = 0.05
moss_source_weight  = 0.1
max_iterations      = 8
```

`transition_weight=0.25` 在十首批处理前固定，没有对该权重做网格搜索。但 001–010 已在前一开发阶段被查看，因此仍全部属于 development/smoke 集合，不能作为无泄漏测试集。

## 结果

| 指标 | 一次融合 | v0.1 调性闭环 | v0.2 链式闭环 |
|---|---:|---:|---:|
| 调性准确率 | 70.00% | 70.00% | 70.00% |
| 和弦完全匹配率 | 65.53% | 69.32% | 69.96% |
| 和弦根音准确率 | 83.70% | 83.81% | 84.35% |
| 和弦覆盖率 | 99.82% | 99.82% | 99.82% |

相对一次融合，v0.2 的和弦完全匹配率提高 `4.43 pp`，根音准确率提高 `0.65 pp`。相对 v0.1，分别再提高 `0.64 pp` 和 `0.54 pp`。

v0.2 在十首上全部到达固定点，共重选 386 个和弦分支。相对 v0.1，十首均不下降；相对一次融合，九首提高，010 仍下降约 `1.25 pp`。010 必须作为下一阶段误差分析对象保留。

## 复现

```bash
python -m audio2musicaog.cli infer-batch-pop909 \
  --output-root /root/autodl-tmp/mossmusic/research_outputs/pop909 \
  --transition-weight 0.25
```

产物：

- 推理 Schema：`research/musicaog_inference_v02.schema.json`；
- 汇总：`research_outputs/pop909/bidirectional-summary-v02.json`；
- 每首推理图：`<ID>/bidirectional-v02.json`；
- 每首指标：`<ID>/bidirectional-v02-metrics.json`。

## 尚未解决

当前一阶转移表是人工指定的开发基线，不是从训练集学习的概率。下一步不应继续手调该表，而应先建立新的冻结测试划分，并在训练/开发集合上学习和弦转移势。同时需要把 MIR 段落边界从单一峰值列表升级为 Or 候选，才能实现可评测的段落 Top-down 推理。
