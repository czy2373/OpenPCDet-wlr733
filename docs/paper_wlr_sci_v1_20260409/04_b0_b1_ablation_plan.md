# 4. B0 / B1 在论文中的定位与消融写法

## 4.1 定位建议

`B0/B1` 非常值得写进论文，但不建议写成“主方法本体”。更合适的定位是：

- `B0`：图像主导单帧后验决策的扩展模块
- `B1`：短时序一致性修正的扩展模块

一句话说：

> 主方法负责把图像语义稳健注入 `LiDAR-BEV`；`B0/B1` 负责验证图像主导 keep/drop 和时序 suppress 的进一步可行性。

## 4.2 为什么不建议把 `B0/B1` 升成论文主标题

原因有三个：

1. 当前最稳的主结果仍然来自前端融合主线本身。
2. `B0/B1` 的收益已经清楚，但还带有较强的任务口径和规则设计痕迹。
3. 如果把 `B0/B1` 升成主标题，审稿人会自然追问：
   - blind-spot 是否真正解决
   - image-only birth 是否稳定
   - `YOLO -> 3D` 是否已经闭环

而这几件事现在都还不适合写满。

## 4.3 建议的章节位置

`B0/B1` 最适合放在：

- `Experiments` 的扩展实验小节
- `Ablation and Discussion` 章节

不建议把它们写进：

- 摘要的第一主句
- 题目
- 主方法图中的中心框

## 4.4 `B0` 的建议写法

### 核心表述

`B0` 用于验证：

- 在 `LiDAR` 先给出候选 `3D` 框的前提下，
- 图像能否通过后验重打分直接影响最终 keep/drop。

### 适合写进正文的结论

- `B0` 明确证明了图像不仅可以做前端特征补充，也可以在 detector 输出后直接主导保留/抑制决策。
- 在 `DAIR` 上，`B0` 的结论最干净，最适合用作机制验证主证据。
- 在 `WLR` 上，`B0` 已正式接入评测链，但当前更适合作为主线扩展，而不是单独取代前端主线。

### 适合放的表

| 数据集 | 对比 | 结论 |
| --- | --- | --- |
| `DAIR` | raw vs `B0` | 图像主导 keep/drop 成立，`FP` 下降最明显 |
| `WLR` | raw vs `B0 promote015` | 已能在正式主链产生小幅稳定正收益 |

## 4.5 `B1` 的建议写法

### 核心表述

`B1` 用于验证：

- 短时序一致性是否能够对单帧结果做进一步修正，
- 尤其是在远距连续目标和低支持坏框上起作用。

### 适合写进正文的结论

- `B1` 当前最稳定的作用不是 rescue，而是 suppress。
- 它已经开始识别一类明确的坏 keep 模式，而不是泛泛地“加个 temporal 看看”。
- 在 `DAIR` 上，`B1` 的机制主要作用于 `50m+`，但整体尚未超过 `B0`。
- 在 `WLR` 上，`B1 temporal suppress` 已做出 `0 TP / -6 FP` 的正式主链收益。

### 适合放的表

| 数据集 | 对比 | 推荐写法 |
| --- | --- | --- |
| `DAIR` | `B0` vs `B1` | 说明时序一致性的作用区间和副作用 |
| `WLR` | `B0` vs `B1 suppress` | 强调其稳定减误检价值 |

## 4.6 建议的消融结构

### 消融 A：前端融合是否成立

- `lidar-only`
- front-end fusion

### 消融 B：图像主导单帧决策是否成立

- front-end fusion
- front-end fusion + `B0`

### 消融 C：时序 suppress 是否成立

- front-end fusion + `B0`
- front-end fusion + `B0 + B1`

### 消融 D：讨论性消融

- `camera-first confirmed` 极窄正样本
- 说明“存在可行信号，但尚未到可做主结果的阶段”

## 4.7 论文里的语气建议

### 推荐语气

- `B0 further validates ...`
- `B1 provides an additional temporal suppression mechanism ...`
- `These extensions support the main pipeline while also revealing the boundary of the current system.`

### 不推荐语气

- `We solve blind-spot compensation by B0/B1`
- `We build a full camera-first 3D detection framework`

这两种写法都容易把论文拉到当前还没完全做完的问题上。

## 4.8 最适合保留的一句话

可以考虑把 `B0/B1` 总结成下面这一句：

> Beyond front-end fusion, B0 and B1 show that image evidence can also be exploited at the decision and temporal levels, where B0 validates image-led keep/drop and B1 provides a stable temporal suppression gain, especially for low-support false keeps.
