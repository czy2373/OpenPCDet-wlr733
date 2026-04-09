# 5. 后续融合阶段改进方向

这份文档不是为本次论文主结果服务，而是对应导师提出的并行任务：

> 在写论文的同时，不要只在后处理层继续补丁式加规则，而是开始把改进重点前移到融合阶段，从根源上重构图像与 `LiDAR` 的职责分工。

## 5.1 为什么当前系统会给人“后面一直打补丁”的感觉

当前系统已经形成了：

- 前端融合主线
- `B0` 单帧后处理
- `B1` 时序后处理
- blind-spot / `camera-first` 审计线

这些工作并不是没有价值，反而说明问题被拆得越来越细。

但它们共同暴露出一个更根本的事实：

> 当前系统中，图像的职责还没有在更早的阶段被清楚建模，因此很多应该在融合阶段就解决的可靠性问题，被拖到了后处理阶段。

典型表现：

- 图像在前端能影响结果，但主要还是辅助修正项。
- 真正困难的样本往往出现在“图像看到了，但 `LiDAR` proposal 不可靠或不存在”。
- 这类问题一旦留到 `B0/B1` 再补，就容易越来越像规则堆叠。

## 5.2 当前最根本的问题不是“图像没接进去”

根据现有结果，当前最根本的问题更准确地说是：

1. 图像是否可信，没有在融合阶段显式建模。
2. 投影误差、同步误差、标定误差，会直接污染图像 `BEV` 特征。
3. 图像负责什么，`LiDAR` 负责什么，没有在结构上彻底拆开。

因此，下一阶段真正值得做的，不是继续把 `B0/B1` 规则越来越复杂，而是把下面三件事前移：

- 图像可靠性建模
- 几何可靠性建模
- existence / geometry 职责拆分

## 5.3 最值得推进的 3 条方向

### 方向 1：做 uncertainty-aware / confidence-aware 融合

当前 `ImageToBEV + BEVFusion` 更像：

- 先投影
- 再做门控注入

但门控本身还没有显式知道：

- 当前帧同步是否可信
- 当前标定是否可信
- 当前投影落点是否稳定
- 当前图像区域是否处于远距/畸变/遮挡状态

下一代融合更适合增加一层显式的 confidence source，例如：

- per-frame calibration confidence
- projection validity / visibility mask
- image support confidence
- range-aware reliability prior

一句话说：

> 不是所有图像特征都该一视同仁地注入 `BEV`，而应该先判断“这份图像证据此刻值不值得相信”。

### 方向 2：把 existence 和 geometry 的职责更早拆开

当前很多后处理规则都指向同一个事实：

- 图像更擅长提供 existence / appearance 证据。
- `LiDAR` 更擅长提供 geometry / 3D box 约束。

这组职责拆分现在已经在 `camera-first birth` 文档里出现，但主要还停留在后处理状态机层。

下一步更值得做的是把这种思想前移到更早阶段，例如：

- 图像分支先输出 candidate existence / objectness
- `LiDAR` 分支负责几何 refinement
- 二者在中间通过显式 query 或 object-level fusion 汇合

这样做的目标不是立刻做完整 image-only 3D 检测，而是减少后端“先没有对象、后面再硬补对象”的被动局面。

### 方向 3：把融合从“静态投影”升级成“几何感知的目标级融合”

当前主线的优点是简单、稳、可解释，但限制也很明显：

- 图像特征主要是栅格级注入
- 很难直接表达“这个 2D 目标和这个弱 3D 几何是不是同一个对象”

因此下一阶段最值得考虑的升级，不一定是更深的图像 backbone，而可能是：

- object-level association
- ROI / proposal-conditioned fusion
- query-based cross-view refinement

也就是说，把融合的一部分从“全局栅格特征混合”推进到“候选目标级别的跨模态确认”。

## 5.4 结合现有代码，最现实的短中期落点

### 短期：不改主干训练，先补融合可靠性审计

可以先做：

- 每帧投影质量统计
- `ImageToBEV` 有效采样比例统计
- ROI 内外图像特征可靠性审计
- 远距/边缘/遮挡区域的注入质量分析

这一步的目的，是先把“哪里在融合阶段就已经坏了”看清楚。

### 中期：在 fusion gate 前增加显式 reliability branch

一种很自然的改法是：

- 不直接把 `img_bev` 原样送进 `BEVFusion`
- 先估一个 `reliability / confidence map`
- 再用它和现有 gate 一起控制注入

这样能够把大量“投影几何不可靠”的噪声提前拦住，而不是等到 `B0/B1` 再兜底。

### 中期：把 `camera-first` 思想从后处理挪到候选级中间层

不是直接做完整 image-only birth，而是先做：

- image-conditioned proposal scoring
- weak proposal geometry refinement
- proposal-level cross-modal confirmation

这一步会比直接做 image-only 3D birth 更稳，也更符合你当前已有基础。

## 5.5 论文里如何写这条“后续改进方向”

建议用下面这种语气：

> The current system already demonstrates the value of image features at the fusion, decision, and temporal levels. However, the remaining hard cases consistently suggest that the responsibilities of image evidence and LiDAR geometry are still resolved too late in the pipeline. Future work will therefore move the improvement focus from rule-heavy post-processing to earlier fusion stages, especially by modeling projection reliability, image confidence, and camera-first existence cues in a geometry-aware manner.

## 5.6 当前最值得并行推进的小任务

如果要边写论文边推进下一代结构，最值得做的不是大改网络，而是先完成下面三项：

1. 给现有 `ImageToBEV / BEVFusion` 增加可靠性审计输出。
2. 做一版最小的 confidence-aware fusion probe。
3. 把 `camera-first existence` 的思想先转成候选级确认模块，而不是继续扩大后处理规则。

这三项都能直接服务后续论文续作，也不会干扰你当前这篇文章的主线收口。
