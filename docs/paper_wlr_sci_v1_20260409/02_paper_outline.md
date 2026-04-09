# 2. 论文大纲建议

## 2.1 题目方向

当前更适合的题目风格是“稳主线 + 工程落地”，而不是“盲区补框已经打通”。可以考虑下面几类方向：

### 方向 A：主线型

`Roadside LiDAR-Camera Fusion on PointPillars with BEV-Aligned Image Injection and ROI-Aware Fusion`

### 方向 B：落地型

`A Practical Roadside LiDAR-Camera Fusion Pipeline with BEV-Aligned Image Features and Geometry-Aware Evaluation`

### 方向 C：中文工作题目

`基于 PointPillars 的路侧激光雷达-图像 BEV 对齐融合方法研究`

## 2.2 摘要建议

摘要里只建议放四件事：

1. 路侧场景的问题和动机。
2. 你的主方法：`PointPillars + ImageToBEV + ROI + BEVFusion`。
3. 你的双数据验证口径：`DAIR` + `WLR`。
4. 你的主结论：图像确实稳定参与，主要收益是误检抑制，`B0/B1` 作为扩展进一步验证图像主导和时序一致性价值。

不要在摘要里过早展开：

- `camera-first birth`
- 大量规则名
- 太细的 handoff/schema 细节

## 2.3 正文结构建议

### 第 1 章：Introduction

要回答的问题：

- 路侧 LiDAR-camera 融合为什么值得做。
- 路侧场景相比车载场景为什么更依赖同步、标定和投影质量。
- 现有前端融合为什么往往更擅长语义增强而非直接补漏。

最后一段建议明确本文的边界：

> 本文聚焦于一条可稳定落地的前端特征级融合主线，并通过 `B0/B1` 扩展验证图像主导决策和时序一致性的潜力，但不把 blind-spot camera-first 补框作为本文主结果。

### 第 2 章：Related Work

建议拆成 4 小段：

1. `PointPillars` 与 `BEV` 检测
2. LiDAR-camera 前端特征级融合
3. 路侧感知中的同步、标定与投影问题
4. 图像主导后处理与时序一致性

这里的 Related Work 不需要把 `camera-first birth` 写太重，因为你这次不打算把它作为正式贡献。

### 第 3 章：Method

这一章建议只把正式主线写干净。

可拆成：

1. 整体框架
2. `PointPillars` 几何主干
3. 轻量图像分支
4. `ImageToBEV`
5. `ROI-aware BEVFusion`
6. `WLR` 工程口径：同步、标定、`cam_matched`、评测链
7. 可选：`B0/B1` 扩展模块简述

注意：

- `B0/B1` 在本章可以先只放一小节，避免抢走主方法篇幅。
- 这章里不要把 `camera-first birth` 写成已完成模块。

### 第 4 章：Experiments

建议分成两大部分：

1. `DAIR-V2X-I`
   - 证明主线成立
   - `alpha` 扫描
   - 主结果表
   - 可视化例子
2. `WLR-733`
   - 数据和工程口径说明
   - 主线落地结果
   - 前端 only vs `B0/B1` 扩展
   - 典型可视化

### 第 5 章：Ablation and Discussion

这章是 `B0/B1` 最适合的位置。

建议包含：

- `B0` 单帧图像主导是否成立
- `B1` 时序 suppress 是否有效
- 为什么当前收益主要体现在 `FP` 抑制，而不是显著补回 `Recall`
- 为什么 blind-spot / camera-first 当前仍是开放问题

### 第 6 章：Conclusion

结论只强调三件事：

- 主线成立
- 工程收口必要
- `B0/B1` 提供了扩展方向，但根本问题仍指向更早期的融合/职责拆分

## 2.4 建议插图

论文里最值得保留的图，建议优先做这些：

1. 总体框架图
   - `PointPillars + ImageToBEV + ROI + BEVFusion`
2. `ImageToBEV` 投影示意图
3. `DAIR` 的 `alpha` 对比图
4. `WLR` 上的 `cam_matched` 可视化例图
5. `B0/B1` 的审计式对比示例

不建议论文主文里放太多：

- 大量规则流图
- 太细的目录结构图
- 过多过程性 debug 图

## 2.5 写作上的收放建议

### 该写重的地方

- 前端融合主线
- `DAIR` 上的有效性
- `WLR` 上的工程落地
- “图像主要抑制误检”的稳定结论

### 该写轻的地方

- `B0/B1` 的大量策略细节
- 规则名称和参数名
- `camera-first` 还没收敛的细枝末节

### 不宜写成主结果的地方

- “我们已经彻底解决 blind-spot 补框”
- “我们已经形成完整 image-only 3D 检测系统”

这两点目前都不够稳，写满了会让审稿人追着问最难的那部分。
