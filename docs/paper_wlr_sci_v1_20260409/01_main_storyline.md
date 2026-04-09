# 1. 论文主线收口

## 1.1 一句话主线

本文最适合收成的主线是：

> 面向路侧感知场景，在 `PointPillars` 的 `LiDAR-BEV` 主干上，引入基于 `ImageToBEV + ROI + BEVFusion` 的前端特征级融合，并在自有 `WLR-733` 数据上通过同步、标定、`cam_matched` 和评测链收口，验证该主线能够稳定利用图像语义抑制误检、提升整体结果可信度。

## 1.2 本次论文最适合强调的 3 个贡献点

### 贡献 1：面向路侧场景的前端特征级融合主线

- 保留 `PointPillars` 的几何主干。
- 使用轻量图像主干提取视觉特征。
- 通过 `ImageToBEV` 将图像特征投影到与 `LiDAR` 一致的 `BEV` 网格。
- 通过 `ROI-aware` 融合让图像语义在有效区域内稳定注入 `LiDAR-BEV`。

这一点对应的核心不是“做了一个很复杂的新网络”，而是把图像语义以可解释、可控、可对齐的方式接进了路侧 `PointPillars` 主线。

### 贡献 2：公开数据和自有数据的双层验证口径

- `DAIR-V2X-I` 用来回答“方法本身是否成立”。
- `WLR-733` 用来回答“这条方法能否在真实自有路侧数据上稳定落地”。

这种双层口径很适合论文写法，因为它天然避免了两类质疑：

- 只在自有数据上有效，泛化性不足。
- 只在公开数据上成立，但在真实落地场景里工程链条不完整。

### 贡献 3：将路侧融合从“模型模块”推进为“完整可复核链路”

论文里可以明确写出：对 `WLR-733` 而言，决定结果能否可信的，不只是融合模块本身，还包括：

- 时间同步
- 空间标定
- `cam_matched` 图像口径
- `ROI`
- 评测与可视化链路

这里的价值不是把工程问题硬凑成贡献，而是说明：

> 对路侧 LiDAR-camera 融合而言，几何底座不稳时，图像分支即使结构正确，也可能在投到 `BEV` 后变成噪声。

## 1.3 本次论文不宜当作主贡献的内容

以下内容建议写进扩展实验、讨论或后续工作，而不要拿来撑主标题：

- 宽放开的 blind-spot compensation
- 广义 image-only birth
- 完整的 `camera-first existence + LiDAR-assisted geometry` 正式系统
- 稳定的 `YOLO -> 3D` 补框闭环
- 大量后处理规则堆叠后的最终最优点

原因很简单：这些内容现在已经有方向，但还没有收敛成最稳、最干净、最适合投稿主结果的形式。

## 1.4 论文主叙事建议

建议按下面这条叙事线来写：

1. 路侧场景中，`LiDAR` 几何稳定但语义不足，图像语义强但受同步、标定、投影误差影响大。
2. 因此需要一种能够把图像语义稳健注入 `LiDAR-BEV` 的前端特征级融合主线。
3. 在公开路侧数据 `DAIR-V2X-I` 上，这条主线已经证明成立，且主要收益体现在误检抑制。
4. 在真实自有数据 `WLR-733` 上，进一步把同步、标定、`cam_matched` 和评测链收口后，该主线能够稳定运行。
5. 进一步的 `B0/B1` 验证说明：图像不仅能做前端语义补充，还能在单帧 keep/drop 和时序 suppress 上发挥作用。
6. 但系统最难的问题仍然是 blind-spot / camera-first 补框，因此这部分留作后续研究方向。

## 1.5 一句更适合写进摘要的版本

可以考虑把摘要主句压成下面这种风格：

> We study roadside LiDAR-camera fusion on top of PointPillars, and build a BEV-aligned image injection pipeline with ImageToBEV, ROI-aware fusion, and a complete synchronization-calibration-evaluation workflow. Experiments on DAIR-V2X-I and WLR-733 show that the proposed pipeline stably changes detector outputs and mainly improves precision by suppressing false positives, while B0/B1 further validate the value of image-led decision and temporal consistency as extension modules.
