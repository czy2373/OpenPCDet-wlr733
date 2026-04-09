# 3. 当前可直接用于论文的实验依据

本文件只保留当前最值得进入论文的结果，不追求把所有过程性实验全部抄进来。

## 3.1 主结论先行

### 结论 A：前端特征级融合主线已成立

- `DAIR-V2X-I` 已经证明：
  - 图像分支不是“接上了但无效”。
  - `alpha` 的变化会稳定影响结果。
  - 当前收益主要体现在误检抑制，而不是显著补回 `Recall`。

### 结论 B：`WLR-733` 上的主要瓶颈并不只是融合强度

- 当主线迁移到 `WLR-733` 后，更大的瓶颈逐渐转移到：
  - 时间同步
  - 标定
  - 投影误差
  - 图像配对
  - 评测链口径

### 结论 C：`B0/B1` 有价值，但更适合做扩展和消融

- `B0` 已证明图像可以直接参与 keep/drop。
- `B1` 已证明时序 suppress 有明确价值。
- 但当前最稳的主叙事仍然是前端融合主线本身。

## 3.2 DAIR: 主线成立的核心证据

主要材料来源：

- [/root/组会3.md](/root/组会3.md)
- [/root/DAIR_B0_B1主线结论_20260319.md](/root/DAIR_B0_B1主线结论_20260319.md)
- [/root/B0_DAIR图像主导验证记录_20260319.md](/root/B0_DAIR图像主导验证记录_20260319.md)

### 3.2.1 `alpha` 扫描

| alpha | TP | FP | FN | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.00 | 20056 | 3935 | 8817 | 0.836 | 0.695 |
| 0.01 | 20067 | 3813 | 8806 | 0.840 | 0.695 |
| 0.03 | 20082 | 3216 | 8791 | 0.862 | 0.696 |
| 0.05 | 20084 | 2789 | 8785 | 0.878 | 0.696 |
| 0.10 | 19977 | 3111 | 8901 | 0.865 | 0.692 |

建议论文中的写法：

- 当 `alpha` 从 `0.00` 提高到 `0.05` 时，`Precision` 持续提升、`FP` 持续下降。
- 当 `alpha` 继续增大到 `0.10` 后，`Precision` 开始回落，说明图像注入过强时会将噪声带入 `BEV`。
- 因而，这条主线在 `DAIR` 上不仅成立，而且呈现出清晰可解释的融合强度趋势。

### 3.2.2 `B0/B1` 的机制证据

| 版本 | TP | FP | FN | Precision | Recall | 说明 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Raw lidar-only | 20071 | 854 | 8817 | 0.959188 | 0.694787 | 对齐后的稳定 proposal 基线 |
| B0 | 19824 | 763 | 9064 | 0.962938 | 0.686236 | 当前最佳图像主导单帧工作点 |
| 原始 B1 | 19849 | 813 | 9039 | 0.960652 | 0.687102 | `Recall` 略升，但 `FP` 回涨 |
| B1 keepguard_r50 | 19818 | 774 | 9070 | 0.962413 | 0.686029 | 时序保护集中作用在 `50m+` |

建议论文中的写法：

- `B0` 证明“图像主导最终保留/抑制、LiDAR 提供 3D 几何”这条路线成立。
- `B1` 证明短时序一致性确实能作用于远距连续目标，但截至当前还没有整体超过 `B0`。

## 3.3 WLR: 工程落地与主线结果

主要材料来源：

- [/root/组会.md](/root/组会.md)
- [/root/OpenPCDet/docs/WLR_MAINLINE_STATUS_CN.md](/root/OpenPCDet/docs/WLR_MAINLINE_STATUS_CN.md)
- [/root/OpenPCDet/docs/WLR_NEXT_ARCH.md](/root/OpenPCDet/docs/WLR_NEXT_ARCH.md)
- [/root/WLR项目阶段总结_20260320.md](/root/WLR项目阶段总结_20260320.md)

### 3.3.1 `WLR` 前端主线的核心定位

当前正式入口统一为：

- 配置：[pp_wlr_main.yaml](/root/OpenPCDet/tools/cfgs/kitti_models/pp_wlr_main.yaml)
- 结果根目录：[pp_wlr_main](/root/OpenPCDet/output/cfgs/kitti_models/pp_wlr_main)
- 可视化目录：[wlr_main_vis_20260402](/root/OpenPCDet/tools/output/wlr_main_vis_20260402)

这条线的含义是：

- 前端主线仍然是 `PointPillars + ImageToBEV + ROI + BEVFusion`
- 正式图像口径统一使用 `cam_matched`
- `B0/B1` 已作为主线扩展接进评测链

### 3.3.2 当前正式主线结果

基于 [default 主线结果目录](/root/OpenPCDet/output/cfgs/kitti_models/pp_wlr_main/blindspot_birth_cammatched_top2fixed_20260331/eval/epoch_15/val_mid20/default)：

- `TP = 288`
- `FP = 76`
- `FN = 1004`
- `Precision = 0.791`
- `Recall = 0.223`

对应审计摘要：

- [B0 summary](/root/OpenPCDet/output/cfgs/kitti_models/pp_wlr_main/blindspot_birth_cammatched_top2fixed_20260331/eval/epoch_15/val_mid20/default/image_led_b0/summary.json)
- [B1 summary](/root/OpenPCDet/output/cfgs/kitti_models/pp_wlr_main/blindspot_birth_cammatched_top2fixed_20260331/eval/epoch_15/val_mid20/default/image_led_b1/summary.json)

该结果更适合这样解释：

- 前端融合主线是稳定的。
- `B0/B1` 已接进正式评测链。
- 盲区补框入口仍然保持保守，没有把 image-only 线宽放进主指标。

### 3.3.3 前端 only vs `B0/B1` 扩展

来自 [/root/组会.md](/root/组会.md) 的压缩对比：

| 版本 | TP | FP | FN | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| 前端 detector 输出，不走 `B0/B1` | 283 | 81 | 1009 | 0.777 | 0.219 |
| 前端 + `B0 + B1` | 288 | 76 | 1004 | 0.791 | 0.223 |

净变化：

- `+5 TP`
- `-5 FP`
- `-5 FN`

这一组数字很适合论文里用一句话总结：

> 在 `WLR-733` 的当前正式口径下，`B0/B1` 扩展在不改动前端主检测器结构的前提下，对最终结果带来了小幅但稳定的正向改进。

### 3.3.4 `B0/B1` 在 `WLR` 上的更细结论

#### `B0`

在 [WLR 下一步结构改造建议](/root/OpenPCDet/docs/WLR_NEXT_ARCH.md) 中，当前一组较稳定的 promotion-only 门控结果为：

- raw: `TP=292, FP=67, FN=1000`
- `B0 promote015`: `TP=293, FP=67, FN=999`

这说明：

- `B0` 已不再是“能不能接进主线”的问题。
- 当前更重要的是区分 safe 默认线和 experiment 线。

#### `B1`

`B1 temporal suppress` 当前最稳定的结论不是 rescue，而是 suppress：

- 在正式主链上已实现 `0 TP / -6 FP` 的稳定减误检收益。
- 被 suppress 的坏 keep 模式非常一致：
  - `support_hist = 0`
  - `matched_yolo_conf = 0`
  - `max_gt_iou = 0`

这个结论非常适合论文写作，因为它是清楚、可解释、可审计的。

## 3.4 `camera-first confirmed` 的当前定位

主要材料来源：

- [/root/OpenPCDet/docs/组会4.md](/root/OpenPCDet/docs/组会4.md)
- [/root/OpenPCDet/docs/WLR_CAMERA_FIRST_BIRTH_PLAN.md](/root/OpenPCDet/docs/WLR_CAMERA_FIRST_BIRTH_PLAN.md)
- [camera_first_confirmed_strict_20260402c 结果目录](/root/OpenPCDet/output/cfgs/kitti_models/pp_wlr_main_camera_first_confirmed/default/eval/epoch_15/val_mid20/camera_first_confirmed_strict_20260402c)

当前最重要的事实只有两点：

1. `camera-first confirmed` 第一版极窄规则已经跑通。
2. 但它仍然是“后续研究方向的最小正样本”，不是当前论文该大讲特讲的主结果。

可以保留在论文 Discussion 里的一句话版本：

> We also observed the first workable camera-first confirmed case under a very strict rule set, suggesting the feasibility of camera-led existence modeling; however, this line remains too narrow to be treated as a stable main result in the current paper.

## 3.5 当前最适合进论文的图表清单

### 正文推荐

- `DAIR alpha` 扫描表
- `DAIR lidar-only vs fusion` 主结果表
- `WLR` 前端 only vs `B0/B1` 扩展对比表
- 一张 `WLR cam_matched` 可视化图
- 一张 `B1 suppress` 的典型案例图

### 讨论/附录推荐

- `B0` keep/drop 审计图
- `camera-first confirmed` 的单例示意图
- `birth_audit.csv` 分桶截图
