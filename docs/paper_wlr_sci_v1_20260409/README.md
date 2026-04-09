# WLR SCI 论文整理包 v1

这套整理包的目标，是把当前项目里已经足够稳定、可以支撑一篇 SCI 三区/四区论文的内容先收成一条清晰主线，同时把 `B0/B1`、`camera-first birth` 和后续融合阶段改进单独放到合适位置，避免写论文时把“已成立结论”和“仍在探索内容”混在一起。

## 当前总判断

- 当前最适合投稿的主线，不是“盲区补框已经完全解决”，而是：
  - `PointPillars + ImageToBEV + ROI + BEVFusion` 这条路侧前端特征级融合主线已经成立。
  - `DAIR-V2X-I` 已经把“图像能稳定参与并主要抑制误检”这件事证明清楚。
  - `WLR-733` 已经把这条主线落到自有数据上，并补齐了同步、标定、`cam_matched`、评测链和可视化链。
- `B0/B1` 适合放到论文中，但更合适的定位是：
  - `B0` 作为图像主导 keep/drop 的机制验证和单帧扩展。
  - `B1` 作为时序 suppress 的机制验证和消融支撑。
- `camera-first birth`、广义 blind-spot compensation、稳定的 `YOLO -> 3D` 闭环，当前更适合写成后续工作，而不是本次论文主结果。

## 建议阅读顺序

1. [01_main_storyline.md](./01_main_storyline.md)
2. [02_paper_outline.md](./02_paper_outline.md)
3. [03_experiment_evidence.md](./03_experiment_evidence.md)
4. [04_b0_b1_ablation_plan.md](./04_b0_b1_ablation_plan.md)
5. [05_future_fusion_direction.md](./05_future_fusion_direction.md)
6. [06_ppt_update_notes.md](./06_ppt_update_notes.md)
7. [07_asset_links.md](./07_asset_links.md)

## 这套整理包主要参考的现有材料

- [/root/组会.pptx](/root/组会.pptx)
- [/root/组会.md](/root/组会.md)
- [/root/组会2.md](/root/组会2.md)
- [/root/组会3.md](/root/组会3.md)
- [/root/OpenPCDet/docs/组会4.md](/root/OpenPCDet/docs/组会4.md)
- [/root/OpenPCDet/docs/WLR_MAINLINE_STATUS_CN.md](/root/OpenPCDet/docs/WLR_MAINLINE_STATUS_CN.md)
- [/root/OpenPCDet/docs/WLR_NEXT_ARCH.md](/root/OpenPCDet/docs/WLR_NEXT_ARCH.md)
- [/root/OpenPCDet/docs/WLR_CAMERA_FIRST_BIRTH_PLAN.md](/root/OpenPCDet/docs/WLR_CAMERA_FIRST_BIRTH_PLAN.md)
- [/root/DAIR_B0_B1主线结论_20260319.md](/root/DAIR_B0_B1主线结论_20260319.md)
- [/root/WLR项目阶段总结_20260320.md](/root/WLR项目阶段总结_20260320.md)

## 使用建议

- 如果目的是尽快起草论文，优先只采用 `01 + 02 + 03`。
- 如果目的是继续完善组会/PPT，把 `06` 一起看。
- 如果目的是边写论文边规划下一代结构，把 `05` 一起看。
- 如果需要找图、点云、结果目录，直接看 `07`。
