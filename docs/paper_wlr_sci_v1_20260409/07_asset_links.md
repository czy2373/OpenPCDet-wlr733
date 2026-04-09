# 7. 资产与结果链接说明

这套整理包不复制大文件，只通过软链接和绝对路径记录当前最重要的图片、点云、可视化和结果目录。

## 7.1 本目录下已准备的软链接

软链接目录：

- [assets_links](/root/OpenPCDet/docs/paper_wlr_sci_v1_20260409/assets_links)

建议保留的链接对象：

- `cam_matched`
- `pointclouds`
- `wlr_main_vis_20260402`
- `pp_wlr_main`
- `pp_wlr_main_camera_first_confirmed`
- `group_meeting_pptx`

## 7.2 最常用图片与可视化入口

### 当前组会 PPT

- [/root/组会.pptx](/root/组会.pptx)

### `WLR` 主线可视化

- [/root/OpenPCDet/tools/output/wlr_main_vis_20260402](/root/OpenPCDet/tools/output/wlr_main_vis_20260402)

### `cam_matched` 图像资产

- [/root/pointpillar/cam_matched](/root/pointpillar/cam_matched)

### 点云目录

- [/root/pointclouds](/root/pointclouds)

## 7.3 最关键的结果目录

### 正式主线结果

- [/root/OpenPCDet/output/cfgs/kitti_models/pp_wlr_main/blindspot_birth_cammatched_top2fixed_20260331/eval/epoch_15/val_mid20/default](/root/OpenPCDet/output/cfgs/kitti_models/pp_wlr_main/blindspot_birth_cammatched_top2fixed_20260331/eval/epoch_15/val_mid20/default)

关键文件：

- [image_led_b0/summary.json](/root/OpenPCDet/output/cfgs/kitti_models/pp_wlr_main/blindspot_birth_cammatched_top2fixed_20260331/eval/epoch_15/val_mid20/default/image_led_b0/summary.json)
- [image_led_b1/summary.json](/root/OpenPCDet/output/cfgs/kitti_models/pp_wlr_main/blindspot_birth_cammatched_top2fixed_20260331/eval/epoch_15/val_mid20/default/image_led_b1/summary.json)
- [image_led_b1/birth_audit.csv](/root/OpenPCDet/output/cfgs/kitti_models/pp_wlr_main/blindspot_birth_cammatched_top2fixed_20260331/eval/epoch_15/val_mid20/default/image_led_b1/birth_audit.csv)
- [result.pkl](/root/OpenPCDet/output/cfgs/kitti_models/pp_wlr_main/blindspot_birth_cammatched_top2fixed_20260331/eval/epoch_15/val_mid20/default/result.pkl)

### `camera-first confirmed` 研究线

- [/root/OpenPCDet/output/cfgs/kitti_models/pp_wlr_main_camera_first_confirmed/default/eval/epoch_15/val_mid20/camera_first_confirmed_strict_20260402c](/root/OpenPCDet/output/cfgs/kitti_models/pp_wlr_main_camera_first_confirmed/default/eval/epoch_15/val_mid20/camera_first_confirmed_strict_20260402c)

关键文件：

- [image_led_b1/summary.json](/root/OpenPCDet/output/cfgs/kitti_models/pp_wlr_main_camera_first_confirmed/default/eval/epoch_15/val_mid20/camera_first_confirmed_strict_20260402c/image_led_b1/summary.json)
- [image_led_b1/camera_first_tentative_summary.json](/root/OpenPCDet/output/cfgs/kitti_models/pp_wlr_main_camera_first_confirmed/default/eval/epoch_15/val_mid20/camera_first_confirmed_strict_20260402c/image_led_b1/camera_first_tentative_summary.json)
- [image_led_b1/trajectory_summary.json](/root/OpenPCDet/output/cfgs/kitti_models/pp_wlr_main_camera_first_confirmed/default/eval/epoch_15/val_mid20/camera_first_confirmed_strict_20260402c/image_led_b1/trajectory_summary.json)

## 7.4 关键说明

### 正式论文主结果优先看哪里

优先看下面三类内容：

1. 正式主线结果目录
2. `WLR` 主线可视化目录
3. `DAIR` / `WLR` 的总结文档

### 后续工作优先看哪里

优先看：

- `camera-first confirmed` 目录
- `birth_audit.csv`
- `WLR_CAMERA_FIRST_BIRTH_PLAN.md`

### 不建议直接当论文主结果的内容

- 各类 smoke 目录
- 中间试验目录
- 只用于结构验证的 handoff 结果
