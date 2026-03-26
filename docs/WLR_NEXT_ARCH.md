# WLR 下一步结构改造建议

这份文档用于把 `WLR-733` 端下一步的结构改造目标写清楚，避免后续开发继续停留在“继续调前端融合参数”而没有真正推进系统结构演化。

适用背景：

- 当前 `PointPillar + ImageToBEV + BEVFusion` 前端融合主线已经能跑通
- `DAIR-V2X-I` 上主线已经验证过“图像能带来稳定的误检抑制收益”
- `WLR-733` 继续推进时，主要瓶颈逐渐转移到同步、标定、投影误差、图像配对和后验决策层

核心结论：

- WLR 下一步不应该只是继续堆前端融合模块
- 更合理的方向是从“单一前端融合”走向“分层协同结构”

## 1. 当前问题的本质

结合当前项目阶段，WLR 端的主要问题已经不再是“图像有没有接进去”，而是：

- 同步关系是否稳定
- LiDAR frame 与 camera frame 的映射是否可信
- 每帧标定是否足够稳定
- 图像投到 BEV 后是不是因为几何误差而变成噪声
- 哪些职责应该由前端融合承担，哪些职责应该拆到后端图像决策层和时序层

因此，继续只在 `BEV_FUSION`、`alpha`、`gate` 上反复调参，收益会越来越有限。

## 2. 推荐目标结构

建议把 WLR 主线明确拆成下面四层：

```text
WLR 原始数据 / 同步资产 / 标定候选
        |
        v
[几何底座层 Geometry Base]
- split / frame map / per-frame calib / image path / candidate calib
- 统一写入 info.pkl 与 dataset meta
        |
        v
[主检测层 Main Detector]
- PointPillar
- TinyImgBackbone
- ImageToBEV
- BEVFusion
- ROI / track ROI
        |
        v
pred_dicts / det_annos / result.pkl
        |
        v
[后端图像主导决策层 B0]
- YOLO 2D evidence
- keep/drop rescoring
- 近场 / 低分框 / 图像支持复核
        |
        v
[时序一致性层 B1]
- short track support
- rescue / suppress
- keep-protect
        |
        v
[最终评测层 Evaluation]
- dataset.evaluation(...)
- 审计 / 可视化 / paper table
```

这个结构的重点不在于“加更多模块”，而在于“把职责拆开”：

- 几何问题由几何底座层负责
- 语义增强由前端融合负责
- keep/drop 决策由后端图像主导层负责
- 连续性问题由时序层负责

## 3. 当前代码分别落在哪些层

### 3.1 几何底座层

当前已经有基础，但还没有完全统一收口：

- [`tools/prepare_wlr733_sync_splits.py`](../tools/prepare_wlr733_sync_splits.py)
- [`tools/prepare_wlr_candidate_eval_infos.py`](../tools/prepare_wlr_candidate_eval_infos.py)
- [`pcdet/datasets/wlr733/wlr733_dataset.py`](../pcdet/datasets/wlr733/wlr733_dataset.py)

已经具备的能力：

- 同步 split 生成
- `infos.pkl` 生成
- `IMAGE_ROOT` 支持
- 按帧标定读取
- 候选标定 info 注入

当前不足：

- `cam_frame_id`
- `image_path`
- `sync source`
- `candidate_name`
- `calib_source`

这些信息还没有形成统一 schema 写入每帧 info，导致后处理链路还依赖外部 `frame_map_csv`。

### 3.2 前端融合层

这是目前最成熟的主线：

- [`pcdet/models/detectors/pointpillar.py`](../pcdet/models/detectors/pointpillar.py)
- [`tools/cfgs/kitti_models/pointpillar_S1.yaml`](../tools/cfgs/kitti_models/pointpillar_S1.yaml)
- [`tools/cfgs/kitti_models/pointpillar_S2.yaml`](../tools/cfgs/kitti_models/pointpillar_S2.yaml)
- [`tools/cfgs/kitti_models/pointpillar_S2_imgboost_v2.yaml`](../tools/cfgs/kitti_models/pointpillar_S2_imgboost_v2.yaml)

已经具备的能力：

- `TinyImgBackbone`
- `ImageToBEV`
- `BEVFusion`
- `CALIB_REFINE`
- `fusion ROI`
- `track ROI`
- `alpha` / `gate` / debug 输出

这层建议继续保留，但不再无限膨胀。

### 3.3 后端图像主导决策层 B0

目前 WLR 已经有离线脚本版：

- [`tools/run_yolo_to_csv.py`](../tools/run_yolo_to_csv.py)
- [`tools/rescore_wlr_with_yolo.py`](../tools/rescore_wlr_with_yolo.py)

现状判断：

- 已经有功能
- 但还不是正式主线模块
- 当前更像独立实验脚本链路

一个关键现象是：虽然已经有

- [`tools/cfgs/kitti_models/pointpillar_S2_imgboost_v2_imageled_wlr.yaml`](../tools/cfgs/kitti_models/pointpillar_S2_imgboost_v2_imageled_wlr.yaml)
- [`tools/cfgs/kitti_models/pointpillar_S2_imgboost_v2_imageled_wlr_rescue.yaml`](../tools/cfgs/kitti_models/pointpillar_S2_imgboost_v2_imageled_wlr_rescue.yaml)

这些配置，但代码里目前并没有真正把 `MODEL.IMAGE_LED` 作为正式 detector 或 evaluation 逻辑消费。

### 3.4 时序一致性层 B1

WLR 端当前还没有正式版本，但 DAIR 端已经有成熟参考：

- [`tools/rescore_dair_with_yolo_temporal.py`](../tools/rescore_dair_with_yolo_temporal.py)
- [`tools/iou_tracker.py`](../tools/iou_tracker.py)

这说明 WLR 的 B1 不需要从零设计，更合理的做法是复用 DAIR temporal 链路里的通用逻辑，只做 WLR 适配。

## 4. 推荐目录结构

为了避免继续把所有逻辑都塞进 `pointpillar.py`，建议把结构整理成下面这样：

```text
pcdet/
├── datasets/
│   └── wlr733/
│       ├── wlr733_dataset.py
│       └── wlr733_meta.py              # 建议新增
├── postprocess/                        # 建议新增
│   ├── __init__.py
│   ├── image_led_base.py
│   ├── wlr_image_led_b0.py
│   └── wlr_temporal_b1.py
└── models/
    └── detectors/
        └── pointpillar.py

tools/
├── prepare_wlr733_sync_splits.py
├── prepare_wlr_candidate_eval_infos.py
├── run_yolo_to_csv.py
├── rescore_wlr_with_yolo.py           # 逐步变成 thin wrapper / debug 入口
├── rescore_wlr_with_yolo_temporal.py  # 后续建议新增
└── overlay_pred_on_image.py
```

## 5. 每一层应该承担什么职责

### 5.1 `pcdet/datasets/wlr733/`

职责：

- 提供统一样本元信息
- 统一 frame id / cam frame id / image path / calib / candidate calib source
- 给训练、评测、B0、B1 提供一致输入

建议的 info schema：

```python
info = {
    "point_cloud": {
        "lidar_idx": "1802/000123",
        "num_features": 4,
    },
    "image": {
        "cam_frame_id": "000456",
        "image_path": "cam_matched_sync_v1/000456.jpg",
        "image_shape": [1080, 1920],
    },
    "sync": {
        "lidar_frame_id": "000123",
        "cam_frame_id": "000456",
        "source": "canonical_frame_map_v1",
    },
    "calib": {
        "cam_K": ...,
        "T_cam_from_lidar": ...,
        "candidate_name": "baseline",
        "calib_source": "candidate_eval_info",
    },
    "annotations": {
        ...
    }
}
```

改造完成后，WLR 后处理尽量不要再额外依赖 `frame_map_csv`。

### 5.2 `pcdet/models/detectors/pointpillar.py`

职责：

- 只负责前端融合和检测
- 不再承担 YOLO 读取、外部 CSV 读取、时序追踪、B0/B1 后处理逻辑

建议保留：

- `USE_FUSION`
- `IMAGE_TO_BEV`
- `BEV_FUSION`
- `CALIB_REFINE`
- `ROI`

建议不要继续往这里放：

- B0 的 YOLO 匹配规则
- B1 的 temporal support / rescue 规则
- 外部 `FRAME_MAP_CSV` 或 `YOLO_CSV` 路径

### 5.3 `pcdet/postprocess/`

这是 WLR 下一步最值得新增的一层。

职责：

- 接收 `det_annos`
- 读取样本元信息和图像证据
- 输出 `det_annos_b0` / `det_annos_b1`
- 让 evaluation 可以直接对后处理后的结果进行评测

建议拆成三部分：

#### `image_led_base.py`

职责：

- 通用投影
- 通用 2D 匹配
- 通用 score 规则
- 通用可视化和审计字段辅助函数

#### `wlr_image_led_b0.py`

职责：

- WLR 单帧 image-led 后处理
- 输入：`det_annos + info_map + yolo_map`
- 输出：`rescored_det_annos + audit_rows + summary`

#### `wlr_temporal_b1.py`

职责：

- 在 B0 输出基础上做短时序支持
- 复用 `IoUTracker`
- 输出 B1 score、rescue/suppress 决策和时序审计

## 6. 正式主线里，B0/B1 应该挂在哪

最合适的挂载点不是 `pointpillar.py`，而是 evaluation pipeline：

- [`tools/eval_utils/eval_utils.py`](../tools/eval_utils/eval_utils.py)

当前流程大致是：

```text
model(batch_dict)
-> pred_dicts
-> dataset.generate_prediction_dicts(...)
-> det_annos
-> 写 result.pkl
-> dataset.evaluation(det_annos, ...)
```

建议未来改成：

```text
model(batch_dict)
-> pred_dicts
-> dataset.generate_prediction_dicts(...)
-> det_annos_raw
-> optional WLR B0 postprocess
-> det_annos_b0
-> optional WLR B1 postprocess
-> det_annos_final
-> 写 result.pkl
-> dataset.evaluation(det_annos_final, ...)
```

这样做的好处：

- detector 主干职责清晰
- B0/B1 逻辑独立演化
- 可以灵活比较 raw / B0 / B1 三种结果
- 不需要把 YOLO、时序、审计逻辑塞进 detector forward

## 7. 推荐改造顺序

### 阶段 1：元信息统一

优先修改：

- [`tools/prepare_wlr733_sync_splits.py`](../tools/prepare_wlr733_sync_splits.py)
- [`tools/prepare_wlr_candidate_eval_infos.py`](../tools/prepare_wlr_candidate_eval_infos.py)
- [`pcdet/datasets/wlr733/wlr733_dataset.py`](../pcdet/datasets/wlr733/wlr733_dataset.py)

目标：

- info 里补齐 `cam_frame_id`
- info 里补齐 `image_path`
- info 里补齐 `sync`
- info 里补齐 `candidate_name`
- info 里补齐 `calib_source`

这是 WLR 下一步最值得先做的结构改变，因为它会统一后面所有链路的输入。

### 阶段 2：WLR-B0 模块化

优先修改：

- 新增 `pcdet/postprocess/image_led_base.py`
- 新增 `pcdet/postprocess/wlr_image_led_b0.py`
- 改 [`tools/eval_utils/eval_utils.py`](../tools/eval_utils/eval_utils.py)

目标：

- 让 B0 从独立脚本变成正式评测后处理阶段
- 保留 [`tools/rescore_wlr_with_yolo.py`](../tools/rescore_wlr_with_yolo.py) 作为 debug wrapper 和可视化入口

### 阶段 3：WLR-B1 正式化

优先修改：

- 新增 `pcdet/postprocess/wlr_temporal_b1.py`
- 新增 `tools/rescore_wlr_with_yolo_temporal.py`
- 复用 [`tools/rescore_dair_with_yolo_temporal.py`](../tools/rescore_dair_with_yolo_temporal.py)
- 复用 [`tools/iou_tracker.py`](../tools/iou_tracker.py)

目标：

- 把“时序一致性修正”从 PPT 里的方向，变成 WLR 的正式主线模块

### 阶段 4：盲区补偿 / 低分框复核

建议先不要直接做成训练新 head，而是先做后处理版：

- near-range only
- track ROI only
- low-score candidate only

验证有效后，再决定是否进入 detector 主线或训练流程。

## 8. 当前最值得先做的两件事

如果只允许先做两件事，建议优先级如下：

1. 把 `frame_map + per-frame calib + candidate info` 统一收进 WLR info / dataset 元信息层
2. 在 WLR 上正式做出 `B1 temporal`，而不是继续只停留在单帧 B0

## 9. 一句话原则

WLR 端下一步的结构改变，不应该是继续把图像功能都堆进前端融合层，而应该把系统明确拆成：

- 几何底座层
- 前端融合层
- 后端图像主导决策层
- 时序一致性层

其中最先该做的是：

- 统一 WLR 的同步 / 标定 / frame map 元信息
- 把 WLR-B0/B1 挂到评测后处理链路
- 避免继续把所有逻辑塞进 `pointpillar.py`
