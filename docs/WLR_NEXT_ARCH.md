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

#### 阶段 1 完成后，info 至少应该长什么样

建议把“最低可用 schema”先收敛到下面这组字段，不要一开始就加太多花哨 meta：

```python
info = {
    "point_cloud": {
        "lidar_idx": "1802/000123",
        "num_features": 4,
    },
    "image": {
        "cam_frame_id": "000456",
        "image_path": "000456.jpg",      # 相对 IMAGE_ROOT
        "image_shape": [1080, 1920],     # [H, W]，没有就先写 null
    },
    "sync": {
        "lidar_frame_id": "000123",
        "cam_frame_id": "000456",
        "source": "cam_lidar_match_canonical.csv",
    },
    "calib": {
        "cam_K": ...,
        "T_cam_from_lidar": ...,
        "candidate_name": "native",      # 原始 info 建议写 native
        "calib_source": "per_frame_npz", # 或 global_calib_npz / candidate_eval_info
        "cam_k_source": "per_frame_npz", # 或 global_calib_npz / candidate
    },
    "annotations": {
        ...
    }
}
```

字段约定建议明确成下面这样：

| 字段 | 建议来源 | 为什么现在就要收进 info |
| --- | --- | --- |
| `image.cam_frame_id` | `frame_map_csv`；如果没有外部映射，就退化成与 lidar 同名 | B0/B1、可视化、YOLO 对齐都需要它 |
| `image.image_path` | 相对 `IMAGE_ROOT` 的路径，例如 `000456.jpg` | 后处理和 dataset 不必再自己猜图像文件名 |
| `image.image_shape` | 从真实图像读取；如果成本高，第一版可先写 `null` | 投影和审计时能避免重复读图拿尺寸 |
| `sync.lidar_frame_id` | `lidar_idx` 去目录和后缀后的规范 id | 后处理统一按规范 key 对齐 |
| `sync.cam_frame_id` | 同 `image.cam_frame_id` | 显式表达同步关系，而不是藏在文件名假设里 |
| `sync.source` | 例如 `cam_lidar_match_canonical.csv` / `identity_sync_filename` | 后续审计能知道同步关系来自哪里 |
| `calib.candidate_name` | 原始同步 info 写 `native`；候选评测 info 写 `baseline` / `top_01` 等 | 后处理和 paper table 直接知道当前跑的是哪套标定 |
| `calib.calib_source` | `per_frame_npz` / `global_calib_npz` / `candidate_eval_info` | 后面排查投影误差时能快速知道标定口径 |
| `calib.cam_k_source` | `per_frame_npz` / `global_calib_npz` / `candidate` | 区分外参与内参是不是来自同一路径 |

这里最重要的原则是：

- `cam_frame_id` 不再只存在于外部 `frame_map_csv`
- `image_path` 不再只靠 `_find_image_path(key)` 猜
- `candidate_name` 和 `calib_source` 不再只存在于文件名或命令行上下文

#### 当前代码的真实缺口

目前这三个入口文件其实已经能覆盖阶段 1 的绝大部分工作，但职责还没收拢：

- [`tools/prepare_wlr733_sync_splits.py`](../tools/prepare_wlr733_sync_splits.py) 目前只负责生成 split，并把 `IMAGE_ROOT` 塞进 dataset config，然后直接调用 `generate_infos(split_name)`。
- [`pcdet/datasets/wlr733/wlr733_dataset.py`](../pcdet/datasets/wlr733/wlr733_dataset.py) 里的 `generate_infos()` 当前只写了 `point_cloud` 和 `annotations`，没有把图像、同步、标定元信息写进去。
- [`tools/prepare_wlr_candidate_eval_infos.py`](../tools/prepare_wlr_candidate_eval_infos.py) 里的 `clone_infos_with_calib()` 目前只覆盖 `calib.cam_K` 和 `calib.T_cam_from_lidar`，没有把 `candidate_name`、`calib_source`、`cam_k_source` 补齐。
- [`tools/rescore_wlr_with_yolo.py`](../tools/rescore_wlr_with_yolo.py) 之所以还必须传 `--frame_map_csv` 和 `--image_dir`，本质上就是因为前面的 info schema 还不够完整。

也就是说，阶段 1 不是要发明一套全新系统，而是把已经散落在脚本参数、文件名假设和外部 csv 里的信息，正式回收到 `info.pkl` 里。

#### 文件级改造清单

下面这份清单是按“改完就能直接开始减外部依赖”的顺序排列的。

##### 1. 改 `pcdet/datasets/wlr733/wlr733_dataset.py`

优先级最高，因为 `generate_infos()` 是所有 WLR info 的统一出口。

建议直接修改这些现有函数：

| 函数 | 建议修改内容 |
| --- | --- |
| `generate_infos()` | 这是阶段 1 的主改造点。除了 `point_cloud` / `annotations`，要补出 `image`、`sync`、`calib` 三个 block。 |
| `_find_image_path()` | 先读 `info["image"]["image_path"]`，只有 legacy info 没这个字段时，才回退到按 `key + ext` 猜文件。 |
| `_load_frame_calib()` | 继续优先读取 `info["calib"]`，但补充对 `candidate_name` / `calib_source` 的透传和兜底逻辑。 |
| `__getitem__()` | 除了 `images` / `cam_K` / `T_cam_from_lidar`，建议把 `cam_frame_id`、`image_path`、`candidate_name`、`calib_source` 一起塞进 `input_dict`，方便 debug 和后处理。 |
| `__init__()` | 当前“过滤没有匹配图像的样本”这一步只按 lidar id 猜图。阶段 1 后建议优先依赖 info 里的 `image.image_path`。 |

`generate_infos()` 里建议实际补的内容：

1. 规范化 `lidar_idx`，拆出 `lidar_frame_id`
2. 查到对应的 `cam_frame_id`
3. 根据 `cam_frame_id` 解析 `image_path`
4. 如果能找到图像，就读取 `image_shape`
5. 读取 per-frame calib 或 global calib，并写入 `calib_source`
6. 对原始同步 info，统一写 `candidate_name="native"`

为了让实现更干净，建议在这个文件里新增 2 到 4 个小 helper，而不是把所有逻辑都塞进 `generate_infos()`：

- `_resolve_cam_frame_id(self, lidar_id, sync_map=None)`
- `_resolve_info_image_path(self, cam_frame_id)`
- `_build_sync_meta(self, lidar_frame_id, cam_frame_id, sync_source)`
- `_build_calib_meta(self, sid)`

这些 helper 不是必须的，但会显著降低 `generate_infos()` 继续失控变长的风险。

##### 2. 改 `tools/prepare_wlr733_sync_splits.py`

这个脚本当前最大的限制是：它生成了“同步 split”，但没有把“同步关系本身”写进 info。

建议直接修改这些位置：

| 位置 | 建议修改内容 |
| --- | --- |
| `main()` | 增加可选参数 `--frame_map_csv` 和 `--sync_source_name`。如果提供了 csv，就把 `lidar -> cam` 对应关系读进来。 |
| `save_infos()` | 不再只裸调 `ds.generate_infos(split_name)`，而是把 `sync_map`、`sync_source_name` 一起传进去。 |

建议新增一个轻量 helper：

- `load_lidar_to_cam_map(csv_path: Path)`：逻辑可以直接参考 [`tools/rescore_wlr_with_yolo.py`](../tools/rescore_wlr_with_yolo.py) 里的同名思路

这一改完成后，`train_sync` / `val_sync` / `test_sync` 生成出来的 info 就不只是“有同步图像的 split”，而是“显式记录了同步对应关系的 split”。

##### 3. 改 `tools/prepare_wlr_candidate_eval_infos.py`

这个脚本阶段 1 的任务不是“继续加新候选逻辑”，而是把候选口径补写到 schema 里。

建议直接修改这些现有函数：

| 函数 | 建议修改内容 |
| --- | --- |
| `clone_infos_with_calib()` | 不要只重写 `cam_K` 和 `T_cam_from_lidar`，要同时补上 `candidate_name`、`calib_source`、`cam_k_source`，并保留已有 `image` / `sync` block。 |
| `main()` | 输出 summary 时把 `candidate_name`、`calib_source` 一并写进 summary row，方便后面对账。 |
| `candidate_cam_k()` | 保持现有逻辑即可，但返回结果最终要体现在 `cam_k_source` 字段上。 |

这里最关键的一点是：`clone_infos_with_calib()` 不应该把原始 info 退化成“只剩 calib 的壳”，它应该是“保留原始 frame meta，再覆盖当前候选 calib”。

建议写出的口径：

- 原始同步 info：`candidate_name="native"`
- 候选 baseline：`candidate_name="baseline"`
- top-k 候选：`candidate_name="top_01"` / `top_03` ...
- `calib_source="candidate_eval_info"`
- `cam_k_source="candidate"` 或 `global_calib_npz`

##### 4. 阶段 1 完成后，再回头改 `tools/rescore_wlr_with_yolo.py`

这一步不一定要和阶段 1 同一个 commit 做，但建议紧接着做兼容改造。

优先读取顺序建议改成：

1. 先从 `info["image"]["cam_frame_id"]` 读相机帧
2. 再从 `info["image"]["image_path"]` 读图像路径
3. 只有 legacy info 缺字段时，才回退到 `--frame_map_csv` 和 `--image_dir`

这样做完以后，WLR-B0 才真正开始摆脱“外部 csv 驱动”的实验脚本状态。

#### 推荐的最小实现顺序

如果只想先做一版最小闭环，建议按下面顺序落地：

1. 先改 `wlr733_dataset.py::generate_infos()`，确保新 info schema 能生成出来
2. 再改 `prepare_wlr733_sync_splits.py`，让同步 split 的 info 真正带上 `cam_frame_id` / `sync.source`
3. 再改 `prepare_wlr_candidate_eval_infos.py`，保证 candidate info 不会把前面的 meta 覆盖丢失
4. 最后再让 `rescore_wlr_with_yolo.py` 优先消费新 schema

这样每一步都是可验证的，不会陷入“大改一半但没有产物可用”的状态。

#### 阶段 1 的验收标准

只要满足下面这几条，就说明阶段 1 已经真正完成，而不是“多写了几个字段”：

1. 随便打开一个新的 `wlr733_infos_*.pkl`，每帧都能看到 `image`、`sync`、`calib` 三个 block。
2. `image.cam_frame_id` 和 `sync.cam_frame_id` 是一致的，不再需要额外猜。
3. `calib.candidate_name` 和 `calib.calib_source` 能直接告诉你当前评测跑的是哪套标定口径。
4. [`tools/rescore_wlr_with_yolo.py`](../tools/rescore_wlr_with_yolo.py) 在面对新 info 时，可以不依赖 `frame_map_csv` 完成大部分对齐。
5. [`pcdet/datasets/wlr733/wlr733_dataset.py`](../pcdet/datasets/wlr733/wlr733_dataset.py) 的 `__getitem__()` 不再只能靠 lidar frame id 猜图像文件。

这一步一旦做完，后面的 B0 / B1 模块化才有真正稳定的输入层。

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
