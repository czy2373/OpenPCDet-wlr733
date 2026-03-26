# OpenPCDet-wlr733

基于 OpenPCDet 的路侧 3D 检测与 LiDAR-camera 融合研究仓库。  
这份代码以 `PointPillar` 为主干，围绕两个方向持续扩展：

- `WLR-733` 路侧私有数据链路：同步、标定、投影、融合、图像主导后验重打分
- `DAIR-V2X-I` 公开数据验证：统一数据格式、LiDAR-only baseline、fusion 对照、B0/B1 验证

当前仓库更接近“研究主线代码库”而不是纯净 upstream 镜像：除了保留 OpenPCDet 的训练与评测框架，也加入了本项目自己的数据准备、融合模块、投影检查和图像主导决策脚本。

## 项目定位

这份仓库主要回答三个问题：

1. 能否在 `PointPillar` 主干上，把图像语义稳定注入到 LiDAR BEV 表征中
2. 在公开路侧数据集 `DAIR-V2X-I` 上，这条融合主线是否成立
3. 当图像不只做前端特征增强，而是直接参与最终 keep/drop 决策时，是否能进一步抑制误检

对应到实现上，仓库里目前有两条互补主线：

- 前端特征级融合主线：`Tiny Image Backbone -> ImageToBEV -> BEVFusion -> PointPillar`
- 后端图像主导决策主线：`LiDAR result.pkl -> YOLO 2D evidence -> B0/B1 rescoring`

## 核心特性

- 基于 `PointPillar` 的路侧车辆检测主线
- 新增轻量图像 backbone、`ImageToBEV` 投影和 `BEVFusion`
- 支持 ROI-aware 融合、门控注入、alpha 调整与部分调试输出
- 支持 WLR-733 的同步 split 生成和候选标定评测 info 构造
- 支持 DAIR-V2X-I 转换为当前工程统一的 KITTI-like 布局
- 支持 YOLO 驱动的 B0 单帧重打分与 B1 短时序一致性修正
- 支持 3D 框投影到图像、投影叠加检查和离屏 demo

## 仓库结构

```text
OpenPCDet/
├── pcdet/
│   ├── datasets/
│   ├── models/
│   ├── ops/
│   └── utils/
├── tools/
│   ├── cfgs/
│   ├── eval_utils/
│   ├── process_tools/
│   ├── train_utils/
│   └── visual_utils/
├── docs/
├── requirements.txt
├── setup.py
└── TOOLS_PY_GUIDE.md
```

几个最关键的位置：

- [`pcdet/models/detectors/pointpillar.py`](pcdet/models/detectors/pointpillar.py)
  当前融合主线的核心 detector 实现
- [`tools/cfgs/kitti_models/`](tools/cfgs/kitti_models/)
  WLR / DAIR / fusion / lidar-only 等主要实验配置
- [`tools/prepare_dair_i_kitti_like.py`](tools/prepare_dair_i_kitti_like.py)
  DAIR 数据转换入口
- [`tools/prepare_wlr733_sync_splits.py`](tools/prepare_wlr733_sync_splits.py)
  WLR 同步 split 生成入口
- [`tools/rescore_dair_with_yolo.py`](tools/rescore_dair_with_yolo.py)
  DAIR B0 图像主导重打分入口
- [`tools/rescore_dair_with_yolo_temporal.py`](tools/rescore_dair_with_yolo_temporal.py)
  DAIR B1 时序修正入口
- [`tools/rescore_wlr_with_yolo.py`](tools/rescore_wlr_with_yolo.py)
  WLR 图像主导重打分入口
- [`TOOLS_PY_GUIDE.md`](TOOLS_PY_GUIDE.md)
  `tools/**/*.py` 的完整用途说明
- [`docs/LOCAL_SETUP.md`](docs/LOCAL_SETUP.md)
  本地环境搭建、扩展编译、验证与 GitHub 发布操作说明

## 目前重点配置

WLR 主线常用配置：

- `tools/cfgs/kitti_models/pointpillar_wlr733.yaml`
- `tools/cfgs/kitti_models/pointpillar_S1.yaml`
- `tools/cfgs/kitti_models/pointpillar_S2.yaml`
- `tools/cfgs/kitti_models/pointpillar_S2_imgboost_v2.yaml`
- `tools/cfgs/kitti_models/pointpillar_wlr733_lidaronly.yaml`

DAIR 主线常用配置：

- `tools/cfgs/kitti_models/pointpillar_dair_i_lidaronly_aligned.yaml`
- `tools/cfgs/kitti_models/pointpillar_dair_i_fusion_alpha_map_v1.yaml`

## 已实现的项目扩展

相对原始 OpenPCDet，这份仓库当前最主要的扩展包括：

- `PointPillar` 中加入轻量图像分支
- 图像特征到 BEV 的投影模块 `ImageToBEV`
- LiDAR BEV 与 image BEV 的 `BEVFusion`
- ROI 约束融合与部分标定 refine 逻辑
- WLR-733 数据读取、同步图像路径与按帧标定支持
- DAIR-V2X-I 到当前训练布局的转换脚本
- B0/B1 图像主导决策链与可视化审计输出

## 更多文档

- [`TOOLS_PY_GUIDE.md`](TOOLS_PY_GUIDE.md)：`tools/**/*.py` 的用途总览
- [`docs/LOCAL_SETUP.md`](docs/LOCAL_SETUP.md)：本地环境安装、扩展编译、验证和 GitHub 推送说明
- [`docs/WLR_NEXT_ARCH.md`](docs/WLR_NEXT_ARCH.md)：WLR 主线下一步结构改造建议与分层路线图
- [`docs/INSTALL.md`](docs/INSTALL.md)：上游 OpenPCDet 的安装说明
- [`docs/GETTING_STARTED.md`](docs/GETTING_STARTED.md)：上游 OpenPCDet 的入门说明

## 安装

下面是一套已知可工作的参考环境，适合作为起点：

- Ubuntu 22.04 / WSL2
- Python 3.8
- PyTorch 2.1.2 + CUDA 11.8
- `spconv-cu118==2.3.6`

### 1. 创建环境

```bash
conda create -n pcdet python=3.8 -y
conda activate pcdet
```

### 2. 安装 PyTorch

```bash
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 \
  --index-url https://download.pytorch.org/whl/cu118
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
pip install spconv-cu118==2.3.6
pip install matplotlib opencv-python shapely plyfile imageio networkx
pip install onnx onnxruntime fire dpkt ultralytics open3d
```

说明：

- `dpkt` 主要给 `tools/pcap2bin_wlr733.py` 使用
- `ultralytics` 主要给 `tools/run_yolo_to_csv.py` 和 B0/B1 链路使用
- `open3d` 主要给 `tools/demo.py`、`tools/demo_offscreen.py` 和部分可视化功能使用

### 4. 编译并安装 OpenPCDet 扩展

```bash
python setup.py develop
```

### 5. 简单验证

```bash
python - <<'PY'
import importlib
mods = [
    "pcdet",
    "pcdet.ops.iou3d_nms.iou3d_nms_utils",
    "pcdet.ops.roiaware_pool3d.roiaware_pool3d_utils",
]
for m in mods:
    importlib.import_module(m)
    print(m, "OK")
PY
```

## 快速开始

### 1. 训练

```bash
python tools/train.py \
  --cfg_file tools/cfgs/kitti_models/pointpillar_S2_imgboost_v2.yaml \
  --batch_size 4 \
  --workers 0 \
  --extra_tag exp_debug
```

### 2. 评测 checkpoint

```bash
python tools/test.py \
  --cfg_file tools/cfgs/kitti_models/pointpillar_S2_imgboost_v2.yaml \
  --ckpt <ckpt_path> \
  --batch_size 1 \
  --workers 0 \
  --save_to_file
```

### 3. 离屏 demo

```bash
python tools/demo_offscreen.py \
  --cfg_file tools/cfgs/kitti_models/pointpillar_wlr733.yaml \
  --ckpt <ckpt_path> \
  --data_path <point_cloud.bin> \
  --save_png tools/output/demo.png
```

## WLR-733 推荐流程

### A. 准备同步 split

```bash
python tools/prepare_wlr733_sync_splits.py \
  --data_root data/wlr733 \
  --cam_matched_dir /root/pointpillar/cam_matched_sync_v1
```

### B. 如果需要比较候选标定

```bash
python tools/prepare_wlr_candidate_eval_infos.py \
  --data_root data/wlr733 \
  --dataset_cfg tools/cfgs/dataset_configs/wlr733_dataset_sync.yaml
```

### C. 跑主模型训练或评测

```bash
python tools/train.py --cfg_file tools/cfgs/kitti_models/pointpillar_S2.yaml
python tools/test.py --cfg_file tools/cfgs/kitti_models/pointpillar_S2.yaml --ckpt <ckpt_path>
```

### D. 如果要做图像主导后验重打分

```bash
python tools/run_yolo_to_csv.py \
  --image_dir <matched_image_dir> \
  --out_csv output/wlr_yolo.csv

python tools/rescore_wlr_with_yolo.py \
  --result_pkl <result.pkl> \
  --info_pkl <info.pkl> \
  --image_dir <matched_image_dir> \
  --frame_map_csv <lidar_to_cam.csv> \
  --yolo_csv output/wlr_yolo.csv \
  --output_dir output/wlr_b0
```

## DAIR-V2X-I 推荐流程

### A. 转换数据

```bash
python tools/prepare_dair_i_kitti_like.py \
  --source_root <single-infrastructure-side> \
  --target_root data/dair_i_kitti \
  --split_json <split_data.json> \
  --link_images
```

### B. 跑 LiDAR-only baseline

```bash
python tools/test.py \
  --cfg_file tools/cfgs/kitti_models/pointpillar_dair_i_lidaronly_aligned.yaml \
  --ckpt <lidar_ckpt> \
  --save_to_file
```

### C. 跑 fusion 配置

```bash
python tools/train.py \
  --cfg_file tools/cfgs/kitti_models/pointpillar_dair_i_fusion_alpha_map_v1.yaml \
  --extra_tag dair_fusion_exp
```

### D. 做 B0 / B1 图像主导验证

```bash
python tools/run_yolo_to_csv.py \
  --image_dir data/dair_i_kitti/image \
  --out_csv output/dair_yolo.csv

python tools/rescore_dair_with_yolo.py \
  --result_pkl <result.pkl> \
  --info_pkl data/dair_i_kitti/dair_i_infos_val.pkl \
  --calib_dir data/dair_i_kitti/training/calib \
  --image_dir data/dair_i_kitti/image \
  --yolo_csv output/dair_yolo.csv \
  --output_dir output/dair_b0

python tools/rescore_dair_with_yolo_temporal.py \
  --result_pkl <result.pkl> \
  --info_pkl data/dair_i_kitti/dair_i_infos_val.pkl \
  --calib_dir data/dair_i_kitti/training/calib \
  --image_dir data/dair_i_kitti/image \
  --yolo_csv output/dair_yolo.csv \
  --output_dir output/dair_b1
```

## 可视化与调试

如果你现在在排查“投影是否对齐、图像证据是否靠谱、标定是否明显偏了”，最常用的是这些文件：

- [`tools/overlay_pred_on_image.py`](tools/overlay_pred_on_image.py)
- [`tools/demo.py`](tools/demo.py)
- [`tools/demo_offscreen.py`](tools/demo_offscreen.py)
- [`TOOLS_PY_GUIDE.md`](TOOLS_PY_GUIDE.md)

## 数据与产物说明

仓库默认不包含以下内容：

- 原始数据集
- 训练输出和评测输出
- 模型权重
- 中间缓存产物

对应目录通常会被 `.gitignore` 忽略：

- `data/`
- `output/`
- `tools/output/`
- `*.pth`
- `*.pt`

## 注意事项

- 这份仓库是研究代码，优先服务于当前主线实验和工程验证
- 自定义脚本和配置较多，推荐先看 [`TOOLS_PY_GUIDE.md`](TOOLS_PY_GUIDE.md)
- WLR 部分流程依赖本地数据组织方式和外部图像匹配结果，直接复现前请先检查路径、split、标定和 image root
- B0/B1 流程依赖额外的 YOLO 2D 检测结果，并不是纯 OpenPCDet 原生评测流程

## 致谢

本仓库基于 [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) 继续开发。  
感谢原项目提供的 3D 检测训练、推理与评测框架基础。

## License

本仓库沿用上游项目的许可证，详情见 [`LICENSE`](LICENSE)。
