# WLR-733 数据集与 OpenPCDet 本地环境配置说明

### WLR-733 Dataset and OpenPCDet Local Environment Setup Guide

---

## 🔧 一、系统环境概要
| 项目 | 内容 |
|------|------|
| 主机系统 | Windows 10 / 11 |
| WSL 发行版 | Ubuntu 22.04 (WSL2) |
| GPU | NVIDIA GeForce RTX 3070 Ti |
| 驱动 | 581.42 (支持 CUDA 13.0) |
| CUDA Toolkit (WSL) | 11.8 |
| Python | 3.8 (通过 Conda) |
| Conda 环境名 | pcdet |

---

## ⚙️ 二、创建 Conda 环境
```bash
conda create -n pcdet python=3.8 -y
conda activate pcdet
```

---

## ⚡️ 三、安装 PyTorch + CUDA 11.8
```bash
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 \
  --index-url https://download.pytorch.org/whl/cu118
```

---

## 🛠️ 四、安装基础依赖
```bash
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export PIP_DEFAULT_TIMEOUT=600
pip install "numpy<2.0" numba==0.56.4 tqdm pyyaml matplotlib opencv-python fire onnx onnxruntime shapely plyfile
pip install easydict tensorboardX==2.6 SharedArray==3.2.4
pip install scikit-image==0.19.3 imageio==2.31.1 networkx==2.8.8
```

---

## 🛠️ 五、安装 spconv
```bash
pip install spconv-cu118==2.3.6
```

验证 spconv 版本：
```bash
python - <<'PY'
import spconv
print("spconv version:", spconv.__version__)
PY
```

---

## 💡 六、编译 OpenPCDet 核心算子
```bash
cd ~/OpenPCDet
python setup.py develop
```

验证算子是否正常加载：
```bash
python - <<'PY'
import importlib
for m in [
 "pcdet.ops.iou3d_nms.iou3d_nms_utils",
 "pcdet.ops.roiaware_pool3d.roiaware_pool3d_utils",
 "pcdet.ops.roipoint_pool3d.roipoint_pool3d_utils",
]:
    try:
        importlib.import_module(m)
        print(m, "OK")
    except Exception as e:
        print(m, "FAIL:", e)
PY
```

---

## 🚀 七、运行 Demo / 评测

#### 离屏 Demo（BEV 可视化）
```bash
xvfb-run -a -s "-screen 0 1280x720x24" \
python tools/demo_offscreen.py \
  --cfg_file tools/cfgs/kitti_models/pointpillar_wlr733.yaml \
  --ckpt output/cfgs/kitti_models/pointpillar_wlr733/default/ckpt/checkpoint_epoch_80.pth \
  --data_path data/wlr733/training/velodyne/000003.bin \
  --save_png tools/output/000003_bev.png \
  --set_cfgs MODEL.POST_PROCESSING.SCORE_THRESH 0.30 \
  --target_w 1100 --target_h 620
```

#### 测试评测
```bash
python tools/test.py \
  --cfg_file tools/cfgs/kitti_models/pointpillar_wlr733.yaml \
  --ckpt output/cfgs/kitti_models/pointpillar_wlr733/default/ckpt/checkpoint_epoch_80.pth \
  --eval
```

#### 训练
```bash
python tools/train.py \
  --cfg_file tools/cfgs/kitti_models/pointpillar_wlr733.yaml \
  --epochs 80 \
  --batch_size 8 \
  --workers 4
```

---

## 🔗 八、将本地项目推送到 GitHub

#### 1.删除原他仓库信息
```bash
cd ~/OpenPCDet
rm -rf .git
```

#### 2.初始化并创建 .gitignore
```bash
git init
git branch -M main

cat > .gitignore <<'EOF'
__pycache__/
*.pyc
*.pyo
*.pyd
.venv/
.env
build/
dist/
*.egg-info/
**/*.so
output/
tools/output/
data/
*.pth
*.pt
*.onnx
*.ckpt
.vscode/
.idea/
EOF
```

#### 3.首次提交
```bash
git add .
git commit -m "Initial commit: import OpenPCDet source (WSL environment)"
```

#### 4.生成 SSH key 并添加到 GitHub
```bash
ssh-keygen -t ed25519 -C "2311735466@qq.com"
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
cat ~/.ssh/id_ed25519.pub
```

将输出的 key 复制到 [GitHub Settings → SSH keys](https://github.com/settings/keys)

#### 5.推送到你的仓库
```bash
git remote add origin git@github.com:czy2373/OpenPCDet-wlr733.git
git push -u origin main
```

---

## 🔍 九、验证结果
```bash
python -c "import torch; print(torch.cuda.is_available())"
# True

nvcc -V
# release 11.8, V11.8.x

python -m pcdet
# 应无报错
```

---

## 🔍 十、最终目录结构
```
OpenPCDet/
├── data/
├── output/
├── tools/
├── pcdet/
├── docs/
│   └── WLR-733 数据集与 OpenPCDet 本地环境配置说明.md
├── requirements.txt
└── README.md
```

---

## 🕛 更新记录 (Update History)
| 日期 | 内容 |
|------|------|
| 2025-10-15 | 新增《WLR-733 数据集与 OpenPCDet 本地环境配置说明》，包含环境搭建、依赖安装、编译验证以及 GitHub 上传步骤。 |

---

该文档可直接重命名为 `README.md` 后推送到 GitHub，以作为仓库主页显示使用。

