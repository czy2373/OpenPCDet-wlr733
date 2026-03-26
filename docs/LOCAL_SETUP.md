# 本地环境搭建与发布说明

这份文档保留了本仓库更偏“本地开发环境”和“仓库发布操作”的内容。  
如果你只是想快速了解项目本身，建议先看仓库首页的 [`README.md`](../README.md)。

## 1. 参考环境

下面是一套已知可工作的参考环境：

| 项目 | 内容 |
| --- | --- |
| 系统 | Ubuntu 22.04 / WSL2 |
| GPU | NVIDIA GeForce RTX 3070 Ti |
| CUDA Toolkit | 11.8 |
| Python | 3.8 |
| Conda 环境名 | `pcdet` |
| PyTorch | 2.1.2 + cu118 |
| spconv | `spconv-cu118==2.3.6` |

说明：

- 这是当前项目常用的参考配置，不代表唯一可用组合
- 如果你本地 CUDA / PyTorch 版本不同，请优先保证和 `spconv` 兼容

## 2. 创建 Conda 环境

```bash
conda create -n pcdet python=3.8 -y
conda activate pcdet
```

## 3. 安装 PyTorch

```bash
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 \
  --index-url https://download.pytorch.org/whl/cu118
```

## 4. 安装基础依赖

先装仓库自带依赖：

```bash
pip install -r requirements.txt
```

再补项目当前常用依赖：

```bash
pip install spconv-cu118==2.3.6
pip install matplotlib opencv-python shapely plyfile imageio networkx
pip install onnx onnxruntime fire dpkt ultralytics open3d
```

几个依赖的用途：

- `dpkt`：给 `tools/pcap2bin_wlr733.py` 使用
- `ultralytics`：给 `tools/run_yolo_to_csv.py`、B0/B1 链路使用
- `open3d`：给 `tools/demo.py`、`tools/demo_offscreen.py` 和可视化功能使用

如果下载速度慢，可按需切换 pip 镜像，例如：

```bash
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export PIP_DEFAULT_TIMEOUT=600
```

## 5. 编译 OpenPCDet 扩展

在仓库根目录执行：

```bash
cd /path/to/OpenPCDet
python setup.py develop
```

## 6. 编译后验证

### 6.1 检查核心模块能否导入

```bash
python - <<'PY'
import importlib
mods = [
    "pcdet",
    "pcdet.ops.iou3d_nms.iou3d_nms_utils",
    "pcdet.ops.roiaware_pool3d.roiaware_pool3d_utils",
    "pcdet.ops.roipoint_pool3d.roipoint_pool3d_utils",
]
for m in mods:
    try:
        importlib.import_module(m)
        print(m, "OK")
    except Exception as e:
        print(m, "FAIL:", e)
PY
```

### 6.2 检查 CUDA / Torch

```bash
python - <<'PY'
import torch
print("cuda available:", torch.cuda.is_available())
print("torch version:", torch.__version__)
PY
```

### 6.3 检查 spconv

```bash
python - <<'PY'
import spconv
print("spconv version:", spconv.__version__)
PY
```

## 7. 常用运行命令

### 7.1 训练

```bash
python tools/train.py \
  --cfg_file tools/cfgs/kitti_models/pointpillar_S2_imgboost_v2.yaml \
  --batch_size 4 \
  --workers 0 \
  --extra_tag exp_debug
```

### 7.2 评测

```bash
python tools/test.py \
  --cfg_file tools/cfgs/kitti_models/pointpillar_S2_imgboost_v2.yaml \
  --ckpt <ckpt_path> \
  --batch_size 1 \
  --workers 0 \
  --save_to_file
```

### 7.3 离屏 Demo

```bash
python tools/demo_offscreen.py \
  --cfg_file tools/cfgs/kitti_models/pointpillar_wlr733.yaml \
  --ckpt <ckpt_path> \
  --data_path <point_cloud.bin> \
  --save_png tools/output/demo.png
```

如果你当前主要在跑 WLR / DAIR / B0-B1 主线，更建议结合 [`TOOLS_PY_GUIDE.md`](../TOOLS_PY_GUIDE.md) 一起看。

## 8. 推送到 GitHub 的推荐做法

### 8.1 先看远程仓库配置

```bash
git remote -v
git branch --show-current
git status
```

如果已经有 `origin`，通常不需要重新 `git init` 或删掉 `.git`。

### 8.2 建议先推一个清理分支

比起直接把所有内容推到 `main`，更建议先推一个清理分支检查：

```bash
git checkout -b mainline_clean_YYYYMMDD
git add -u
git add README.md TOOLS_PY_GUIDE.md docs pcdet tools
git status
git commit -m "feat: publish WLR/DAIR fusion mainline"
git push -u origin mainline_clean_YYYYMMDD
```

这样你可以先在 GitHub 上检查：

- README 首页是否清晰
- 有没有误传本地资料
- 有没有把数据、权重、输出目录带上去

### 8.3 如果确认无误，再合并到 `main`

```bash
git checkout main
git merge --ff-only mainline_clean_YYYYMMDD
git push origin main
```

## 9. SSH Key 参考步骤

如果 `git push` 提示类似：

```text
Permission denied (publickey)
```

说明 GitHub SSH key 可能还没配好，可以参考下面步骤：

### 9.1 生成 SSH key

```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
```

### 9.2 启动 agent 并添加 key

```bash
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
```

### 9.3 查看公钥

```bash
cat ~/.ssh/id_ed25519.pub
```

然后把输出内容复制到 GitHub 的 SSH keys 页面。

### 9.4 测试 SSH 是否生效

```bash
ssh -T git@github.com
```

## 10. `.gitignore` 建议

当前项目一般不应该提交这些内容：

- `data/`
- `output/`
- `tools/output/`
- `*.pth`
- `*.pt`
- `*.onnx`
- `*.ckpt`
- 本地标注工具相关文件
- 日志和缓存文件

如果你准备公开仓库，推送前务必再次检查：

```bash
git status
git diff --cached --stat
```

## 11. 适合公开上传的内容

通常更适合公开的是：

- `pcdet/` 下改过的主线代码
- `tools/` 下与你主线直接相关的脚本
- `tools/cfgs/` 下主线配置
- `README.md`
- `TOOLS_PY_GUIDE.md`
- `docs/` 下说明文档

通常更应该谨慎处理的是：

- 私有数据集
- 训练输出
- checkpoint
- 本地实验缓存
- 明显只服务本机的配置文件

## 12. 一个简单原则

如果一个文件回答的是“项目本身是什么、怎么跑、怎么复现”，更适合留在仓库里。  
如果一个文件回答的是“我这台机器怎么配、我本地怎么调、我本地的中间产物是什么”，就应该谨慎决定是否公开。
