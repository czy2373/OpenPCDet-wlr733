# `tools/**/*.py` 文件说明

这份文档用于说明当前 `OpenPCDet/tools` 目录下每个 Python 文件的大致作用。

说明范围：

- 只覆盖当前仓库里真实存在的 `tools/**/*.py`
- 重点解释“这个文件拿来干什么”，不展开到每个实现细节
- 以 2026-03-25 这份仓库状态为准

## 阅读方式

- `入口脚本`：通常是你会直接执行 `python xxx.py` 的文件
- `支撑模块`：通常不直接运行，而是被其他脚本 import 或调用
- `自定义流程`：和你当前 WLR / DAIR / 图像主导决策主线直接相关的脚本
- `通用工具`：更接近原始 OpenPCDet 的训练、评测、可视化支撑文件

## 1. 核心入口脚本

| 文件 | 类型 | 作用 | 典型使用场景 |
| --- | --- | --- | --- |
| `tools/_init_path.py` | 支撑模块 | 把仓库父目录加入 `sys.path`，让顶层脚本能正常 import `pcdet` 和同级模块。 | 主要被 `train.py`、`test.py` 自动导入，通常不单独运行。 |
| `tools/train.py` | 入口脚本 | 主训练入口。负责解析配置、构建 dataloader 和模型、执行训练、保存 checkpoint，并可在训练后触发评测。 | 训练模型、微调模型、跑不同 yaml 配置时使用。 |
| `tools/test.py` | 入口脚本 | 主评测入口。负责载入 checkpoint、构建测试 dataloader、执行验证或测试，并写出 `result.pkl` 和评测日志。 | 做验证集评测、测试集推理、不同 checkpoint 对比时使用。 |
| `tools/demo.py` | 入口脚本 | 交互式点云 demo。对单个文件或文件夹中的点云运行检测，并用 Open3D 或 Mayavi 可视化结果。 | 想快速看某个点云样本的检测效果时使用。 |
| `tools/demo_offscreen.py` | 入口脚本 | 离屏渲染版 demo。对点云运行检测后，直接保存渲染图片或点云文件，不依赖交互窗口。 | 服务器环境、远程环境、无 GUI 环境下做可视化时很有用。 |

## 2. 数据准备与格式转换

| 文件 | 类型 | 作用 | 典型使用场景 |
| --- | --- | --- | --- |
| `tools/prepare_dair_i_kitti_like.py` | 自定义流程 | 把 DAIR-V2X-I 路侧数据转成当前工程能直接读的 KITTI-like 结构。它会处理点云转换、标注整理、数据划分、标定文件和图像链接或复制。 | 在 DAIR 上训练或验证之前，先把原始数据整理成当前工程统一格式。 |
| `tools/prepare_wlr733_sync_splits.py` | 自定义流程 | 基于有标注帧、LiDAR 帧和已匹配相机帧的交集，生成 WLR-733 同步版数据划分，并写出 `ImageSets/*.txt` 和对应 `infos.pkl`。 | 给 WLR 同步相机-LiDAR 训练和验证准备稳定、可复现的数据划分。 |
| `tools/prepare_wlr_candidate_eval_infos.py` | 自定义流程 | 生成带“候选标定参数”的 WLR 评测 `infos.pkl`，把不同外参候选直接写进每帧 info 里，从而能比较候选而不改磁盘上的公共标定文件。 | 做标定候选比较、本地 refine 结果评测、不同外参口径对比时使用。 |
| `tools/merge_kitti_like_roadside.py` | 自定义流程 | 把两个 KITTI-like 的路侧数据集合并成一个新数据集，并通过前缀避免帧号冲突，同时生成新的 split 和 info 文件。 | 想把两个 roadside 数据源拼成一个统一训练集时使用。 |
| `tools/pcap2bin_wlr733.py` | 自定义工具 | 提供 WLR-733 原始 UDP 数据包解析逻辑，以及把 `.pcap` 转成 KITTI 风格 `.bin` 点云的函数。 | 更偏原始数据阶段，适合从抓包文件里恢复点云。注意：目前它只有函数，没有完整的命令行 `main()` 入口。 |
| `tools/process_tools/create_integrated_database.py` | 通用工具 | 把 GT database 中分散存储的目标点云整合成一个全局 numpy 数据文件，并更新每个目标在全局文件里的 offset。 | 数据预处理、GT database 压缩整合时使用。 |

## 3. 图像主导重打分与时序决策

| 文件 | 类型 | 作用 | 典型使用场景 |
| --- | --- | --- | --- |
| `tools/run_yolo_to_csv.py` | 自定义流程 | 对图像目录跑 YOLO，并导出检测 CSV，包含帧号、框坐标、置信度、类别和模型名。 | 给后续 B0/B1 图像主导重打分提供 2D 图像证据。 |
| `tools/rescore_dair_with_yolo.py` | 自定义流程 | DAIR 的 B0 图像主导重打分脚本。输入 LiDAR 的 `result.pkl`，把 3D 框投到图像平面，再和 YOLO 2D 检测做匹配，重新计算分数，并输出新结果、指标、审计表和可视化。 | 验证“图像是否能直接影响 keep/drop 决策”时使用。 |
| `tools/rescore_dair_with_yolo_temporal.py` | 自定义流程 | DAIR 的 B1 时序版本。在 B0 单帧重打分基础上，再加短时轨迹一致性规则。 | 想从单帧图像决策继续推进到时序支持或时序抑制时使用。 |
| `tools/rescore_wlr_with_yolo.py` | 自定义流程 | WLR 版本的 B0 图像主导重打分。会结合 WLR 的 info、相机帧映射、标定和 YOLO 检测，对 LiDAR 结果重新打分并导出指标和审计信息。 | 在 WLR 上验证图像主导 keep/drop 决策时使用。 |
| `tools/iou_tracker.py` | 支撑模块 | 一个轻量级 BEV IoU tracker，用于跨帧把检测关联成短轨迹，并维护 track id、存活时间和 miss 次数。 | 主要被 `rescore_dair_with_yolo_temporal.py` 用来支撑 B1 时序逻辑。 |

## 4. 可视化与人工排查

| 文件 | 类型 | 作用 | 典型使用场景 |
| --- | --- | --- | --- |
| `tools/overlay_pred_on_image.py` | 自定义工具 | 把预测得到的 3D 框投影到相机图像上并保存，可选择画分数和 heading arrow。 | 排查投影质量、标定对齐情况、误检漏检案例时非常有用。 |
| `tools/visual_utils/open3d_vis_utils.py` | 支撑模块 | 基于 Open3D 的三维可视化工具函数，用来画点云和 3D 框。 | Open3D 环境下的 demo 或调试脚本会调用它。 |
| `tools/visual_utils/visualize_utils.py` | 支撑模块 | 基于 Mayavi 的三维可视化工具函数，包含点旋转、框角点计算和框绘制等。 | 当不用 Open3D、改走 Mayavi 可视化时使用。 |

## 5. 评测支撑

| 文件 | 类型 | 作用 | 典型使用场景 |
| --- | --- | --- | --- |
| `tools/eval_utils/eval_utils.py` | 支撑模块 | `test.py` 背后的通用评测循环。负责累计 recall 统计、写 `result.pkl`、调用数据集自带 evaluation，并记录最终指标。 | 一般不直接手动运行，而是通过 `tools/test.py` 间接使用。 |

## 6. 训练支撑

| 文件 | 类型 | 作用 | 典型使用场景 |
| --- | --- | --- | --- |
| `tools/train_utils/train_utils.py` | 支撑模块 | 训练循环核心工具，负责单 epoch 训练、日志记录、checkpoint 保存、AMP 支持，以及训练后期关闭增强的 hook。 | `tools/train.py` 的主要后端支撑模块。 |
| `tools/train_utils/optimization/__init__.py` | 支撑模块 | 根据配置构建 optimizer 和 lr scheduler，支持 Adam、SGD、OneCycle、cosine anneal 等。 | 训练启动时由 `train.py` 调用。 |
| `tools/train_utils/optimization/fastai_optim.py` | 支撑模块 | 改自 SECOND / fastai 风格的 optimizer wrapper，负责参数分组、master FP32 权重和优化步更新。 | 主要服务于 OneCycle 这类训练策略。 |
| `tools/train_utils/optimization/learning_schedules_fastai.py` | 支撑模块 | 各种学习率和动量调度实现，包括 OneCycle、cosine warmup、cosine annealing。 | 通过 `optimization/__init__.py` 间接被训练流程使用。 |

## 7. 结合你当前项目，最关键的是哪些文件

如果只看你当前这条 WLR / DAIR / 图像融合主线，`tools/` 下最核心的通常是这些：

- `tools/train.py`
- `tools/test.py`
- `tools/prepare_dair_i_kitti_like.py`
- `tools/prepare_wlr733_sync_splits.py`
- `tools/prepare_wlr_candidate_eval_infos.py`
- `tools/run_yolo_to_csv.py`
- `tools/rescore_dair_with_yolo.py`
- `tools/rescore_dair_with_yolo_temporal.py`
- `tools/rescore_wlr_with_yolo.py`
- `tools/overlay_pred_on_image.py`

这些文件分别对应你当前主线里的几个关键环节：

- 数据格式整理
- 同步样本划分
- 候选标定评测
- LiDAR-only / fusion 主线训练与评测
- 图像主导 B0 / B1 决策验证
- 人工投影检查与可视化审计

## 8. 几个实用备注

- `tools/train.py` 和 `tools/test.py` 仍然是最正式的训练、评测入口。
- `tools/pcap2bin_wlr733.py` 更像原始数据处理助手，不像一条已经完全封装好的正式流程。
- `tools/iou_tracker.py` 文件虽然小，但 B1 时序逻辑依赖它，不能随便删。
- `tools/overlay_pred_on_image.py` 对你现在这类“投影对不对、标定准不准、图像证据靠不靠谱”的问题很关键。
- WLR 和 DAIR 相关脚本已经不只是临时脚本，它们实际上构成了你当前项目的自定义流程层。

## 9. 后续可以继续补充的方向

如果你愿意，这份文档下一步还可以继续扩展成下面几种版本：

- 哪些文件属于 WLR 主线，哪些属于 DAIR 主线
- 哪些文件是训练前要跑的，哪些是评测后要跑的
- 哪些文件可以安全归档，哪些文件不要删
- 从原始数据到最终指标的完整推荐执行顺序

## 10. 按主线分类来看这些文件

### WLR 主线相关

这些文件更偏向你自己的 WLR-733 工程主线：

- `tools/prepare_wlr733_sync_splits.py`
  负责生成同步版样本划分和对应的 `infos.pkl`
- `tools/prepare_wlr_candidate_eval_infos.py`
  负责把候选标定参数写入评测 info，方便比较不同外参方案
- `tools/pcap2bin_wlr733.py`
  负责从原始 `.pcap` 解析出点云，属于更底层的数据准备工具
- `tools/rescore_wlr_with_yolo.py`
  负责在 WLR 上验证图像主导的 keep/drop 决策
- `tools/overlay_pred_on_image.py`
  常用于排查 WLR 投影效果、标定质量和图像匹配是否靠谱

### DAIR 主线相关

这些文件更偏向公开数据验证和迁移：

- `tools/prepare_dair_i_kitti_like.py`
  把 DAIR-V2X-I 路侧数据整理成当前工程能直接读的格式
- `tools/rescore_dair_with_yolo.py`
  DAIR 上的 B0 单帧图像主导重打分
- `tools/rescore_dair_with_yolo_temporal.py`
  DAIR 上的 B1 时序一致性修正
- `tools/run_yolo_to_csv.py`
  为 DAIR 或 WLR 的 B0/B1 脚本提供 YOLO 2D 证据

### 通用训练与评测主线

这些文件不区分 WLR / DAIR，属于整个检测系统的通用主干：

- `tools/train.py`
- `tools/test.py`
- `tools/eval_utils/eval_utils.py`
- `tools/train_utils/train_utils.py`
- `tools/train_utils/optimization/__init__.py`
- `tools/train_utils/optimization/fastai_optim.py`
- `tools/train_utils/optimization/learning_schedules_fastai.py`

### 可视化与调试支撑

这些文件更多是辅助理解和人工排查：

- `tools/demo.py`
- `tools/demo_offscreen.py`
- `tools/overlay_pred_on_image.py`
- `tools/visual_utils/open3d_vis_utils.py`
- `tools/visual_utils/visualize_utils.py`

### 数据融合与组合类工具

这些文件不一定天天用，但在扩展数据来源时有价值：

- `tools/merge_kitti_like_roadside.py`
- `tools/process_tools/create_integrated_database.py`

## 11. 按任务阶段来看，哪些文件是训练前、训练中、训练后会用到的

### 训练前

主要是把数据整理好、把样本和标定口径定下来：

- `tools/prepare_dair_i_kitti_like.py`
- `tools/prepare_wlr733_sync_splits.py`
- `tools/prepare_wlr_candidate_eval_infos.py`
- `tools/merge_kitti_like_roadside.py`
- `tools/pcap2bin_wlr733.py`
- `tools/process_tools/create_integrated_database.py`

### 训练中

主要是跑主模型训练和调优化器：

- `tools/train.py`
- `tools/train_utils/train_utils.py`
- `tools/train_utils/optimization/__init__.py`
- `tools/train_utils/optimization/fastai_optim.py`
- `tools/train_utils/optimization/learning_schedules_fastai.py`

### 训练后 / 评测时

主要是导出结果、做主线验证、做图像主导验证：

- `tools/test.py`
- `tools/eval_utils/eval_utils.py`
- `tools/run_yolo_to_csv.py`
- `tools/rescore_dair_with_yolo.py`
- `tools/rescore_dair_with_yolo_temporal.py`
- `tools/rescore_wlr_with_yolo.py`
- `tools/overlay_pred_on_image.py`

### 结果理解与展示

主要是看效果、看投影、看单样本：

- `tools/demo.py`
- `tools/demo_offscreen.py`
- `tools/overlay_pred_on_image.py`
- `tools/visual_utils/open3d_vis_utils.py`
- `tools/visual_utils/visualize_utils.py`

## 12. 推荐执行顺序：从数据到最终结果

下面这个顺序，更接近你现在项目的真实使用方式。

### A. 如果你在跑 DAIR 主线

1. 先整理数据：
   `tools/prepare_dair_i_kitti_like.py`

2. 跑 LiDAR-only 或 fusion 主模型：
   `tools/train.py`

3. 对 checkpoint 做验证导出结果：
   `tools/test.py`

4. 如果要验证图像主导决策：
   先跑 `tools/run_yolo_to_csv.py`

5. 做 B0 单帧重打分：
   `tools/rescore_dair_with_yolo.py`

6. 如果还要加时序一致性：
   `tools/rescore_dair_with_yolo_temporal.py`

7. 如有必要，做投影人工检查：
   `tools/overlay_pred_on_image.py`

### B. 如果你在跑 WLR 主线

1. 如果还在原始数据阶段：
   `tools/pcap2bin_wlr733.py`

2. 生成同步版 split 和 info：
   `tools/prepare_wlr733_sync_splits.py`

3. 如果要比较候选标定方案：
   `tools/prepare_wlr_candidate_eval_infos.py`

4. 跑 LiDAR-only 或 fusion 主模型：
   `tools/train.py`

5. 对 checkpoint 做评测，生成 `result.pkl`：
   `tools/test.py`

6. 如果要验证图像主导 keep/drop：
   先跑 `tools/run_yolo_to_csv.py`

7. 再跑 WLR 图像主导重打分：
   `tools/rescore_wlr_with_yolo.py`

8. 遇到投影不准、匹配不稳、标定怀疑有问题时：
   `tools/overlay_pred_on_image.py`

### C. 如果只是快速看一个模型或样本

可以直接走这条轻量路线：

1. `tools/demo.py`
2. 或者 `tools/demo_offscreen.py`
3. 如果要把结果叠到图像上，再用 `tools/overlay_pred_on_image.py`

## 13. 一个简短判断标准：什么时候应该先看哪个文件

- 想训练模型：先看 `tools/train.py`
- 想评测 checkpoint：先看 `tools/test.py`
- 想整理 DAIR：先看 `tools/prepare_dair_i_kitti_like.py`
- 想整理 WLR 同步数据：先看 `tools/prepare_wlr733_sync_splits.py`
- 想比较 WLR 标定候选：先看 `tools/prepare_wlr_candidate_eval_infos.py`
- 想验证图像是否能直接改 keep/drop：先看 `tools/rescore_dair_with_yolo.py` 或 `tools/rescore_wlr_with_yolo.py`
- 想加时序支持：先看 `tools/rescore_dair_with_yolo_temporal.py`
- 想看投影和可视化：先看 `tools/overlay_pred_on_image.py`
