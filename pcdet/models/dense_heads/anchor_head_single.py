import numpy as np
import torch.nn as nn

from .anchor_head_template import AnchorHeadTemplate


class AnchorHeadSingle(AnchorHeadTemplate):
    # 这是类的初始化函数，它接受多个参数来配置和构建头部网络。
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range,
                 predict_boxes_when_training=True, **kwargs):
        
        # 使用 super() 调用父类的初始化方法，并传递一系列参数来配置头部网络。
        super().__init__(
            model_cfg=model_cfg, num_class=num_class, class_names=class_names, grid_size=grid_size, point_cloud_range=point_cloud_range,
            predict_boxes_when_training=predict_boxes_when_training
        )

        # 1. 计算每个位置的锚框总数
        self.num_anchors_per_location = sum(self.num_anchors_per_location)

        # 2. 搭建“分类”预测层
        self.conv_cls = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.num_class,
            kernel_size=1
        )
        # 创建了一个卷积层 self.conv_cls，用于预测每个锚点位置的类别概率。

        # 3. 搭建“边界框回归”预测层
        self.conv_box = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.box_coder.code_size,
            kernel_size=1
        )
        # 创建了另一个卷积层 self.conv_box，用于预测每个锚点位置的边界框坐标。

        # 4. 搭建“方向分类”预测层 
        if self.model_cfg.get('USE_DIRECTION_CLASSIFIER', None) is not None:
            self.conv_dir_cls = nn.Conv2d(
                input_channels,
                self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS,
                kernel_size=1
            )
        else:
            self.conv_dir_cls = None
        # 根据配置选择是否创建方向分类器的卷积层。

        self.init_weights()
        # 调用 init_weights() 方法来初始化网络层的权重。

    def init_weights(self):
        pi = 0.01
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))  # self.conv_cls.bias 的初始化使用了常数初始化方法，根据逻辑来设置偏置的初始值。
        nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)   # self.conv_box.weight 使用了正态分布初始化方法，给权重设置了均值和标准差。
    # 这是 init_weights() 方法，用于初始化网络层的权重。它使用了 PyTorch 的初始化方法来设置权重。

    def forward(self, data_dict):
        # 5.  从 batch_dict 中取出 BaseBEVBackbone 计算出的“终极特征图” spatial_features_2d
        spatial_features_2d = data_dict['spatial_features_2d']

        # 6. 对特征图进行 1x1 卷积预测
        cls_preds = self.conv_cls(spatial_features_2d)  # 生成类别预测 
        box_preds = self.conv_box(spatial_features_2d)  # 边界框预测

        # 7. 整理预测结果的形状
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]

        self.forward_ret_dict['cls_preds'] = cls_preds
        self.forward_ret_dict['box_preds'] = box_preds


        # 8. 如果启用了方向分类器
        if self.conv_dir_cls is not None:
            dir_cls_preds = self.conv_dir_cls(spatial_features_2d)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            self.forward_ret_dict['dir_cls_preds'] = dir_cls_preds
        else:
            dir_cls_preds = None
        
        # 9. (训练时) 分配"标准答案"并计算损失
        if self.training:
            targets_dict = self.assign_targets(
                gt_boxes=data_dict['gt_boxes']
            )
            self.forward_ret_dict.update(targets_dict)

        # 10. (非训练时) 生成最终的检测框
        if not self.training or self.predict_boxes_when_training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=data_dict['batch_size'],
                cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds
            )
            data_dict['batch_cls_preds'] = batch_cls_preds
            data_dict['batch_box_preds'] = batch_box_preds
            data_dict['cls_preds_normalized'] = False

        return data_dict
