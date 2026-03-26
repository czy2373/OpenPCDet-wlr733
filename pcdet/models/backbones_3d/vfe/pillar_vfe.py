import torch
import torch.nn as nn
import torch.nn.functional as F

from .vfe_template import VFETemplate


class PFNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False):
        super().__init__()
        
        self.last_vfe = last_layer
        self.use_norm = use_norm
        if not self.last_vfe:
            # 1. 如果这不是最后一层，输出通道数减半
            out_channels = out_channels // 2

        if self.use_norm:
            # 2. 定义带归一化的全连接层
            self.linear = nn.Linear(in_channels, out_channels, bias=False)
            self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        else:
            # 3. 定义不带归一化的全连接层
            self.linear = nn.Linear(in_channels, out_channels, bias=True)

        # 4. 用于处理超大批次的分割阈值
        self.part = 50000 

    def forward(self, inputs):
        if inputs.shape[0] > self.part:
            # nn.Linear performs randomly when batch size is too large
            num_parts = inputs.shape[0] // self.part
            part_linear_out = [self.linear(inputs[num_part*self.part:(num_part+1)*self.part])
                               for num_part in range(num_parts+1)]
            x = torch.cat(part_linear_out, dim=0)
        else:
            #逐点线性层
            x = self.linear(inputs)
        torch.backends.cudnn.enabled = False
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1) if self.use_norm else x
        torch.backends.cudnn.enabled = True
        # 3. 激活函数 (ReLU)
        x = F.relu(x)
        # 4. 最大池化 (Max Pooling) - 关键步骤！s
        x_max = torch.max(x, dim=1, keepdim=True)[0]

        if self.last_vfe:
            # 5a. 如果是最后一层，直接返回“全局特征”
            return x_max
        else:
            # 5b. 如果是中间层，融合“局部”与“全局”特征
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated


class PillarVFE(VFETemplate):

    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)
    #这是PillarVFE类的定义，它继承自VFETemplate类。
    # 在初始化函数中，它接收模型配置(model_cfg)、点云特征的数量(num_point_features)、体素的大小(voxel_size)和点云范围(point_cloud_range)等参数。

        self.use_norm = self.model_cfg.USE_NORM
        self.with_distance = self.model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ
        # 这些变量存储了模型配置中的一些标志位，用来控制特征编码过程中的不同选项。

        num_point_features += 6 if self.use_absolute_xyz else 3
        if self.with_distance:
            num_point_features += 1
        # 根据标志位的设置，调整输入点云特征的数量。
        # 如果设置了"use_absolute_xyz"标志位，将会增加6个坐标特征（x、y、z的原始坐标和相对坐标），否则增加3个坐标特征。
        # 如果设置了"with_distance"标志位，还会增加一个距离特征。
  
        self.num_filters = self.model_cfg.NUM_FILTERS  # self.num_filters 变为 [64]。
        assert len(self.num_filters) > 0  # 它确保从配置文件中读取的 NUM_FILTERS 列表不是空的。如果因为配置错误导致这个列表是空的，程序会在这里立刻报错并停止，从而防止后续创建网络层时出错。
        num_filters = [num_point_features] + list(self.num_filters) # num_filters 变为 [10] + [64]，最终得到列表 [10, 64]。
        #从模型配置中获取特征编码的卷积层的通道数配置。将输入点云特征的数量(num_point_features)作为第一个通道数，然后依次添加后续的通道数。

        pfn_layers = []
        for i in range(len(num_filters) - 1): # 这里的 len(num_filters) - 1 是 len([10, 64]) - 1 = 1 , 列表 [10, 64] 的长度 len() 是 2
            # 循环只会执行一次 (i=0)
            in_filters = num_filters[i]   # num_filters[0] 是 10
            out_filters = num_filters[i + 1]  # num_filters[1] 是 64
            pfn_layers.append(
                PFNLayer(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2)) # 创建一个 PFNLayer(10, 64, ...)
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)
        # 创建PFNLayer的列表，PFNLayer是特征编码模块的基本单元。根据通道数配置，构建多个PFNLayer，并添加到列表中。

        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]
        # 存储体素的大小和点云范围，并计算体素的中心偏移量。这些偏移量将在后续的特征编码中使用。

    def get_output_feature_dim(self):
        return self.num_filters[-1]
    # 返回特征编码后的输出特征维度，即最后一个卷积层的通道数。

    def get_paddings_indicator(self, actual_num, max_num, axis=0):
        actual_num = torch.unsqueeze(actual_num, axis + 1)
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        paddings_indicator = actual_num.int() > max_num
        return paddings_indicator
    # 定义了一个辅助函数，用于生成一个指示填充位置的张量。
    # 根据输入的实际数量(actual_num)和最大数量(max_num)，生成一个形状相同的张量，其中填充位置为False，其他位置为True。

    def forward(self, batch_dict, **kwargs):
        voxel_features, voxel_num_points, coords = batch_dict['voxels'], batch_dict['voxel_num_points'], batch_dict['voxel_coords']
        # 前向传播函数。接收一个批次的输入数据(batch_dict)，包括体素特征(voxel_features)、体素中点的数量(voxel_num_points)和体素坐标(coords)。

        points_mean = voxel_features[:, :, :3].sum(dim=1, keepdim=True) / voxel_num_points.type_as(voxel_features).view(-1, 1, 1)  #每个pillar内部所有点的平均值
        f_cluster = voxel_features[:, :, :3] - points_mean
        # 计算点云的平均位置(points_mean)，并将每个点的坐标减去平均位置得到聚类特征(f_cluster)。

        f_center = torch.zeros_like(voxel_features[:, :, :3])  #计算每个点与其pillar几何中心之间的差
        f_center[:, :, 0] = voxel_features[:, :, 0] - (coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
        f_center[:, :, 1] = voxel_features[:, :, 1] - (coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
        f_center[:, :, 2] = voxel_features[:, :, 2] - (coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)
        # 计算点云中心特征(f_center)，通过将每个点的x、y、z坐标减去相应的体素偏移量得到。

        if self.use_absolute_xyz:
            # 1. 如果使用绝对坐标：
            # features 列表包含 [原始特征(x,y,z,r), f_cluster, f_center]
            features = [voxel_features, f_cluster, f_center]
        else:
            # 2. 如果不使用绝对坐标（只用反射强度r）：
            # features 列表包含 [原始特征(r), f_cluster, f_center]
            features = [voxel_features[..., 3:], f_cluster, f_center]

        if self.with_distance:
            # 3. 如果配置了 WITH_DISTANCE:
            # 计算点到原点的距离
            points_dist = torch.norm(voxel_features[:, :, :3], 2, 2, keepdim=True)
            features.append(points_dist)

        # 4. 执行拼接:
        # 此时 features 是一个包含3个或4个张量(Tensor)的列表
        # torch.cat(features, dim=-1) 
        # 会在最后一个维度（特征维度）上把它们全部拼接起来
        features = torch.cat(features, dim=-1) #

        voxel_count = features.shape[1]
        mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(voxel_features)
        features *= mask
        # 根据体素中点的数量生成一个掩码(mask)，并将其应用于输入特征。即将不包含点的位置的特征置为0。

        for pfn in self.pfn_layers:
            features = pfn(features)
        features = features.squeeze()
        batch_dict['pillar_features'] = features
        return batch_dict
        # 通过遍历PFNLayer列表，对输入特征进行多层卷积操作，得到编码后的特征。
        # 最后，将编码后的特征保存在batch_dict中的’pillar_features’键下，并返回更新后的batch_dict。