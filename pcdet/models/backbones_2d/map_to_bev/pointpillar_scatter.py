import torch
import torch.nn as nn


class PointPillarScatter(nn.Module):
    def __init__(self, model_cfg, grid_size, **kwargs):
        super().__init__()

        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES # 从 .yaml 中读取 NUM_BEV_FEATURES: 64 通道数（C）
        self.nx, self.ny, self.nz = grid_size  # 从数据集中读取 BEV 网格的空间尺寸。nx 是宽度 (W)，ny 是高度 (H)
        assert self.nz == 1 # 它确认我们是在 2D（z 轴为 1）上操作,这印证了 PointPillars 的核心思想：将 3D 问题“压扁”成 2D 问题。

    def forward(self, batch_dict, **kwargs):
        #1.获取pillar_vfe.py计算出的[N, 64]特征向量和[N, 4] 的坐标张量，格式为 [batch_index, z, y, x]
        pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']
        # pillar_features是点云数据在体素中的特征表示，coords是体素的坐标。

        # 这是一个 Python 列表。我们将为批次中的每一帧图像生成一张 BEV 伪图像，然后把这些伪图像（形状还是有点奇怪的）暂时存到这个列表中。
        batch_spatial_features = []  # 创建一个空列表batch_spatial_features用于存储每个批次的空间特征。
        # 确定当前批次中有多少帧图像需要处理
        batch_size = coords[:, 0].max().int().item() + 1  # 通过计算coords中的最大批次索引，然后将其转换为整数值。

        for batch_idx in range(batch_size):  # 针对每个批次索引，依次处理每个批次的点云数据
            spatial_feature = torch.zeros(  # torch.zeros(...) 创建了一个全零的巨大张量
                self.num_bev_features,  # e.g. 64 (C)
                self.nz * self.nx * self.ny,  # e.g. 1 * 496 * 432 (H*W)
                dtype=pillar_features.dtype,
                device=pillar_features.device)
            # 创建一个空的空间特征张量spatial_feature，其形状为(num_bev_features, nz * nx * ny)，用于存储该批次的空间特征。
            
            # coords 是 [batch_index, z, y, x]
            batch_mask = coords[:, 0] == batch_idx  # 创建一个布尔型掩码batch_mask，用于过滤出属于当前批次的体素和对应的特征
            this_coords = coords[batch_mask, :] # 根据batch_mask将坐标coords中属于当前批次的部分提取出来，得到this_coords
            
            #this_coords[:, 1] 是 z 坐标。this_coords[:, 2] 是 y 坐标。this_coords[:, 3] 是 x 坐标。
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3] #因为 self.nz == 1，z 坐标（this_coords[:, 1]）永远是 0,所以 indices = 0 + y * nx + x。
            indices = indices.type(torch.long)
            pillars = pillar_features[batch_mask, :]
            pillars = pillars.t()
            spatial_feature[:, indices] = pillars # 把 pillars的第j列（一个[64, 1]的特征）赋值给spatial_feature的第indices[j]列。
            batch_spatial_features.append(spatial_feature)

        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features * self.nz, self.ny, self.nx)
        batch_dict['spatial_features'] = batch_spatial_features
        return batch_dict


class PointPillarScatter3d(nn.Module):
    def __init__(self, model_cfg, grid_size, **kwargs):
        super().__init__()
        
        self.model_cfg = model_cfg
        self.nx, self.ny, self.nz = self.model_cfg.INPUT_SHAPE
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES 
        self.num_bev_features_before_compression = self.model_cfg.NUM_BEV_FEATURES // self.nz

    def forward(self, batch_dict, **kwargs):
        pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']
        
        batch_spatial_features = []
        batch_size = coords[:, 0].max().int().item() + 1
        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(
                self.num_bev_features_before_compression,
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)

            batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 1] * self.ny * self.nx + this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            pillars = pillar_features[batch_mask, :]
            pillars = pillars.t()
            spatial_feature[:, indices] = pillars
            batch_spatial_features.append(spatial_feature)
         
        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features_before_compression * self.nz, self.ny, self.nx)
        batch_dict['spatial_features'] = batch_spatial_features
        return batch_dict