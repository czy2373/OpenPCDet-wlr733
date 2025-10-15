#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import numpy as np
import torch
import open3d as o3d

from pathlib import Path

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from pcdet.datasets.dataset import DatasetTemplate
from pcdet.utils.box_utils import boxes_to_corners_3d
from open3d.visualization import rendering


class DemoDataset(DatasetTemplate):
    """Minimal dataset for a single point cloud"""
    def __init__(self, dataset_cfg, class_names, training=False, root_path=None, ext='.bin'):
        super().__init__(dataset_cfg=dataset_cfg, class_names=class_names, root_path=root_path, training=training)
        self.ext = ext
        self.sample_file_list = []

    def set_sample(self, file_path):
        self.sample_file_list = [file_path]
        self.total_samples = 1

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        file_path = self.sample_file_list[index]
        if file_path.endswith('.bin'):
            points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)  # x,y,z,intensity
        elif file_path.endswith('.pcd'):
            pcd = o3d.io.read_point_cloud(file_path)
            pts = np.asarray(pcd.points, dtype=np.float32)
            inten = np.zeros((pts.shape[0], 1), dtype=np.float32)
            points = np.concatenate([pts, inten], axis=1)
        else:
            raise ValueError(f'Unsupported file type: {file_path}')

        input_dict = {
            'points': points,
            'frame_id': Path(file_path).stem
        }
        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def make_o3d_bbox_lines(boxes_lidar, color=(1.0, 0.0, 0.0)):
    """boxes_lidar: (N,7) [x,y,z,dx,dy,dz,yaw], return list of LineSet"""
    corners = boxes_to_corners_3d(torch.from_numpy(boxes_lidar).float()).numpy()  # (N,8,3)
    lines = [
        [0,1],[1,2],[2,3],[3,0],  # bottom
        [4,5],[5,6],[6,7],[7,4],  # top
        [0,4],[1,5],[2,6],[3,7]   # pillars
    ]
    colors = [color for _ in range(len(lines))]
    line_sets = []
    for cn in corners:
        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(cn)
        ls.lines  = o3d.utility.Vector2iVector(lines)
        ls.colors = o3d.utility.Vector3dVector(colors)
        line_sets.append(ls)
    return line_sets


def render_offscreen(points_xyzit, boxes_lidar, save_png, save_ply=None, width=1600, height=1200):
    """
    points_xyzit: (N,4) or (N,3) np.float32
    boxes_lidar:  (M,7) np.float32
    """
    # point colors by intensity (or z if intensity missing)
    if points_xyzit.shape[1] >= 4:
        inten = points_xyzit[:, 3]
        inten = (inten - inten.min()) / (inten.ptp() + 1e-6)
        colors = np.stack([inten, inten, inten], axis=1)
    else:
        z = points_xyzit[:, 2]
        z_norm = (z - z.min()) / (z.ptp() + 1e-6)
        colors = np.stack([z_norm, z_norm, z_norm], axis=1)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_xyzit[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(colors)

    if save_ply:
        o3d.io.write_point_cloud(save_ply, pcd, write_ascii=False, compressed=True)

    # Scene + Offscreen renderer
    from open3d.visualization import rendering
    renderer = rendering.OffscreenRenderer(width, height)
    scene = renderer.scene
    scene.set_background([1, 1, 1, 1])  # white bg

    mat_pcd = rendering.MaterialRecord()
    mat_pcd.shader = "defaultUnlit"
    mat_pcd.point_size = 2.5

    scene.add_geometry("pcd", pcd, mat_pcd)

    # add bboxes
    if boxes_lidar is not None and len(boxes_lidar) > 0:
        for i, ls in enumerate(make_o3d_bbox_lines(boxes_lidar, color=(1.0, 0.0, 0.0))):
            mat_ls = rendering.MaterialRecord()
            mat_ls.shader = "unlitLine"
            mat_ls.line_width = 2.0
            scene.add_geometry(f"bbox_{i}", ls, mat_ls)

    # camera
    aabb = pcd.get_axis_aligned_bounding_box()
    center = aabb.get_center()
    extent = aabb.get_extent()
    radius = np.linalg.norm(extent) * 0.7 + 30.0

    eye = center + np.array([0.0, 0.0, radius])   # 正上方
    up  = np.array([1.0, 0.0, 0.0])               # 让 +X 朝上；想让 +Y 朝上就用 [0,1,0]
    scene.camera.look_at(center, eye, up)

    # 俯视可把 FoV 稍微收窄，画面更集中；保持 Vertical FOVType
    scene.camera.set_projection(35.0, width / height, 0.1, 1000.0, rendering.Camera.FovType.Vertical)

    img = renderer.render_to_image()
    o3d.io.write_image(save_png, img)
    print(f"[Saved] {save_png}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_file', type=str, required=True)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True, help='bin or pcd')
    parser.add_argument('--save_png', type=str, default='demo.png')
    parser.add_argument('--save_ply', type=str, default=None)
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    logger = common_utils.create_logger()
    logger.info('---- Demo Offscreen ----')

    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False, root_path=None
    )
    demo_dataset.set_sample(args.data_path)

    # build model
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=False)
    model.cuda()
    model.eval()

    data_dict = demo_dataset[0]
    data_dict = demo_dataset.collate_batch([data_dict])
    load_data_to_gpu(data_dict)

    with torch.no_grad():
        pred_dicts, _ = model.forward(data_dict)

    # gather outputs
    pred = pred_dicts[0]
    boxes = pred.get('pred_boxes', torch.empty((0, 7))).detach().cpu().numpy().astype(np.float32)
    points = data_dict['points'][:, 1:5].cpu().numpy().astype(np.float32)  # [bs,x,y,z,i] -> use x,y,z,i

    out_png = Path(args.save_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_ply = args.save_ply
    if out_ply:
        Path(out_ply).parent.mkdir(parents=True, exist_ok=True)

    render_offscreen(points, boxes, save_png=str(out_png), save_ply=out_ply)


if __name__ == '__main__':
    main()
