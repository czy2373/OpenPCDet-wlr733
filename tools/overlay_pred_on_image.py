import os
import cv2
import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils


def load_data_to_device(batch_dict, device):
    if device == "cuda":
        load_data_to_gpu(batch_dict)
        return

    for key, val in batch_dict.items():
        if key == 'camera_imgs':
            batch_dict[key] = val.float()
        elif not isinstance(val, np.ndarray):
            continue
        elif key in ['frame_id', 'metadata', 'calib', 'image_paths', 'ori_shape', 'img_process_infos']:
            continue
        elif key in ['images']:
            batch_dict[key] = torch.from_numpy(val).float().contiguous()
        elif key in ['image_shape']:
            batch_dict[key] = torch.from_numpy(val).int()
        else:
            batch_dict[key] = torch.from_numpy(val).float()


def iou_xyxy(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter + 1e-9
    return inter / union


def nms_xyxy(boxes, scores, iou_thr=0.35, topk=35):
    if len(boxes) == 0:
        return []
    order = np.argsort(-scores)
    keep = []
    for idx in order:
        ok = True
        for kept in keep:
            if iou_xyxy(boxes[idx], boxes[kept]) > iou_thr:
                ok = False
                break
        if ok:
            keep.append(idx)
        if len(keep) >= topk:
            break
    return keep


def boxes3d_corners_lidar(boxes3d):
    """
    boxes3d: (M,7) [x,y,z,dx,dy,dz,heading] in LiDAR frame
    return corners: (M,8,3)
    """
    M = boxes3d.shape[0]
    x, y, z, dx, dy, dz, yaw = [boxes3d[:, i] for i in range(7)]

    x_corners = np.stack([ dx/2,  dx/2, -dx/2, -dx/2,  dx/2,  dx/2, -dx/2, -dx/2], axis=1)
    y_corners = np.stack([ dy/2, -dy/2, -dy/2,  dy/2,  dy/2, -dy/2, -dy/2,  dy/2], axis=1)
    z_corners = np.stack([ dz/2,  dz/2,  dz/2,  dz/2, -dz/2, -dz/2, -dz/2, -dz/2], axis=1)

    corners = np.stack([x_corners, y_corners, z_corners], axis=2).astype(np.float32)  # (M,8,3)

    cos, sin = np.cos(yaw), np.sin(yaw)
    R = np.zeros((M, 3, 3), dtype=np.float32)
    R[:, 0, 0] = cos
    R[:, 0, 1] = -sin
    R[:, 1, 0] = sin
    R[:, 1, 1] = cos
    R[:, 2, 2] = 1.0

    corners = corners @ np.transpose(R, (0, 2, 1))  # (M,8,3)
    corners[:, :, 0] += x[:, None]
    corners[:, :, 1] += y[:, None]
    corners[:, :, 2] += z[:, None]
    return corners


def draw_projected_box(img, pts2d, color=(0, 255, 0), thickness=2):
    pts = pts2d.astype(np.int32)
    edges = [
        (0,1),(1,2),(2,3),(3,0),
        (4,5),(5,6),(6,7),(7,4),
        (0,4),(1,5),(2,6),(3,7)
    ]
    for i, j in edges:
        cv2.line(img, tuple(pts[i]), tuple(pts[j]), color, thickness, lineType=cv2.LINE_AA)


def draw_heading_arrow(
    img,
    box3d,
    K,
    T_cam_from_lidar,
    color=(0, 165, 255),
    thickness=2,
    arrow_scale=0.6,
    flip_heading=False,
    layout="centerline",
):
    """
    Draw a heading arrow that indicates the predicted box heading in image space.

    Supported layouts:
    - centerline: arrow along the vehicle center axis
    - side_ypos: arrow along one longitudinal side of the box
    - side_yneg: arrow along the opposite longitudinal side of the box
    """
    box3d = np.asarray(box3d, dtype=np.float32).reshape(7)
    yaw = float(box3d[6])
    if flip_heading:
        yaw += np.pi

    box3d_draw = box3d.copy()
    box3d_draw[6] = yaw
    corners3d = boxes3d_corners_lidar(box3d_draw[None, :])[0]

    # Local +x face and -x face define the box longitudinal axis.
    front_center = corners3d[[0, 1, 4, 5]].mean(axis=0)
    rear_center = corners3d[[2, 3, 6, 7]].mean(axis=0)
    center = 0.5 * (front_center + rear_center)

    # Longitudinal lines on the two side surfaces.
    side_ypos_front = corners3d[[0, 4]].mean(axis=0)
    side_ypos_rear = corners3d[[3, 7]].mean(axis=0)
    side_yneg_front = corners3d[[1, 5]].mean(axis=0)
    side_yneg_rear = corners3d[[2, 6]].mean(axis=0)

    if layout == "centerline":
        start_3d = rear_center
        end_3d = front_center
    elif layout == "side_ypos":
        start_3d = side_ypos_rear
        end_3d = side_ypos_front
    elif layout == "side_yneg":
        start_3d = side_yneg_rear
        end_3d = side_yneg_front
    else:
        raise ValueError(f"unsupported heading layout: {layout}")

    if arrow_scale != 1.0:
        mid = 0.5 * (start_3d + end_3d)
        half_vec = (end_3d - start_3d) * (0.5 * float(arrow_scale))
        end_3d = mid + half_vec
        start_3d = mid - half_vec

    pts2d, depth = project_points_raw(np.stack([start_3d, end_3d], axis=0), K, T_cam_from_lidar)
    if pts2d.shape[0] != 2:
        return
    if float(depth[0]) <= 0.1 or float(depth[1]) <= 0.1:
        return

    h, w = img.shape[:2]
    start = tuple(np.clip(pts2d[0], [0, 0], [w - 1, h - 1]).astype(int))
    end = tuple(np.clip(pts2d[1], [0, 0], [w - 1, h - 1]).astype(int))
    if start == end:
        return

    cv2.line(img, start, end, color, 1, lineType=cv2.LINE_AA)
    cv2.arrowedLine(
        img,
        start,
        end,
        color,
        thickness,
        line_type=cv2.LINE_AA,
        tipLength=0.18,
    )
    cv2.circle(img, start, 3, color, -1, lineType=cv2.LINE_AA)


def parse_seq_and_frame(data_path: str):
    frame_id = os.path.splitext(os.path.basename(data_path))[0]
    parent = os.path.basename(os.path.dirname(data_path))
    seq = parent if parent.isdigit() else ""
    return seq, frame_id


def load_calib_npz(calib_npz: str):
    data = np.load(calib_npz)
    K = np.asarray(data["K"], dtype=np.float32)
    T = np.asarray(data["T_cam_from_lidar"], dtype=np.float32)
    if K.shape != (3, 3) or T.shape != (4, 4):
        raise ValueError(f"invalid calib shapes in {calib_npz}: K={K.shape}, T={T.shape}")
    return K, T


def _rotx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float32)


def _roty(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)


def _rotz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)


def apply_env_tweak(T_cam_from_lidar: np.ndarray):
    dpitch = float(os.environ.get("CALIB_DPITCH_DEG", "0"))
    droll = float(os.environ.get("CALIB_DROLL_DEG", "0"))
    dyaw = float(os.environ.get("CALIB_DYAW_DEG", "0"))
    dx_cam = float(os.environ.get("CALIB_DX_CAM", "0"))
    dy_cam = float(os.environ.get("CALIB_DY_CAM", "0"))
    dz_cam = float(os.environ.get("CALIB_DZ_CAM", "0"))

    if (abs(dpitch) + abs(droll) + abs(dyaw) + abs(dx_cam) + abs(dy_cam) + abs(dz_cam)) <= 1e-12:
        return T_cam_from_lidar, (dpitch, droll, dyaw, dx_cam, dy_cam, dz_cam)

    T_delta = np.eye(4, dtype=np.float32)
    T_delta[:3, :3] = (
        _rotz(np.deg2rad(dyaw))
        @ _roty(np.deg2rad(dpitch))
        @ _rotx(np.deg2rad(droll))
    ).astype(np.float32)
    T_delta[:3, 3] = np.array([dx_cam, dy_cam, dz_cam], dtype=np.float32)
    return (T_delta @ T_cam_from_lidar).astype(np.float32), (dpitch, droll, dyaw, dx_cam, dy_cam, dz_cam)


def project_points_raw(points_lidar: np.ndarray, K: np.ndarray, T_cam_from_lidar: np.ndarray):
    pts = np.asarray(points_lidar, dtype=np.float32).reshape(-1, 3)
    pts_h = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float32)], axis=1)
    pts_cam = (T_cam_from_lidar @ pts_h.T).T[:, :3]
    depth = pts_cam[:, 2].astype(np.float32)
    depth_safe = np.where(depth > 1e-6, depth, 1e-6).astype(np.float32)

    u = K[0, 0] * (pts_cam[:, 0] / depth_safe) + K[0, 2]
    v = K[1, 1] * (pts_cam[:, 1] / depth_safe) + K[1, 2]
    return np.stack([u, v], axis=1).astype(np.float32), depth


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_file", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data_path", required=True, help="path to .bin")
    parser.add_argument("--img_root", default="/root/pointpillar/cam_matched", help="where 000003.jpg lives")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--calib_npz", default="/root/OpenPCDet/data/wlr733/training/calib/calib.npz",
                        help="camera intrinsic/extrinsic npz used for projection")
    parser.add_argument("--out_dir", default="tools/output/overlay_1802", help="output folder")
    parser.add_argument("--score_thr", type=float, default=0.85, help="score filter for overlay")
    parser.add_argument("--max_depth", type=float, default=80.0, help="mean depth filter (meters)")
    parser.add_argument("--min_w", type=int, default=25, help="min 2D box width")
    parser.add_argument("--min_h", type=int, default=25, help="min 2D box height")
    parser.add_argument("--img_nms_iou", type=float, default=0.35, help="2D NMS IoU")
    parser.add_argument("--topk", type=int, default=35, help="keep top-k after NMS")
    parser.add_argument("--draw_heading_arrow", action="store_true", help="draw predicted heading arrow for each kept 3D box")
    parser.add_argument("--heading_arrow_scale", type=float, default=0.6, help="arrow length as a fraction of box length")
    parser.add_argument("--flip_heading_arrow", action="store_true", help="flip heading arrow by 180 degrees for display")
    parser.add_argument(
        "--heading_arrow_layout",
        default="centerline",
        choices=["centerline", "side_ypos", "side_yneg"],
        help="where to place the heading arrow relative to the vehicle body",
    )
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    logger = common_utils.create_logger()
    logger.info("loading cfg...")

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] overlay_pred_on_image device={device}")

    # 2) 原图（不resize）
    seq, frame_id = parse_seq_and_frame(args.data_path)
    img_path = os.path.join(args.img_root, f"{frame_id}.jpg")
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise FileNotFoundError(f"image not found: {img_path}")
    H, W = img_bgr.shape[:2]
    K, T = load_calib_npz(args.calib_npz)
    T, calib_delta = apply_env_tweak(T)
    print(f"[Projection calib] {args.calib_npz} | dxyz/rpy={calib_delta}")

    # 3) 单帧推理（只用 points）
    class DemoDataset(DatasetTemplate):
        def __init__(self, dataset_cfg, class_names, training=False, root_path=None, logger=None):
            super().__init__(dataset_cfg=dataset_cfg, class_names=class_names,
                             training=training, root_path=root_path, logger=logger)
        def __len__(self):
            return 1
        def __getitem__(self, index):
            points = np.fromfile(args.data_path, dtype=np.float32).reshape(-1, 4)
            input_dict = {"frame_id": frame_id, "points": points}
            data_dict = self.prepare_data(data_dict=input_dict)
            return data_dict

    dataset = DemoDataset(dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES,
                          training=False, root_path=None, logger=logger)
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=(device == "cpu"))
    model.to(device).eval()

    data_dict = dataset[0]
    batch_dict = dataset.collate_batch([data_dict])
    load_data_to_device(batch_dict, device=device)

    with torch.no_grad():
        pred_dicts, _ = model(batch_dict)

    pred = pred_dicts[0]
    if "pred_boxes" not in pred or pred["pred_boxes"].shape[0] == 0:
        print("[WARN] no boxes predicted")
        return

    boxes = pred["pred_boxes"].detach().cpu().numpy().astype(np.float32)    # (M,7)
    scores = pred["pred_scores"].detach().cpu().numpy().astype(np.float32)  # (M,)

    corners_all = boxes3d_corners_lidar(boxes)  # (M,8,3)

    # 4) 先做投影生成2D框 + 过滤 + 2D NMS（解决“框太多/挤角落/远处一堆”）
    boxes2d = []
    scores2d = []
    valid_ids = []

    for i in range(corners_all.shape[0]):
        s = float(scores[i])
        if s < args.score_thr:
            continue

        pts2d, depth = project_points_raw(corners_all[i], K, T)
        depth = depth.reshape(-1)

        # 至少 6 个角点在相机前方
        if np.sum(depth > 0.1) < 6:
            continue

        # 深度过滤（均值太远就不画）
        if float(depth.mean()) > float(args.max_depth):
            continue

        x1 = float(np.min(pts2d[:, 0])); y1 = float(np.min(pts2d[:, 1]))
        x2 = float(np.max(pts2d[:, 0])); y2 = float(np.max(pts2d[:, 1]))

        # clip到图像内部（让2D NMS有效）
        x1 = np.clip(x1, 0, W - 1); y1 = np.clip(y1, 0, H - 1)
        x2 = np.clip(x2, 0, W - 1); y2 = np.clip(y2, 0, H - 1)

        if (x2 - x1) < args.min_w or (y2 - y1) < args.min_h:
            continue

        boxes2d.append([x1, y1, x2, y2])
        scores2d.append(s)
        valid_ids.append(i)

    boxes2d = np.array(boxes2d, dtype=np.float32)
    scores2d = np.array(scores2d, dtype=np.float32)

    keep_local = nms_xyxy(boxes2d, scores2d, iou_thr=args.img_nms_iou, topk=args.topk)
    keep_idx = [valid_ids[j] for j in keep_local]

    print(f"[DBG] overlay filter: raw={len(scores)} -> after_filter={len(valid_ids)} -> after_nms={len(keep_idx)}")

    # 5) 画框（只画keep_idx）
    for i in keep_idx:
        pts2d, depth = project_points_raw(corners_all[i], K, T)

        # 如果整个框完全离屏太远，跳过（防止角落堆一坨）
        if np.all((pts2d[:, 0] < -200) | (pts2d[:, 0] > W + 200) |
                  (pts2d[:, 1] < -200) | (pts2d[:, 1] > H + 200)):
            continue

        draw_projected_box(img_bgr, pts2d, color=(0, 255, 0), thickness=2)
        if args.draw_heading_arrow:
            draw_heading_arrow(
                img_bgr,
                boxes[i],
                K,
                T,
                color=(0, 165, 255),
                thickness=2,
                arrow_scale=float(args.heading_arrow_scale),
                flip_heading=bool(args.flip_heading_arrow),
                layout=args.heading_arrow_layout,
            )
        p0 = tuple(np.clip(pts2d[0], [0, 0], [W - 1, H - 1]).astype(int))
        cv2.putText(img_bgr, f"{scores[i]:.2f}", p0,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    # 6) 保存
    out_dir = args.out_dir
    if seq != "":
        out_dir = os.path.join(out_dir, seq)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{frame_id}.jpg")
    cv2.imwrite(out_path, img_bgr)
    print("[Saved overlay]", os.path.abspath(out_path))
    print(f"[Projection calib] {args.calib_npz}")


if __name__ == "__main__":
    main()
