#!/usr/bin/env python3
"""
Image-led B0 validation on DAIR:
- LiDAR result.pkl provides 3D geometry
- YOLO 2D detections drive keep / rescore / suppress decisions
"""

import argparse
import csv
import json
import math
import pickle
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


def stem_frame_id(x) -> str:
    s = str(x).strip()
    s = s.split("/")[-1].split("\\")[-1]
    s = Path(s).stem
    return s


def norm_frame_id(x) -> str:
    s = stem_frame_id(x)
    s = s.lstrip("0") or "0"
    return s


def unique_keep_order(items):
    out = []
    seen = set()
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def resolve_calib_file(calib_dir: Path, frame_id: str) -> Path:
    norm = norm_frame_id(frame_id)
    candidates = unique_keep_order(
        [
            stem_frame_id(frame_id),
            norm,
            f"{int(norm):06d}" if norm.isdigit() else norm,
        ]
    )
    for cand in candidates:
        path = calib_dir / f"{cand}.npz"
        if path.exists():
            return path
    raise FileNotFoundError(f"calib npz not found for frame_id={frame_id} under {calib_dir}")


def resolve_image_file(image_dir: Path, frame_id: str) -> Path:
    norm = norm_frame_id(frame_id)
    candidates = unique_keep_order(
        [
            stem_frame_id(frame_id),
            norm,
            f"{int(norm):06d}" if norm.isdigit() else norm,
        ]
    )
    for cand in candidates:
        for ext in (".jpg", ".jpeg", ".png"):
            path = image_dir / f"{cand}{ext}"
            if path.exists():
                return path
    raise FileNotFoundError(f"image not found for frame_id={frame_id} under {image_dir}")


def load_calib(calib_path: Path):
    data = np.load(str(calib_path))
    return np.asarray(data["K"], dtype=np.float32), np.asarray(data["T_cam_from_lidar"], dtype=np.float32)


def load_result_list(result_pkl: Path):
    with open(result_pkl, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, list):
        raise TypeError(f"result.pkl should contain a list, got {type(data)}")
    return data


def load_gt_map(info_pkl: Path):
    with open(info_pkl, "rb") as f:
        infos = pickle.load(f)
    out = {}
    for info in infos:
        raw_id = info.get("point_cloud", {}).get("lidar_idx", info.get("frame_id", ""))
        key = norm_frame_id(raw_id)
        ann = info.get("annotations", {}) or {}
        gt_boxes = np.asarray(ann.get("gt_boxes_lidar", np.zeros((0, 7), dtype=np.float32)), dtype=np.float32).reshape(-1, 7)
        out[key] = gt_boxes
    return out


def load_yolo_map(csv_path: Path, keep_classes):
    out = defaultdict(list)
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cls_id = row.get("cls", "").strip()
            if keep_classes and cls_id != "":
                try:
                    if int(cls_id) not in keep_classes:
                        continue
                except ValueError:
                    continue
            key = norm_frame_id(row.get("frame_id", ""))
            out[key].append(
                {
                    "frame_id": stem_frame_id(row.get("frame_id", "")),
                    "bbox": np.array(
                        [
                            float(row["x1"]),
                            float(row["y1"]),
                            float(row["x2"]),
                            float(row["y2"]),
                        ],
                        dtype=np.float32,
                    ),
                    "conf": float(row.get("conf", 0.0)),
                    "cls": int(cls_id) if cls_id not in ("", None) else -1,
                    "name": str(row.get("name", "")),
                }
            )
    return out


def boxes3d_corners_lidar(boxes3d: np.ndarray) -> np.ndarray:
    boxes3d = np.asarray(boxes3d, dtype=np.float32).reshape(-1, 7)
    x, y, z, dx, dy, dz, yaw = [boxes3d[:, i] for i in range(7)]

    x_corners = np.stack([dx / 2, dx / 2, -dx / 2, -dx / 2, dx / 2, dx / 2, -dx / 2, -dx / 2], axis=1)
    y_corners = np.stack([dy / 2, -dy / 2, -dy / 2, dy / 2, dy / 2, -dy / 2, -dy / 2, dy / 2], axis=1)
    z_corners = np.stack([dz / 2, dz / 2, dz / 2, dz / 2, -dz / 2, -dz / 2, -dz / 2, -dz / 2], axis=1)
    corners = np.stack([x_corners, y_corners, z_corners], axis=2).astype(np.float32)

    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    rot = np.zeros((boxes3d.shape[0], 3, 3), dtype=np.float32)
    rot[:, 0, 0] = cos_yaw
    rot[:, 0, 1] = -sin_yaw
    rot[:, 1, 0] = sin_yaw
    rot[:, 1, 1] = cos_yaw
    rot[:, 2, 2] = 1.0

    corners = corners @ np.transpose(rot, (0, 2, 1))
    corners[:, :, 0] += x[:, None]
    corners[:, :, 1] += y[:, None]
    corners[:, :, 2] += z[:, None]
    return corners


def project_points(points_lidar: np.ndarray, cam_k: np.ndarray, t_cam_from_lidar: np.ndarray):
    pts = np.asarray(points_lidar, dtype=np.float32).reshape(-1, 3)
    pts_h = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float32)], axis=1)
    pts_cam = (t_cam_from_lidar @ pts_h.T).T[:, :3]
    depth = pts_cam[:, 2].astype(np.float32)
    depth_safe = np.where(depth > 1e-6, depth, 1e-6)
    u = cam_k[0, 0] * (pts_cam[:, 0] / depth_safe) + cam_k[0, 2]
    v = cam_k[1, 1] * (pts_cam[:, 1] / depth_safe) + cam_k[1, 2]
    return np.stack([u, v], axis=1).astype(np.float32), depth


def project_box_to_image(box_lidar, cam_k, t_cam_from_lidar, image_hw):
    corners = boxes3d_corners_lidar(np.asarray(box_lidar, dtype=np.float32).reshape(1, 7))[0]
    pts2d, depth = project_points(corners, cam_k, t_cam_from_lidar)
    valid = np.isfinite(pts2d).all(axis=1) & (depth > 0.1)
    if valid.sum() < 2:
        return {
            "valid": False,
            "bbox": None,
            "corners2d": pts2d,
            "corner_valid": valid,
        }

    visible = pts2d[valid]
    x1 = float(np.min(visible[:, 0]))
    y1 = float(np.min(visible[:, 1]))
    x2 = float(np.max(visible[:, 0]))
    y2 = float(np.max(visible[:, 1]))

    if image_hw is not None:
        h, w = image_hw
        x1 = max(0.0, min(x1, w - 1.0))
        y1 = max(0.0, min(y1, h - 1.0))
        x2 = max(0.0, min(x2, w - 1.0))
        y2 = max(0.0, min(y2, h - 1.0))

    if (x2 - x1) < 2.0 or (y2 - y1) < 2.0:
        return {
            "valid": False,
            "bbox": None,
            "corners2d": pts2d,
            "corner_valid": valid,
        }

    return {
        "valid": True,
        "bbox": np.array([x1, y1, x2, y2], dtype=np.float32),
        "corners2d": pts2d,
        "corner_valid": valid,
    }


def iou_xyxy(box_a, box_b) -> float:
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(box_a[2]) - float(box_a[0])) * max(0.0, float(box_a[3]) - float(box_a[1]))
    area_b = max(0.0, float(box_b[2]) - float(box_b[0])) * max(0.0, float(box_b[3]) - float(box_b[1]))
    union = area_a + area_b - inter
    return float(inter / max(union, 1e-6))


def center_dist_norm(box_a, box_b) -> float:
    ca_x = 0.5 * (float(box_a[0]) + float(box_a[2]))
    ca_y = 0.5 * (float(box_a[1]) + float(box_a[3]))
    cb_x = 0.5 * (float(box_b[0]) + float(box_b[2]))
    cb_y = 0.5 * (float(box_b[1]) + float(box_b[3]))
    dist = math.hypot(ca_x - cb_x, ca_y - cb_y)
    diag = math.hypot(max(float(box_a[2]) - float(box_a[0]), 1.0), max(float(box_a[3]) - float(box_a[1]), 1.0))
    return float(dist / max(diag, 1.0))


def clip01(x: float) -> float:
    return float(min(max(x, 0.0), 1.0))


def greedy_match_records(records, yolo_dets, match_iou_thresh, match_center_thresh):
    used_yolo = set()
    order = sorted(range(len(records)), key=lambda idx: (-records[idx]["lidar_score"], idx))
    for rec_idx in order:
        record = records[rec_idx]
        proj_bbox = record["proj_bbox"]
        best = None
        if proj_bbox is not None:
            for yolo_idx, yolo in enumerate(yolo_dets):
                if yolo_idx in used_yolo:
                    continue
                iou2d = iou_xyxy(proj_bbox, yolo["bbox"])
                center_norm = center_dist_norm(proj_bbox, yolo["bbox"])
                if iou2d < match_iou_thresh and center_norm > match_center_thresh:
                    continue
                cand = (
                    iou2d,
                    -center_norm,
                    yolo["conf"],
                    -yolo_idx,
                )
                if best is None or cand > best[0]:
                    best = (
                        cand,
                        yolo_idx,
                        iou2d,
                        center_norm,
                    )
        if best is not None:
            _, yolo_idx, iou2d, center_norm = best
            used_yolo.add(yolo_idx)
            record["matched"] = True
            record["matched_yolo"] = yolo_dets[yolo_idx]
            record["iou2d"] = float(iou2d)
            record["center_dist_norm"] = float(center_norm)
        else:
            record["matched"] = False
            record["matched_yolo"] = None
            record["iou2d"] = 0.0
            record["center_dist_norm"] = 1.0


def oriented_iou_bev(boxes_a: np.ndarray, boxes_b: np.ndarray, use_rotated_iou: bool = True) -> np.ndarray:
    if boxes_a.shape[0] == 0 or boxes_b.shape[0] == 0:
        return np.zeros((boxes_a.shape[0], boxes_b.shape[0]), dtype=np.float32)
    if use_rotated_iou:
        try:
            from pcdet.ops.iou3d_nms.iou3d_nms_utils import boxes_bev_iou_cpu

            return np.asarray(boxes_bev_iou_cpu(boxes_a[:, :7], boxes_b[:, :7]), dtype=np.float32)
        except Exception:
            pass
    xa1 = boxes_a[:, 0] - boxes_a[:, 3] / 2
    ya1 = boxes_a[:, 1] - boxes_a[:, 4] / 2
    xa2 = boxes_a[:, 0] + boxes_a[:, 3] / 2
    ya2 = boxes_a[:, 1] + boxes_a[:, 4] / 2

    xb1 = boxes_b[:, 0] - boxes_b[:, 3] / 2
    yb1 = boxes_b[:, 1] - boxes_b[:, 4] / 2
    xb2 = boxes_b[:, 0] + boxes_b[:, 3] / 2
    yb2 = boxes_b[:, 1] + boxes_b[:, 4] / 2

    inter_x1 = np.maximum(xa1[:, None], xb1[None, :])
    inter_y1 = np.maximum(ya1[:, None], yb1[None, :])
    inter_x2 = np.minimum(xa2[:, None], xb2[None, :])
    inter_y2 = np.minimum(ya2[:, None], yb2[None, :])

    inter_w = np.maximum(inter_x2 - inter_x1, 0.0)
    inter_h = np.maximum(inter_y2 - inter_y1, 0.0)
    inter = inter_w * inter_h

    area_a = np.maximum(xa2 - xa1, 0.0) * np.maximum(ya2 - ya1, 0.0)
    area_b = np.maximum(xb2 - xb1, 0.0) * np.maximum(yb2 - yb1, 0.0)
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-6)


def greedy_match_eval(pred_boxes, pred_scores, gt_boxes, iou_thresh, use_rotated_iou):
    n_pred = pred_boxes.shape[0]
    n_gt = gt_boxes.shape[0]
    if n_pred == 0:
        return np.zeros((0,), dtype=bool), np.zeros((0,), dtype=bool), np.zeros((n_gt,), dtype=bool)

    order = np.argsort(-pred_scores)
    boxes_sorted = pred_boxes[order]
    iou = oriented_iou_bev(boxes_sorted, gt_boxes, use_rotated_iou=use_rotated_iou)

    gt_used = np.zeros((n_gt,), dtype=bool)
    tp = np.zeros((n_pred,), dtype=bool)
    fp = np.zeros((n_pred,), dtype=bool)
    for i in range(n_pred):
        if n_gt == 0:
            fp[i] = True
            continue
        gt_idx = int(np.argmax(iou[i]))
        best = float(iou[i, gt_idx])
        if best >= iou_thresh and not gt_used[gt_idx]:
            tp[i] = True
            gt_used[gt_idx] = True
        else:
            fp[i] = True

    inv = np.empty_like(order)
    inv[order] = np.arange(n_pred)
    return tp[inv], fp[inv], gt_used


def evaluate_det_list(det_list, gt_map, score_thresh, iou_thresh, use_rotated_iou):
    pred_map = {}
    for anno in det_list:
        pred_map[norm_frame_id(anno.get("frame_id", ""))] = anno

    common = sorted(set(pred_map.keys()) & set(gt_map.keys()))
    if not common:
        raise RuntimeError("No common frame ids between predictions and GT infos")

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_kept = 0
    for key in common:
        anno = pred_map[key]
        gt_boxes = gt_map[key]
        pred_boxes = np.asarray(anno.get("boxes_lidar", np.zeros((0, 7), dtype=np.float32)), dtype=np.float32).reshape(-1, 7)
        pred_scores = np.asarray(anno.get("score", np.zeros((pred_boxes.shape[0],), dtype=np.float32)), dtype=np.float32).reshape(-1)
        keep = pred_scores >= score_thresh
        pred_boxes = pred_boxes[keep]
        pred_scores = pred_scores[keep]
        total_kept += int(keep.sum())

        tp_mask, fp_mask, gt_used = greedy_match_eval(
            pred_boxes,
            pred_scores,
            gt_boxes,
            iou_thresh=iou_thresh,
            use_rotated_iou=use_rotated_iou,
        )
        total_tp += int(tp_mask.sum())
        total_fp += int(fp_mask.sum())
        total_fn += int(gt_boxes.shape[0] - gt_used.sum())

    precision = float(total_tp / max(total_tp + total_fp, 1))
    recall = float(total_tp / max(total_tp + total_fn, 1))
    return {
        "num_common_frames": len(common),
        "score_thresh": float(score_thresh),
        "iou_thresh": float(iou_thresh),
        "use_rotated_iou": bool(use_rotated_iou),
        "tp": int(total_tp),
        "fp": int(total_fp),
        "fn": int(total_fn),
        "precision": precision,
        "recall": recall,
        "kept_predictions": int(total_kept),
    }


def filter_anno_with_new_scores(anno, keep_mask, new_scores):
    keep_mask = np.asarray(keep_mask, dtype=bool)
    num = int(np.asarray(anno.get("score", []), dtype=np.float32).reshape(-1).shape[0])
    out = {}
    for key, val in anno.items():
        if isinstance(val, np.ndarray) and val.shape[:1] == (num,):
            out[key] = val[keep_mask].copy()
        elif isinstance(val, list) and len(val) == num:
            out[key] = [val[i] for i in np.where(keep_mask)[0]]
        else:
            out[key] = val

    out["score"] = np.asarray(new_scores, dtype=np.float32)[keep_mask]
    if "name" in out:
        out["name"] = np.asarray(out["name"])
    return out


def draw_projected_box(img, corners2d, corner_valid, color, thickness):
    pts = np.asarray(corners2d, dtype=np.float32)
    valid = np.asarray(corner_valid, dtype=bool)
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    for i, j in edges:
        if not (valid[i] and valid[j]):
            continue
        p1 = tuple(np.round(pts[i]).astype(np.int32))
        p2 = tuple(np.round(pts[j]).astype(np.int32))
        cv2.line(img, p1, p2, color, thickness, lineType=cv2.LINE_AA)


def draw_label(img, text, xy, color):
    x, y = int(xy[0]), int(xy[1])
    cv2.rectangle(img, (x, max(0, y - 16)), (x + 160, y + 4), (0, 0, 0), -1)
    cv2.putText(img, text, (x + 2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def render_frame_triptych(frame_bundle, out_path, vis_scale):
    image = cv2.imread(str(frame_bundle["image_path"]))
    if image is None:
        raise FileNotFoundError(frame_bundle["image_path"])

    raw_img = image.copy()
    yolo_img = image.copy()
    rescored_img = image.copy()

    for record in frame_bundle["records"]:
        if record["decision_before"] == "keep" and record["proj_valid"]:
            draw_projected_box(raw_img, record["corners2d"], record["corner_valid"], (0, 255, 0), 2)
            x1, y1, _, _ = record["proj_bbox"]
            draw_label(raw_img, f"L {record['lidar_score']:.2f}", (x1, y1), (0, 255, 0))

    for yolo in frame_bundle["yolo_dets"]:
        x1, y1, x2, y2 = np.round(yolo["bbox"]).astype(np.int32)
        cv2.rectangle(yolo_img, (x1, y1), (x2, y2), (0, 215, 255), 2, cv2.LINE_AA)
        draw_label(yolo_img, f"Y {yolo['conf']:.2f}", (x1, y1), (0, 215, 255))

    for record in frame_bundle["records"]:
        if record["decision_after"] != "keep" or not record["proj_valid"]:
            continue
        if record["decision_before"] == "drop":
            color = (0, 140, 255)
        elif record["matched"]:
            color = (0, 255, 0)
        else:
            color = (255, 200, 0)
        draw_projected_box(rescored_img, record["corners2d"], record["corner_valid"], color, 2)
        x1, y1, _, _ = record["proj_bbox"]
        draw_label(rescored_img, f"R {record['final_score']:.2f}", (x1, y1), color)

    top_h = 34
    panels = []
    titles = [
        f"Raw lidar keep >= {frame_bundle['raw_score_thresh']:.2f}",
        "YOLO vehicle detections",
        f"Rescored keep >= {frame_bundle['final_score_thresh']:.2f}",
    ]
    for img, title in zip([raw_img, yolo_img, rescored_img], titles):
        banner = np.zeros((top_h, img.shape[1], 3), dtype=np.uint8)
        cv2.putText(banner, title, (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        panels.append(np.vstack([banner, img]))

    canvas = np.concatenate(panels, axis=1)
    footer = np.zeros((42, canvas.shape[1], 3), dtype=np.uint8)
    footer_text = (
        f"frame={frame_bundle['frame_id']}  "
        f"suppress={frame_bundle['num_suppress']}  "
        f"promote={frame_bundle['num_promote']}  "
        f"matched_keep={frame_bundle['num_matched_keep']}"
    )
    cv2.putText(footer, footer_text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    canvas = np.vstack([canvas, footer])

    if vis_scale != 1.0:
        canvas = cv2.resize(canvas, None, fx=vis_scale, fy=vis_scale, interpolation=cv2.INTER_AREA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)


def summarize_audit(audit_rows):
    summary = {
        "num_detections": len(audit_rows),
        "num_matched_yolo": 0,
        "num_valid_projection": 0,
        "num_before_keep": 0,
        "num_after_keep": 0,
        "num_keep_to_drop": 0,
        "num_drop_to_keep": 0,
        "num_supported_keep": 0,
    }
    for row in audit_rows:
        if int(row["proj_valid"]) == 1:
            summary["num_valid_projection"] += 1
        if float(row["matched_yolo_conf"]) > 0:
            summary["num_matched_yolo"] += 1
        if row["decision_before"] == "keep":
            summary["num_before_keep"] += 1
        if row["decision_after"] == "keep":
            summary["num_after_keep"] += 1
        if row["decision_before"] == "keep" and row["decision_after"] == "drop":
            summary["num_keep_to_drop"] += 1
        if row["decision_before"] == "drop" and row["decision_after"] == "keep":
            summary["num_drop_to_keep"] += 1
        if row["change_reason"] in ("yolo_promoted_keep", "yolo_supported_keep"):
            summary["num_supported_keep"] += 1
    return summary


def select_frame_bundles(frame_bundles, max_support, max_suppress):
    bundles = list(frame_bundles.values())
    suppress_rank = sorted(
        bundles,
        key=lambda x: (x["num_suppress"], x["num_changes"], x["num_matched_keep"]),
        reverse=True,
    )
    support_rank = sorted(
        bundles,
        key=lambda x: (x["num_promote"], x["num_matched_keep"], x["num_changes"]),
        reverse=True,
    )

    selected_suppress = [b for b in suppress_rank if b["num_suppress"] > 0][:max_suppress]
    selected_support = [b for b in support_rank if (b["num_promote"] > 0 or b["num_matched_keep"] > 0)][:max_support]

    if len(selected_suppress) < max_suppress:
        seen = {b["frame_id"] for b in selected_suppress}
        for bundle in suppress_rank:
            if bundle["frame_id"] in seen:
                continue
            selected_suppress.append(bundle)
            seen.add(bundle["frame_id"])
            if len(selected_suppress) >= max_suppress:
                break

    if len(selected_support) < max_support:
        seen = {b["frame_id"] for b in selected_support}
        for bundle in support_rank:
            if bundle["frame_id"] in seen:
                continue
            selected_support.append(bundle)
            seen.add(bundle["frame_id"])
            if len(selected_support) >= max_support:
                break

    return selected_suppress[:max_suppress], selected_support[:max_support]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result_pkl", required=True, help="raw lidar-only result.pkl from tools/test.py")
    ap.add_argument("--info_pkl", required=True, help="DAIR val infos pkl")
    ap.add_argument("--calib_dir", required=True, help="training/calib directory with per-frame npz")
    ap.add_argument("--image_dir", required=True, help="DAIR image directory")
    ap.add_argument("--yolo_csv", required=True, help="YOLO detection CSV from run_yolo_to_csv.py")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--yolo_classes", default="2,5,7", help="vehicle-like COCO ids to keep")
    ap.add_argument("--match_iou_thresh", type=float, default=0.05)
    ap.add_argument("--match_center_dist_thresh", type=float, default=0.35)
    ap.add_argument("--raw_score_thresh", type=float, default=0.35)
    ap.add_argument("--final_score_thresh", type=float, default=0.35)
    ap.add_argument("--near_range_m", type=float, default=60.0)
    ap.add_argument("--eval_iou_thresh", type=float, default=0.5)
    ap.add_argument("--disable_rotated_iou", action="store_true")
    ap.add_argument("--max_support_frames", type=int, default=10)
    ap.add_argument("--max_suppress_frames", type=int, default=10)
    ap.add_argument("--vis_scale", type=float, default=0.5)
    return ap.parse_args()


def main():
    args = parse_args()
    result_pkl = Path(args.result_pkl)
    info_pkl = Path(args.info_pkl)
    calib_dir = Path(args.calib_dir)
    image_dir = Path(args.image_dir)
    yolo_csv = Path(args.yolo_csv)
    output_dir = Path(args.output_dir)
    vis_dir = output_dir / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    keep_classes = {int(x) for x in args.yolo_classes.split(",") if x.strip()}
    raw_result_list = load_result_list(result_pkl)
    gt_map = load_gt_map(info_pkl)
    yolo_map = load_yolo_map(yolo_csv, keep_classes=keep_classes)

    audit_rows = []
    rescored_result_list = []
    frame_bundles = {}

    for anno in raw_result_list:
        frame_id = stem_frame_id(anno.get("frame_id", ""))
        frame_key = norm_frame_id(frame_id)

        image_path = resolve_image_file(image_dir, frame_id)
        calib_path = resolve_calib_file(calib_dir, frame_id)
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(image_path)
        image_hw = image.shape[:2]
        cam_k, t_cam_from_lidar = load_calib(calib_path)

        boxes_lidar = np.asarray(anno.get("boxes_lidar", np.zeros((0, 7), dtype=np.float32)), dtype=np.float32).reshape(-1, 7)
        lidar_scores = np.asarray(anno.get("score", np.zeros((boxes_lidar.shape[0],), dtype=np.float32)), dtype=np.float32).reshape(-1)
        yolo_dets = yolo_map.get(frame_key, [])

        records = []
        for det_idx in range(boxes_lidar.shape[0]):
            box = boxes_lidar[det_idx]
            proj = project_box_to_image(box, cam_k, t_cam_from_lidar, image_hw=image_hw)
            range_m = float(np.linalg.norm(box[:2]))
            records.append(
                {
                    "det_idx": det_idx,
                    "box_lidar": box,
                    "lidar_score": float(lidar_scores[det_idx]),
                    "range_m": range_m,
                    "proj_valid": bool(proj["valid"]),
                    "proj_bbox": proj["bbox"],
                    "corners2d": proj["corners2d"],
                    "corner_valid": proj["corner_valid"],
                }
            )

        greedy_match_records(
            records,
            yolo_dets,
            match_iou_thresh=float(args.match_iou_thresh),
            match_center_thresh=float(args.match_center_dist_thresh),
        )

        final_scores = np.zeros((boxes_lidar.shape[0],), dtype=np.float32)
        keep_mask = np.zeros((boxes_lidar.shape[0],), dtype=bool)
        num_suppress = 0
        num_promote = 0
        num_changes = 0
        num_matched_keep = 0

        for record in records:
            det_idx = record["det_idx"]
            lidar_score = float(record["lidar_score"])
            matched_yolo = record["matched_yolo"]
            iou2d = float(record["iou2d"])
            center_norm = float(record["center_dist_norm"])
            range_m = float(record["range_m"])

            if matched_yolo is not None:
                matched_yolo_conf = float(matched_yolo["conf"])
                final_score = clip01(0.75 * matched_yolo_conf + 0.25 * lidar_score + 0.20 * iou2d - 0.10 * center_norm)
            elif range_m <= float(args.near_range_m):
                matched_yolo_conf = 0.0
                final_score = clip01(0.25 * lidar_score)
            else:
                matched_yolo_conf = 0.0
                final_score = clip01(0.60 * lidar_score)

            decision_before = "keep" if lidar_score >= float(args.raw_score_thresh) else "drop"
            decision_after = "keep" if final_score >= float(args.final_score_thresh) else "drop"

            if matched_yolo is not None and decision_after == "keep" and decision_before == "drop":
                change_reason = "yolo_promoted_keep"
            elif matched_yolo is not None and decision_after == "keep":
                change_reason = "yolo_supported_keep"
            elif matched_yolo is not None and decision_after == "drop":
                change_reason = "weak_yolo_match_drop"
            elif not record["proj_valid"] and range_m <= float(args.near_range_m):
                change_reason = "no_valid_projection_near_penalty"
            elif not record["proj_valid"] and range_m > float(args.near_range_m):
                change_reason = "no_valid_projection_far_penalty"
            elif range_m <= float(args.near_range_m):
                change_reason = "no_yolo_near_penalty"
            else:
                change_reason = "no_yolo_far_penalty"

            record["matched_yolo_conf"] = matched_yolo_conf
            record["final_score"] = float(final_score)
            record["decision_before"] = decision_before
            record["decision_after"] = decision_after
            record["change_reason"] = change_reason

            final_scores[det_idx] = final_score
            keep_mask[det_idx] = decision_after == "keep"

            if decision_before == "keep" and decision_after == "drop":
                num_suppress += 1
                num_changes += 1
            if decision_before == "drop" and decision_after == "keep":
                num_promote += 1
                num_changes += 1
            if decision_after == "keep" and matched_yolo is not None:
                num_matched_keep += 1

            proj_bbox = record["proj_bbox"] if record["proj_bbox"] is not None else [np.nan, np.nan, np.nan, np.nan]
            audit_rows.append(
                {
                    "frame_id": frame_id,
                    "det_idx": int(det_idx),
                    "lidar_score": f"{lidar_score:.6f}",
                    "final_score": f"{final_score:.6f}",
                    "matched_yolo_conf": f"{matched_yolo_conf:.6f}",
                    "matched_yolo_cls": matched_yolo["cls"] if matched_yolo is not None else -1,
                    "matched_yolo_name": matched_yolo["name"] if matched_yolo is not None else "",
                    "iou2d": f"{iou2d:.6f}",
                    "center_dist_norm": f"{center_norm:.6f}",
                    "range_m": f"{range_m:.6f}",
                    "proj_valid": int(record["proj_valid"]),
                    "proj_x1": f"{float(proj_bbox[0]):.3f}",
                    "proj_y1": f"{float(proj_bbox[1]):.3f}",
                    "proj_x2": f"{float(proj_bbox[2]):.3f}",
                    "proj_y2": f"{float(proj_bbox[3]):.3f}",
                    "decision_before": decision_before,
                    "decision_after": decision_after,
                    "change_reason": change_reason,
                }
            )

        rescored_result_list.append(filter_anno_with_new_scores(anno, keep_mask, final_scores))
        frame_bundles[frame_id] = {
            "frame_id": frame_id,
            "image_path": image_path,
            "yolo_dets": yolo_dets,
            "records": records,
            "raw_score_thresh": float(args.raw_score_thresh),
            "final_score_thresh": float(args.final_score_thresh),
            "num_suppress": int(num_suppress),
            "num_promote": int(num_promote),
            "num_changes": int(num_changes),
            "num_matched_keep": int(num_matched_keep),
        }

    raw_metrics = evaluate_det_list(
        raw_result_list,
        gt_map,
        score_thresh=float(args.raw_score_thresh),
        iou_thresh=float(args.eval_iou_thresh),
        use_rotated_iou=not bool(args.disable_rotated_iou),
    )
    rescored_metrics = evaluate_det_list(
        rescored_result_list,
        gt_map,
        score_thresh=0.0,
        iou_thresh=float(args.eval_iou_thresh),
        use_rotated_iou=not bool(args.disable_rotated_iou),
    )
    audit_summary = summarize_audit(audit_rows)

    rescored_pkl_path = output_dir / "rescored_result.pkl"
    with open(rescored_pkl_path, "wb") as f:
        pickle.dump(rescored_result_list, f)

    audit_csv_path = output_dir / "decision_audit.csv"
    audit_fieldnames = [
        "frame_id",
        "det_idx",
        "lidar_score",
        "final_score",
        "matched_yolo_conf",
        "matched_yolo_cls",
        "matched_yolo_name",
        "iou2d",
        "center_dist_norm",
        "range_m",
        "proj_valid",
        "proj_x1",
        "proj_y1",
        "proj_x2",
        "proj_y2",
        "decision_before",
        "decision_after",
        "change_reason",
    ]
    with open(audit_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=audit_fieldnames)
        writer.writeheader()
        writer.writerows(audit_rows)

    selected_suppress, selected_support = select_frame_bundles(
        frame_bundles,
        max_support=int(args.max_support_frames),
        max_suppress=int(args.max_suppress_frames),
    )
    selected_manifest = {
        "suppression_frames": [bundle["frame_id"] for bundle in selected_suppress],
        "support_frames": [bundle["frame_id"] for bundle in selected_support],
    }
    for idx, bundle in enumerate(selected_suppress, 1):
        render_frame_triptych(bundle, vis_dir / f"suppress_{idx:02d}_{bundle['frame_id']}.jpg", vis_scale=float(args.vis_scale))
    for idx, bundle in enumerate(selected_support, 1):
        render_frame_triptych(bundle, vis_dir / f"support_{idx:02d}_{bundle['frame_id']}.jpg", vis_scale=float(args.vis_scale))

    metrics_json = {
        "inputs": {
            "result_pkl": str(result_pkl),
            "info_pkl": str(info_pkl),
            "calib_dir": str(calib_dir),
            "image_dir": str(image_dir),
            "yolo_csv": str(yolo_csv),
        },
        "config": {
            "yolo_classes": sorted(list(keep_classes)),
            "match_iou_thresh": float(args.match_iou_thresh),
            "match_center_dist_thresh": float(args.match_center_dist_thresh),
            "raw_score_thresh": float(args.raw_score_thresh),
            "final_score_thresh": float(args.final_score_thresh),
            "near_range_m": float(args.near_range_m),
            "eval_iou_thresh": float(args.eval_iou_thresh),
            "use_rotated_iou": not bool(args.disable_rotated_iou),
        },
        "raw_metrics": raw_metrics,
        "rescored_metrics": rescored_metrics,
        "audit_summary": audit_summary,
        "selected_frames": selected_manifest,
        "outputs": {
            "rescored_result_pkl": str(rescored_pkl_path),
            "decision_audit_csv": str(audit_csv_path),
            "visualization_dir": str(vis_dir),
        },
    }
    metrics_json_path = output_dir / "rescored_metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_json, f, indent=2, ensure_ascii=False)

    metrics_txt_path = output_dir / "rescored_metrics.txt"
    lines = [
        "DAIR image-led B0 rescoring",
        "",
        f"raw       : tp={raw_metrics['tp']} fp={raw_metrics['fp']} fn={raw_metrics['fn']} "
        f"precision={raw_metrics['precision']:.6f} recall={raw_metrics['recall']:.6f} "
        f"kept={raw_metrics['kept_predictions']}",
        f"rescored  : tp={rescored_metrics['tp']} fp={rescored_metrics['fp']} fn={rescored_metrics['fn']} "
        f"precision={rescored_metrics['precision']:.6f} recall={rescored_metrics['recall']:.6f} "
        f"kept={rescored_metrics['kept_predictions']}",
        "",
        f"audit detections     : {audit_summary['num_detections']}",
        f"matched yolo         : {audit_summary['num_matched_yolo']}",
        f"valid projections    : {audit_summary['num_valid_projection']}",
        f"before keep          : {audit_summary['num_before_keep']}",
        f"after keep           : {audit_summary['num_after_keep']}",
        f"keep -> drop         : {audit_summary['num_keep_to_drop']}",
        f"drop -> keep         : {audit_summary['num_drop_to_keep']}",
        f"yolo supported keep  : {audit_summary['num_supported_keep']}",
        "",
        f"rescored_result.pkl  : {rescored_pkl_path}",
        f"decision_audit.csv   : {audit_csv_path}",
        f"visualizations       : {vis_dir}",
    ]
    metrics_txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    selected_manifest_path = output_dir / "selected_frames.json"
    with open(selected_manifest_path, "w", encoding="utf-8") as f:
        json.dump(selected_manifest, f, indent=2, ensure_ascii=False)

    print(json.dumps(metrics_json, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
