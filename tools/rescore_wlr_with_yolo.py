#!/usr/bin/env python3
"""
Image-led B0 validation on WLR:
- LiDAR result.pkl provides 3D geometry
- matched camera YOLO detections drive keep / rescore / suppress decisions
"""

import argparse
import csv
import json
import pickle
from pathlib import Path

import cv2
import numpy as np

import rescore_dair_with_yolo as b0


def load_wlr_info_map(info_pkl: Path):
    with open(info_pkl, "rb") as f:
        infos = pickle.load(f)

    out = {}
    for info in infos:
        lidar_id = b0.stem_frame_id(info.get("point_cloud", {}).get("lidar_idx", info.get("frame_id", "")))
        key = b0.norm_frame_id(lidar_id)
        ann = info.get("annotations", {}) or {}
        calib = info.get("calib", {}) or {}
        out[key] = {
            "lidar_frame_id": lidar_id,
            "gt_boxes": np.asarray(
                ann.get("gt_boxes_lidar", np.zeros((0, 7), dtype=np.float32)),
                dtype=np.float32,
            ).reshape(-1, 7),
            "cam_k": np.asarray(calib.get("cam_K", np.eye(3, dtype=np.float32)), dtype=np.float32),
            "t_cam_from_lidar": np.asarray(
                calib.get("T_cam_from_lidar", np.eye(4, dtype=np.float32)),
                dtype=np.float32,
            ),
        }
    return out


def load_gt_map_from_info_map(info_map):
    return {key: val["gt_boxes"] for key, val in info_map.items()}


def evaluate_det_list_wlr3d(det_list, gt_map, score_thresh, iou_thresh):
    try:
        import torch
        from pcdet.ops.iou3d_nms import iou3d_nms_utils
    except Exception as e:
        raise RuntimeError(f"WLR 3D IoU eval unavailable: {e}") from e

    pred_map = {}
    for anno in det_list:
        pred_map[b0.norm_frame_id(anno.get("frame_id", ""))] = anno

    common = sorted(set(pred_map.keys()) & set(gt_map.keys()))
    if not common:
        raise RuntimeError("No common frame ids between predictions and GT infos")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_kept = 0
    all_iou = []

    for key in common:
        anno = pred_map[key]
        gt_boxes = gt_map[key]
        pred_boxes = np.asarray(anno.get("boxes_lidar", np.zeros((0, 7), dtype=np.float32)), dtype=np.float32).reshape(-1, 7)
        pred_scores = np.asarray(anno.get("score", np.zeros((pred_boxes.shape[0],), dtype=np.float32)), dtype=np.float32).reshape(-1)
        keep = pred_scores >= score_thresh
        pred_boxes = pred_boxes[keep]
        total_kept += int(keep.sum())

        if pred_boxes.shape[0] == 0 or gt_boxes.shape[0] == 0:
            continue

        boxes_pred = torch.from_numpy(pred_boxes).to(device)
        boxes_gt = torch.from_numpy(gt_boxes).to(device)
        iou = iou3d_nms_utils.boxes_iou3d_gpu(boxes_pred, boxes_gt).detach().cpu().numpy()
        if iou.size == 0:
            continue

        all_iou.append(iou)
        max_iou = iou.max(axis=1)
        tp_i = int((max_iou > iou_thresh).sum())
        total_tp += tp_i
        total_fp += int((max_iou <= iou_thresh).sum())
        total_fn += max(0, int(gt_boxes.shape[0] - tp_i))

    precision = float(total_tp / max(total_tp + total_fp, 1))
    recall = float(total_tp / max(total_tp + total_fn, 1))
    mean_iou = float(np.mean([i.max() for i in all_iou if i.size > 0])) if all_iou else 0.0
    return {
        "num_common_frames": len(common),
        "score_thresh": float(score_thresh),
        "iou_thresh": float(iou_thresh),
        "tp": int(total_tp),
        "fp": int(total_fp),
        "fn": int(total_fn),
        "precision": precision,
        "recall": recall,
        "kept_predictions": int(total_kept),
        "mean_iou": mean_iou,
    }


def load_lidar_to_cam_map(csv_path: Path):
    out = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lidar_raw = row.get("lidar_frame", row.get("frame_id", "")).strip()
            cam_raw = row.get("cam_frame", "").strip()
            if lidar_raw == "" or cam_raw == "":
                continue
            out[b0.norm_frame_id(lidar_raw)] = b0.stem_frame_id(cam_raw)
    return out


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result_pkl", required=True, help="raw WLR result.pkl from tools/test.py")
    ap.add_argument("--info_pkl", required=True, help="WLR infos pkl with GT and calib")
    ap.add_argument("--image_dir", required=True, help="matched camera image directory")
    ap.add_argument("--frame_map_csv", required=True, help="lidar_frame -> cam_frame mapping csv")
    ap.add_argument("--yolo_csv", required=True, help="YOLO detection CSV on camera frames")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--yolo_classes", default="2,5,7", help="vehicle-like COCO ids to keep")
    ap.add_argument("--match_iou_thresh", type=float, default=0.05)
    ap.add_argument("--match_center_dist_thresh", type=float, default=0.35)
    ap.add_argument("--raw_score_thresh", type=float, default=0.25)
    ap.add_argument("--final_score_thresh", type=float, default=0.25)
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
    image_dir = Path(args.image_dir)
    frame_map_csv = Path(args.frame_map_csv)
    yolo_csv = Path(args.yolo_csv)
    output_dir = Path(args.output_dir)
    vis_dir = output_dir / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    keep_classes = {int(x) for x in args.yolo_classes.split(",") if x.strip()}
    raw_result_list = b0.load_result_list(result_pkl)
    info_map = load_wlr_info_map(info_pkl)
    gt_map = load_gt_map_from_info_map(info_map)
    lidar_to_cam_map = load_lidar_to_cam_map(frame_map_csv)
    yolo_map = b0.load_yolo_map(yolo_csv, keep_classes=keep_classes)

    audit_rows = []
    rescored_result_list = []
    frame_bundles = {}

    for anno in raw_result_list:
        lidar_frame_id = b0.stem_frame_id(anno.get("frame_id", ""))
        lidar_key = b0.norm_frame_id(lidar_frame_id)
        if lidar_key not in info_map:
            raise KeyError(f"lidar frame {lidar_frame_id} missing from info_pkl")
        if lidar_key not in lidar_to_cam_map:
            raise KeyError(f"lidar frame {lidar_frame_id} missing from frame_map_csv")

        cam_frame_id = lidar_to_cam_map[lidar_key]
        cam_key = b0.norm_frame_id(cam_frame_id)
        meta = info_map[lidar_key]
        image_path = b0.resolve_image_file(image_dir, cam_frame_id)
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(image_path)
        image_hw = image.shape[:2]
        cam_k = meta["cam_k"]
        t_cam_from_lidar = meta["t_cam_from_lidar"]

        boxes_lidar = np.asarray(anno.get("boxes_lidar", np.zeros((0, 7), dtype=np.float32)), dtype=np.float32).reshape(-1, 7)
        lidar_scores = np.asarray(anno.get("score", np.zeros((boxes_lidar.shape[0],), dtype=np.float32)), dtype=np.float32).reshape(-1)
        yolo_dets = yolo_map.get(cam_key, [])

        records = []
        for det_idx in range(boxes_lidar.shape[0]):
            box = boxes_lidar[det_idx]
            proj = b0.project_box_to_image(box, cam_k, t_cam_from_lidar, image_hw=image_hw)
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

        b0.greedy_match_records(
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
                final_score = b0.clip01(0.75 * matched_yolo_conf + 0.25 * lidar_score + 0.20 * iou2d - 0.10 * center_norm)
            elif range_m <= float(args.near_range_m):
                matched_yolo_conf = 0.0
                final_score = b0.clip01(0.25 * lidar_score)
            else:
                matched_yolo_conf = 0.0
                final_score = b0.clip01(0.60 * lidar_score)

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
                    "lidar_frame_id": lidar_frame_id,
                    "cam_frame_id": cam_frame_id,
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

        rescored_result_list.append(b0.filter_anno_with_new_scores(anno, keep_mask, final_scores))
        frame_bundles[lidar_frame_id] = {
            "frame_id": f"{lidar_frame_id} -> cam {cam_frame_id}",
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

    raw_metrics = evaluate_det_list_wlr3d(
        raw_result_list,
        gt_map,
        score_thresh=float(args.raw_score_thresh),
        iou_thresh=float(args.eval_iou_thresh),
    )
    rescored_metrics = evaluate_det_list_wlr3d(
        rescored_result_list,
        gt_map,
        score_thresh=0.0,
        iou_thresh=float(args.eval_iou_thresh),
    )
    audit_summary = b0.summarize_audit(audit_rows)

    rescored_pkl_path = output_dir / "rescored_result.pkl"
    with open(rescored_pkl_path, "wb") as f:
        pickle.dump(rescored_result_list, f)

    audit_csv_path = output_dir / "decision_audit.csv"
    if audit_rows:
        with open(audit_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(audit_rows[0].keys()))
            writer.writeheader()
            writer.writerows(audit_rows)

    selected_suppress, selected_support = b0.select_frame_bundles(
        frame_bundles,
        max_support=int(args.max_support_frames),
        max_suppress=int(args.max_suppress_frames),
    )
    selected_manifest = {
        "suppression_frames": [bundle["frame_id"] for bundle in selected_suppress],
        "support_frames": [bundle["frame_id"] for bundle in selected_support],
    }
    for idx, bundle in enumerate(selected_suppress, 1):
        b0.render_frame_triptych(bundle, vis_dir / f"suppress_{idx:02d}.jpg", vis_scale=float(args.vis_scale))
    for idx, bundle in enumerate(selected_support, 1):
        b0.render_frame_triptych(bundle, vis_dir / f"support_{idx:02d}.jpg", vis_scale=float(args.vis_scale))

    metrics_json = {
        "inputs": {
            "result_pkl": str(result_pkl),
            "info_pkl": str(info_pkl),
            "image_dir": str(image_dir),
            "frame_map_csv": str(frame_map_csv),
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
            "eval_metric": "wlr_3d_iou",
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
        "WLR image-led B0 rescoring",
        "",
        f"raw       : tp={raw_metrics['tp']} fp={raw_metrics['fp']} fn={raw_metrics['fn']} "
        f"precision={raw_metrics['precision']:.6f} recall={raw_metrics['recall']:.6f} "
        f"mean_iou={raw_metrics['mean_iou']:.6f} "
        f"kept={raw_metrics['kept_predictions']}",
        f"rescored  : tp={rescored_metrics['tp']} fp={rescored_metrics['fp']} fn={rescored_metrics['fn']} "
        f"precision={rescored_metrics['precision']:.6f} recall={rescored_metrics['recall']:.6f} "
        f"mean_iou={rescored_metrics['mean_iou']:.6f} "
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

    with open(output_dir / "selected_frames.json", "w", encoding="utf-8") as f:
        json.dump(selected_manifest, f, indent=2, ensure_ascii=False)

    print(json.dumps(metrics_json, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
