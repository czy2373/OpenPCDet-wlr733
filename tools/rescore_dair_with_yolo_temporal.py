#!/usr/bin/env python3
"""
Temporal B1 validation on DAIR:
- reuse B0 image-led single-frame rescoring
- add short-track consistency correction on top of B0 scores
"""

import argparse
import json
import math
import pickle
from collections import deque
from pathlib import Path

import cv2
import numpy as np

import rescore_dair_with_yolo as b0
from iou_tracker import IoUTracker


def frame_sort_key(frame_id: str):
    stem = b0.stem_frame_id(frame_id)
    try:
        return (0, int(stem))
    except Exception:
        return (1, stem)


def frame_num(frame_id: str):
    stem = b0.stem_frame_id(frame_id)
    try:
        return int(stem)
    except Exception:
        return None


def should_reset_tracker(prev_frame_id: str, cur_frame_id: str, max_frame_gap: int) -> bool:
    if prev_frame_id is None:
        return False
    prev_num = frame_num(prev_frame_id)
    cur_num = frame_num(cur_frame_id)
    if prev_num is None or cur_num is None:
        return False
    if cur_num <= prev_num:
        return True
    return (cur_num - prev_num) > int(max_frame_gap)


def make_default_track_state(history_len: int):
    return {
        "support_history": deque(maxlen=history_len),
        "match_streak": 0,
        "miss_streak": 0,
        "seen_count": 0,
        "last_frame_id": "",
        "last_b0_score": 0.0,
        "last_center_xy": None,
        "last_dims": None,
        "last_proj_area": None,
    }


def labels_from_anno(anno, num_boxes: int):
    labels = np.asarray(anno.get("pred_labels", np.zeros((num_boxes,), dtype=np.int64)), dtype=np.int64).reshape(-1)
    if labels.shape[0] != num_boxes:
        labels = np.ones((num_boxes,), dtype=np.int64)
    if num_boxes > 0 and labels.shape[0] == 0:
        labels = np.ones((num_boxes,), dtype=np.int64)
    return labels


def support_flag_from_record(record):
    return int(record["matched_yolo_conf"] > 0.0 and record["decision_b0"] == "keep")


def bbox_area_xyxy(bbox):
    if bbox is None:
        return 0.0
    x1, y1, x2, y2 = [float(x) for x in bbox]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def relative_size_delta(cur_dims, prev_dims):
    if prev_dims is None:
        return 0.0
    cur_dims = np.asarray(cur_dims, dtype=np.float32).reshape(-1)
    prev_dims = np.asarray(prev_dims, dtype=np.float32).reshape(-1)
    if cur_dims.shape[0] != prev_dims.shape[0]:
        return 0.0
    denom = np.maximum(np.abs(prev_dims), 1e-3)
    return float(np.max(np.abs(cur_dims - prev_dims) / denom))


def geometry_consistency_stats(record, state):
    cur_center_xy = np.asarray(record["box_lidar"][:2], dtype=np.float32)
    cur_dims = np.asarray(record["box_lidar"][3:6], dtype=np.float32)
    prev_center_xy = state.get("last_center_xy")
    prev_dims = state.get("last_dims")
    prev_proj_area = state.get("last_proj_area")
    cur_proj_area = bbox_area_xyxy(record["proj_bbox"]) if record["proj_valid"] else 0.0

    if prev_center_xy is None:
        center_step_m = 0.0
    else:
        center_step_m = float(np.linalg.norm(cur_center_xy - np.asarray(prev_center_xy, dtype=np.float32)))

    size_rel_delta = relative_size_delta(cur_dims, prev_dims)

    if prev_proj_area is None or prev_proj_area <= 0.0 or cur_proj_area <= 0.0:
        proj_area_ratio = 1.0
    else:
        lo = min(float(prev_proj_area), float(cur_proj_area))
        hi = max(float(prev_proj_area), float(cur_proj_area))
        proj_area_ratio = float(hi / max(lo, 1e-6))

    return {
        "center_step_m": center_step_m,
        "size_rel_delta": size_rel_delta,
        "proj_area_ratio": proj_area_ratio,
        "cur_center_xy": cur_center_xy,
        "cur_dims": cur_dims,
        "cur_proj_area": cur_proj_area,
    }


def rescue_gate_decision(
    record,
    confirmed,
    prev_support_hist,
    prev_match_streak,
    prev_miss_streak,
    prev_seen_count,
    geom_stats,
    args,
):
    if bool(args.disable_rescue_gate):
        return True, "gate_disabled", "disabled"

    if float(record["range_m"]) < float(args.temporal_min_range):
        return False, "range_lt_temporal_min", "range_reject"
    matched_yolo_conf = float(record["matched_yolo_conf"])

    confirm_mode = "confirmed"
    if not bool(confirmed):
        soft_confirm_ok = (
            bool(args.rescue_allow_unconfirmed_with_yolo)
            and matched_yolo_conf >= float(args.rescue_soft_confirm_min_yolo_conf)
            and prev_seen_count >= int(args.rescue_soft_confirm_min_seen_count)
            and float(prev_support_hist) >= float(args.rescue_soft_confirm_min_support_hist)
            and int(prev_match_streak) >= int(args.rescue_soft_confirm_min_match_streak)
            and int(prev_miss_streak) <= int(args.rescue_soft_confirm_max_prev_miss_streak)
        )
        if not soft_confirm_ok:
            return False, "track_unconfirmed", "unconfirmed_reject"
        confirm_mode = "soft_yolo_confirm"
    required_seen_count = int(args.rescue_min_seen_count)
    if confirm_mode == "soft_yolo_confirm":
        required_seen_count = min(required_seen_count, int(args.rescue_soft_confirm_min_seen_count))
    if prev_seen_count < required_seen_count:
        return False, "seen_count_low", confirm_mode

    if matched_yolo_conf >= float(args.rescue_min_current_yolo_conf):
        min_support_hist = float(args.rescue_min_support_hist_with_yolo)
        min_match_streak = int(args.rescue_min_match_streak_with_yolo)
        min_b0_score = float(args.rescue_min_b0_score_with_yolo)
    else:
        min_support_hist = float(args.rescue_min_support_hist)
        min_match_streak = int(args.rescue_min_match_streak)
        min_b0_score = float(args.rescue_min_b0_score)

    if float(prev_support_hist) < min_support_hist:
        return False, "support_hist_low", confirm_mode
    if int(prev_match_streak) < min_match_streak:
        return False, "match_streak_low", confirm_mode
    if int(prev_miss_streak) > int(args.rescue_max_prev_miss_streak):
        return False, "prev_miss_too_high", confirm_mode
    if float(record["b0_score"]) < min_b0_score:
        return False, "b0_score_low", confirm_mode
    if bool(args.rescue_require_proj_valid) and not bool(record["proj_valid"]):
        return False, "proj_invalid", confirm_mode
    if float(geom_stats["center_step_m"]) > float(args.rescue_max_center_step_m):
        return False, "center_step_too_large", confirm_mode
    if float(geom_stats["size_rel_delta"]) > float(args.rescue_max_size_rel_delta):
        return False, "size_delta_too_large", confirm_mode
    if float(geom_stats["proj_area_ratio"]) > float(args.rescue_max_proj_area_ratio):
        return False, "proj_area_ratio_too_large", confirm_mode
    return True, "pass", confirm_mode


def temporal_change_reason(record):
    if int(record.get("rescue_gate_applied", 0)) == 1 and int(record.get("rescue_gate_passed", 1)) == 0:
        return "temporal_rescue_blocked"
    if int(record.get("keep_protect_applied", 0)) == 1:
        return "temporal_keep_protected"
    if record["decision_b0"] == "drop" and record["decision_b1"] == "keep":
        return "temporal_rescue_keep"
    if record["decision_b0"] == "keep" and record["decision_b1"] == "drop":
        return "temporal_suppress_drop"
    if record["decision_b1"] == "keep" and record["temporal_bonus"] > record["temporal_penalty"] + 1e-9:
        return "temporal_supported_keep"
    if record["decision_b1"] == "drop" and record["temporal_penalty"] > record["temporal_bonus"] + 1e-9:
        return "temporal_penalized_drop"
    return "temporal_no_change"


def build_b0_records_for_frame(anno, yolo_dets, image_hw, cam_k, t_cam_from_lidar, args):
    boxes_lidar = np.asarray(anno.get("boxes_lidar", np.zeros((0, 7), dtype=np.float32)), dtype=np.float32).reshape(-1, 7)
    lidar_scores = np.asarray(anno.get("score", np.zeros((boxes_lidar.shape[0],), dtype=np.float32)), dtype=np.float32).reshape(-1)

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

    b0_scores = np.zeros((boxes_lidar.shape[0],), dtype=np.float32)
    b0_keep_mask = np.zeros((boxes_lidar.shape[0],), dtype=bool)

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
        decision_b0 = "keep" if final_score >= float(args.b0_score_thresh) else "drop"

        if matched_yolo is not None and decision_b0 == "keep" and decision_before == "drop":
            change_reason = "yolo_promoted_keep"
        elif matched_yolo is not None and decision_b0 == "keep":
            change_reason = "yolo_supported_keep"
        elif matched_yolo is not None and decision_b0 == "drop":
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
        record["b0_score"] = float(final_score)
        record["decision_before"] = decision_before
        record["decision_b0"] = decision_b0
        record["change_reason"] = change_reason

        b0_scores[det_idx] = final_score
        b0_keep_mask[det_idx] = decision_b0 == "keep"

    return records, boxes_lidar, b0_scores, b0_keep_mask


def compute_temporal_scores(records, track_assignments, track_states, frame_id, args):
    b1_scores = np.zeros((len(records),), dtype=np.float32)
    b1_keep_mask = np.zeros((len(records),), dtype=bool)

    for record, track_info in zip(records, track_assignments):
        track_id = int(track_info["track_id"])
        state = track_states.get(track_id)
        if state is None:
            state = make_default_track_state(history_len=int(args.history_len))
            track_states[track_id] = state

        prev_support_hist = float(np.mean(state["support_history"])) if len(state["support_history"]) > 0 else 0.0
        prev_match_streak = int(state["match_streak"])
        prev_miss_streak = int(state["miss_streak"])
        prev_seen_count = int(state["seen_count"])
        geom_stats = geometry_consistency_stats(record, state)

        support_now = support_flag_from_record(record)
        confirmed = prev_seen_count >= int(args.confirm_min_hits)

        apply_temporal = confirmed or support_now == 1 or float(record["range_m"]) >= float(args.temporal_min_range)
        if apply_temporal:
            temporal_bonus = float(args.bonus_support_w) * prev_support_hist + float(args.bonus_streak_w) * min(prev_match_streak, int(args.streak_cap))
            temporal_penalty = float(args.penalty_miss_w) * min(prev_miss_streak, int(args.streak_cap))
        else:
            temporal_bonus = 0.0
            temporal_penalty = 0.0

        b1_score = b0.clip01(float(record["b0_score"]) + temporal_bonus - temporal_penalty)
        keep_thresh = float(args.continue_thresh) if confirmed else float(args.birth_thresh)
        decision_b1_pre_gate = "keep" if b1_score >= keep_thresh else "drop"
        decision_b1 = decision_b1_pre_gate
        rescue_gate_applied = 0
        rescue_gate_passed = 1
        rescue_gate_reason = "not_needed"
        rescue_confirm_mode = "not_needed"
        keep_protect_applied = 0
        keep_protect_reason = "not_needed"

        if record["decision_b0"] == "drop" and decision_b1_pre_gate == "keep":
            rescue_gate_applied = 1
            rescue_gate_passed, rescue_gate_reason, rescue_confirm_mode = rescue_gate_decision(
                record=record,
                confirmed=confirmed,
                prev_support_hist=prev_support_hist,
                prev_match_streak=prev_match_streak,
                prev_miss_streak=prev_miss_streak,
                prev_seen_count=prev_seen_count,
                geom_stats=geom_stats,
                args=args,
            )
            if not rescue_gate_passed:
                decision_b1 = "drop"
                b1_score = max(0.0, math.nextafter(keep_thresh, 0.0))

        protect_keep = (
            bool(args.protect_b0_keep_with_current_yolo)
            and record["decision_b0"] == "keep"
            and decision_b1 == "drop"
            and float(record["range_m"]) >= float(args.keep_protect_min_range)
            and float(record["matched_yolo_conf"]) >= float(args.keep_protect_min_yolo_conf)
        )
        if protect_keep:
            decision_b1 = "keep"
            keep_protect_applied = 1
            keep_protect_reason = "current_yolo_support"
            b1_score = max(float(record["b0_score"]), float(keep_thresh))

        record["track_id"] = track_id
        record["support_hist"] = prev_support_hist
        record["match_streak"] = prev_match_streak
        record["miss_streak"] = prev_miss_streak
        record["confirmed_track"] = int(confirmed)
        record["keep_thresh_b1"] = keep_thresh
        record["temporal_bonus"] = temporal_bonus
        record["temporal_penalty"] = temporal_penalty
        record["center_step_m"] = float(geom_stats["center_step_m"])
        record["size_rel_delta"] = float(geom_stats["size_rel_delta"])
        record["proj_area_ratio"] = float(geom_stats["proj_area_ratio"])
        record["decision_b1_pre_gate"] = decision_b1_pre_gate
        record["rescue_gate_applied"] = int(rescue_gate_applied)
        record["rescue_gate_passed"] = int(rescue_gate_passed)
        record["rescue_gate_reason"] = rescue_gate_reason
        record["rescue_confirm_mode"] = rescue_confirm_mode
        record["keep_protect_applied"] = int(keep_protect_applied)
        record["keep_protect_reason"] = keep_protect_reason
        record["b1_score"] = float(b1_score)
        record["decision_b1"] = decision_b1
        record["temporal_change_reason"] = temporal_change_reason(record)

        b1_scores[record["det_idx"]] = b1_score
        b1_keep_mask[record["det_idx"]] = decision_b1 == "keep"

        if support_now == 1:
            new_match_streak = prev_match_streak + 1
            new_miss_streak = 0
        else:
            new_match_streak = 0
            new_miss_streak = prev_miss_streak + 1

        state["support_history"].append(support_now)
        state["match_streak"] = new_match_streak
        state["miss_streak"] = new_miss_streak
        state["seen_count"] = prev_seen_count + 1
        state["last_frame_id"] = frame_id
        state["last_b0_score"] = float(record["b0_score"])
        state["last_center_xy"] = geom_stats["cur_center_xy"]
        state["last_dims"] = geom_stats["cur_dims"]
        state["last_proj_area"] = float(geom_stats["cur_proj_area"])

    return b1_scores, b1_keep_mask


def draw_b0_box(img, record):
    if record["decision_b0"] != "keep" or not record["proj_valid"]:
        return
    if record["decision_before"] == "drop":
        color = (0, 140, 255)
    elif record["matched_yolo_conf"] > 0.0:
        color = (0, 255, 0)
    else:
        color = (255, 200, 0)
    b0.draw_projected_box(img, record["corners2d"], record["corner_valid"], color, 2)
    x1, y1, _, _ = record["proj_bbox"]
    b0.draw_label(img, f"B0 {record['b0_score']:.2f}", (x1, y1), color)


def draw_b1_box(img, record):
    if record["decision_b1"] != "keep" or not record["proj_valid"]:
        return
    if record["decision_b0"] == "drop":
        color = (255, 180, 0)
    elif record["temporal_bonus"] > record["temporal_penalty"] + 1e-9:
        color = (0, 255, 255)
    else:
        color = (0, 255, 0)
    b0.draw_projected_box(img, record["corners2d"], record["corner_valid"], color, 2)
    x1, y1, _, _ = record["proj_bbox"]
    label = f"T{record['track_id']} {record['b0_score']:.2f}->{record['b1_score']:.2f}"
    b0.draw_label(img, label, (x1, y1), color)


def render_frame_quad(frame_bundle, out_path, vis_scale):
    image = cv2.imread(str(frame_bundle["image_path"]))
    if image is None:
        raise FileNotFoundError(frame_bundle["image_path"])

    raw_img = image.copy()
    yolo_img = image.copy()
    b0_img = image.copy()
    b1_img = image.copy()

    for record in frame_bundle["records"]:
        if record["decision_before"] == "keep" and record["proj_valid"]:
            b0.draw_projected_box(raw_img, record["corners2d"], record["corner_valid"], (0, 255, 0), 2)
            x1, y1, _, _ = record["proj_bbox"]
            b0.draw_label(raw_img, f"L {record['lidar_score']:.2f}", (x1, y1), (0, 255, 0))

    for yolo in frame_bundle["yolo_dets"]:
        x1, y1, x2, y2 = np.round(yolo["bbox"]).astype(np.int32)
        cv2.rectangle(yolo_img, (x1, y1), (x2, y2), (0, 215, 255), 2, cv2.LINE_AA)
        b0.draw_label(yolo_img, f"Y {yolo['conf']:.2f}", (x1, y1), (0, 215, 255))

    for record in frame_bundle["records"]:
        draw_b0_box(b0_img, record)
        draw_b1_box(b1_img, record)

    top_h = 34
    titles = [
        f"Raw lidar keep >= {frame_bundle['raw_score_thresh']:.2f}",
        "YOLO vehicle detections",
        f"B0 keep >= {frame_bundle['b0_score_thresh']:.2f}",
        f"B1 temporal keep",
    ]
    panels = []
    for img, title in zip([raw_img, yolo_img, b0_img, b1_img], titles):
        banner = np.zeros((top_h, img.shape[1], 3), dtype=np.uint8)
        cv2.putText(banner, title, (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
        panels.append(np.vstack([banner, img]))

    canvas = np.concatenate(panels, axis=1)
    footer = np.zeros((44, canvas.shape[1], 3), dtype=np.uint8)
    footer_text = (
        f"frame={frame_bundle['frame_id']}  "
        f"b0_drop->b1_keep={frame_bundle['num_temporal_rescue']}  "
        f"b0_keep->b1_drop={frame_bundle['num_temporal_suppress']}  "
        f"confirmed={frame_bundle['num_confirmed_keep']}"
    )
    cv2.putText(footer, footer_text, (12, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    canvas = np.vstack([canvas, footer])

    if vis_scale != 1.0:
        canvas = cv2.resize(canvas, None, fx=vis_scale, fy=vis_scale, interpolation=cv2.INTER_AREA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)


def summarize_b1_audit(audit_rows):
    summary = {
        "num_detections": len(audit_rows),
        "b0_drop_to_b1_keep": 0,
        "b0_keep_to_b1_drop": 0,
        "rescue_gate_blocked": 0,
        "confirmed_tracks": 0,
        "matched_yolo": 0,
        "support_now": 0,
    }
    for row in audit_rows:
        if int(row["confirmed_track"]) == 1:
            summary["confirmed_tracks"] += 1
        if float(row["matched_yolo_conf"]) > 0.0:
            summary["matched_yolo"] += 1
        if int(row["support_now"]) == 1:
            summary["support_now"] += 1
        if int(row.get("rescue_gate_applied", 0)) == 1 and int(row.get("rescue_gate_passed", 1)) == 0:
            summary["rescue_gate_blocked"] += 1
        if row["decision_b0"] == "drop" and row["decision_b1"] == "keep":
            summary["b0_drop_to_b1_keep"] += 1
        if row["decision_b0"] == "keep" and row["decision_b1"] == "drop":
            summary["b0_keep_to_b1_drop"] += 1
    return summary


def select_frame_bundles(frame_bundles, max_rescue, max_suppress):
    bundles = list(frame_bundles.values())
    rescue_rank = sorted(bundles, key=lambda x: (x["num_temporal_rescue"], x["num_b1_keep"], x["num_confirmed_keep"]), reverse=True)
    suppress_rank = sorted(bundles, key=lambda x: (x["num_temporal_suppress"], x["num_b1_keep"], x["num_confirmed_keep"]), reverse=True)

    rescue_frames = [b for b in rescue_rank if b["num_temporal_rescue"] > 0][:max_rescue]
    suppress_frames = [b for b in suppress_rank if b["num_temporal_suppress"] > 0][:max_suppress]
    return rescue_frames, suppress_frames


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result_pkl", required=True)
    ap.add_argument("--info_pkl", required=True)
    ap.add_argument("--calib_dir", required=True)
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--yolo_csv", required=True)
    ap.add_argument("--output_dir", required=True)

    ap.add_argument("--yolo_classes", default="2,5,7")
    ap.add_argument("--match_iou_thresh", type=float, default=0.05)
    ap.add_argument("--match_center_dist_thresh", type=float, default=0.35)
    ap.add_argument("--raw_score_thresh", type=float, default=0.35)
    ap.add_argument("--b0_score_thresh", type=float, default=0.35)
    ap.add_argument("--near_range_m", type=float, default=60.0)
    ap.add_argument("--eval_iou_thresh", type=float, default=0.5)
    ap.add_argument("--disable_rotated_iou", action="store_true")

    ap.add_argument("--tracker_iou", type=float, default=0.3)
    ap.add_argument("--tracker_max_age", type=int, default=5)
    ap.add_argument("--history_len", type=int, default=5)
    ap.add_argument("--max_frame_gap", type=int, default=5)
    ap.add_argument("--confirm_min_hits", type=int, default=2)
    ap.add_argument("--birth_thresh", type=float, default=0.35)
    ap.add_argument("--continue_thresh", type=float, default=0.28)
    ap.add_argument("--bonus_support_w", type=float, default=0.12)
    ap.add_argument("--bonus_streak_w", type=float, default=0.03)
    ap.add_argument("--penalty_miss_w", type=float, default=0.08)
    ap.add_argument("--streak_cap", type=int, default=2)
    ap.add_argument("--temporal_min_range", type=float, default=50.0)
    ap.add_argument("--disable_rescue_gate", action="store_true")
    ap.add_argument("--rescue_min_seen_count", type=int, default=2)
    ap.add_argument("--rescue_min_support_hist", type=float, default=0.6)
    ap.add_argument("--rescue_min_match_streak", type=int, default=1)
    ap.add_argument("--rescue_min_support_hist_with_yolo", type=float, default=0.4)
    ap.add_argument("--rescue_min_match_streak_with_yolo", type=int, default=0)
    ap.add_argument("--rescue_max_prev_miss_streak", type=int, default=0)
    ap.add_argument("--rescue_min_b0_score", type=float, default=0.25)
    ap.add_argument("--rescue_min_b0_score_with_yolo", type=float, default=0.20)
    ap.add_argument("--rescue_min_current_yolo_conf", type=float, default=0.25)
    ap.add_argument("--rescue_allow_unconfirmed_with_yolo", action="store_true")
    ap.add_argument("--rescue_soft_confirm_min_yolo_conf", type=float, default=0.22)
    ap.add_argument("--rescue_soft_confirm_min_seen_count", type=int, default=1)
    ap.add_argument("--rescue_soft_confirm_min_support_hist", type=float, default=1.0)
    ap.add_argument("--rescue_soft_confirm_min_match_streak", type=int, default=1)
    ap.add_argument("--rescue_soft_confirm_max_prev_miss_streak", type=int, default=0)
    ap.add_argument("--protect_b0_keep_with_current_yolo", action="store_true")
    ap.add_argument("--keep_protect_min_range", type=float, default=50.0)
    ap.add_argument("--keep_protect_min_yolo_conf", type=float, default=0.25)
    ap.add_argument("--no_rescue_require_proj_valid", dest="rescue_require_proj_valid", action="store_false")
    ap.add_argument("--rescue_max_center_step_m", type=float, default=5.0)
    ap.add_argument("--rescue_max_size_rel_delta", type=float, default=0.35)
    ap.add_argument("--rescue_max_proj_area_ratio", type=float, default=2.5)

    ap.add_argument("--max_rescue_frames", type=int, default=10)
    ap.add_argument("--max_suppress_frames", type=int, default=10)
    ap.add_argument("--vis_scale", type=float, default=0.5)
    ap.set_defaults(
        rescue_require_proj_valid=True,
        rescue_allow_unconfirmed_with_yolo=True,
        protect_b0_keep_with_current_yolo=True,
    )
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
    raw_result_list = b0.load_result_list(result_pkl)
    raw_result_list = sorted(raw_result_list, key=lambda anno: frame_sort_key(anno.get("frame_id", "")))
    gt_map = b0.load_gt_map(info_pkl)
    yolo_map = b0.load_yolo_map(yolo_csv, keep_classes=keep_classes)

    tracker = IoUTracker(iou_thresh=float(args.tracker_iou), max_age=int(args.tracker_max_age))
    track_states = {}
    reset_count = 0
    prev_frame_id = None

    audit_rows = []
    b0_result_list = []
    b1_result_list = []
    frame_bundles = {}

    for anno in raw_result_list:
        frame_id = b0.stem_frame_id(anno.get("frame_id", ""))
        frame_key = b0.norm_frame_id(frame_id)

        if should_reset_tracker(prev_frame_id, frame_id, max_frame_gap=int(args.max_frame_gap)):
            tracker = IoUTracker(iou_thresh=float(args.tracker_iou), max_age=int(args.tracker_max_age))
            track_states = {}
            reset_count += 1

        image_path = b0.resolve_image_file(image_dir, frame_id)
        calib_path = b0.resolve_calib_file(calib_dir, frame_id)
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(image_path)
        image_hw = image.shape[:2]
        cam_k, t_cam_from_lidar = b0.load_calib(calib_path)
        yolo_dets = yolo_map.get(frame_key, [])

        records, boxes_lidar, b0_scores, b0_keep_mask = build_b0_records_for_frame(
            anno, yolo_dets, image_hw=image_hw, cam_k=cam_k, t_cam_from_lidar=t_cam_from_lidar, args=args
        )
        labels = labels_from_anno(anno, num_boxes=boxes_lidar.shape[0])
        track_assignments = tracker.update(boxes_lidar, labels, b0_scores) if boxes_lidar.shape[0] > 0 else []

        b1_scores, b1_keep_mask = compute_temporal_scores(
            records,
            track_assignments,
            track_states=track_states,
            frame_id=frame_id,
            args=args,
        )

        b0_result_list.append(b0.filter_anno_with_new_scores(anno, b0_keep_mask, b0_scores))
        b1_result_list.append(b0.filter_anno_with_new_scores(anno, b1_keep_mask, b1_scores))

        num_temporal_rescue = 0
        num_temporal_suppress = 0
        num_b1_keep = 0
        num_confirmed_keep = 0

        for record in records:
            if record["decision_b0"] == "drop" and record["decision_b1"] == "keep":
                num_temporal_rescue += 1
            if record["decision_b0"] == "keep" and record["decision_b1"] == "drop":
                num_temporal_suppress += 1
            if record["decision_b1"] == "keep":
                num_b1_keep += 1
                if int(record["confirmed_track"]) == 1:
                    num_confirmed_keep += 1

            proj_bbox = record["proj_bbox"] if record["proj_bbox"] is not None else [np.nan, np.nan, np.nan, np.nan]
            audit_rows.append(
                {
                    "frame_id": frame_id,
                    "det_idx": int(record["det_idx"]),
                    "track_id": int(record.get("track_id", -1)),
                    "lidar_score": f"{float(record['lidar_score']):.6f}",
                    "b0_score": f"{float(record['b0_score']):.6f}",
                    "b1_score": f"{float(record['b1_score']):.6f}",
                    "matched_yolo_conf": f"{float(record['matched_yolo_conf']):.6f}",
                    "iou2d": f"{float(record['iou2d']):.6f}",
                    "center_dist_norm": f"{float(record['center_dist_norm']):.6f}",
                    "range_m": f"{float(record['range_m']):.6f}",
                    "proj_valid": int(record["proj_valid"]),
                    "proj_x1": f"{float(proj_bbox[0]):.3f}",
                    "proj_y1": f"{float(proj_bbox[1]):.3f}",
                    "proj_x2": f"{float(proj_bbox[2]):.3f}",
                    "proj_y2": f"{float(proj_bbox[3]):.3f}",
                    "support_now": support_flag_from_record(record),
                    "support_hist": f"{float(record['support_hist']):.6f}",
                    "match_streak": int(record["match_streak"]),
                    "miss_streak": int(record["miss_streak"]),
                    "confirmed_track": int(record["confirmed_track"]),
                    "keep_thresh_b1": f"{float(record['keep_thresh_b1']):.6f}",
                    "temporal_bonus": f"{float(record['temporal_bonus']):.6f}",
                    "temporal_penalty": f"{float(record['temporal_penalty']):.6f}",
                    "center_step_m": f"{float(record['center_step_m']):.6f}",
                    "size_rel_delta": f"{float(record['size_rel_delta']):.6f}",
                    "proj_area_ratio": f"{float(record['proj_area_ratio']):.6f}",
                    "decision_before": record["decision_before"],
                    "decision_b0": record["decision_b0"],
                    "decision_b1_pre_gate": record["decision_b1_pre_gate"],
                    "decision_b1": record["decision_b1"],
                    "change_reason_b0": record["change_reason"],
                    "rescue_gate_applied": int(record["rescue_gate_applied"]),
                    "rescue_gate_passed": int(record["rescue_gate_passed"]),
                    "rescue_gate_reason": record["rescue_gate_reason"],
                    "rescue_confirm_mode": record["rescue_confirm_mode"],
                    "keep_protect_applied": int(record["keep_protect_applied"]),
                    "keep_protect_reason": record["keep_protect_reason"],
                    "temporal_change_reason": record["temporal_change_reason"],
                }
            )

        frame_bundles[frame_id] = {
            "frame_id": frame_id,
            "image_path": image_path,
            "yolo_dets": yolo_dets,
            "records": records,
            "raw_score_thresh": float(args.raw_score_thresh),
            "b0_score_thresh": float(args.b0_score_thresh),
            "num_temporal_rescue": int(num_temporal_rescue),
            "num_temporal_suppress": int(num_temporal_suppress),
            "num_b1_keep": int(num_b1_keep),
            "num_confirmed_keep": int(num_confirmed_keep),
        }
        prev_frame_id = frame_id

    raw_metrics = b0.evaluate_det_list(
        raw_result_list,
        gt_map,
        score_thresh=float(args.raw_score_thresh),
        iou_thresh=float(args.eval_iou_thresh),
        use_rotated_iou=not bool(args.disable_rotated_iou),
    )
    b0_metrics = b0.evaluate_det_list(
        b0_result_list,
        gt_map,
        score_thresh=0.0,
        iou_thresh=float(args.eval_iou_thresh),
        use_rotated_iou=not bool(args.disable_rotated_iou),
    )
    b1_metrics = b0.evaluate_det_list(
        b1_result_list,
        gt_map,
        score_thresh=0.0,
        iou_thresh=float(args.eval_iou_thresh),
        use_rotated_iou=not bool(args.disable_rotated_iou),
    )
    temporal_summary = summarize_b1_audit(audit_rows)

    b1_result_path = output_dir / "b1_result.pkl"
    with open(b1_result_path, "wb") as f:
        pickle.dump(b1_result_list, f)

    b0_result_path = output_dir / "b0_result.pkl"
    with open(b0_result_path, "wb") as f:
        pickle.dump(b0_result_list, f)

    audit_csv_path = output_dir / "b1_decision_audit.csv"
    if audit_rows:
        import csv

        with open(audit_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(audit_rows[0].keys()))
            writer.writeheader()
            writer.writerows(audit_rows)

    rescue_frames, suppress_frames = select_frame_bundles(
        frame_bundles,
        max_rescue=int(args.max_rescue_frames),
        max_suppress=int(args.max_suppress_frames),
    )
    selected_manifest = {
        "temporal_rescue_frames": [bundle["frame_id"] for bundle in rescue_frames],
        "temporal_suppress_frames": [bundle["frame_id"] for bundle in suppress_frames],
    }
    for idx, bundle in enumerate(rescue_frames, 1):
        render_frame_quad(bundle, vis_dir / f"rescue_{idx:02d}_{bundle['frame_id']}.jpg", vis_scale=float(args.vis_scale))
    for idx, bundle in enumerate(suppress_frames, 1):
        render_frame_quad(bundle, vis_dir / f"suppress_{idx:02d}_{bundle['frame_id']}.jpg", vis_scale=float(args.vis_scale))

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
            "b0_score_thresh": float(args.b0_score_thresh),
            "near_range_m": float(args.near_range_m),
            "eval_iou_thresh": float(args.eval_iou_thresh),
            "tracker_iou": float(args.tracker_iou),
            "tracker_max_age": int(args.tracker_max_age),
            "history_len": int(args.history_len),
            "max_frame_gap": int(args.max_frame_gap),
            "confirm_min_hits": int(args.confirm_min_hits),
            "birth_thresh": float(args.birth_thresh),
            "continue_thresh": float(args.continue_thresh),
            "bonus_support_w": float(args.bonus_support_w),
            "bonus_streak_w": float(args.bonus_streak_w),
            "penalty_miss_w": float(args.penalty_miss_w),
            "streak_cap": int(args.streak_cap),
            "temporal_min_range": float(args.temporal_min_range),
            "disable_rescue_gate": bool(args.disable_rescue_gate),
            "rescue_min_seen_count": int(args.rescue_min_seen_count),
            "rescue_min_support_hist": float(args.rescue_min_support_hist),
            "rescue_min_match_streak": int(args.rescue_min_match_streak),
            "rescue_min_support_hist_with_yolo": float(args.rescue_min_support_hist_with_yolo),
            "rescue_min_match_streak_with_yolo": int(args.rescue_min_match_streak_with_yolo),
            "rescue_max_prev_miss_streak": int(args.rescue_max_prev_miss_streak),
            "rescue_min_b0_score": float(args.rescue_min_b0_score),
            "rescue_min_b0_score_with_yolo": float(args.rescue_min_b0_score_with_yolo),
            "rescue_min_current_yolo_conf": float(args.rescue_min_current_yolo_conf),
            "rescue_allow_unconfirmed_with_yolo": bool(args.rescue_allow_unconfirmed_with_yolo),
            "rescue_soft_confirm_min_yolo_conf": float(args.rescue_soft_confirm_min_yolo_conf),
            "rescue_soft_confirm_min_seen_count": int(args.rescue_soft_confirm_min_seen_count),
            "rescue_soft_confirm_min_support_hist": float(args.rescue_soft_confirm_min_support_hist),
            "rescue_soft_confirm_min_match_streak": int(args.rescue_soft_confirm_min_match_streak),
            "rescue_soft_confirm_max_prev_miss_streak": int(args.rescue_soft_confirm_max_prev_miss_streak),
            "protect_b0_keep_with_current_yolo": bool(args.protect_b0_keep_with_current_yolo),
            "keep_protect_min_range": float(args.keep_protect_min_range),
            "keep_protect_min_yolo_conf": float(args.keep_protect_min_yolo_conf),
            "rescue_require_proj_valid": bool(args.rescue_require_proj_valid),
            "rescue_max_center_step_m": float(args.rescue_max_center_step_m),
            "rescue_max_size_rel_delta": float(args.rescue_max_size_rel_delta),
            "rescue_max_proj_area_ratio": float(args.rescue_max_proj_area_ratio),
            "use_rotated_iou": not bool(args.disable_rotated_iou),
        },
        "raw_metrics": raw_metrics,
        "b0_metrics": b0_metrics,
        "b1_metrics": b1_metrics,
        "temporal_summary": {
            **temporal_summary,
            "tracker_resets": int(reset_count),
        },
        "selected_frames": selected_manifest,
        "outputs": {
            "b0_result_pkl": str(b0_result_path),
            "b1_result_pkl": str(b1_result_path),
            "b1_decision_audit_csv": str(audit_csv_path),
            "visualization_dir": str(vis_dir),
        },
    }
    metrics_json_path = output_dir / "b1_metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_json, f, indent=2, ensure_ascii=False)

    metrics_txt_path = output_dir / "b1_metrics.txt"
    lines = [
        "DAIR image-led temporal B1 rescoring",
        "",
        f"raw       : tp={raw_metrics['tp']} fp={raw_metrics['fp']} fn={raw_metrics['fn']} "
        f"precision={raw_metrics['precision']:.6f} recall={raw_metrics['recall']:.6f} kept={raw_metrics['kept_predictions']}",
        f"b0        : tp={b0_metrics['tp']} fp={b0_metrics['fp']} fn={b0_metrics['fn']} "
        f"precision={b0_metrics['precision']:.6f} recall={b0_metrics['recall']:.6f} kept={b0_metrics['kept_predictions']}",
        f"b1        : tp={b1_metrics['tp']} fp={b1_metrics['fp']} fn={b1_metrics['fn']} "
        f"precision={b1_metrics['precision']:.6f} recall={b1_metrics['recall']:.6f} kept={b1_metrics['kept_predictions']}",
        "",
        f"b0 drop -> b1 keep : {temporal_summary['b0_drop_to_b1_keep']}",
        f"b0 keep -> b1 drop : {temporal_summary['b0_keep_to_b1_drop']}",
        f"rescue blocked      : {temporal_summary['rescue_gate_blocked']}",
        f"confirmed tracks    : {temporal_summary['confirmed_tracks']}",
        f"matched yolo        : {temporal_summary['matched_yolo']}",
        f"support_now         : {temporal_summary['support_now']}",
        f"tracker_resets      : {reset_count}",
        "",
        f"b1_result.pkl       : {b1_result_path}",
        f"b1_decision_audit   : {audit_csv_path}",
        f"visualizations      : {vis_dir}",
    ]
    metrics_txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with open(output_dir / "selected_frames.json", "w", encoding="utf-8") as f:
        json.dump(selected_manifest, f, indent=2, ensure_ascii=False)

    print(json.dumps(metrics_json, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
