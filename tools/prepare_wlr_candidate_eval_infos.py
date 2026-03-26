#!/usr/bin/env python3
"""
Build WLR eval info pickles with candidate-specific calibration embedded per frame.

This lets us compare baseline / top-k extrinsic candidates at inference time
without overwriting the shared calib.npz on disk.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import yaml
from easydict import EasyDict

from pcdet.datasets.wlr733.wlr733_dataset import WLR733Dataset


def parse_csv_spec(spec: str) -> list[str]:
    return [chunk.strip() for chunk in str(spec).split(",") if chunk.strip()]


def rotx_deg(angle_deg: float) -> np.ndarray:
    angle_rad = np.deg2rad(float(angle_deg))
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float32)


def roty_deg(angle_deg: float) -> np.ndarray:
    angle_rad = np.deg2rad(float(angle_deg))
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)


def rotz_deg(angle_deg: float) -> np.ndarray:
    angle_rad = np.deg2rad(float(angle_deg))
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)


def apply_env_delta_to_t(
    base_t_cam_from_lidar: np.ndarray,
    *,
    droll_deg: float,
    dpitch_deg: float,
    dyaw_deg: float,
    dx_cam_m: float,
    dy_cam_m: float,
    dz_cam_m: float,
) -> np.ndarray:
    t_delta = np.eye(4, dtype=np.float32)
    t_delta[:3, :3] = (
        rotz_deg(dyaw_deg)
        @ roty_deg(dpitch_deg)
        @ rotx_deg(droll_deg)
    ).astype(np.float32)
    t_delta[:3, 3] = np.asarray([dx_cam_m, dy_cam_m, dz_cam_m], dtype=np.float32)
    return (t_delta @ np.asarray(base_t_cam_from_lidar, dtype=np.float32)).astype(np.float32)


def parse_extra_candidate_env_specs(specs: list[str], base_t_cam_from_lidar: np.ndarray) -> dict[str, dict]:
    candidate_map = {}
    for raw_spec in specs:
        parts = [chunk.strip() for chunk in str(raw_spec).split(",")]
        if len(parts) != 7:
            raise ValueError(
                "extra candidate env spec must be: "
                "name,droll_deg,dpitch_deg,dyaw_deg,dx_cam_m,dy_cam_m,dz_cam_m"
            )
        name = parts[0]
        droll_deg, dpitch_deg, dyaw_deg, dx_cam_m, dy_cam_m, dz_cam_m = [float(x) for x in parts[1:]]
        candidate_map[name] = {
            "source": "env_delta",
            "delta": {
                "droll_deg": droll_deg,
                "dpitch_deg": dpitch_deg,
                "dyaw_deg": dyaw_deg,
                "dx_cam_m": dx_cam_m,
                "dy_cam_m": dy_cam_m,
                "dz_cam_m": dz_cam_m,
            },
            "T_cam_from_lidar": apply_env_delta_to_t(
                base_t_cam_from_lidar,
                droll_deg=droll_deg,
                dpitch_deg=dpitch_deg,
                dyaw_deg=dyaw_deg,
                dx_cam_m=dx_cam_m,
                dy_cam_m=dy_cam_m,
                dz_cam_m=dz_cam_m,
            ).tolist(),
        }
    return candidate_map


def load_candidate_map(local_refine_json: Path) -> dict[str, dict]:
    payload = json.loads(local_refine_json.read_text(encoding="utf-8"))
    out = {"baseline": payload["baseline"]}
    for row in payload.get("top_local_candidates", []):
        out[str(row["rank_label"])] = row
    return out


def candidate_cam_k(candidate_row: dict, default_cam_k: np.ndarray) -> np.ndarray:
    candidate_name = candidate_row.get("rank_label", candidate_row.get("name", "unknown"))
    for key in ("cam_K", "camera_matrix", "K"):
        if key not in candidate_row or candidate_row[key] is None:
            continue
        cam_k = np.asarray(candidate_row[key], dtype=np.float32)
        if cam_k.shape != (3, 3):
            raise ValueError(f"candidate {candidate_name} has bad {key} shape: {cam_k.shape}")
        return cam_k
    return np.asarray(default_cam_k, dtype=np.float32)


def clone_infos_with_calib(infos: list[dict], cam_k: np.ndarray, t_cam_from_lidar: np.ndarray) -> list[dict]:
    cam_k = np.asarray(cam_k, dtype=np.float32)
    t_cam_from_lidar = np.asarray(t_cam_from_lidar, dtype=np.float32)

    out = []
    for info in infos:
        row = dict(info)
        row["calib"] = {
            "cam_K": cam_k.copy(),
            "T_cam_from_lidar": t_cam_from_lidar.copy(),
        }
        out.append(row)
    return out


def count_nonempty_gt(infos: list[dict]) -> tuple[int, int]:
    frames_with_gt = 0
    total_boxes = 0
    for info in infos:
        ann = info.get("annotations") or {}
        gt = ann.get("gt_boxes_lidar", [])
        if getattr(gt, "shape", None) is not None:
            n = int(gt.shape[0])
        elif gt is None:
            n = 0
        else:
            n = int(len(gt))
        frames_with_gt += int(n > 0)
        total_boxes += n
    return frames_with_gt, total_boxes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/wlr733")
    ap.add_argument("--dataset_cfg", default="tools/cfgs/dataset_configs/wlr733_dataset_sync.yaml")
    ap.add_argument("--image_root", default="/root/pointpillar/cam_matched")
    ap.add_argument("--global_calib_npz", default="data/wlr733/training/calib/calib.npz")
    ap.add_argument("--local_refine_json", default="/root/pointpillar/wlr_extrinsic_local_refine_v1/T_best_local_by_metric.json")
    ap.add_argument("--splits", default="val_mid20,all_labeled_31")
    ap.add_argument("--candidates", default="baseline,top_01,top_03")
    ap.add_argument(
        "--extra_candidate_env",
        action="append",
        default=[],
        help=(
            "extra candidate using historical env-style delta: "
            "name,droll_deg,dpitch_deg,dyaw_deg,dx_cam_m,dy_cam_m,dz_cam_m"
        ),
    )
    ap.add_argument("--out_dir", default="data/wlr733")
    ap.add_argument("--summary_json", default="/root/pointpillar/wlr_candidate_eval_assets/candidate_eval_info_summary.json")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    dataset_cfg_path = Path(args.dataset_cfg)
    out_dir = Path(args.out_dir)
    summary_json = Path(args.summary_json)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = EasyDict(yaml.safe_load(dataset_cfg_path.read_text(encoding="utf-8")))
    cfg.INFO_PATH = {}
    cfg.IMAGE_ROOT = str(args.image_root)

    ds = WLR733Dataset(
        dataset_cfg=cfg,
        class_names=cfg.CLASS_NAMES,
        training=True,
        root_path=data_root,
    )

    calib_data = np.load(str(args.global_calib_npz))
    cam_k = np.asarray(calib_data["K"], dtype=np.float32)
    base_t_cam_from_lidar = np.asarray(calib_data["T_cam_from_lidar"], dtype=np.float32)

    candidate_map = load_candidate_map(Path(args.local_refine_json))
    candidate_map.update(parse_extra_candidate_env_specs(args.extra_candidate_env, base_t_cam_from_lidar))
    split_names = parse_csv_spec(args.splits)
    candidate_names = parse_csv_spec(args.candidates)

    summary_rows = []
    for split_name in split_names:
        base_infos = ds.generate_infos(split_name)
        for candidate_name in candidate_names:
            if candidate_name not in candidate_map:
                raise KeyError(f"candidate not found in local refine json: {candidate_name}")

            candidate_row = candidate_map[candidate_name]
            cam_k_cur = candidate_cam_k(candidate_row, cam_k)
            cam_k_source = (
                "candidate"
                if any((key in candidate_row) and (candidate_row[key] is not None) for key in ("cam_K", "camera_matrix", "K"))
                else "global_calib_npz"
            )
            t_cur = np.asarray(candidate_row["T_cam_from_lidar"], dtype=np.float32)
            infos = clone_infos_with_calib(base_infos, cam_k=cam_k_cur, t_cam_from_lidar=t_cur)
            out_path = out_dir / f"wlr733_infos_{split_name}_{candidate_name}.pkl"
            with open(out_path, "wb") as f:
                pickle.dump(infos, f)

            frames_with_gt, total_boxes = count_nonempty_gt(infos)
            summary_rows.append(
                {
                    "split": split_name,
                    "candidate": candidate_name,
                    "info_pkl": str(out_path),
                    "cam_k_source": cam_k_source,
                    "num_samples": int(len(infos)),
                    "frames_with_gt": int(frames_with_gt),
                    "total_gt_boxes": int(total_boxes),
                }
            )
            print(
                json.dumps(
                    {
                        "split": split_name,
                        "candidate": candidate_name,
                        "info_pkl": str(out_path),
                        "cam_k_source": cam_k_source,
                        "num_samples": int(len(infos)),
                        "frames_with_gt": int(frames_with_gt),
                        "total_gt_boxes": int(total_boxes),
                    },
                    ensure_ascii=False,
                )
            )

    summary = {
        "dataset_cfg": str(dataset_cfg_path),
        "image_root": str(args.image_root),
        "global_calib_npz": str(args.global_calib_npz),
        "local_refine_json": str(args.local_refine_json),
        "rows": summary_rows,
    }
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
