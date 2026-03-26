#!/usr/bin/env python3
"""
Prepare reproducible WLR733 splits and info files for synchronized cam images.

What it does:
1) Build usable ID set from label_2, velodyne and synchronized image assets
2) Write ImageSets files:
   - all_labeled_sync.txt
   - train_sync.txt
   - val_sync.txt
   - test_sync.txt
3) Generate info pickles for each split:
   - wlr733_infos_train_sync.pkl
   - wlr733_infos_val_sync.pkl
   - wlr733_infos_test_sync.pkl
   - wlr733_infos_all_labeled_sync.pkl
"""

import argparse
import csv
import pickle
import random
from pathlib import Path
from typing import Dict, List, Optional, Set

import yaml
from easydict import EasyDict

from pcdet.datasets.wlr733.wlr733_dataset import WLR733Dataset


def stem_set(pattern: str, base: Path):
    return {p.stem for p in base.glob(pattern)}


def norm_frame_id(frame_id: str) -> str:
    return Path(str(frame_id).strip()).stem


def canonical_frame_id(frame_id: str) -> str:
    key = norm_frame_id(frame_id)
    if key.isdigit():
        return f"{int(key):06d}"
    return key


def frame_id_variants(frame_id: str) -> List[str]:
    key = norm_frame_id(frame_id)
    out = []
    for candidate in (
        key,
        str(int(key)) if key.isdigit() else key,
        f"{int(key):06d}" if key.isdigit() else key,
    ):
        if candidate not in out:
            out.append(candidate)
    return out


def load_lidar_to_cam_map(csv_path: Path) -> Dict[str, str]:
    out = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lidar_raw = str(row.get("lidar_frame", row.get("frame_id", ""))).strip()
            cam_raw = str(row.get("cam_frame", "")).strip()
            if lidar_raw == "" or cam_raw == "":
                continue
            cam_id = canonical_frame_id(cam_raw)
            for lidar_key in frame_id_variants(lidar_raw):
                out[lidar_key] = cam_id
    return out


def resolve_existing_image_key(
    lidar_id: str,
    cam_ids: Set[str],
    sync_map: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    search_keys = []
    if sync_map is not None:
        cam_id = sync_map.get(norm_frame_id(lidar_id), None)
        if cam_id not in (None, ""):
            search_keys.append(cam_id)
    search_keys.append(lidar_id)

    for frame_id in search_keys:
        for key in frame_id_variants(frame_id):
            if key in cam_ids:
                return key
    return None


def write_split(path: Path, ids):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for x in ids:
            f.write(f"{x}\n")


def save_infos(
    ds: WLR733Dataset,
    root: Path,
    split_name: str,
    sync_map: Optional[Dict[str, str]] = None,
    sync_source: Optional[str] = None,
):
    infos = ds.generate_infos(split_name, sync_map=sync_map, sync_source=sync_source)
    out = root / f"wlr733_infos_{split_name}.pkl"
    with open(out, "wb") as f:
        pickle.dump(infos, f)
    print(f"saved: {out} len={len(infos)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/wlr733")
    ap.add_argument("--cam_matched_dir", default="/root/pointpillar/cam_matched")
    ap.add_argument("--dataset_cfg", default="tools/cfgs/dataset_configs/wlr733_dataset_sync.yaml")
    ap.add_argument("--frame_map_csv", default="", help="optional lidar_frame -> cam_frame mapping csv")
    ap.add_argument("--sync_source_name", default="", help="optional name written to info['sync']['source']")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_ratio", type=float, default=0.7)
    ap.add_argument("--val_ratio", type=float, default=0.2)
    args = ap.parse_args()

    root = Path(args.data_root)
    label_dir = root / "training" / "label_2"
    velodyne_dir = root / "training" / "velodyne" / "1802"
    cam_dir = Path(args.cam_matched_dir)
    image_sets = root / "ImageSets"

    if not label_dir.exists():
        raise FileNotFoundError(label_dir)
    if not velodyne_dir.exists():
        raise FileNotFoundError(velodyne_dir)
    if not cam_dir.exists():
        raise FileNotFoundError(cam_dir)

    label_ids = stem_set("*.txt", label_dir)
    lidar_ids = stem_set("*.bin", velodyne_dir)
    cam_ids = stem_set("*.jpg", cam_dir) | stem_set("*.png", cam_dir)
    sync_map = None
    sync_source = None
    if args.frame_map_csv:
        frame_map_csv = Path(args.frame_map_csv)
        if not frame_map_csv.exists():
            raise FileNotFoundError(frame_map_csv)
        sync_map = load_lidar_to_cam_map(frame_map_csv)
        sync_source = args.sync_source_name or frame_map_csv.name

    labeled_lidar_ids = sorted(label_ids & lidar_ids)
    missing_sync = []
    missing_image = []
    usable = []
    for lidar_id in labeled_lidar_ids:
        lidar_key = norm_frame_id(lidar_id)
        if sync_map is not None and lidar_key not in sync_map:
            missing_sync.append(lidar_id)
            continue
        if resolve_existing_image_key(lidar_id, cam_ids=cam_ids, sync_map=sync_map) is None:
            missing_image.append(lidar_id)
            continue
        usable.append(lidar_id)

    if len(usable) < 3:
        raise RuntimeError(f"usable labeled ids too small: {len(usable)}")

    print(f"label_ids={len(label_ids)} lidar_ids={len(lidar_ids)} cam_ids={len(cam_ids)}")
    print(f"labeled_lidar_intersection={len(labeled_lidar_ids)}")
    if sync_map is not None:
        print(f"sync_map_entries={len(sync_map)} sync_source={sync_source}")
        print(f"missing_sync={len(missing_sync)} missing_image={len(missing_image)}")
    print(f"usable(labeled+lidar+sync_image)={len(usable)}")

    # deterministic shuffle
    rng = random.Random(args.seed)
    ids = usable[:]
    rng.shuffle(ids)

    n = len(ids)
    n_train = max(1, int(round(n * args.train_ratio)))
    n_val = max(1, int(round(n * args.val_ratio)))
    if n_train + n_val >= n:
        n_val = max(1, n - n_train - 1)
    n_test = n - n_train - n_val
    if n_test <= 0:
        n_test = 1
        if n_train > n_val:
            n_train -= 1
        else:
            n_val -= 1

    train_ids = sorted(ids[:n_train])
    val_ids = sorted(ids[n_train:n_train + n_val])
    test_ids = sorted(ids[n_train + n_val:])

    write_split(image_sets / "all_labeled_sync.txt", sorted(usable))
    write_split(image_sets / "train_sync.txt", train_ids)
    write_split(image_sets / "val_sync.txt", val_ids)
    write_split(image_sets / "test_sync.txt", test_ids)

    print(f"split sizes: train={len(train_ids)} val={len(val_ids)} test={len(test_ids)}")
    print(f"train ids: {train_ids}")
    print(f"val ids:   {val_ids}")
    print(f"test ids:  {test_ids}")

    # Generate infos from dataset class (reuses current parsing logic)
    cfg = EasyDict(yaml.safe_load(open(args.dataset_cfg, "r", encoding="utf-8")))
    cfg.INFO_PATH = {}
    cfg.IMAGE_ROOT = str(cam_dir)

    ds = WLR733Dataset(
        dataset_cfg=cfg,
        class_names=cfg.CLASS_NAMES,
        training=True,
        root_path=root,
    )

    save_infos(ds, root, "all_labeled_sync", sync_map=sync_map, sync_source=sync_source)
    save_infos(ds, root, "train_sync", sync_map=sync_map, sync_source=sync_source)
    save_infos(ds, root, "val_sync", sync_map=sync_map, sync_source=sync_source)
    save_infos(ds, root, "test_sync", sync_map=sync_map, sync_source=sync_source)


if __name__ == "__main__":
    main()
