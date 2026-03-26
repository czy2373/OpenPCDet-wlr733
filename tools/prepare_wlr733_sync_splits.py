#!/usr/bin/env python3
"""
Prepare reproducible WLR733 splits and info files for synchronized cam images.

What it does:
1) Build usable ID set = intersection(label_2, velodyne, cam_matched_sync_v1)
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
import pickle
import random
from pathlib import Path

import yaml
from easydict import EasyDict

from pcdet.datasets.wlr733.wlr733_dataset import WLR733Dataset


def stem_set(pattern: str, base: Path):
    return {p.stem for p in base.glob(pattern)}


def write_split(path: Path, ids):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for x in ids:
            f.write(f"{x}\n")


def save_infos(ds: WLR733Dataset, root: Path, split_name: str):
    infos = ds.generate_infos(split_name)
    out = root / f"wlr733_infos_{split_name}.pkl"
    with open(out, "wb") as f:
        pickle.dump(infos, f)
    print(f"saved: {out} len={len(infos)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/wlr733")
    ap.add_argument("--cam_matched_dir", default="/root/pointpillar/cam_matched_sync_v1")
    ap.add_argument("--dataset_cfg", default="tools/cfgs/dataset_configs/wlr733_dataset.yaml")
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

    usable = sorted(label_ids & lidar_ids & cam_ids)
    if len(usable) < 3:
        raise RuntimeError(f"usable labeled ids too small: {len(usable)}")

    print(f"label_ids={len(label_ids)} lidar_ids={len(lidar_ids)} cam_ids={len(cam_ids)}")
    print(f"usable(labeled+lidar+cam)={len(usable)}")

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

    save_infos(ds, root, "all_labeled_sync")
    save_infos(ds, root, "train_sync")
    save_infos(ds, root, "val_sync")
    save_infos(ds, root, "test_sync")


if __name__ == "__main__":
    main()
