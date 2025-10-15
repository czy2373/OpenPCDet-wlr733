
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Final, robust FPS (Hz) measurement for OpenPCDet on custom WLR-733 dataset.

Fixes:
- Proper function scoping (no indentation errors; no free-variable errors)
- Works with custom dataset via --dataset_module/--dataset_class
- Compatible with different build_dataloader return signatures
- Provides a real logger
- Ensures voxel_coords has batch column [M,4] AFTER load_data_to_gpu
- Ensures batch_size exists in batch_dict

Usage:
  python measure_fps_wlr733_final.py \
    --cfg_file tools/cfgs/kitti_models/pointpillar_wlr733.yaml \
    --ckpt output/cfgs/kitti_models/pointpillar_wlr733/default/ckpt/checkpoint_epoch_80.pth \
    --dataset_module pcdet/datasets/wlr733/wlr733_dataset.py \
    --dataset_class WLR733Dataset \
    --add_sys_path . \
    --batch_size 1 --workers 4 --warmup 50 --max_frames 500 --include_io
"""
import os
import sys
import time
import json
import logging
import importlib.util
from pathlib import Path
from typing import Any, Tuple

import torch
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu


def make_logger() -> logging.Logger:
    logger = logging.getLogger("fps_eval")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        h = logging.StreamHandler()
        h.setLevel(logging.INFO)
        h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(h)
    return logger


def import_class_from_py(py_path: str, class_name: str):
    """Import a class from arbitrary .py file path."""
    py_path = str(Path(py_path).resolve())
    if not os.path.isfile(py_path):
        raise FileNotFoundError(f"--dataset_module not found: {py_path}")
    module_name = Path(py_path).stem + "_dyn"
    spec = importlib.util.spec_from_file_location(module_name, py_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for: {py_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)  # type: ignore
    if not hasattr(mod, class_name):
        raise ImportError(f"Class {class_name} not found in {py_path}")
    return getattr(mod, class_name)


def get_loader_and_dataset(logger: logging.Logger, ds_cfg, class_names, batch_size: int, workers: int) -> Tuple[Any, Any]:
    """Build dataloader; handle different return signatures."""
    ret = build_dataloader(
        dataset_cfg=ds_cfg,
        class_names=class_names,
        batch_size=batch_size,
        dist=False,
        workers=workers,
        training=False,
        logger=logger,
    )
    if isinstance(ret, tuple):
        if len(ret) == 3:
            loader, _sampler, ds = ret
        elif len(ret) == 2:
            loader, ds = ret
        else:
            loader, ds = ret, None
    else:
        loader, ds = ret, None
    if ds is None:
        ds = getattr(loader, "dataset", None)
    return loader, ds


def main():
    import argparse

    ap = argparse.ArgumentParser("Final FPS measurement for OpenPCDet (custom dataset supported)")
    ap.add_argument("--cfg_file", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--set_cfgs", nargs="+", default=None)
    ap.add_argument("--max_frames", type=int, default=1000)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--include_io", action="store_true")
    ap.add_argument("--save_json", default="fps_result.json")
    # explicit dataset hooks
    ap.add_argument("--dataset_module", default=None, help="Path to dataset .py, e.g., pcdet/datasets/wlr733/wlr733_dataset.py")
    ap.add_argument("--dataset_class", default=None, help="Class name, e.g., WLR733Dataset")
    ap.add_argument("--add_sys_path", action="append", default=[], help="Append dirs to sys.path (repeatable)")
    args = ap.parse_args()

    for d in args.add_sys_path:
        if d and os.path.isdir(d):
            sys.path.insert(0, os.path.abspath(d))

    logger = make_logger()

    cfg_from_yaml_file(args.cfg_file, cfg)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    # dataloader (+ maybe dataset)
    dataloader, dataset = get_loader_and_dataset(logger, cfg.DATA_CONFIG, cfg.CLASS_NAMES, args.batch_size, args.workers)

    # if dataset not provided by loader, and user specified module+class, build explicitly
    if dataset is None and args.dataset_module and args.dataset_class:
        DS = import_class_from_py(args.dataset_module, args.dataset_class)
        try:
            dataset = DS(
                dataset_cfg=cfg.DATA_CONFIG,
                class_names=cfg.CLASS_NAMES,
                training=False,
                root_path=Path(cfg.DATA_CONFIG.DATA_PATH),
            )
        except TypeError:
            dataset = DS(dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False)

    if dataset is None:
        raise RuntimeError("Could not obtain dataset. Pass --dataset_module and --dataset_class for your custom dataset.")

    # build model
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    model.load_params_from_file(filename=args.ckpt, to_cpu=False, logger=logger)
    model.cuda().eval()

    def one_iter(batch_dict):
        # move to GPU
        load_data_to_gpu(batch_dict)

        # FIX 1: ensure voxel_coords shape is [M,4] (add batch_idx=0) AFTER GPU load
        if "voxel_coords" in batch_dict:
            coords = batch_dict["voxel_coords"]
            if hasattr(coords, "shape") and coords.shape[-1] == 3:
                zeros = torch.zeros((coords.shape[0], 1), dtype=coords.dtype, device=coords.device)
                batch_dict["voxel_coords"] = torch.cat((coords, zeros), dim=1)

        # FIX 2: ensure batch_size exists
        if "batch_size" not in batch_dict:
            try:
                bs = int(batch_dict["voxel_coords"][:, -1].max().item()) + 1 if "voxel_coords" in batch_dict else 1
            except Exception:
                bs = 1
            batch_dict["batch_size"] = bs
            
        # FIX 3: 仅测速，删除 GT，避免 post_processing 里做召回/IoU
        for k in ("gt_boxes", "gt_boxes2d", "gt_names", "gt_classes", "score_labels"):
            if k in batch_dict:
                batch_dict.pop(k)

        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            _preds, _ret = model(batch_dict)  # includes post-processing (e.g., NMS)
        torch.cuda.synchronize()
        t1 = time.time()
        return t1 - t0

    # iterate & measure
    n_measured, time_sum = 0, 0.0
    frames_total = 0
    data_iter = iter(dataloader)

    while frames_total < args.max_frames:
        try:
            if args.include_io:
                t_io0 = time.time()
                batch_dict = next(data_iter)
                t_io1 = time.time()
                io_cost = t_io1 - t_io0
                infer_cost = one_iter(batch_dict)
                total_cost = io_cost + infer_cost
            else:
                batch_dict = next(data_iter)
                total_cost = one_iter(batch_dict)
        except StopIteration:
            data_iter = iter(dataloader)
            continue

        if frames_total >= args.warmup:
            time_sum += total_cost
            n_measured += 1
        frames_total += 1

    if n_measured == 0:
        print("No frames measured. Reduce --warmup or increase --max_frames.")
        return

    avg = time_sum / n_measured
    fps = 1.0 / avg if avg > 0 else 0.0

    result = {
        "frames_measured": int(n_measured),
        "avg_time_ms": avg * 1000.0,
        "speed_hz": fps,
        "timing_includes_io": bool(args.include_io),
        "cfg_file": args.cfg_file,
        "ckpt": args.ckpt,
        "batch_size": int(args.batch_size),
    }
    print(json.dumps(result, indent=2))
    with open(args.save_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Saved JSON to {args.save_json}")


if __name__ == "__main__":
    main()
