#!/usr/bin/env python3
"""
Run YOLO inference on an image folder and export a detection CSV.

Output columns:
frame_id,x1,y1,x2,y2,conf,cls,name,model
"""

import argparse
import csv
from pathlib import Path


def read_frames_txt(path: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    out = set()
    for line in p.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s:
            out.add(s.split("/")[-1].split("\\")[-1].split(".")[0])
    return out


def collect_images(image_dir: str, exts=(".jpg", ".jpeg", ".png")):
    d = Path(image_dir)
    if not d.exists():
        raise FileNotFoundError(d)
    files = []
    for ext in exts:
        files.extend(d.glob(f"*{ext}"))
    return sorted(files, key=lambda p: p.stem)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--model", default="yolov8l.pt")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.2)
    ap.add_argument("--iou", type=float, default=0.6)
    ap.add_argument("--device", default="0")
    ap.add_argument("--classes", default="2,5,7", help="COCO class ids, e.g. 2,5,7 for car,bus,truck")
    ap.add_argument("--frames_txt", default="", help="optional frame id list txt")
    ap.add_argument("--max_images", type=int, default=0)
    args = ap.parse_args()

    class_ids = [int(x) for x in args.classes.split(",") if x.strip()]
    keep_ids = None
    if args.frames_txt.strip():
        keep_ids = read_frames_txt(args.frames_txt.strip())

    imgs = collect_images(args.image_dir)
    if keep_ids is not None:
        imgs = [p for p in imgs if p.stem in keep_ids]
    if int(args.max_images) > 0:
        imgs = imgs[: int(args.max_images)]
    if len(imgs) == 0:
        raise RuntimeError("no images matched")

    from ultralytics import YOLO

    model = YOLO(args.model)

    out_rows = []
    for i, p in enumerate(imgs, 1):
        res = model.predict(
            source=str(p),
            imgsz=int(args.imgsz),
            conf=float(args.conf),
            iou=float(args.iou),
            classes=class_ids,
            device=args.device,
            verbose=False,
        )[0]

        names = res.names
        boxes = res.boxes
        if boxes is None or boxes.xyxy is None:
            continue

        xyxy = boxes.xyxy.detach().cpu().numpy()
        confs = boxes.conf.detach().cpu().numpy() if boxes.conf is not None else []
        clss = boxes.cls.detach().cpu().numpy().astype(int) if boxes.cls is not None else []

        for j in range(len(xyxy)):
            x1, y1, x2, y2 = [float(v) for v in xyxy[j]]
            cls_id = int(clss[j]) if len(clss) > j else -1
            conf = float(confs[j]) if len(confs) > j else 0.0
            name = names[cls_id] if isinstance(names, dict) and cls_id in names else str(cls_id)
            out_rows.append(
                {
                    "frame_id": p.stem,
                    "x1": f"{x1:.3f}",
                    "y1": f"{y1:.3f}",
                    "x2": f"{x2:.3f}",
                    "y2": f"{y2:.3f}",
                    "conf": f"{conf:.6f}",
                    "cls": str(cls_id),
                    "name": str(name),
                    "model": str(args.model),
                }
            )

        if i % 50 == 0:
            print(f"processed {i}/{len(imgs)} images, det_rows={len(out_rows)}")

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    cols = ["frame_id", "x1", "y1", "x2", "y2", "conf", "cls", "name", "model"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(out_rows)

    print("saved:", out_csv)
    print("stats:", f"images={len(imgs)}", f"detections={len(out_rows)}", f"model={args.model}")


if __name__ == "__main__":
    main()
