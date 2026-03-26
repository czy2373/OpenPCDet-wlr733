import argparse
import json
import os
import pickle
import shutil
import struct
from pathlib import Path

import numpy as np
import tqdm


VEHICLE_TYPES = {
    'car', 'van', 'bus', 'truck', 'trunk'
}


def parse_args():
    parser = argparse.ArgumentParser(description='Convert DAIR-V2X-I to current OpenPCDet fusion layout')
    parser.add_argument('--source_root', type=str, required=True, help='DAIR single-infrastructure-side root')
    parser.add_argument('--target_root', type=str, required=True, help='Output root under OpenPCDet/data')
    parser.add_argument('--split_json', type=str, default=None, help='Split file path, default: <source_root>/split_data.json')
    parser.add_argument('--link_images', action='store_true', help='Symlink images into target image dir')
    parser.add_argument('--copy_images', action='store_true', help='Copy images instead of symlink')
    return parser.parse_args()


def _norm_id(v):
    s = str(v).strip().split('/')[-1].split('\\')[-1]
    s = Path(s).stem
    return s.zfill(6) if s.isdigit() else s


def _mkdir(p):
    Path(p).mkdir(parents=True, exist_ok=True)


def _load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _dtype_from_pcd(type_char, size):
    if type_char == 'F':
        return {4: np.float32, 8: np.float64}[size]
    if type_char == 'I':
        return {1: np.int8, 2: np.int16, 4: np.int32, 8: np.int64}[size]
    if type_char == 'U':
        return {1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}[size]
    raise ValueError(f'Unsupported PCD type={type_char}, size={size}')


def lzf_decompress(data, expected_length):
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError('lzf_decompress expects bytes-like input')

    in_data = memoryview(data)
    in_len = len(in_data)
    out = bytearray(expected_length)
    i = 0
    o = 0

    while i < in_len:
        ctrl = in_data[i]
        i += 1

        if ctrl < 32:
            length = ctrl + 1
            if o + length > expected_length or i + length > in_len:
                raise ValueError('Invalid LZF literal run while decompressing PCD')
            out[o:o + length] = in_data[i:i + length]
            i += length
            o += length
        else:
            length = ctrl >> 5
            ref = o - ((ctrl & 0x1F) << 8) - 1
            if length == 7:
                if i >= in_len:
                    raise ValueError('Invalid LZF backref length while decompressing PCD')
                length += in_data[i]
                i += 1
            if i >= in_len:
                raise ValueError('Invalid LZF backref offset while decompressing PCD')
            ref -= in_data[i]
            i += 1
            length += 2

            if ref < 0 or o + length > expected_length:
                raise ValueError('Invalid LZF backref bounds while decompressing PCD')

            for _ in range(length):
                out[o] = out[ref]
                o += 1
                ref += 1

    if o != expected_length:
        raise ValueError(f'LZF decompressed length mismatch: got {o}, expect {expected_length}')
    return bytes(out)


def read_pcd_xyzi(pcd_path):
    with open(pcd_path, 'rb') as f:
        header = {}
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f'Invalid PCD file: {pcd_path}')
            s = line.decode('ascii', errors='ignore').strip()
            if not s or s.startswith('#'):
                continue
            parts = s.split()
            key = parts[0].upper()
            header[key] = parts[1:]
            if key == 'DATA':
                data_mode = parts[1].lower()
                break

        fields = header['FIELDS']
        sizes = list(map(int, header['SIZE']))
        types = header['TYPE']
        counts = list(map(int, header.get('COUNT', ['1'] * len(fields))))
        points = int(header.get('POINTS', [header.get('WIDTH', ['0'])[0]])[0])

        flat_fields = []
        flat_dtypes = []
        for name, size, type_char, count in zip(fields, sizes, types, counts):
            dtype = _dtype_from_pcd(type_char, size)
            if count == 1:
                flat_fields.append(name)
                flat_dtypes.append((name, dtype))
            else:
                for i in range(count):
                    sub_name = f'{name}_{i}'
                    flat_fields.append(sub_name)
                    flat_dtypes.append((sub_name, dtype))

        if data_mode == 'binary':
            arr = np.fromfile(f, dtype=np.dtype(flat_dtypes), count=points)
            feat = {k: arr[k].astype(np.float32) for k in flat_fields}
        elif data_mode == 'binary_compressed':
            header_sizes = f.read(8)
            if len(header_sizes) != 8:
                raise ValueError(f'Invalid binary_compressed header in {pcd_path}')
            compressed_size, uncompressed_size = struct.unpack('II', header_sizes)
            compressed = f.read(compressed_size)
            if len(compressed) != compressed_size:
                raise ValueError(f'Failed to read compressed payload from {pcd_path}')
            raw = lzf_decompress(compressed, uncompressed_size)

            feat = {}
            offset = 0
            for name, size, type_char, count in zip(fields, sizes, types, counts):
                dtype = _dtype_from_pcd(type_char, size)
                n_bytes = points * count * size
                chunk = raw[offset:offset + n_bytes]
                if len(chunk) != n_bytes:
                    raise ValueError(f'Unexpected field byte count for {name} in {pcd_path}')
                arr = np.frombuffer(chunk, dtype=dtype, count=points * count)
                if count == 1:
                    feat[name] = arr.astype(np.float32)
                else:
                    arr = arr.reshape(points, count)
                    for i in range(count):
                        feat[f'{name}_{i}'] = arr[:, i].astype(np.float32)
                offset += n_bytes
        elif data_mode == 'ascii':
            raw = np.loadtxt(f, dtype=np.float32)
            if raw.ndim == 1:
                raw = raw[None, :]
            feat = {k: raw[:, i].astype(np.float32) for i, k in enumerate(flat_fields)}
        else:
            raise ValueError(f'Unsupported PCD DATA mode: {data_mode}')

    xyz = []
    for key in ['x', 'y', 'z']:
        if key not in feat:
            raise KeyError(f'Missing field {key} in {pcd_path}')
        xyz.append(feat[key])

    intensity = None
    for key in ['intensity', 'reflectance', 'i', 'intensities']:
        if key in feat:
            intensity = feat[key]
            break
    if intensity is None:
        intensity = np.zeros_like(xyz[0], dtype=np.float32)

    return np.stack([xyz[0], xyz[1], xyz[2], intensity], axis=1).astype(np.float32)


def parse_cam_k(calib_json):
    if 'cam_K' in calib_json:
        cam_k = np.asarray(calib_json['cam_K'], dtype=np.float32).reshape(3, 3)
    else:
        cam_k = np.asarray(calib_json['K'], dtype=np.float32).reshape(3, 3)
    return cam_k


def parse_extrinsic(calib_json):
    rot = np.asarray(calib_json['rotation'], dtype=np.float32).reshape(3, 3)
    trans = np.asarray(calib_json['translation'], dtype=np.float32).reshape(-1)
    if trans.shape[0] != 3:
        raise ValueError(f'Unexpected translation shape: {trans.shape}')
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = rot
    T[:3, 3] = trans
    return T


def object_to_kitti_line(obj):
    obj_type = str(obj.get('type', '')).strip()
    obj_type_l = obj_type.lower()
    if obj_type_l not in VEHICLE_TYPES:
        return None, None

    box2d = obj.get('2d_box', {}) or {}
    dims = obj.get('3d_dimensions', {}) or {}
    loc = obj.get('3d_location', {}) or {}

    h = float(dims.get('h', 0.0))
    w = float(dims.get('w', 0.0))
    l = float(dims.get('l', 0.0))
    x = float(loc.get('x', 0.0))
    y = float(loc.get('y', 0.0))
    z = float(loc.get('z', 0.0))
    ry = float(obj.get('rotation', 0.0))

    x1 = float(box2d.get('xmin', 0.0))
    y1 = float(box2d.get('ymin', 0.0))
    x2 = float(box2d.get('xmax', 0.0))
    y2 = float(box2d.get('ymax', 0.0))

    line = f'vehicle 0 0 0 {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} {h:.4f} {w:.4f} {l:.4f} {x:.4f} {y:.4f} {z:.4f} {ry:.6f}'
    gt_box = [x, y, z, l, w, h, ry]
    return line, gt_box


def build_infos(ids, anno_by_id):
    infos = []
    for sid in ids:
        ann = anno_by_id.get(sid, {'name': [], 'gt_boxes_lidar': np.zeros((0, 7), dtype=np.float32)})
        infos.append({
            'point_cloud': {'num_features': 4, 'lidar_idx': sid},
            'annotations': {
                'name': ann['name'],
                'gt_boxes_lidar': ann['gt_boxes_lidar'],
            }
        })
    return infos


def infer_items_from_filesystem(source_root):
    image_dir = source_root / 'image'
    velodyne_dir = source_root / 'velodyne'
    cam_dir = source_root / 'calib' / 'camera_intrinsic'
    ext_dir = source_root / 'calib' / 'virtuallidar_to_camera'
    label_dir = source_root / 'label' / 'virtuallidar'

    image_ids = {_norm_id(p.stem) for p in image_dir.glob('*.jpg')}
    velo_ids = {_norm_id(p.stem) for p in velodyne_dir.glob('*.pcd')}
    cam_ids = {_norm_id(p.stem) for p in cam_dir.glob('*.json')}
    ext_ids = {_norm_id(p.stem) for p in ext_dir.glob('*.json')}
    label_ids = {_norm_id(p.stem) for p in label_dir.glob('*.json')}

    valid_ids = sorted(image_ids & velo_ids & cam_ids & ext_ids & label_ids)
    items = {}
    for sid in valid_ids:
        items[sid] = {
            'image_path': f'image/{sid}.jpg',
            'pointcloud_path': f'velodyne/{sid}.pcd',
            'calib_camera_intrinsic_path': f'calib/camera_intrinsic/{sid}.json',
            'calib_virtuallidar_to_camera_path': f'calib/virtuallidar_to_camera/{sid}.json',
            'label_lidar_std_path': f'label/virtuallidar/{sid}.json',
        }
    return items


def main():
    args = parse_args()
    source_root = Path(args.source_root)
    target_root = Path(args.target_root)
    split_json = Path(args.split_json) if args.split_json else source_root / 'split_data.json'

    if not source_root.exists():
        raise FileNotFoundError(f'source_root not found: {source_root}')
    if not split_json.exists():
        raise FileNotFoundError(f'split_json not found: {split_json}')

    image_dir = target_root / 'image'
    velodyne_dir = target_root / 'training' / 'velodyne'
    label_dir = target_root / 'training' / 'label_2'
    calib_dir = target_root / 'training' / 'calib'
    imagesets_dir = target_root / 'ImageSets'
    for p in [image_dir, velodyne_dir, label_dir, calib_dir, imagesets_dir]:
        _mkdir(p)

    split_info = _load_json(split_json)

    info_by_id = {}
    data_info_path = source_root / 'data_info.json'
    if data_info_path.exists():
        data_info = _load_json(data_info_path)
        for item in data_info:
            pointcloud_path = item.get('pointcloud_path', '')
            sid = _norm_id(Path(pointcloud_path).stem) if pointcloud_path else ''
            if sid:
                info_by_id[sid] = item

    inferred_items = infer_items_from_filesystem(source_root)
    if len(inferred_items) > len(info_by_id):
        print(f'Using filesystem-derived index: {len(inferred_items)} items (data_info has {len(info_by_id)})')
        info_by_id = inferred_items

    anno_by_id = {}
    used_ids = set()

    split_ids = {}
    all_ids = []
    for split_name in ['train', 'val', 'test']:
        ids = [_norm_id(x) for x in split_info.get(split_name, [])]
        split_ids[split_name] = ids
        all_ids.extend(ids)

    all_ids = sorted(set(all_ids))
    for sid in tqdm.tqdm(all_ids, desc='Convert DAIR-I'):
        item = info_by_id.get(sid)
        if item is None:
            continue

        src_img = source_root / item['image_path']
        src_pcd = source_root / item['pointcloud_path']
        src_label = source_root / item['label_lidar_std_path']
        src_cam = source_root / item['calib_camera_intrinsic_path']
        src_ext = source_root / item['calib_virtuallidar_to_camera_path']

        if args.copy_images:
            shutil.copy2(src_img, image_dir / f'{sid}{src_img.suffix.lower()}')
        elif args.link_images:
            dst_img = image_dir / f'{sid}{src_img.suffix.lower()}'
            if dst_img.exists() or dst_img.is_symlink():
                dst_img.unlink()
            os.symlink(src_img, dst_img)

        points = read_pcd_xyzi(src_pcd)
        points.astype(np.float32).tofile(velodyne_dir / f'{sid}.bin')

        cam_k = parse_cam_k(_load_json(src_cam))
        T_cam_from_lidar = parse_extrinsic(_load_json(src_ext))
        np.savez(calib_dir / f'{sid}.npz', K=cam_k.astype(np.float32), T_cam_from_lidar=T_cam_from_lidar.astype(np.float32))

        labels = _load_json(src_label)
        lines = []
        gt_boxes = []
        names = []
        for obj in labels:
            line, gt_box = object_to_kitti_line(obj)
            if line is None:
                continue
            lines.append(line)
            gt_boxes.append(gt_box)
            names.append('vehicle')

        with open(label_dir / f'{sid}.txt', 'w', encoding='utf-8') as f:
            for line in lines:
                f.write(line + '\n')

        anno_by_id[sid] = {
            'name': names,
            'gt_boxes_lidar': np.asarray(gt_boxes, dtype=np.float32).reshape(-1, 7) if gt_boxes else np.zeros((0, 7), dtype=np.float32)
        }
        used_ids.add(sid)

    for split_name in ['train', 'val', 'test']:
        ids = [sid for sid in split_ids.get(split_name, []) if sid in used_ids]
        with open(imagesets_dir / f'{split_name}.txt', 'w', encoding='utf-8') as f:
            for sid in ids:
                f.write(f'{sid}\n')
        infos = build_infos(ids, anno_by_id)
        with open(target_root / f'dair_i_infos_{split_name}.pkl', 'wb') as f:
            pickle.dump(infos, f)
        print(f'Saved {split_name}: {len(ids)} samples -> {target_root / f"dair_i_infos_{split_name}.pkl"}')

    print(f'Done. target_root={target_root}')


if __name__ == '__main__':
    main()
