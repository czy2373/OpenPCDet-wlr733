import argparse
import os
import pickle
import shutil
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description='Merge two kitti-like roadside datasets with prefixed ids')
    p.add_argument('--src1-root', required=True)
    p.add_argument('--src1-train-info', required=True)
    p.add_argument('--src1-val-info', required=True)
    p.add_argument('--src1-train-split', required=True)
    p.add_argument('--src1-val-split', required=True)
    p.add_argument('--src1-prefix', required=True)
    p.add_argument('--src2-root', required=True)
    p.add_argument('--src2-train-info', required=True)
    p.add_argument('--src2-val-info', required=True)
    p.add_argument('--src2-train-split', required=True)
    p.add_argument('--src2-val-split', required=True)
    p.add_argument('--src2-prefix', required=True)
    p.add_argument('--target-root', required=True)
    p.add_argument('--copy', action='store_true', help='copy files instead of symlink')
    return p.parse_args()


def mkdir(p):
    Path(p).mkdir(parents=True, exist_ok=True)


def read_split(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [x.strip() for x in f.readlines() if x.strip()]


def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def dump_pkl(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def norm_id(x):
    x = str(x).strip().split('/')[-1].split('\\')[-1]
    return Path(x).stem


def resolve_image(root, sid):
    cands = [
        Path(root) / 'image' / f'{sid}.jpg',
        Path(root) / 'image' / f'{sid}.png',
        Path(root) / 'training' / 'image_2' / f'{sid}.jpg',
        Path(root) / 'training' / 'image_2' / f'{sid}.png',
    ]
    for p in cands:
        if p.exists():
            return p
    raise FileNotFoundError(f'image not found for {sid} under {root}')


def resolve_velodyne(root, sid):
    cands = [
        Path(root) / 'training' / 'velodyne' / f'{sid}.bin',
        Path(root) / 'training' / 'velodyne' / f'{sid}.pcd',
    ]
    for p in cands:
        if p.exists():
            return p
    raise FileNotFoundError(f'velodyne not found for {sid} under {root}')


def resolve_label(root, sid):
    cands = [
        Path(root) / 'training' / 'label_2' / f'{sid}.txt',
    ]
    for p in cands:
        if p.exists():
            return p
    raise FileNotFoundError(f'label_2 not found for {sid} under {root}')


def resolve_calib(root, sid):
    cands = [
        Path(root) / 'training' / 'calib' / f'{sid}.npz',
        Path(root) / 'training' / 'calib' / 'calib_fixed.npz',
        Path(root) / 'training' / 'calib' / 'calib.npz',
    ]
    for p in cands:
        if p.exists():
            return p
    raise FileNotFoundError(f'calib not found for {sid} under {root}')


def link_or_copy(src, dst, do_copy=False):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if do_copy:
        shutil.copy2(src, dst)
    else:
        os.symlink(src, dst)


def build_info_index(infos):
    out = {}
    for info in infos:
        sid = norm_id(info.get('point_cloud', {}).get('lidar_idx', info.get('frame_id', '')))
        if sid:
            out[sid] = info
    return out


def clone_info_with_prefix(info, prefixed_id):
    new_info = dict(info)
    point_cloud = dict(new_info.get('point_cloud', {}))
    point_cloud['lidar_idx'] = prefixed_id
    new_info['point_cloud'] = point_cloud
    if 'frame_id' in new_info:
        new_info['frame_id'] = prefixed_id
    return new_info


def merge_one_source(src_root, train_ids, val_ids, prefix, train_infos, val_infos, target_root, do_copy):
    info_train_idx = build_info_index(train_infos)
    info_val_idx = build_info_index(val_infos)

    merged_train_infos = []
    merged_val_infos = []
    merged_train_ids = []
    merged_val_ids = []

    for split_name, ids, info_idx, out_infos, out_ids in [
        ('train', train_ids, info_train_idx, merged_train_infos, merged_train_ids),
        ('val', val_ids, info_val_idx, merged_val_infos, merged_val_ids),
    ]:
        for sid_raw in ids:
            sid = norm_id(sid_raw)
            prefixed = f'{prefix}_{sid}'

            img_src = resolve_image(src_root, sid)
            velo_src = resolve_velodyne(src_root, sid)
            label_src = resolve_label(src_root, sid)
            calib_src = resolve_calib(src_root, sid)

            img_dst = target_root / 'image' / f'{prefixed}{img_src.suffix.lower()}'
            velo_dst = target_root / 'training' / 'velodyne' / f'{prefixed}{velo_src.suffix.lower()}'
            label_dst = target_root / 'training' / 'label_2' / f'{prefixed}.txt'
            calib_dst = target_root / 'training' / 'calib' / f'{prefixed}.npz'

            link_or_copy(img_src, img_dst, do_copy)
            link_or_copy(velo_src, velo_dst, do_copy)
            link_or_copy(label_src, label_dst, do_copy)
            link_or_copy(calib_src, calib_dst, do_copy)

            out_ids.append(prefixed)
            if sid in info_idx:
                out_infos.append(clone_info_with_prefix(info_idx[sid], prefixed))
            else:
                out_infos.append({
                    'point_cloud': {'num_features': 4, 'lidar_idx': prefixed},
                    'annotations': {'name': [], 'gt_boxes_lidar': []}
                })

    return merged_train_ids, merged_val_ids, merged_train_infos, merged_val_infos


def main():
    args = parse_args()

    target_root = Path(args.target_root)
    mkdir(target_root / 'image')
    mkdir(target_root / 'training' / 'velodyne')
    mkdir(target_root / 'training' / 'label_2')
    mkdir(target_root / 'training' / 'calib')
    mkdir(target_root / 'ImageSets')

    src1_train_ids = read_split(args.src1_train_split)
    src1_val_ids = read_split(args.src1_val_split)
    src2_train_ids = read_split(args.src2_train_split)
    src2_val_ids = read_split(args.src2_val_split)

    src1_train_infos = load_pkl(args.src1_train_info)
    src1_val_infos = load_pkl(args.src1_val_info)
    src2_train_infos = load_pkl(args.src2_train_info)
    src2_val_infos = load_pkl(args.src2_val_info)

    s1_tr_ids, s1_va_ids, s1_tr_infos, s1_va_infos = merge_one_source(
        args.src1_root, src1_train_ids, src1_val_ids, args.src1_prefix,
        src1_train_infos, src1_val_infos, target_root, args.copy
    )
    s2_tr_ids, s2_va_ids, s2_tr_infos, s2_va_infos = merge_one_source(
        args.src2_root, src2_train_ids, src2_val_ids, args.src2_prefix,
        src2_train_infos, src2_val_infos, target_root, args.copy
    )

    merged_train_ids = s1_tr_ids + s2_tr_ids
    merged_val_ids = s1_va_ids + s2_va_ids
    merged_train_infos = s1_tr_infos + s2_tr_infos
    merged_val_infos = s1_va_infos + s2_va_infos

    for name, ids in [
        ('train.txt', merged_train_ids),
        ('val.txt', merged_val_ids),
        ('train_src1.txt', s1_tr_ids),
        ('val_src1.txt', s1_va_ids),
        ('train_src2.txt', s2_tr_ids),
        ('val_src2.txt', s2_va_ids),
    ]:
        with open(target_root / 'ImageSets' / name, 'w', encoding='utf-8') as f:
            for sid in ids:
                f.write(f'{sid}\n')

    dump_pkl(merged_train_infos, target_root / 'merged_infos_train.pkl')
    dump_pkl(merged_val_infos, target_root / 'merged_infos_val.pkl')
    dump_pkl(s1_va_infos, target_root / 'src1_infos_val.pkl')
    dump_pkl(s2_va_infos, target_root / 'src2_infos_val.pkl')

    print(f'Merged train samples: {len(merged_train_ids)}')
    print(f'Merged val samples: {len(merged_val_ids)}')
    print(f'Target root: {target_root}')


if __name__ == '__main__':
    main()
