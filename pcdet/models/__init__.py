from collections import namedtuple

import numpy as np
import torch

from .detectors import build_detector

try:
    import kornia
except:
    kornia = None
    # print('Warning: kornia is not installed. This package is only required by CaDDN')



def build_network(model_cfg, num_class, dataset):
    model = build_detector(
        model_cfg=model_cfg, num_class=num_class, dataset=dataset
    )
    return model


def load_data_to_gpu(batch_dict):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for key, val in batch_dict.items():
        if key == 'camera_imgs':
            batch_dict[key] = val.to(device)
        elif not isinstance(val, np.ndarray):
            continue
        elif key in ['frame_id', 'metadata', 'calib', 'image_paths', 'cam_frame_id', 'candidate_name', 'calib_source', 'ori_shape', 'img_process_infos']:
            continue
        elif key in ['images']:
            if kornia is not None:
                batch_dict[key] = kornia.image_to_tensor(val).float().to(device).contiguous()
            else:
                batch_dict[key] = torch.from_numpy(val).float().to(device).contiguous()
        elif key in ['image_shape']:
            batch_dict[key] = torch.from_numpy(val).int().to(device)
        else:
            batch_dict[key] = torch.from_numpy(val).float().to(device)


def model_fn_decorator():
    ModelReturn = namedtuple('ModelReturn', ['loss', 'tb_dict', 'disp_dict'])

    def model_func(model, batch_dict):
        load_data_to_gpu(batch_dict)
        ret_dict, tb_dict, disp_dict = model(batch_dict)

        loss = ret_dict['loss'].mean()
        if hasattr(model, 'update_global_step'):
            model.update_global_step()
        else:
            model.module.update_global_step()

        return ModelReturn(loss, tb_dict, disp_dict)

    return model_func
