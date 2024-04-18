import numpy as np
import torch

def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)

def move_model_to_device(model, device):
    model.to(device)
    return model

def move_model_to_cpu(model):
    return model.cpu()

def clear_cuda_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def get_stamp(cfg, crt_cfg=None):
    lrs = cfg.lr_scheduler['name'][0] if 'lr_scheduler' in cfg.keys else 'n'
    stamp = cfg.model['name'] + '-' + \
            cfg.model['ensemble_info']['name'][0] + \
            lrs + \
            cfg.loss['name'][0] + \
            cfg.train_info['mode'][0] + \
            cfg.train_info['data_aug']['name'][0]

    if crt_cfg is not None:
        clrs = crt_cfg.lr_scheduler['name'][0] if 'lr_scheduler' in crt_cfg.keys else 'n'
        stamp2 = crt_cfg.model['name'] + \
                 crt_cfg.model['ensemble_info']['name'][0] + \
                 clrs + \
                 crt_cfg.loss['name'][0] + \
                 crt_cfg.train_info['mode'][0] + \
                 crt_cfg.train_info['data_aug']['name'][0] + \
                 str(crt_cfg.dataset['sampler'])[0]
        stamp = stamp + '_' + stamp2

    return stamp
