import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.utils import normalized
import logging


def model_eval(model, dataset, cls_num_list, backbone=None, ensemble_name='none', ensemble_num=None, crt=False):
    model.eval()

    if ensemble_name == 'dropout':
        model.apply(apply_dropout)

    if backbone is not None:
        backbone.eval()
    else:
        def backbone(x,crt=False):
            return x

    labels = []
    probs = []
    with torch.no_grad():
        for i, (x, y) in enumerate(dataset):
            # batch_size = len(y.numpy())
            labels.extend(list(y.numpy()))
            img = x.cuda()

            if ensemble_name == 'none':
                logit = model(backbone(img, crt=crt)).detach()
                prob = F.softmax(logit, dim=-1)
                probs.extend(list(prob.cpu().numpy()))

            elif ensemble_name == 'tail':
                logit = model(backbone(img, crt=crt))
                prob = torch.stack([F.softmax(l.detach(), dim=-1) for l in logit], dim=1)
                probs.extend(list(prob.squeeze().cpu().numpy()))
            else:
                raise Exception('Unsupported config[\'ensemble_info\'][\'name\'],' +
                                'expect in [\'none\', \'tail\']')

    return np.array(probs), labels


def apply_dropout(m):
    if type(m) == nn.Dropout:
        m.train()
