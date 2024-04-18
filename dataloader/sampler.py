import numpy as np
from torch.utils.data import WeightedRandomSampler, RandomSampler


def get_sampler(mode, dataset, cls_img_num):
    if mode is None:
        sampler = RandomSampler(dataset)
    elif mode == 'Weighted':
        cls_weight = 1. / np.array(cls_img_num)
        sample_weight = cls_weight[dataset.targets]
        sampler = WeightedRandomSampler(sample_weight, len(sample_weight))
    else:
        raise Exception("Unsupported sampler, expect in [None, \'Weighted\']")

    return sampler
