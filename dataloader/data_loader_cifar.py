import os
import pickle
from PIL import Image

import numpy as np
import torchvision.datasets as dset
from torchvision import transforms
from torch.utils.data import DataLoader as DLoader

from .dataloader.sampler import get_sampler

normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                 std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4),
        transforms.ToTensor(),
        normalize
    ]),
    'test': transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
}


def load_data(datapath, data_transforms, params):
    # kwargs = {'num_workers': params['num_workers'], 'pin_memory': params['pin_memory'], 'drop_last': True}
    kwargs = {'num_workers': params['num_workers'], 'pin_memory': params['pin_memory']}
    train_dataset, val_dataset, test_dataset, dset_info = data_loader(params['name'],
                                                                      datapath,
                                                                      data_transforms,
                                                                      imb_factor=params['imb_factor'])

    # --------------------------Define Sample Strategy--------------------------#
    sampler = get_sampler(params['sampler'], train_dataset, dset_info['per_class_img_num'])

    # ---------------------Create Batch Dataloader------------------------------#
    train_set = DLoader(train_dataset, batch_size=params['batch_size'], sampler=sampler, **kwargs)
    val_set = DLoader(val_dataset, batch_size=params['batch_size'], shuffle=False, **kwargs)
    test_set = DLoader(test_dataset, batch_size=params['batch_size'], shuffle=False, **kwargs)

    return train_set, val_set, test_set, dset_info


def data_loader(dataset, datapath, data_transforms, imb_factor=None):
    # ------------------Load Original CIFAR110/1100 data------------------------#
    if dataset == 'CIFAR10':
        train_dataset = dset.CIFAR10(root=datapath, train=True, download=True, transform=data_transforms['train'])
        val_dataset = dset.CIFAR10(root=datapath, train=False, download=True, transform=data_transforms['test'])
        test_dataset = dset.CIFAR10(root=datapath, train=False, transform=data_transforms['test'])
    elif dataset == 'CIFAR100':
        train_dataset = dset.CIFAR100(root=datapath, train=True, download=True, transform=data_transforms['train'])
        val_dataset = dset.CIFAR100(root=datapath, train=False, download=True, transform=data_transforms['test'])
        test_dataset = dset.CIFAR100(root=datapath, train=False, transform=data_transforms['test'])
    else:
        raise Exception('Dataset must in CIFAR10/100')

    # ---------------------Obtain Dataset Basic Info----------------------------#
    num_classes = len(train_dataset.classes)
    # print("Dataset: %s, # of classes: %i" % (dataset,num_classes))
    class_loc_list = [[] for i in range(num_classes)]
    for i, label in enumerate(train_dataset.targets):
        class_loc_list[label].append(i)
    data_list_val = []
    data_list_train = []
    # -----------------Create Long-Tailed CIFAR10/100 Data----------------------#
    if dataset in ['CIFAR10', 'CIFAR100']:
        class_num_list = get_img_num_per_cls(dataset, imb_factor)
        if imb_factor is not None:
            _ = [np.random.shuffle(x) for x in class_loc_list]
            _ = [data_list_val.extend(list(x[y:])) for x, y in zip(class_loc_list, class_num_list)]
            _ = [data_list_train.extend(list(x[:y])) for x, y in zip(class_loc_list, class_num_list)]

            train_dataset.data = np.delete(train_dataset.data, data_list_val, axis=0)
            train_dataset.targets = np.delete(train_dataset.targets, data_list_val, axis=0)

    dset_info = {'class_num': num_classes,
                 # 'per_class_loc': class_loc_list,
                 'per_class_img_num': class_num_list}

    return train_dataset, val_dataset, test_dataset, dset_info


        # ======================== Imbalanced Cifar Data ===============================
def get_img_num_per_cls(dataset, imb_factor=None):
    """
    Get a list of image numbers for each class, given cifar version
    Num of imgs follows emponential distribution
    img max: 5000 / 500 * e^(-lambda * 0);
    img min: 5000 / 500 * e^(-lambda * int(cifar_version - 1))
    exp(-lambda * (int(cifar_version) - 1)) = img_max / img_min
    args:
      cifar_version: str, '10', '100', '20'
      imb_factor: float, imbalance factor: img_min/img_max,
        None if geting default cifar data number
    output:
      img_num_per_cls: a list of number of images per class
    """
    if dataset == 'CIFAR10':
        cls_num = 10
    elif dataset == 'CIFAR100':
        cls_num = 100
    else:
        raise Exception('Function only valid for CIFAR10 and CIFAR100 dataset')

    img_max = 50000 / cls_num
    if imb_factor is None:
        return [int(img_max)] * cls_num
    img_num_per_cls = []
    for cls_idx in range(cls_num):
        num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
        img_num_per_cls.append(int(num))
    return img_num_per_cls

# if __name__ == '__main__':
#     print(get_img_num_per_cls(dataset = "CIFAR10", imb_factor = 0.1))
