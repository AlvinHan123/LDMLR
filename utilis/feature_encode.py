import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import torch
from torch import optim, nn
from torch.utils.data import TensorDataset, DataLoader

from networks.resnet import resnet50
from utilis.test_ft import test_ft


def feature_encode(dataset, dataset_info, model_fixed_path, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load pre-trained long-tailed model, load the parameters to the same device
    model = test_ft(datapath = dataset_info["path"],
                    args= args,
                    modelpath = model_fixed_path,
                    crt_modelpath = None, test_cfg=None)

    model.to(device)
    modules = list(model.children())[:-1]

    if dataset_info["name"] == 'CIFAR10' or dataset_info["name"] == 'CIFAR100':
        modules.append(torch.nn.AvgPool2d(kernel_size=8))
        model = torch.nn.Sequential(*modules)
    elif args.dataset == "ImageNet":
        model = torch.nn.Sequential(*modules)

    model.to(device)

    feature_set = []
    label_set = []

    for i, (x, y) in enumerate(dataset):
        # ----------------------------Compute features----------------------------#
        img = x.cuda()
        labels = y.cuda()
        features = model(img)
        # Reshape the features to 2D
        features = features.view(features.size(0), -1)
        feature_set.append(features.detach().cpu())
        label_set.append(labels.detach().cpu())

    # Stack all features and labels
    features_tensor = torch.cat(feature_set, dim=0)
    labels_tensor = torch.cat(label_set, dim=0)
    # Create new TensorDataset and DataLoader
    new_dataset = TensorDataset(features_tensor, labels_tensor)
    new_dataloader = DataLoader(new_dataset, batch_size=256)  # You can set your own batch_size

    return new_dataset, new_dataloader
