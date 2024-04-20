import glob
import os

import numpy as np
import torchvision.datasets as dset
from torchvision import transforms
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import WeightedRandomSampler, RandomSampler

from PIL import Image


def default_loader(path):
    return Image.open(path).convert("RGB")


# customized Dataset function
class MyDataset(Dataset):
    def __init__(self, im_list, transform, label_dict, loader=default_loader):
        super(MyDataset, self).__init__()
        imgs = []
        for im_item in im_list:
            im_label_name = im_item.split("/")[-2]
            imgs.append([im_item, label_dict[im_label_name]])

        self.imgs = imgs
        self.transform = transform
        self.loader = loader
    def __getitem__(self, index):
        im_path, im_label = self.imgs[index]
        im_data = self.loader(im_path)
        if self.transform is not None:
            im_data = self.transform(im_data)
        return im_data, im_label

    def __len__(self):
        return len(self.imgs)


class FeatureDataset(Dataset):
    def __init__(self, features, labels):
        """
        Args:
            features (Tensor): Features tensor of shape (2043, 64).
            labels (Tensor): Labels tensor of shape (2043).
        """
        self.features = features
        self.labels = labels

    def __len__(self):
        """
        Returns the total number of samples.
        """
        return len(self.labels)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (feature, label, index) where label is the label of the feature and index is the index of the sample.
        """
        feature = self.features[index]
        label = self.labels[index]

        return feature, label, index


def data_loader_custm(dataset, datapath, data_transforms, imb_factor):
    # the generated img with imb factor 0.1
    data_list_trian = glob.glob(datapath + "/*/*.png")

    #data_list_trian_fold_cifar10 = "./data/TRAIN"
    data_list_trian_fold_cifar10 = datapath
    label_name_cifar10 = sorted(os.listdir(data_list_trian_fold_cifar10))
    label_dict_cifar10 = {}
    for i,ele in enumerate(label_name_cifar10):
        label_dict_cifar10[ele] = i

    data_list_trian_fold_cifar100 = datapath
    label_name_cifar100 = sorted(os.listdir(data_list_trian_fold_cifar100))

    label_dict_cifar100 = {}
    for i,ele in enumerate(label_name_cifar100):
        label_dict_cifar100[ele] = i

    # ------------------Load Original CIFAR110/1100 data------------------------#
    if dataset == 'CIFAR10' and imb_factor == 0.01:
        # original cifar 10 testset
        data_list_test = glob.glob("./data/TEST_CIFAR10/*/*.png")

        train_dataset = MyDataset(data_list_trian, transform=data_transforms['train'], label_dict=label_dict_cifar10)
        val_dataset = dset.CIFAR10(root="./DATASET", train=False, download=True, transform=data_transforms['test'])
        test_dataset = MyDataset(data_list_test,transform=data_transforms['test'], label_dict=label_dict_cifar10)

        class_num_list = [5000, 2997, 1796, 1077, 645, 387, 232, 139, 83, 50] # for long-tail imb_factor: 0.01
        num_classes = len(class_num_list)
    elif dataset == 'CIFAR10' and imb_factor == 0.1:
        data_list_test = glob.glob("./data/TEST_CIFAR10/*/*.png")

        train_dataset = MyDataset(data_list_trian, transform=data_transforms['train'], label_dict=label_dict_cifar10)
        val_dataset = dset.CIFAR10(root="./DATASET", train=False, download=True, transform=data_transforms['test'])
        test_dataset = MyDataset(data_list_test,transform=data_transforms['test'], label_dict=label_dict_cifar10)

        class_num_list = [5000, 3871, 2997, 2320, 1796, 1391, 1077, 834, 645, 500]
        num_classes = len(class_num_list)
    elif dataset == "CIFAR100" and imb_factor == 0.01:
        data_list_test = glob.glob("./data/TEST_CIFAR100/*/*.png")

        train_dataset = MyDataset(data_list_trian, transform=data_transforms['train'], label_dict=label_dict_cifar100)
        val_dataset = dset.CIFAR100(root="./DATASET", train=False, download=True, transform=data_transforms['test'])
        test_dataset = MyDataset(data_list_test, transform=data_transforms['test'], label_dict=label_dict_cifar100)

        class_num_list=[500, 477, 455, 434, 415, 396, 378, 361, 344, 328, 314, 299, 286, 273, 260, 248, 237, 226, 216, 206, 197, 188, 179,
                        171, 163, 156, 149, 142, 135, 129, 123, 118, 112, 107, 102, 98, 93, 89, 85, 81, 77, 74, 70, 67, 64, 61, 58, 56, 53,
                        51, 48, 46, 44, 42, 40, 38, 36, 35, 33, 32, 30, 29, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 15, 14, 13,
                        13, 12, 12, 11, 11, 10, 10, 9, 9, 8, 8, 7, 7, 7, 6, 6, 6, 6, 5, 5, 5, 5]   # cifar100 0.01
        num_classes = len(class_num_list)
    elif dataset == "CIFAR100" and imb_factor == 0.1:
        data_list_test = glob.glob("./data/TEST_CIFAR100/*/*.png")

        train_dataset = MyDataset(data_list_trian, transform=data_transforms['train'], label_dict=label_dict_cifar100)
        val_dataset = dset.CIFAR100(root="./DATASET", train=False, download=True, transform=data_transforms['test'])
        test_dataset = MyDataset(data_list_test, transform=data_transforms['test'], label_dict=label_dict_cifar100)

        class_num_list = [500, 488, 477, 466, 455, 445, 434, 424, 415, 405, 396, 387, 378, 369, 361, 352, 344, 336, 328,
                          321, 314, 306, 299, 292, 286, 279, 273, 266, 260, 254, 248, 243, 237, 232, 226, 221, 216, 211,
                          206, 201, 197, 192, 188, 183, 179, 175, 171, 167, 163, 159, 156, 152, 149, 145, 142, 139, 135,
                          132, 129, 126, 123, 121, 118, 115, 112, 110, 107, 105, 102, 100, 98, 95, 93, 91, 89, 87, 85, 83,
                          81, 79, 77, 75, 74, 72, 70, 69, 67, 66, 64, 63, 61, 60, 58, 57, 56, 54, 53, 52, 51, 50]  # cifar100 0.1
        num_classes = len(class_num_list)
    else:
        raise Exception('Dataset must in CIFAR10/100')

    dset_info = {'class_num': num_classes,
                 # 'per_class_loc': class_loc_list,
                 'per_class_img_num': class_num_list}

    return train_dataset, val_dataset, test_dataset, dset_info


def load_data(dataset_name, datapath, data_transforms, imb_factor):
    train_dataset, val_dataset, test_dataset, dset_info = data_loader_custm(dataset_name,
                                                                      datapath,
                                                                      data_transforms,
                                                                      imb_factor=imb_factor)

    # ---------------------Create Batch Dataloader------------------------------#
    train_set = DataLoader(train_dataset, batch_size=64,shuffle=True)
    val_set = DataLoader(val_dataset, batch_size=64,shuffle = False)
    test_set = DataLoader(test_dataset, batch_size=64,shuffle = False)

    return train_set, val_set, test_set, dset_info


def Custom_dataset(args):
    if args.dataset == "CIFAR10":
        class_num = 10
    if args.dataset == "CIFAR100":
        class_num = 100

    dataset = {"name" : args.dataset,
               "class_num" : class_num,
               "imb_factor" : args.imb_factor,
                "path" : args.datapath,
                "batch_size": 64,
                "sampler": None,
                "number_worker": 0,
                "pin_memory": True}
    return dataset


def Custom_dataset_test(args):
    if args.dataset == "CIFAR10":
        class_num = 10
    if args.dataset == "CIFAR100":
        class_num = 100
    dataset = {"name" : args.dataset,
               "class_num" : class_num,
               "imb_factor" : args.imb_factor,
                "batch_size": 64,
                "sampler": None,
                "number_worker": 0,
                "pin_memory": True}
    return dataset


def data_loader_wrapper_cust(cust_dataset):
    # Dataloader for CIFAR10/100, data info loaded in --datapath + '/' + dataset
    if cust_dataset["name"] in ['CIFAR10', 'CIFAR100']:
        # from data_loader.data_loader_cifar import load_data
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
    else:
        raise Exception('Unsupported dataset, expect in CIFAR10,CIFAR100,ImageNet,Places,iNaturalist2018')
    # dataset, datapath, data_transforms
    return load_data(cust_dataset["name"], cust_dataset["path"], data_transforms=data_transforms, imb_factor=cust_dataset["imb_factor"])
