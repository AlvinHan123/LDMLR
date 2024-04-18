from torchvision import transforms


def data_loader_wrapper(config):
    # Dataloader for CIFAR10/100, data info loaded in --datapath + '/' + dataset
    if config['name'] in ['CIFAR10', 'CIFAR100']:
        from .dataloader.data_loader_cifar import load_data
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

    # Dataloader for ImageNet datasets, data info loaded in --datapath + '/' + dataset
    elif config['name'] == 'ImageNet':
        from .dataloader.data_loader_ImageNet import load_data
        # Data transformation with augmentation
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        }

    # Dataloader for iNaturalist2017, 2018 datasets, data info loaded in --datapath + '/' + dataset
    elif config['name'] in ['iNaturalist2018', 'iNaturalist2017']:
        from data_loader.data_loader_iNaturalist import load_data
        # Data transformation with augmentation
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.466, 0.471, 0.380], [0.195, 0.194, 0.192])
            ]),
            'test': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.466, 0.471, 0.380], [0.195, 0.194, 0.192])
            ])
        }

    # Dataloader for Places365 datasets, data info loaded in --datapath + '/' + dataset
    elif config['name'] == 'Places':
        from data_loader.data_loader_Places import load_data
        # Data transformation with augmentation
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        }
    else:
        raise Exception('Unsupported dataset, expect in CIFAR10,CIFAR100,ImageNet,Places,iNaturalist2018')

    return load_data(config['path'], data_transforms, config)


# Get the image normalization factor mean and std for each dataset
def get_norm_params(dataset):
    if dataset in ['CIFAR10', 'CIFAR100']:
        mean = [x / 255.0 for x in [125.3, 123.0, 113.9]]
        std = [x / 255.0 for x in [63.0, 62.1, 66.7]]
        return mean, std
    elif dataset in ['ImageNet', 'Places']:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        return mean, std
    elif dataset in ['iNaturalist2018', 'iNaturalist2017']:
        mean = [0.466, 0.471, 0.380]
        std = [0.195, 0.194, 0.192]
        return mean, std
    else:
        raise Exception('Unsupported dataset %s' % dataset)

def Custom_dataset_ImageNet(args):

    dataset = {"name" : args.dataset,
               "class_num" : 1000,
               "imb_factor" : args.imb_factor,
                "path" : args.datapath,
                "batch_size": 128,
                "sampler": None,
                "number_worker": 0,
                "pin_memory": True}
    return dataset