import os
import argparse

import numpy as np
import torch
import time
import logging
import json
import copy

from utilis.tester_ft import tester_ft
from utilis.config_parse import config_setup
from dataloader.data_loader_wrapper import data_loader_wrapper
from dataloader.Custom_Dataloader import Custom_dataset, data_loader_wrapper_cust


def test_ft(datapath, args, modelpath=None, crt_modelpath=None, test_cfg=None):
    # ---------------------------Load Saved Model---------------------------#
    if modelpath is not None:
        try:
            config = dict(torch.load(modelpath))
        except:
            raise Exception('Unable to load checkpoint via torch.load at %s , please check the path.' % modelpath)

    if crt_modelpath is not None:
        try:
            config = dict(torch.load(crt_modelpath))
        except:
            raise Exception('Unable to load checkpoint via torch.load at %s , please check the path.' % crt_modelpath)

    if (crt_modelpath is None) and (modelpath is None):
        raise Exception('Checkpoint of the model should be given in either --modelpath or --crt_modelpath.')
    elif (crt_modelpath is not None) and (modelpath is not None):
        print('warning: both --modelpath and --crt_modelpath are given, will ignore --modelpath.')

    # imb_logname = int(1 / config['dataset']['imb_factor']) if config['dataset']['imb_factor'] is not None else 'None'
    path = crt_modelpath if crt_modelpath is not None else modelpath
    # Generate print version of config (without [\'state_dict\'], [\'train_info\'][\'class_num_list\'])
    cfg_print = copy.deepcopy({k: config[k] for k in set(list(config.keys())) - set(['state_dict'])})
    cfg_print['train_info'].pop('class_num_list', None)

    # ----------------Loading the dataset, create dataloader----------------#
    config['dataset']['path'] = datapath + config['dataset']['name']   # 这块儿是不是要改一下
    if torch.cuda.device_count() == 1:
        config['dataset']['num_workers'] = 4

    if args.dataset == "CIFAR10" or args.dataset == "CIFAR100":
        cust_dataset = Custom_dataset(args=args)
        train_set, val_set, test_set, dset_info = data_loader_wrapper_cust(cust_dataset)
    elif args.dataset == "ImageNet":
        cfg, finish = config_setup(args.config,
                                   args.checkpoint,
                                   args.datapath,
                                   update=False)
        train_set, val_set, test_set, dset_info = data_loader_wrapper(cfg.dataset)
    config['train_info']['class_num_list'] = dset_info['per_class_img_num']

    # -------------------------Test the Model-------------------------------#
    logging.info('Test performance on test set.')
    model = tester_ft(test_set, train_set, val_set, config, args)
    return model


def default(obj):
    if isinstance(obj, np.ndarray):
        return obj.round(4).tolist()
    elif isinstance(obj, np.float32):
        return round(float(obj), 4)
    raise TypeError('Not serializable')
