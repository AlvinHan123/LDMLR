import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from dataloader.Custom_Dataloader import Custom_dataset, data_loader_wrapper_cust
from dataloader.data_loader_wrapper import data_loader_wrapper
from dataloader.data_loader_wrapper import Custom_dataset_ImageNet
from utilis.feature_encode import feature_encode
from torch import nn
from utilis.diffusion_model import diffusion_train
from utilis.utils import clear_cuda_cache
from utilis.config_parse import config_setup
from utilis.diffusion_model_colab import diffusion_train_colab
from utilis.test_ft import test_ft
from fine_tune_tr import fine_tune_fc
import argparse
from datetime import datetime
import torch
import logging

# python main.py --datapath /home/pengxiao/4T_data/pengxiao_space/LT-Baselines-vae/LSC/DATASET_ImageNet_LT/CIFAR10_LT001_v5 --model_fixed /4T/pengxiao_space/WCDAS/WCDAS_code/pretrained_models/CIFAR10_LT001_v5_fixed.checkpoint

parser = argparse.ArgumentParser(description='Long-Tailed Diffusion Model training   ----Author: Pengxiao Han')
parser.add_argument('--datapath', default=None, type=str, help='dataset path')
parser.add_argument('--config', default="./config/cifar10/cifar10_LSC_Mixup.txt", help='path to config file')

parser.add_argument('--epoch', default=400, type=int, help='epoch number to train')
parser.add_argument('--dataset', default="CIFAR10", type=str, help='dataset name it may be CIFAR10, CIFAR100 or ImageNet')
parser.add_argument('--imb_factor', default=0.01, type=float, help='long-tailed imbalance factor')
parser.add_argument('--diffusion_epoch', default=201, type=int, help='diffusion epoch to train')
parser.add_argument('--model_fixed', default=None, type=str, help='the encoder model path')
parser.add_argument('--feature_ratio', default=0.20, type=float, help='The ratio of generating feature')
parser.add_argument('--diffusion_step', default=1000, type=int, help='The steps of diffusion')
parser.add_argument('--checkpoint', default=None, type=str, help='model path to resume previous training, default None')
parser.add_argument('--batch_size_fc', default=1024, type=int, help='CNN fully connected layer batch size')
parser.add_argument('--learning_rate_fc', default=0.001, type=float, help='CNN fully connected layer learning rate')

parser.add_argument('--is_diffusion_pretrained', default = None, help='pre-trained diffusion model path. Training from scratch if None')
parser.add_argument('--generation_mmf', default=None, type=str, help='CNN fully connected layer batch size')

def main():
    args = parser.parse_args()
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = args.dataset + '_' + args.dataset + '_' +  str(args.imb_factor) + f"_{current_time}.log"
    logging.basicConfig(filename=log_filename, level=logging.INFO)
    logging.info("--epoch: {} --dataset: {} --imb_factor:{} --feature_ratio:{} "
          "--diffusion_step:{} (DDPM official)".format(args.epoch, args.dataset, args.imb_factor,
                                                                             args.feature_ratio, args.diffusion_step))
    print("--epoch: {} --dataset: {} --imb_factor:{} --feature_ratio:{} "
          "--diffusion_step:{} (DDPM official)".format(args.epoch, args.dataset, args.imb_factor, args.feature_ratio, args.diffusion_step))
    # 1. load data -------------------------------------------------------------------------------------------------------------
    # ouput: dataloader of cifar10/cifar100/ImageNet
    cfg, finish = config_setup(args.config,
                               args.checkpoint,
                               args.datapath,
                               update=False)
    if args.dataset == "CIFAR10" or args.dataset == "CIFAR100":
        # customized dataset
        dataset_info = Custom_dataset(args)
        train_set, _, test_set, dset_info = data_loader_wrapper_cust(dataset_info)
    elif args.dataset == "ImageNet":
        dataset_info = Custom_dataset_ImageNet(args)
        train_set, val_set, test_set, dset_info = data_loader_wrapper(cfg.dataset)

    # 2. Encoder - encode images into features (batch_size, feature_dim)--------------------------------------------------------
    # input: image dataloader (batch_size, 3, 32, 32); output: feature dataloader (batch_size, 64)
    feature_dataset_tr, feature_dataloader_tr = feature_encode(train_set, dataset_info, args.model_fixed, args)

    # 3. training a diffusion model and generate features
    # input: cifar training set, cifar testing set, feature dataloader; output: generated features by diffusion ------------
    generated_features, fake_classes = diffusion_train(train_set, test_set, feature_dataloader_tr, dataset_info, dset_info, args)  # FIXME Official DDPM/DDIM

    # fine-tuning a fully-connected layer using generated features
    fine_tune_fc(generated_features, fake_classes, feature_dataset_tr, test_set, dataset_info, args, dset_info)

    print(" ------------Finish--------------")


if __name__ == '__main__':
    main()
