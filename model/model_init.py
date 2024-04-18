import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.losses import *
import logging
import importlib

#fixme
model_paths = {
    'stage2':        'networks.stage_2',
    'none_cifar':    'networks.resnet_cifar',
    'none':          'networks.resnet',
    'tail_cifar':    'networks.resnet_cifar_ensemble',
    'tail':          'networks.resnet_ensemble'
}


def model_init(cfg, dataset, state_dict=None):
    #names = ['resnet10', 'resnet50', 'resnext50', 'resnet152', 'resnet32', 'cRT', 'LWS', 'MiSLAS']

    # Check if last fc layer in resnet is nn.Linear or NormLayer
    if 'fc_norm' not in cfg.keys():
        cfg['fc_norm'] = False
        logging.info("No fc_norm in model config, set default training with fc_layer = nn.Linear")
    elif not cfg['fc_norm']:
        logging.info("fc_norm = False in model config, fc_layer = nn.Linear")
    elif cfg['fc_norm']:
        logging.info("fc_norm = True in model config, fc_layer = NormLayer")
    else:
        raise Exception('Unsupported fc_norm option in model config, expect boolean')

    # --------------------------Load Stage-2 Models-----------------------------#
    # Classifier Re-Training (CRT)
    if cfg['name'] == 'cRT':
        logging.info('Loading cRT model for second stage training')
        module = getattr(importlib.import_module(model_paths['stage2']), cfg['name'])
        model = module(cfg['input_dim'],
                       cfg['output_dim'],
                       cfg['ensemble_info']['name'],
                       cfg['ensemble_info']['ensemble_num'],
                       cfg['fc_norm'])

    # Learnable Weight Scaling (LWS)

    # --------------------------Regular ResNet models---------------------------#
    elif cfg['ensemble_info']['name'] == 'none':
        logging.info('\'none\' given in [\'ensemble_info\'][\'name\'], load regular network.')
        if dataset in ['CIFAR10', 'CIFAR100']:
            module = getattr(importlib.import_module(model_paths['none_cifar']), cfg['name'])
            model = module(cfg['output_dim'], fc_norm=cfg['fc_norm'])
        else:
            module = getattr(importlib.import_module(model_paths['none']), cfg['name'])
            model = module(cfg['output_dim'], fc_norm=cfg['fc_norm'])

    # --------------Load Multi-Tailed ResNet Ensemble Models--------------------#
    elif cfg['ensemble_info']['name'] == 'tail':
        logging.info('\'tail\' given in [\'ensemble_info\'][\'name\'], load Multi-Tailed network.')
        ensemble_num = cfg['ensemble_info']['ensemble_num']
        if dataset in ['CIFAR10', 'CIFAR100']:
            module = getattr(importlib.import_module(model_paths['tail_cifar']), cfg['name'])
            model = module(cfg['output_dim'], ensemble_num, fc_norm=cfg['fc_norm'])
        else:
            module = getattr(importlib.import_module(model_paths['tail']), cfg['name'])
            model = module(cfg['output_dim'], ensemble_num, fc_norm=cfg['fc_norm'])

    # ----------------Raise Error with Unknown Ensemble name--------------------#
    else:
        raise Exception('Unsupported cfg[\'ensemble_info\'][\'name\'],' +
                        'expect in [\'none\', \'tail\']')

    if state_dict is not None:
        try:
            model.load_state_dict(state_dict)
        except:
            raise Exception('Failed to load state dict to current model.')

    return model


def optimizer_init(cfg, model, state_dict=None):
    if cfg['name'] == 'SGD':
        optimizer = torch.optim.SGD([{'params': model.parameters(), 'initial_lr': cfg['lr']}],
                                    lr=cfg['lr'],
                                    momentum=cfg['momentum'],
                                    nesterov=cfg['nesterov'],
                                    weight_decay=cfg['wd'])
    else:
        raise Exception('Currently only SGD optimizer is supported.')
    if state_dict is not None:
        try:
            optimizer.load_state_dict(state_dict)
        except:
            raise Exception('Failed to load state dict to current optimizer.')

    return optimizer


def lr_scheduler_init(cfg, optimizer, state_dict=None):
    if cfg['name'] == 'cos':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['T_max'])
    elif cfg['name'] == 'step':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg['milestones'])
    elif cfg['name'] == 'warmup_step':
        gamma = 0.1

        def lr_lambda(epoch):
            if epoch >= cfg["milestones"][1]:
                lr = gamma * gamma
            elif epoch >= cfg["milestones"][0]:
                lr = gamma
            else:
                lr = 1

            """Warmup"""
            warmup_epoch = cfg["warmup_epoch"]
            if epoch < warmup_epoch:
                lr = lr * float(1 + epoch) / warmup_epoch
            return lr

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        raise Exception('Unsupported lr_scheduler, expect in [\'cos\',\'step\',\'warmup_step\']')

    if state_dict is not None:
        try:
            lr_scheduler.load_state_dict(state_dict)
        except:
            raise Exception('Failed to load state dict to current lr_scheduler.')

    return lr_scheduler


def loss_init(cfg, class_num_list, milestones, epoch, per_cls_weights=None, logit_scale=1):
    # --------------------------Loss Init-----------------------------------#
    if cfg['name'] == 'CE':
        criterion = CELoss(weight=per_cls_weights, scale=logit_scale).cuda()
    elif cfg['name'] == 'Focal':
        criterion = FocalLoss(gamma=cfg['focal_gamma'], weight=per_cls_weights, scale=logit_scale).cuda()
    elif cfg['name'] == 'LDAM':
        criterion = LDAMLoss(class_num_list, weight=per_cls_weights, s=logit_scale).cuda()
    else:
        raise Exception('Unsupported Loss type.')

    return criterion
