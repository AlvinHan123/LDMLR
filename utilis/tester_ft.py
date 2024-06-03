import sys
import os
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.model_init import model_init


def tester_ft(dataset, trainset, valset, config, args):

    # --------------------------Load Backbone---------------------------------#
    if 'backbone' in config.keys():
        if 'backbone' in config['state_dict'].keys():
            backbone_state_dict = config['state_dict']['backbone']
        else:
            backbone_state_dict = torch.load(config['backbone']['path'])['state_dict']['model']
        backbone = model_init(config['backbone'],
                              config['dataset']['name'],
                              backbone_state_dict)
    else:
        backbone = None
    # --------------------------Load Model------------------------------------#
    if 'model' in config['state_dict'].keys():
        model_state_dict = config['state_dict']['model']
    else:
        model_state_dict = torch.load(config['checkpoint']['save_path'])['state_dict']['model']

    model = model_init(config['model'],
                       config['dataset']['name'],
                       model_state_dict)
    return model
