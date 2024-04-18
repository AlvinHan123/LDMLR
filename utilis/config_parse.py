import torch
import ast
import copy
from functools import reduce
from operator import getitem
from utilis.sane_check import cfg_check, consistency_check
import json


def config_setup(config_path, checkpoint_path, data_path, update=False):
    # --------------------Stage1 Args Parsing---------------------------#
    if (config_path is None) and (checkpoint_path is None):
        print("No stage-1 info in --config and --checkpoint is provided.")
        cfg = None
        # Determine if stage-1 model have finished training.
        finish = True
    else:
        print('Loading stage-1 info from  %s and %s.' % (str(config_path), str(checkpoint_path)))
        cfg = Checkpoint(config_path, checkpoint_path, update=update)
        cfg_check(cfg)
        cfg.update(['dataset', 'path'], data_path + cfg.dataset['name'])
        cfg.update(['lr_scheduler', 'T_max'], cfg.train_info['epoch'])
        # Determine if stage-1 model have finished training.
        finish = cfg.finished
    return cfg, finish


class Checkpoint:
    def __init__(self, config_path, checkpoint_path, update=False):
        if config_path is None and checkpoint_path is None:
            raise Exception('Either config or checkpoint_path should be given to enable training.')

        if checkpoint_path is not None:
            checkpoint = dict(torch.load(checkpoint_path))
            if checkpoint['train_info']['current_epoch'] == checkpoint['train_info']['epoch']:
                self._resume = False
                self._config_finished = True
            else:
                self._resume = True
                self._config_finished = False
            self._cfg = checkpoint
            print('model path is given, loaded.')

        if config_path is not None:
            self._resume = False
            self._config_finished = False
            with open(config_path, 'r') as f:
                config = ast.literal_eval(f.read().replace(' ', '').replace('\n', ''))
            print('config path is given, loaded.')

            # Preferred to use checkpoint_path for training, update the train_info and loss if update=True
            if checkpoint_path is not None:
                if update:
                    print("Both config and model path given, update=True, update [\'train_info\'], model[\'loss\']")
                    self.update(['train_info'], config['train_info'])
                    self.update(['loss'], config['loss'])
                else:
                    print("update=False but both config_path and checkpoint_path given, use checkpoint_path")
            else:
                print('Only config path is given, train with config_path')
                self._cfg = config
                if 'state_dict' not in self._cfg.keys():
                    self.update(['state_dict'], {})
                if 'ensemble_info' not in self._cfg['model'].keys():
                    self.update(['model', 'ensemble_info'], self._cfg['backbone']['ensemble_info'])


    @property
    def resume(self):
        return self._resume

    @property
    def finished(self):
        return self._config_finished

    @property
    def all(self):
        return copy.deepcopy(self._cfg)

    @property
    def dataset(self):
        return copy.deepcopy(self._cfg['dataset'])

    @property
    def backbone(self):
        return copy.deepcopy(self._cfg['backbone'])

    @property
    def model(self):
        return copy.deepcopy(self._cfg['model'])

    @property
    def lr_scheduler(self):
        return copy.deepcopy(self._cfg['lr_scheduler'])

    @property
    def optimizer(self):
        return copy.deepcopy(self._cfg['optimizer'])

    @property
    def loss(self):
        return copy.deepcopy(self._cfg['loss'])

    @property
    def train_info(self):
        return copy.deepcopy(self._cfg['train_info'])

    @property
    def state_dict(self):
        # return copy.deepcopy(self._cfg['state_dict'])
        return self._cfg['state_dict']

    @property
    def checkpoint(self):
        return copy.deepcopy(self._cfg['checkpoint'])

    @property
    def keys(self):
        return self._cfg.keys()

    @property
    def config(self):
        cfg = copy.deepcopy({x: self._cfg[x] for x in self._cfg if x not in ['state_dict']})
        return cfg

    def update(self, keys, value):
        self._cfg, add = set_nested_item(self._cfg, keys, value)

    def print_old(self):
        info = "\n {:<8} {:<15} {:<10}\n".format('Key', 'Label', 'Value')
        for i, k in enumerate(self._cfg.keys()):
            if k != 'state_dict':
                if 'class_num_list' not in self._cfg[k].keys():
                    val = str(self._cfg[k])
                else:
                    val = str({x: self._cfg[k][x] for x in self._cfg[k] if x not in ['class_num_list']})
                    # cls_num_list = str(self._cfg[k]['cls_num_list'])
                info += "{:<8} {:<15} {:<10}\n".format(i, k, val)

        # return str(self.config)
        return info

    def print(self):
        cfg = self.config
        cfg['train_info'].pop('class_num_list', None)
        return json.dumps(cfg, indent=4)

    def get_state_dict(self, key, path):
        if key == 'model':
            self.update(['state_dict'][key], torch.load(path))
        else:
            raise Exception('Only [\'state_dict\'][\'model\'] is allowed to update in current version.')

    def save(self, path):
        torch.save(self._cfg, path)



def set_nested_item(dataDict, mapList, val):
    """Set item in nested dictionary"""
    add = False
    if mapList[-1] in reduce(getitem, mapList[:-1], dataDict):
        add = True
    reduce(getitem, mapList[:-1], dataDict)[mapList[-1]] = val
    return dataDict, add





