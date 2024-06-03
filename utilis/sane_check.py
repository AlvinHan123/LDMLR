import numpy as np
import ast


def arg_path_check(cfg_path, ckpt_path, crt_cfg_path, crt_ckpt_path):
    def check_none(path):
        return False if path is None else True

    character = list(map(check_none, [cfg_path, ckpt_path, crt_cfg_path, crt_ckpt_path]))

    if character in [[1, 0, 0, 0], [1, 0, 1, 0], [0, 1, 1, 0], [0, 1, 0, 0]]:
        print('config and checkpoint sane check passed.')
    elif character in [[1, 1, 1, 1], [1, 1, 0, 0], [0, 0, 1, 1]]:
        print('[warning] Both config and checkpoint path are provided for stage-1 or stage-2 training, '
              'will only consider info in checkpoint, set update=True to update checkpoint with configs')
    elif character == [0, 0, 0, 1]:
        print('[warning] Only --crt_checkpoint is provided, '
              'will resume stage-2 training or directly test for 2-stage model. ')
    elif character == [0, 0, 1, 0]:
        raise Exception('--crt_config is required to provided along with '
                        '--checkpoint or --config to enable training')
    else:
        raise Exception('Unsupported --config, --checkpoint, '
                        '--crt_config, --crt_checkpoint combination: ' + str(character))


def consistency_check(cfg, crt_cfg):
    if cfg.model != crt_cfg.backbone:
        print('Found config[\'model\'] != crt_config[\'backbone\'], will go with config[\'model\']')
        print('crt_config: ', crt_cfg.backbone)
        print('config: ', cfg.model)
        crt_cfg.update(['backbone'], cfg.model)
        crt_cfg.update(['model', 'ensemble_info'], cfg.model['ensemble_info'])

    #if cfg.dataset != crt_cfg.dataset:
    #    print('Found config[\'dataset\'] != crt_config[\'dataset\'], will go with config[\'model\']')
    #    crt_cfg.update(['dataset'], cfg.dataset)
    #    print(crt_cfg.dataset, cfg.dataset)

    print('consistency check finished.')


def cfg_check(cfg):
    # Perform a dict check with depth = 2
    path = './utilis/config_range.txt'
    with open(path, 'r') as f:
        config = ast.literal_eval(f.read().replace(' ', '').replace('\n', ''))
    cfg_config = cfg.config

    type_list = ['float', 'int', 'str', 'bool', 'list', None]
    for k in config.keys():
        for k1 in config[k]:
            k_type = type(config[k][k1])
            if k_type != dict:
                check_key(config[k][k1], cfg_config[k][k1], type_list)
            else:
                for k2 in config[k][k1]:
                    check_key(config[k][k1][k2], cfg_config[k][k1][k2], type_list)
    print('Params range check based on %s have passed.' % path)


def check_key(key_range, target, type_list):
    k_type = type(key_range)
    if k_type == str and key_range in type_list:
        pass
    elif k_type == list and np.sum([(x in type_list) for x in key_range]) == len(key_range):
        pass
    else:
        if not range_check(key_range, target):
            raise Exception('Config config[\'%s\'] not in range %s' % (target, str(key_range)))


def range_check(val_range, val):
    if isinstance(val_range, list):
        if val in val_range:
            return True
    else:
        return False


def type_check(val_range, val):
    if not isinstance(val, list):
        if "float" in val_range and type(val) == float:
            return True
        elif "int" in val_range and type(val) == int:
            return True
        elif "bool" in val_range and (type(val) == bool):
            return True
        elif "str" in val_range and type(val) == str:
            return True
        else:
            return False
    else:
        if "float" in val_range and all(isinstance(x, float) for x in val):
            return True
        elif "int" in val_range and all(isinstance(x, int) for x in val):
            return True
        elif "bool" in val_range and all(isinstance(x, bool) for x in val):
            return True
        elif "str" in val_range and all(isinstance(x, str) for x in val):
            return True
        else:
            return False
