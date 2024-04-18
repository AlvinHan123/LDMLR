import torch
import torch.nn as nn
from networks.NormLayer import NormedLinear


class cRT(nn.Module):
    def __init__(self, input: int, output: int, ensemble_name: str = 'none', ensemble_num: int = None, fc_norm=False):
        super(cRT, self).__init__()
        self.ensemble_name = ensemble_name
        self.ensemble_num = ensemble_num
        if self.ensemble_name == 'none':
            if fc_norm:
                self.model = NormedLinear(input, output)
            else:
                self.model = nn.Linear(input, output)
        elif self.ensemble_name == 'tail':
            self.model = nn.ModuleList()
            if fc_norm:
                for i in range(ensemble_num):
                    self.model.append(NormedLinear(input, output))
            else:
                for i in range(ensemble_num):
                    self.model.append(nn.Linear(input, output))
        else:
            raise Exception('Unsupported config[\'ensemble_info\'][\'name\'],' +
                            'expect in [\'none\', \'tail\']')

        self.relu = nn.ReLU()
        # torch.nn.init.kaiming_normal_(self.model.weight)

    def forward(self, x):
        if self.ensemble_name in ['none', 'mask', 'dropout']:
            x = self.model(x)
        elif self.ensemble_name == 'tail':
            x = [f(out) for f, out in zip(self.model, x)]

        return x


class LWS(nn.Module):
    def __init__(self, output: int, ensemble_name: str = 'none', ensemble_num: int = None):
        super(LWS, self).__init__()
        self.ensemble_name = ensemble_name
        self.ensemble_num = ensemble_num
        if self.ensemble_name == 'none':
            self.model = ScaleLayer(output)
        elif self.ensemble_name == 'tail':
            self.model = nn.ModuleList()
            for i in range(ensemble_num):
                self.model.append(ScaleLayer(output))
        else:
            raise Exception('Unsupported config[\'ensemble_info\'][\'name\'],' +
                            'expect in [\'none\', \'tail\']')

    def forward(self, x):
        if self.ensemble_name == 'none':
            x = self.model(x)
        elif self.ensemble_name == 'tail':
            x = [f(out) for f, out in zip(self.model, x)]

        return x


class ScaleLayer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        dim_num = len(x.shape) - 1
        return x * self.scale[(None,) * dim_num + (...,)]

