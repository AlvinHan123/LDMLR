import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F


# Normlayer used in LDAM-DRW https://github.com/kaidic/LDAM-DRW
class NormedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return out
