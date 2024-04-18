import torch.nn as nn
import torch.nn.functional as F

class DeepNet(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(DeepNet, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        out = self.fc(x)
        # out = F.log_softmax(out, dim=1)
        return out
