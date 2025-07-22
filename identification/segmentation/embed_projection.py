import torch.nn as nn

class Projected(nn.Module):
    def __init__(self, in_dim, out_dim=512):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.proj(x)

