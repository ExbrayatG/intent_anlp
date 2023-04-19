import torch.nn as nn
import torch


class OneLayerMLP(nn.Module):
    def __init__(self, D_in, H, D_out, bias=True):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(D_in, H, bias=bias), nn.ReLU(), nn.Linear(H, D_out, bias=bias)
        )

    def forward(self, x):
        return self.network(x)
