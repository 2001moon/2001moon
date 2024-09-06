import torch
import torch.nn as nn
from model.invblock import ActNorm, InvConv2d, CouplingLayer

class Hinet_stage(nn.Module):
    def __init__(self, num_flows=3, in_channels=24, mid_channels=64):
        super(Hinet_stage, self).__init__()
        self.flows = nn.ModuleList()
        for _ in range(num_flows):
            self.flows.append(ActNorm(in_channels))
            self.flows.append(InvConv2d(in_channels))
            self.flows.append(CouplingLayer(in_channels, mid_channels))

    def forward(self, x, reverse=False):
        if reverse:
            for flow in reversed(self.flows):
                x = flow(x, reverse)
        else:
            for flow in self.flows:
                x = flow(x)
        return x
