import torch
import torch.nn as nn
import torch.nn.functional as F

class ActNorm(nn.Module):
    def __init__(self, in_channels):
        super(ActNorm, self).__init__()
        self.scale = nn.Parameter(torch.ones(1, in_channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, in_channels, 1, 1))
    
    def forward(self, x, reverse=False):
        if reverse:
            x = (x - self.bias) / self.scale
        else:
            x = x * self.scale + self.bias
        return x

class InvConv2d(nn.Module):
    def __init__(self, in_channels):
        super(InvConv2d, self).__init__()
        weight = torch.randn(in_channels, in_channels)
        q, _ = torch.qr(weight)
        self.weight = nn.Parameter(q)
    
    def forward(self, x, reverse=False):
        if reverse:
            weight = self.weight.inverse()
        else:
            weight = self.weight
        weight = weight.unsqueeze(2).unsqueeze(3)
        return F.conv2d(x, weight)

class CouplingLayer(nn.Module):
    def __init__(self, in_channels, mid_channels):
        super(CouplingLayer, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels // 2, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels//2, kernel_size=3, padding=1)
        )
    
    def forward(self, x, reverse=False):
        x1, x2 = torch.chunk(x, 2, dim=1)
        if reverse:
            x2 = (x2 - self.net(x1)) / torch.exp(self.net(x1))
        else:
            x2 = x2 * torch.exp(self.net(x1)) + self.net(x1)
        return torch.cat((x1, x2), dim=1)
