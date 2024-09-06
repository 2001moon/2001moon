import torch
import torch.nn as nn

import config as c
from model.hinet import Hinet_stage

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        # 使用新的基于 Glow 的 Hinet_stage 实例
        self.model = Hinet_stage(in_channels=24)

    def forward(self, x, rev=False):
        # 根据 rev 参数决定前向传播的方式
        if not rev:
            # 正向传播，直接将输入 x 传递给模型
            out = self.model(x)
        else:
            # 反向传播，传递 rev=True 给模型
            out = self.model(x, rev=True)

        return out

def init_model(mod):
    # 初始化模型参数
    for key, param in mod.named_parameters():
        # 按照参数名划分层次结构
        split = key.split('.')
        if param.requires_grad:
            # 初始化所有需要梯度的参数，使用正态分布初始化，并乘以一个缩放因子 c.init_scale
            param.data = c.init_scale * torch.randn(param.data.shape).cuda()
            if split[-2] == 'conv5':
                # 如果参数属于 'conv5' 卷积层，则将参数初始化为 0
                param.data.fill_(0.)
