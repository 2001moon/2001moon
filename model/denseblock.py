from model.rrdb_denselayer import *


class Dense(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, input, output):
        super(Dense, self).__init__()

        self.dense = ResidualDenseBlock_out(input, output, nf=c.nf, gc=c.gc) # 带有残差连接的密集块

    def forward(self, x): # 定义了前向传播的计算过程
        out = self.dense(x) # 输入 x 通过 self.dense（即 ResidualDenseBlock_out 实例）进行处理，结果保存在 out 中

        return out


