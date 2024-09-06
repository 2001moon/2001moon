import torch
import torch.nn as nn
import modules.module_util as mutil
import config as c


# Dense connection
class ResidualDenseBlock_out(nn.Module): # 密集残差块
    def __init__(self, input, output, nf=c.nf, gc=c.gc, bias=True, use_snorm=False): 
# input: 输入通道数。output: 输出通道数。nf: 基础特征数量（从全局配置 c 中读取）。gc: 增长通道（Growth Channel），即中间通道数量（从全局配置 c 中读取）。
# bias: 卷积层是否使用偏置项，默认为 True。
# use_snorm: 是否使用谱归一化（Spectral Normalization），默认为 False。
        super(ResidualDenseBlock_out, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        if use_snorm: # 如果 use_snorm 为 True，则使用谱归一化的卷积层
            self.conv1 = nn.utils.spectral_norm(nn.Conv2d(nf, gc, 3, 1, 1, bias=bias))
            self.conv2 = nn.utils.spectral_norm(nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias))
            self.conv3 = nn.utils.spectral_norm(nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias))
            self.conv4 = nn.utils.spectral_norm(nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias))
            self.conv5 = nn.utils.spectral_norm(nn.Conv2d(nf + 4 * gc, output, 3, 1, 1, bias=bias))
        else: # 否则，使用普通的卷积层
            self.conv1 = nn.Conv2d(input, 32, 3, 1, 1, bias=bias)
            self.conv2 = nn.Conv2d(input + 32, 32, 3, 1, 1, bias=bias)
            self.conv3 = nn.Conv2d(input + 2 * 32, 32, 3, 1, 1, bias=bias)
            self.conv4 = nn.Conv2d(input + 3 * 32, 32, 3, 1, 1, bias=bias)
            self.conv5 = nn.Conv2d(input + 4 * 32, output, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(inplace=True) # 使用 Leaky ReLU 激活函数
        # initialization
        # mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)
        mutil.initialize_weights([self.conv5], 0.) # 使用自定义的权重初始化方法 mutil.initialize_weights 初始化卷积层的权重

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x)) # 输入 x 通过第一个卷积层 conv1，并应用 Leaky ReLU 激活函数，得到 x1。
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1))) # 输入 x 和 x1 连接在一起（在通道维度上），通过第二个卷积层 conv2，并应用 Leaky ReLU 激活函数，得到 x2
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1))) # 输入 x、x1 和 x2 连接在一起，通过第三个卷积层 conv3，并应用 Leaky ReLU 激活函数，得到 x3
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1))) # 输入 x、x1、x2 和 x3 连接在一起，通过第四个卷积层 conv4，并应用 Leaky ReLU 激活函数，得到 x4
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1)) # 最后，输入 x、x1、x2、x3 和 x4 连接在一起，通过第五个卷积层 conv5，得到最终输出 x5
        return x5
