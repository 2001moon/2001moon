import torch.nn as nn
import torch
import json
from PIL import Image
import os
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def guide_loss(output, bicubic_image): # 均方误差损失
    loss_fn = torch.nn.MSELoss(reduce=True, size_average=False)
    loss = loss_fn(output, bicubic_image)
    return loss.to(device)

def l1_loss(output, bicubic_image):
    loss_fn = torch.nn.L1Loss(reduce=True, size_average=True)
    loss = loss_fn(output, bicubic_image)
    return loss.to(device)

def get_parameter_number(net):
    # 计算模型中所有参数的总数
    total_num = sum(p.numel() for p in net.parameters())
    # 计算模型中所有可训练参数的总数
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    # 返回一个字典，包含总参数数量和可训练参数数量
    return {'Total': total_num, 'Trainable': trainable_num}


def load(name, net):
    state_dicts = torch.load(name)
    network_state_dict = {k: v for k, v in state_dicts['net'].items() if 'tmp_var' not in k}
    net.load_state_dict(network_state_dict)

class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.register_buffer('mean', torch.Tensor(mean))
        self.register_buffer('std', torch.Tensor(std))

    def forward(self, input):
        # Broadcasting
        input = input / 255.0
        mean = self.mean.reshape(1, 3, 1, 1).to(device)
        std = self.std.reshape(1, 3, 1, 1).to(device)
        return (input - mean) / std

def index(i):
    class_idx = json.load(open("./util/imagenet_class_index.json"))
    class2label = [class_idx[str(k)][0] for k in range(len(class_idx))]
    return class2label[i]

def cindex(name):
    class_idx = json.load(open("./util/imagenet_class_index.json"))
    for i in range(len(class_idx)): 
        if(class_idx[str(i)][0] == name):# 如果找到与给定名称匹配的类别，返回其索引
            return i

def normal_r(output_r):
    r_max = torch.max(output_r)
    r_min = torch.min(output_r)
    r_mean = r_max - r_min
    output_r = (output_r - r_min) / r_mean
    return output_r

def to_rgb(image):
    rgb_image = Image.new("RGB", image.size) # 创建一个新的RGB图像，尺寸与输入图像相同
    rgb_image.paste(image) # 将输入图像粘贴到新的RGB图像中
    return rgb_image # 返回转换后的RGB图像

def imglist(path,mat):

    dirpath = []
    for parent,dirname,filenames in os.walk(path):
        for filename in filenames:
            if(filename.endswith(mat)):
                dirpath.append(os.path.join(parent,filename))

    return dirpath

def l_cal(img1,img2):
    noise = (img1 - img2).flatten(start_dim=0)
    l2 = torch.sum(torch.pow(torch.norm(noise, p=2, dim=0), 2)) # 计算 noise 张量的 L2 范数，即噪声张量中每个元素的平方和的平方根。
    l_inf = torch.sum(torch.norm(noise, p=float('inf'), dim=0)) # 对所有平方值进行求和，得到总的 L2 误差
    return l2,l_inf
