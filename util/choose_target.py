import os
from PIL import Image
import torchvision.transforms as transforms
from torchvision import models
import torch
import json
import config as c
from args import get_args_parser
args = get_args_parser()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if args.models == 'Resnet50':
    model = models.resnet50(pretrained=True)
if args.models == 'Inception_v3':
    model = models.inception_v3(pretrained=True)
if args.models == 'Densenet121':
    model = models.densenet121(pretrained=True)

with open('./util/imagenet_classes.txt') as f:
    classes = [line.strip() for line in f.readlines()]

def index(i):
    class_idx = json.load(open("./util/imagenet_class_index.json"))
    class2label = [class_idx[str(k)][0] for k in range(len(class_idx))]
    return class2label[i]

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225])
])

def choose_target(number):
    # 设置打开目录和图片目录
    open_dir = args.inputpath
    pic_dir = os.path.join(open_dir, number)
    
    # 获取图片目录下的第一张图片路径
    pic = os.listdir(pic_dir)[0]
    path = os.path.join(pic_dir, pic)
    
    # 打开图片并转换为模型可接受的张量格式
    image = Image.open(path, 'r')
    image_t = transform(image).to(device)
    batch_t = torch.unsqueeze(image_t, 0)

    # 将模型设置为评估模式并传递输入数据到设备（GPU）
    model.eval().to(device)
    
    # 对输入数据进行模型推断
    out = model(batch_t)
    
    # 获取模型输出中的最大值索引和最小值索引
    _, index = torch.max(out, 1)
    _, target = torch.min(out, 1)
    
    # 计算模型输出的 softmax 百分比
    percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
    
    # 打印最大值索引对应的类别及其百分比
    print("recent_label:", [classes[index[0]], percentage[index[0]].item()])
    # 打印最小值索引对应的类别及其百分比
    print("target_label:", [classes[target[0]], percentage[target[0]].item()])

    # 读取 ImageNet 类别索引并获取目标类别的名称
    class_idx = json.load(open("./util/imagenet_class_index.json"))
    class2label = [class_idx[str(k)][0] for k in range(len(class_idx))]
    target = class2label[target[0]]
    print(target)  # 打印目标类别的名称
    return target  # 返回目标类别的名称作为函数的输出











