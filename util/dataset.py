from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from natsort import natsorted
from util import choose_target
from args import get_args_parser
from util.utils import *
args = get_args_parser()

class Dataset(Dataset):
    def __init__(self, transforms_=None):
        # 初始化数据集类
        self.transform = transforms_  # 图像变换操作
        self.TRAIN_PATH = args.inputpath  # 训练数据的路径
        self.format_train = 'png'  # 图像格式
        self.files = natsorted(sorted(imglist(self.TRAIN_PATH, self.format_train)))  # 获取并排序所有图像文件路径

    def __getitem__(self, index):
        try:
            # 打开图像文件，并将其转换为RGB模式
            image = Image.open(self.files[index + args.pass_num])
            image = to_rgb(image)
            
            # 应用图像变换操作，并增加一个维度
            item = self.transform(image)
            item = item.unsqueeze(0)
            
            # 提取文件名，获取类别名和对应的类别索引
            filename = self.files[index + args.pass_num].split("/")
            classname = filename[len(filename) - 2]
            classindex = cindex(classname)
            
            # 选择目标类别及其索引
            targetclass = choose_target.choose_target(classname)
            tarindex = cindex(targetclass)
            
            # 返回图像张量、类别索引和目标类别索引
            return item, classindex, tarindex

        except:
            # 处理异常情况，尝试获取下一个索引的图像
            return self.__getitem__(index + 1)

    def __len__(self):
        # 返回数据集中图像文件的数量
        return len(self.files)


transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
])

# Training data loader
trainloader1 = DataLoader(
    Dataset(transforms_=transform),
    batch_size=1, # 一次处理一张
    shuffle=False,
    pin_memory=False,
    num_workers=args.workers,
    drop_last=True
)

# 定义了一个初始函数
def load_image_as_tensor(image_path, device='cuda'):
    # 定义图像转换
    transform = T.Compose([
        # transforms.Resize((224, 224)),  # 调整图像大小为224x224
        T.ToTensor(),  # 将图像转换为张量
    ])

    # 加载图像
    image = Image.open(image_path).convert('RGB')

    # 应用转换
    image_tensor = transform(image)

    # 添加批次维度并将张量移到指定设备
    image_tensor = image_tensor.unsqueeze(0).to(device)  # 形状变为 (1, 3, 224, 224)

    return image_tensor