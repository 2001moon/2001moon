import torch.nn.functional as F
import sys
from util.dataset import trainloader1
from util import viz
import config as c
import modules.Unet_common as common
import warnings
from torchvision import models
import torchvision.transforms as transforms
from util.vgg_loss import VGGLoss
import time
from torch.autograd import Variable
import numpy as np
from torchvision.utils import save_image
from args import get_args_parser
from util.utils import *
from model.model import *
import pandas as pd

warnings.filterwarnings("ignore")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import torch
import torch.nn.functional as F
import random

def random_crop_and_resize(image, crop_size, output_size):
    """
    随机截取并裁剪图像，然后将其resize成指定尺寸
    """
    _, _, h, w = image.size()
    
    # 随机选择截取的起点
    top = random.randint(0, h - crop_size)
    left = random.randint(0, w - crop_size)
    
    # 截取图像
    image = image[:, :, top:top + crop_size, left:left + crop_size]
    
    # resize回原始尺寸
    resize = transforms.Resize((output_size, output_size))
    image = resize(image)
    
    return image


#####################
# Model initialize: #模型初始化
#####################
# 获取参数解析器
args = get_args_parser()
# 初始化可逆模型实例，并将其移动到指定的设备（GPU/CPU）
INN_net = Model().to(device) # 可逆模型的初始化
# 初始化模型参数
init_model(INN_net)
# 使用 DataParallel 包装模型，以支持多 GPU 训练
INN_net = torch.nn.DataParallel(INN_net, device_ids=[0])
# 获取模型的参数数量
para = get_parameter_number(INN_net)
print(para)
# 过滤出模型中所有需要梯度的参数
params_trainable = list(filter(lambda p: p.requires_grad, INN_net.parameters()))
# 定义优化器，这里使用 Adam 优化器，并传入训练参数和一些优化器超参数
optim1 = torch.optim.Adam(params_trainable, lr=c.lr, betas=c.betas, eps=1e-6, weight_decay=c.weight_decay)
# 定义学习率调度器，用于调整学习率，这里使用 StepLR，每经过指定步数后乘以 gamma 值进行调整
weight_scheduler = torch.optim.lr_scheduler.StepLR(optim1, c.weight_step, gamma=c.gamma)
# 获取优化器的初始状态字典
optim_init = optim1.state_dict()
# 实例化离散小波变换 (DWT) 和逆小波变换 (IWT)
dwt = common.DWT() # 实例化 DWT离散小波变换
iwt = common.IWT() # 逆小波变换



# 读取ImageNet类别索引，并设置标准化层，用于对输入图像进行预处理 
class_idx = json.load(open("./util/imagenet_class_index.json")) # 从指定路径读取ImageNet类别索引文件
idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))] # 创建一个列表 idx2label，包含每个类别的标签名称
class2label = [class_idx[str(k)][0] for k in range(len(class_idx))] # 创建一个列表 class2label，包含每个类别的标签编号

norm_layer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # 创建标准化层，用于将输入图像的每个通道的均值和标准差标准化

# 加载预训练模型
if args.models == 'Resnet50':
    model = nn.Sequential(
        norm_layer,
        models.resnet50(pretrained=True)
    ).to(device)
elif args.models == 'Inception_v3':
    model = nn.Sequential(
        norm_layer,
        models.inception_v3(pretrained=True)
    ).to(device)
elif args.models == 'Densenet121':
    model = nn.Sequential(
        norm_layer,
        models.densenet121(pretrained=True)
    ).to(device)
else:
    sys.exit("Please choose Resnet50 or Inception_v3 or Densenet121")
model = model.eval()


try:
    totalTime = time.time() # 记录总时间
    vgg_loss = VGGLoss(3, 1, False) # 初始化VGG感知损失，用于计算图像感知损失。

    vgg_loss.to(device)
    failnum = 0 # 记录失败次数
    count = 0.0 # 总图像计数
    result_matrix = np.zeros(4)  # 暂时保存一轮数据的攻击成功率
    result = None  # 保存攻击成功率初始化结果变量
    for i_batch, mydata in enumerate(trainloader1):
        start_time = time.time()  # 记录每个批次的开始时间
        X_1 = torch.full((1, 3, 224, 224), 0.5).to(device)  # 初始化一个图像张量，值全为0.5
        # image_path = 'DTMI_Local_FeatureSimilarityLoss_perturbed_images/0.CGT.png'  # 替换为我的图像路径
        # X_1 = load_image_as_tensor(image_path, device)
        X_ori = X_1.to(device)  # 将张量移动到设备上（GPU或CPU）
        X_ori = Variable(X_ori, requires_grad=True)  # 将张量转换为变量，允许其梯度计算（作为隐写图像）
        optim2 = torch.optim.Adam([X_ori], lr=c.lr2)  # 用于优化隐写图像的Adam优化器

        if c.pretrain:
            load(args.pre_model, INN_net)  # 如果预训练，加载预训练模型到INN_net
            optim1.load_state_dict(optim_init)  # 加载初始优化器状态

        data = mydata[0].to(device)  # 获取当前批次的图像数据并移动到设备上
        data = data.squeeze(0)  # 移除批次维度，使数据形状匹配预期
        lablist1 = mydata[1]  # 获取图像的第一个标签列表
        lablist2 = mydata[2]  # 获取图像的第二个标签列表

        n1 = int(lablist1)  # 将第一个标签转换为整数 0
        n2 = int(lablist2)  # 将第二个标签转换为整数 625
        i1 = np.array([n1])  # 将第一个标签转换为NumPy数组
        i2 = np.array([n2])  # 将第二个标签转换为NumPy数组
        source_name = index(n1)  # 根据标签获取源类别名称
        target_name = index(n2)  # 根据标签获取目标类别名称

        labels = torch.from_numpy(i2).to(device)  # 将目标标签转换为张量并移动到设备上
        labels = labels.to(torch.int64).to(device)  # 确保标签类型为int64并在设备上

        cover = data.to(device)  # 将原始图像数据移动到设备上（通道数=3）
        cover_dwt_1 = dwt(cover).to(device)  # 对原始图像进行离散小波变换（通道数=12）（三个高频一个低频）
        cover_dwt_low = cover_dwt_1.narrow(1, 0, c.channels_in).to(device)  # 提取低频分量（通道数=3）

        # 检查并创建保存结果图像的目录
        if not os.path.exists(args.outputpath + source_name + "-" + target_name):
            os.makedirs(args.outputpath + source_name + "-" + target_name)  # 创建目录以保存结果图像。

        # 保存原始图像
        save_image(cover, args.outputpath + source_name + "-" + target_name + '/cover.png')

        # 开始训练过程
        for i_epoch in range(c.epochs):
            #################
            ##     训练:    ##
            #################

            CGT = X_ori.to(device)  # 获取当前隐写图像
            CGT_dwt_1 = dwt(CGT).to(device)  # channels = 12 对隐写图像进行离散小波变换
            CGT_dwt_low_1 = CGT_dwt_1.narrow(1, 0, c.channels_in).to(device)  # channels = 3 提取低频分量
            input_dwt_1 = torch.cat((cover_dwt_1, CGT_dwt_1), 1).to(device)  # channels = 12*2 将原始图像和隐写图像的DWT结果拼接作为输入

            output_dwt_1 = INN_net(input_dwt_1).to(device) # 全局

            output_steg_dwt_2 = output_dwt_1.narrow(1, 0, 4 * c.channels_in).to(device)  # channels = 12 从生成的输出中提取隐写图像的DWT结果
            output_step_low_2 = output_steg_dwt_2.narrow(1, 0, c.channels_in).to(device)  # channels = 3 提取隐写图像的低频分量
            output_steg_dwt_low_1 = output_steg_dwt_2.narrow(1, 0, c.channels_in).to(device)  # channels = 3 提取隐写图像的低频分量
            # 从生成的输出中提取还原图像的DWT结果
            output_r_dwt_1 = output_dwt_1.narrow(1, 4 * c.channels_in, 4 * c.channels_in).to(device) # 后12个channel里面的
            # 获取隐写图像
            output_steg_1 = iwt(output_steg_dwt_2).to(device)  # channels = 12 逆小波变换过来
            # 获取还原图像
            output_r = iwt(output_r_dwt_1).to(device) # channels = 12 逆小波变换过来

            # 对隐写图像和还原图像进行逆小波变换，得到最终图像
            output_steg_1 = torch.clamp(output_steg_1, min=0, max=1).to(device)
            # 限制图像像素值在[0,1]范围内
            eta = torch.clamp(output_steg_1 - cover, min=-args.eps, max=args.eps)
            output_steg_1 = torch.clamp(cover + eta, min=0, max=1)
            # save_image(output_steg_1, args.outputpath + source_name + "-" + target_name + '/guocheng.png')
            #################
            #     loss:     #
            #################
            # 计算引导损失
            g_loss = guide_loss(output_steg_1.cuda(), cover.cuda()).to(device) # 最终生成的隐写图像与原图像的损失

            # 使用VGG网络提取原始图像和隐写图像的特征
            vgg_on_cov = vgg_loss(cover).to(device)  # 原始图像的特征
            vgg_on_steg_1 = vgg_loss(output_steg_1).to(device)  # 隐写图像的特征
            # 计算感知损失
            perc_loss = guide_loss(vgg_on_cov, vgg_on_steg_1).to(device)

            # 计算低频损失
            l_loss = guide_loss(output_step_low_2.cuda(), cover_dwt_low.cuda()).to(device)

            # 使用预训练分类模型对隐写图像进行分类，使用一个模型去分类检测生成的对抗损失
            out = model(output_steg_1 * 255.0).to(device) # 1*1000
            _, pre = torch.max(out.data, 1) # 分类模型得到的输出数据张量，在第一维度（即类别维度）上取最大值，返回每个样本的最大值和对应的索引，每个标签的概率，
            percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100  # 对输出数据张量进行Softmax操作，将其转换为概率分布
            _, indices = torch.sort(out, descending=True) # 对输出数据张量进行排序，返回排序后的值和对应的索引。

            # 计算对抗损失，与目标标签的损失
            adv_cost = nn.CrossEntropyLoss().to(device)
            adv_loss = adv_cost(out, labels).to(device)

            # 计算成功率
            suc_rate = ((pre == labels).sum()).cpu().detach().numpy() # 计算预测概率最大的为最终识别出来的类别

            # 计算总损失
            total_loss = (c.lamda_guide * g_loss +
                        c.lamda_low_frequency * l_loss +
                        args.lamda_adv * adv_loss +
                        c.lamda_per * perc_loss)
            # 记录状态
            ii = int(pre)
            state = "img" + str(i_batch) + ":" + str(suc_rate)
            
            #################
            #     Exit:     #保存结果和退出条件
            #################
            if suc_rate == 1:
                # 如果分类成功率为100%
                if (int(percentage[indices[0]][0]) >= 85): # 对应类别概率最大的，并且需要这个预测概率大于85%
                    # 如果分类置信度大于等于85%
                    print(state)
                    # 打印当前状态信息

                    # print([(class2label[idx], percentage[idx].item()) for idx in indices[0][:5]]) # 打印标签以及置信度
                    # 打印前五个分类结果及其置信度
                    save_image(output_steg_1, args.outputpath + source_name + "-" + target_name + '/' + str(i_epoch) + 'result.png')
                    # 保存隐写图像
                    output_r = normal_r(output_r)
                    # 处理还原图像
                    save_image(output_r, args.outputpath + source_name + "-" + target_name + '/r.png')
                    # 保存还原图像
                    count += 1
                    # 计数增加
                    break
                    # 结束当前批次的训练

                if (i_epoch >= 2000):
                    # 如果训练超过2000轮次仍未达到置信度要求
                    # print([(class2label[idx], percentage[idx].item()) for idx in indices[0][:5]])
                    # 打印前五个分类结果及其置信度
                    save_image(output_steg_1, args.outputpath + source_name + "-" + target_name + '/' + str(i_epoch) + "_" + str(int(percentage[indices[0]][0])) + 'd_result.png')
                    # 保存带有置信度的隐写图像
                    output_r = normal_r(output_r)
                    # 处理还原图像
                    save_image(output_r, args.outputpath + source_name + "-" + target_name + '/r.png')
                    # 保存还原图像
                    count += 1
                    # 计数增加
                    break
                    # 结束当前批次的训练

            if (i_epoch >= 5000): # 主要是记录失败的次数
                # 如果训练超过5000轮次仍未达到成功率要求
                failnum += 1
                # 失败计数增加
                print([(class2label[idx], percentage[idx].item()) for idx in indices[0][:5]])
                # 打印前五个分类结果及其置信度
                save_image(output_steg_1, args.outputpath + source_name + "-" + target_name + '/' + str(i_epoch) + 'dw_result.png')
                # 保存隐写图像
                output_r = normal_r(output_r)
                # 处理还原图像
                save_image(output_r, args.outputpath + source_name + "-" + target_name + '/r.png')
                # 保存还原图像
                count += 1
                # 成功计数增加
                break
                # 结束当前批次的训练

            #################
            #   Backward:   #反向传播和优化，计算损失的梯度，并更新模型参数
            #################
            optim1.zero_grad() # 清零梯度
            optim2.zero_grad() # 清零梯度
            total_loss.backward() # 计算total_loss相对于模型参数的梯度，并将这些梯度存储在参数的.grad属性中。这是反向传播的过程。
            optim1.step()  # 更新optim1中所有参数的值，以最小化total_loss。这是通过优化器（如Adam）根据参数的梯度进行更新的过程。
           


            # 随机截取并裁剪局部图像，然后resize成原始尺寸
            crop_size = 24  # 例如，裁剪112x112的局部,或者24*24
            output_size = 224  # 将局部resize回224x224

            local_image = random_crop_and_resize(CGT, crop_size, output_size)

            # 利用vgg提取出来
            vgg_on_global = vgg_loss(CGT).to(device)  # 全局图像特征
            vgg_on_local = vgg_loss(local_image).to(device)  # 局部图像的特征
            # sim_loss = guide_loss(vgg_on_global, vgg_on_local).to(device) # MSE损失计算相似性
            sim_loss = torch.nn.functional.cosine_similarity(vgg_on_global.view(1, -1), vgg_on_local.view(1, -1)) # 相似性损失
            # sim_loss = adv_cost(vgg_on_global, vgg_on_local).to(device) # 交叉熵损失算出来为负数
            accom_inputs = torch.cat([CGT, local_image], dim=0) # 两个1*3*299*299叠加
            C_out = model(accom_inputs)
            loss_label = torch.cat([labels, labels], dim=0)

            # C_out = model(CGT * 255.0).to(device) # 将生成的图像（CGT）放大到0-255范围，并输入分类模型
            # C_adv_loss = adv_cost(C_out, labels).to(device) # 计算对抗损失（C_adv_loss），即模型输出与目标标签（labels）之间的交叉熵损失。

            C_adv_loss = adv_cost(C_out, loss_label).to(device)
            CGT_loss = C_adv_loss - sim_loss # 如果是相似性损失的话，需要相减
            CGT_loss.backward() # 计算C_adv_loss相对于模型参数的梯度
            # C_adv_loss.backward()
            optim2.step() # 更新optim2中所有参数的值，以最小化、

            weight_scheduler.step() # 调用学习率调度器的步骤方法。调度器根据预设的策略调整优化器的学习率，可能是减少学习率以实现更稳定的训练。
            lr_min = c.lr_min # 获取预设的最低学习率
            lr_now = optim1.param_groups[0]['lr'] # 获取当前优化器（optim1）的学习率
            if lr_now < lr_min: # 如果当前学习率小于最低学习率，则将当前学习率设置为最低学习率。这样可以防止学习率过小，导致训练过程停滞。
                optim1.param_groups[0]['lr'] = lr_min


################### 验证CGT到底是不是目标标签
        # CGT_out = model(CGT * 255.0).to(device) # 1*1000
        # _, pre = torch.max(CGT_out.data, 1) # 分类模型得到的输出数据张量，在第一维度（即类别维度）上取最大值，返回每个样本的最大值和对应的索引，每个标签的概率
        # suc_rate = ((pre == labels).sum()).cpu().detach().numpy() # 计算预测概率最大的为最终识别出来的类别
        save_image(CGT , args.outputpath + source_name + "-" + target_name + '/CGT.png') # 保存隐写图像和统计

##########测试代码
        all_models = ['Inception_v3', 'Resnet50', 'Densenet121', 'GoogLeNet', 'VGG16'] # 
        black_models = [i for i in all_models if i != args.models]  # 获取要测试的黑盒模型名字
        eval_models = []  # 初始化评估模型列表
        for model_name in black_models:
            if model_name == 'Densenet121':
                model_eval = nn.Sequential(norm_layer, models.densenet121(pretrained=True)).to(device)
            elif model_name == 'Inception_v3':
                model_eval = nn.Sequential(norm_layer, models.inception_v3(pretrained=True)).to(device)
            elif model_name == 'Resnet50':
                model_eval = nn.Sequential(norm_layer, models.resnet50(pretrained=True)).to(device)
            elif model_name == 'GoogLeNet':
                model_eval = nn.Sequential(norm_layer, models.googlenet(pretrained=True)).to(device)
            elif model_name == 'VGG16':
                model_eval = nn.Sequential(norm_layer, models.vgg16(pretrained=True)).to(device)
            else:
                sys.exit("Please choose Resnet50, Inception_v3, Densenet121, GoogLeNet, VGG16, VisionTransformer, or SwinTransformer")
            model_eval.cuda()  # 将模型移到GPU
            eval_models.append(model_eval)  # 将模型添加到评估模型列表


        for m_id, model_eval in enumerate(eval_models): # 测试模型
            with torch.no_grad():
                out = model_eval(output_steg_1 * 255.0).to(device) # 1*1000
                _, pre = torch.max(out.data, 1) # 分类模型得到的输出数据张量，在第一维度（即类别维度）上取最大值，返回每个样本的最大值和对应的索引，每个标签的概率，
            success_nums = ((pre == labels).sum()).cpu().detach().numpy()
            if success_nums == 1:
                result_matrix[m_id] += 1 # 记录对抗样本的成功率

        df = pd.DataFrame({
            "Model": black_models,
            "Result": result_matrix})
        
    totalstop_time = time.time()
    # df.to_csv('transferable.csv', args.models, index=False)  # 保存到当前工作目录
    time_cost = totalstop_time - totalTime
    Total_suc_rate = (count-failnum)/count
    print("Total cost time :" + str(time_cost))
    print("Total suc rate :" + str(Total_suc_rate))
except:
    raise

finally:
    viz.signal_stop()