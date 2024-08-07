import time
import sys
import warnings
warnings.filterwarnings('ignore')
sys.path.append("/mnt/nas/share2/home/by/hyena-dna")
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from src.models.sequence.denoise import CNNModel
from src.models.sequence.caduceus import Caduceus
from src.models.sequence.caduceus import  Hyena
from omegaconf import DictConfig
import torch
import math


args = DictConfig({
    "clean_data": False,
    "hidden_dim": 256,
    "cls_expanded_simplex": True,
    "mode": "dirichlet",
    "num_cnn_stacks": 1,
    "dropout": 0.0,
    "cls_free_guidance": False,
    "alpha_max": 8,
    "fix_alpha": 1e6,
    "alpha_scale": 2,
    "prior_pseudocount": 2,
    "seed": 2222,
})
dcgcnn = CNNModel(args, 5, 0,classifier=False, for_representation=True, pretrain=False, dilation=4, kernel_size=3, mlp=True, out_dim=2, length=248, use_outlinear=False, forget=True, num_conv1d=5, d_inner=2, final_conv=True, use_comp=True).to("cuda")

caduceus = Caduceus(mode="ph", size=7.7, out_dim=2).to("cuda")

hyena = Hyena(32, 2).to("cuda")

B = 1
L = 100000

import time
import torch
from torch.utils.data import DataLoader

# 定义模型

total_time = 0
itrs = 100
dcgcnn.eval()

def test():
    total_time = 0
    itrs = 100
    dcgcnn.eval()
    input_data = torch.randint(0, 4, (B, L)).cuda()  # 创建一个形状为[B, L]的序列，序列中都是int，从0-3
    for i in range(itrs):

        # 确保所有初始化的CUDA操作都已完成
        torch.cuda.synchronize()

        # 开始计时
        start = time.time()

        # 运行模型
        output = dcgcnn(input_data)

        end = time.time()
        # 确保所有CUDA操作都已完成
        torch.cuda.synchronize()
        
        total_time += end-start
    return total_time
hh = torch.compile(test)
total_time = hh()

# 打印运行时间
print('dcgcnn Inference time:', math.log(total_time/itrs*1000))
del dcgcnn

import time
import torch
from torch.utils.data import DataLoader

# 定义模型

# total_time = 0
# itrs = 100
# caduceus.eval()
# for i in range(itrs):
#     input_data = torch.randint(0, 4, (B, L)).cuda()  # 创建一个形状为[B, L]的序列，序列中都是int，从0-3

#     # 确保所有初始化的CUDA操作都已完成
#     torch.cuda.synchronize()

#     # 开始计时
#     start = time.time()

#     # 运行模型
#     output = caduceus(input_data)

#     # 确保所有CUDA操作都已完成
#     torch.cuda.synchronize()

#     # 结束计时
#     end = time.time()
#     total_time += end-start

# # 打印运行时间
# print('caduceus Inference time:', math.log(total_time/itrs*1000))
# del caduceus, input_data

import time
import torch
from torch.utils.data import DataLoader

# 定义模型

# 创建一个模拟的输入数据
B = 32  # 批量大小
L = 1000  # 序列长度
total_time = 0
itrs = 100
hyena.eval()
for i in range(itrs):
    input_data = torch.randint(0, 4, (B, L)).cuda()  # 创建一个形状为[B, L]的序列，序列中都是int，从0-3

    # 确保所有初始化的CUDA操作都已完成
    torch.cuda.synchronize()

    # 开始计时
    start = time.time()

    # 运行模型
    output = hyena(input_data)

    # 确保所有CUDA操作都已完成
    torch.cuda.synchronize()

    # 结束计时
    end = time.time()
    total_time += end-start

# 打印运行时间
print('hyena Inference time:', math.log(total_time/itrs*1000))
del hyena, input_data

dcgcnn = CNNModel(args, 5, 0,classifier=False, for_representation=True, pretrain=False, dilation=4, kernel_size=9, mlp=True, out_dim=2, length=248, use_outlinear=False, forget=True, num_conv1d=5, d_inner=2, final_conv=True, use_comp=True).to("cuda")

caduceus = Caduceus(mode="ph", size=7.7, out_dim=2).to("cuda")

hyena = Hyena(4500, 2).to("cuda")


# from thop import profile
# import torch

# input_data = torch.randint(0, 4, (B, L)).cuda()
# # 使用profile函数计算FLOPs
# macs, params = profile(dcgcnn, inputs=(input_data, ))

# print('dcgcnnFLOPs: ', macs)
# print('Params: ', params)
# del dcgcnn, input_data

# from thop import profile
# import torch

# input_data = torch.randint(0, 4, (B, L)).cuda()
# # 使用profile函数计算FLOPs
# macs, params = profile(caduceus, inputs=(input_data, ))

# print('caduceusFLOPs: ', macs)
# print('Params: ', params)
# del caduceus, input_data

# from thop import profile
# import torch

# input_data = torch.randint(0, 4, (B, L)).cuda()
# # 使用profile函数计算FLOPs
# macs, params = profile(hyena, inputs=(input_data, ))

# print('hyenaFLOPs: ', macs)
# print('Params: ', params)
# del hyena, input_data



dcgcnn = CNNModel(args, 5, 0,classifier=False, for_representation=True, pretrain=False, dilation=4, kernel_size=9, mlp=True, out_dim=2, length=248, use_outlinear=False, forget=True, num_conv1d=5, d_inner=2, final_conv=True, use_comp=True).to("cuda")

caduceus = Caduceus(mode="ph", size=7.7, out_dim=2).to("cuda")

hyena = Hyena(4500, 2).to("cuda")

# nelement()：统计Tensor的元素个数
#.parameters()：生成器，迭代的返回模型所有可学习的参数，生成Tensor类型的数据
total = sum([param.nelement() for param in dcgcnn.parameters()])
print("dcgcnn Number of parameter: %.2fM" % (total/1e6))
total = sum([param.nelement() for param in caduceus.parameters()])
print("caduceus Number of parameter: %.2fM" % (total/1e6))
total = sum([param.nelement() for param in hyena.parameters()])
print("hyena Number of parameter: %.2fM" % (total/1e6))
