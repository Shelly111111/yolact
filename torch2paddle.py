import sys
sys.path.append('pytorch')

import torch
import paddle
import numpy as np
from pytorch.yolact import Yolact
# 构建输入
input_data = np.random.rand(1, 3, 550, 550).astype("float32")
# 获取PyTorch Module
torch_module = Yolact()
torch_dict = torch.load('pytorch/weights/yolact_base_54_800000.pth')
paddle_dict = {}
for key in torch_dict:
    try:
        weight = torch_dict[key].cpu().numpy().astype("float32")
    except:
        weight = torch_dict[key].cpu().detach().numpy().astype("float32")
    key = key.replace('running_mean','_mean').replace('running_var','_variance')
    paddle_dict[key]=weight

paddle.save(paddle_dict,'paddle/weights/yolact_base_0_1.pdparams')
