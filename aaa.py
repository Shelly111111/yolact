import paddle
import torch
import torch.nn.functional as F

a=torch.Tensor([0.5,0.1,0.3,0.2,0.4])

print(F.relu(a, inplace=False))

'''
tensor([0.5000, 0.1000, 0.3000, 0.2000, 0.4000])
'''