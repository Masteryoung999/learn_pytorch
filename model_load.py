import torch
import torchvision
from mode_save import *
from torch import nn

# 方式1->保存方式1，加载模型
# model = torch.load("vgg16_method1.pth")
# print(model)

# 方式2->加载模型
vgg16 = torchvision.models.vgg16(weights=None)
vgg16.load_state_dict(torch.load("vgg16_method2.pth"))
# model = torch.load("vgg16_method2.pth")
# print(vgg16)


# 陷阱1
# class Tudui(nn.Module):
#     def __init__(self):
#         super(Tudui, self).__init__()
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=3)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         return x
#

model = torch.load('tudui_method.pth')
print(model)
