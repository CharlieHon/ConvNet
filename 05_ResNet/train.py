import torch
import torch.nn as nn
from ResNet import resnet34, resnet101
import torchvision.models.resnet

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

net = resnet34().to(device)
model_weight_path = './resnet34-pre.pth'

# 载入模型方式1
missing_keys, unexpected_keys = net.load_state_dict(torch.load(model_weight_path), strict=False)
# 调整模型分类器的分类个数
in_channel = net.fc.in_features
net.fc = nn.Linear(in_channel, 5)

"""
载入模型方式2
1. 先加载模型到内存 torch.load(model_weight_path)，得到的是一个字段
2. 将全连接层删掉
3. net.load_state_dict加载参数
这样可以在一开始定义模型时，就将模型类别个数设置为5，resnet34(num_classes=5)
"""
