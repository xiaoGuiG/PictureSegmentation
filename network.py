import torch
from torch import nn
from torchvision import models

NUM_CLASS = 2

class AlexNet(nn.Module):
    def __init__(self, is_pretrained):
        super(AlexNet, self).__init__()
        self.is_pretrained = is_pretrained

        # 加载预训练权重
        if is_pretrained:
            print('------ Pretrained model------ ')
            model = models.alexnet(pretrained=True)
            self.conv = model.features
        else:
            print('------ No pretrained model------ ')
            self.conv = nn.Sequential(
                nn.Conv2d(3, 96, 11, 4),
                nn.ReLU(),
                nn.MaxPool2d(3, 2),
                nn.Conv2d(96, 256, 5, 1, 2),
                nn.ReLU(),
                nn.MaxPool2d(3, 2),
                nn.Conv2d(256, 384, 3, 1, 1),
                nn.ReLU(),
                nn.Conv2d(384, 384, 3, 1, 1),
                nn.ReLU(),
                nn.Conv2d(384, 256, 3, 1, 1),
                nn.ReLU(),
                nn.MaxPool2d(3, 2)
            )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, NUM_CLASS)

    def forward(self, x, feature_only=False):
        x = self.conv(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)  # 展平

        if feature_only:
            return x  # 返回特征向量
        else:
            x = self.fc(x)
            return x
