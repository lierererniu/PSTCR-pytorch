import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.partialconv2d import PartialConv2d
from modules.Attention import AttentionModule
from torchvision import models


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, ResScale, stride=(1, 1)):
        super(Bottleneck, self).__init__()
        self.resscale = ResScale
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=(3, 3), stride=stride,
                               padding=1, bias=False)
        self.relu1 = nn.ReLU()
        # self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=(3, 3), stride=stride,
                               padding=1, bias=False)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.resscale * out
        out += residual
        # out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, b, bands):
        super(ResNet, self).__init__()
        self.times = b
        self.bands = bands
        self.conv1 = nn.Conv2d(in_channels=5, out_channels=self.bands, kernel_size=(3, 3), stride=(1, 1),
                               padding=1)
        self.relu = nn.ReLU()
        self.bottleneck = Bottleneck(self.bands, self.bands, 0.1)
        self.conv2 = nn.Conv2d(in_channels=self.bands, out_channels=1, kernel_size=(3, 3), stride=(1, 1),
                               padding=1)

    def forward(self, input1, input2):
        y1 = torch.cat((input1, input2), dim=1)
        y1 = self.relu(self.conv1(y1))
        for i in range(self.times):
            y1 = self.bottleneck(y1)
        y = self.conv2(y1)
        y = y + input1
        return y

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                # m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()


if __name__ == '__main__':
    model = ResNet(6, 256)
    image1 = torch.randn(16, 1, 40, 40)
    image2 = torch.randn(16, 4, 40, 40)
    mask = torch.randn(16, 1, 40, 40)
    with torch.no_grad():
        output1 = model.forward(image1, image2)
