from collections import OrderedDict
from torch import nn
import torch
import torch.nn.functional as F
from math import exp

class MyLoss(nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()

    def forward(self, output, ms, ps):
        local = torch.norm(torch.mul((1 - ms), output) - torch.mul((1 - ms), ps))
        glob = torch.norm(torch.sub(output, ps))
        return 0.15 * local + glob


class Multi_Spatio(nn.Module):
    def __init__(self, inchanels, out_chanel, stride=(1, 1), padding=1, kernel_size=1):
        super(Multi_Spatio, self).__init__()
        self.conv3 = nn.Conv2d(in_channels=inchanels, out_channels=out_chanel, kernel_size=(3, 3), stride=stride,
                               padding=1)
        self.conv5 = nn.Conv2d(in_channels=inchanels, out_channels=out_chanel, kernel_size=(5, 5), stride=stride,
                               padding=2)
        self.conv7 = nn.Conv2d(in_channels=inchanels, out_channels=out_chanel, kernel_size=(7, 7), stride=stride,
                               padding=3)

    def forward(self, x):
        feature3 = self.conv3(x)
        feature5 = self.conv5(x)
        feature7 = self.conv7(x)
        y = torch.cat((feature3, feature5, feature7), dim=1)
        return y


class Spatio_temporal(nn.Module):
    # 120, 60, 1, 1, 3
    def __init__(self, inchanels, out_chanel, stride=(1, 1), padding=1, kernel_size=(3, 3)):
        super(Spatio_temporal, self).__init__()
        self.conv3_1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=(3, 3), stride=stride,
                               padding=1)
        self.conv5_1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=(5, 5), stride=stride,
                               padding=2)
        self.conv7_1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=(7, 7), stride=stride,
                               padding=3)
        self.conv3_2 = nn.Conv2d(in_channels=4, out_channels=20, kernel_size=(3, 3), stride=stride,
                                 padding=1)
        self.conv5_2 = nn.Conv2d(in_channels=4, out_channels=20, kernel_size=(5, 5), stride=stride,
                                 padding=2)
        self.conv7_2 = nn.Conv2d(in_channels=4, out_channels=20, kernel_size=(7, 7), stride=stride,
                                 padding=3)

        self.baseConv = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels=inchanels, out_channels=out_chanel, kernel_size=kernel_size, stride=stride,
                                padding=padding)),
            ('relu1', nn.ReLU()),
            ('conv2', nn.Conv2d(in_channels=out_chanel, out_channels=out_chanel, kernel_size=kernel_size, stride=stride,
                                padding=padding)),
            ('relu2', nn.ReLU()),
            ('conv3', nn.Conv2d(in_channels=out_chanel, out_channels=out_chanel, kernel_size=kernel_size, stride=stride,
                                padding=padding)),
            ('relu3', nn.ReLU()),
            ('conv4', nn.Conv2d(in_channels=out_chanel, out_channels=out_chanel, kernel_size=kernel_size, stride=stride,
                                padding=padding)),
            ('relu4', nn.ReLU()),
            ('conv5', nn.Conv2d(in_channels=out_chanel, out_channels=out_chanel, kernel_size=kernel_size, stride=stride,
                                padding=padding)),
            ('relu5', nn.ReLU()),
            ('conv6', nn.Conv2d(in_channels=out_chanel, out_channels=out_chanel, kernel_size=kernel_size, stride=stride,
                                padding=padding)),
            ('relu6', nn.ReLU()),
            ('conv7', nn.Conv2d(in_channels=out_chanel, out_channels=out_chanel, kernel_size=kernel_size, stride=stride,
                                padding=padding)),
            ('relu7', nn.ReLU()),
            ('conv8', nn.Conv2d(in_channels=out_chanel, out_channels=out_chanel, kernel_size=kernel_size, stride=stride,
                                padding=padding)),
            ('relu8', nn.ReLU()),
            ('conv9', nn.Conv2d(in_channels=out_chanel, out_channels=out_chanel, kernel_size=kernel_size, stride=stride,
                                padding=padding)),
            ('relu9', nn.ReLU())]))

        self.conv10 = nn.Conv2d(in_channels=out_chanel, out_channels=60, kernel_size=kernel_size, stride=stride,
                                padding=padding)
        self.relu10 = nn.ReLU(inplace=False)

        self.conv11 = nn.Conv2d(in_channels=60, out_channels=out_chanel // out_chanel, kernel_size=kernel_size,
                                stride=stride, padding=padding, bias=True)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, input1, input2):
        feature3 = self.conv3_1(input1)
        feature5 = self.conv5_1(input1)
        feature7 = self.conv7_1(input1)
        y1 = torch.cat((feature3, feature5, feature7), dim=1)
        feature3_1 = self.conv3_2(input2)
        feature5_1 = self.conv5_2(input2)
        feature7_1 = self.conv7_2(input2)
        y2 = torch.cat((feature3_1, feature5_1, feature7_1), dim=1)
        y = torch.cat((y1, y2), dim=1)
        x9 = self.baseConv(y)
        x10 = self.relu10(self.conv10(x9))
        output = self.conv11(x10)
        return output + input1

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()


if __name__ == '__main__':
    model = Spatio_temporal(120, 60, (1, 1), 1, (3, 3))
    model.initialize()
    image1 = torch.randn(16, 1, 40, 40)
    image2 = torch.randn(16, 4, 40, 40)

    with torch.no_grad():
        output1 = model.forward(image1, image2)
    print(output1.size())


