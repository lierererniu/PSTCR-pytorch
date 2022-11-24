import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.partialconv2d import PartialConv2d
from modules.Attention import AttentionModule
from torchvision import models


class VGG16FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        vgg16 = models.vgg16(pretrained=True)
        self.enc_1 = nn.Sequential(*vgg16.features[:5])
        self.enc_2 = nn.Sequential(*vgg16.features[5:10])
        self.enc_3 = nn.Sequential(*vgg16.features[10:17])

        # fix the encoder
        for i in range(3):
            for param in getattr(self, 'enc_{:d}'.format(i + 1)).parameters():
                param.requires_grad = False

    def forward(self, image):
        results = [image]
        for i in range(3):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


class RFRModule(nn.Module):
    def __init__(self, layer_size=6, in_channel=64):
        super(RFRModule, self).__init__()
        self.freeze_enc_bn = False
        self.layer_size = layer_size
        for i in range(3):
            name = 'enc_{:d}'.format(i + 1)
            out_channel = in_channel * 2
            block = [nn.Conv2d(in_channel, out_channel, 3, 2, 1, bias=False),
                     nn.BatchNorm2d(out_channel),
                     nn.ReLU(inplace=True)]
            in_channel = out_channel
            setattr(self, name, nn.Sequential(*block))

        for i in range(3, 6):
            name = 'enc_{:d}'.format(i + 1)
            block = [nn.Conv2d(in_channel, out_channel, 3, 1, 2, dilation=2, bias=False),
                     nn.BatchNorm2d(out_channel),
                     nn.ReLU(inplace=True)]
            setattr(self, name, nn.Sequential(*block))
        self.att = AttentionModule(512)
        for i in range(5, 3, -1):
            name = 'dec_{:d}'.format(i)
            block = [nn.Conv2d(in_channel + in_channel, in_channel, 3, 1, 2, dilation=2, bias=False),
                     nn.BatchNorm2d(in_channel),
                     nn.LeakyReLU(0.2, inplace=True)]
            setattr(self, name, nn.Sequential(*block))

        block = [nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),
                 nn.BatchNorm2d(512),
                 nn.LeakyReLU(0.2, inplace=True)]
        self.dec_3 = nn.Sequential(*block)

        block = [nn.ConvTranspose2d(768, 256, 4, 2, 1, bias=False),
                 nn.BatchNorm2d(256),
                 nn.LeakyReLU(0.2, inplace=True)]
        self.dec_2 = nn.Sequential(*block)

        block = [nn.ConvTranspose2d(384, 64, 4, 2, 1, bias=False),
                 nn.BatchNorm2d(64),
                 nn.LeakyReLU(0.2, inplace=True)]
        self.dec_1 = nn.Sequential(*block)

    def forward(self, input, mask):

        h_dict = {}  # for the output of enc_N

        h_dict['h_0'] = input

        h_key_prev = 'h_0'
        for i in range(1, self.layer_size + 1):
            l_key = 'enc_{:d}'.format(i)
            h_key = 'h_{:d}'.format(i)
            h_dict[h_key] = getattr(self, l_key)(h_dict[h_key_prev])
            h_key_prev = h_key

        h = h_dict[h_key]
        for i in range(self.layer_size - 1, 0, -1):
            enc_h_key = 'h_{:d}'.format(i)
            dec_l_key = 'dec_{:d}'.format(i)
            h = torch.cat([h, h_dict[enc_h_key]], dim=1)
            h = getattr(self, dec_l_key)(h)
            if i == 3:
                h = self.att(h, mask)
        return h


class RFRNet(nn.Module):
    def __init__(self):
        super(RFRNet, self).__init__()

        self.conv10 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(10, 10), stride=(1, 1),
                                padding=3)
        self.relu = nn.ReLU()
        self.conv8 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(8, 8), stride=(1, 1),
                               padding=2)
        self.conv6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(6, 6), stride=(1, 1),
                               padding=3)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(4, 4), stride=(1, 1),
                               padding=4)
        self.AvgPool1 = nn.AdaptiveAvgPool2d((3, 3))

        self.conv9 = nn.Conv2d(in_channels=4, out_channels=64, kernel_size=(9, 9), stride=(1, 1),
                               padding=1)
        self.relu = nn.ReLU()
        self.conv7 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(7, 7), stride=(1, 1),
                               padding=2)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 5), stride=(1, 1),
                               padding=3)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1),
                               padding=4)

        self.AvgPool2 = nn.AdaptiveAvgPool2d((2, 2))
        self.SharedMLP1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1),
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1),
                      padding=1),
            nn.ReLU()
        )
        self.SharedMLP2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1),
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(4, 4), stride=(1, 1),
                      padding=1),
            nn.ReLU()
        )
        self.avgpool1 = nn.AvgPool2d(3, stride=1)
        self.avgpool2 = nn.AvgPool2d(2, stride=1)
        self.Pconv1 = PartialConv2d(3, 64, 7, 2, 3, multi_channel=True, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.Pconv2 = PartialConv2d(64, 64, 7, 1, 3, multi_channel=True, bias=False)
        self.bn20 = nn.BatchNorm2d(64)
        self.Pconv21 = PartialConv2d(64, 64, 7, 1, 3, multi_channel=True, bias=False)
        self.Pconv22 = PartialConv2d(64, 64, 7, 1, 3, multi_channel=True, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.RFRModule = RFRModule()
        self.Tconv = nn.ConvTranspose2d(64, 64, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.tail1 = PartialConv2d(67, 32, 3, 1, 1, multi_channel=True, bias=False)
        self.tail2 = Bottleneck(32, 8)
        self.out = nn.Conv2d(64, 1, 3, 1, 1, bias=False)

    def forward(self, input1, input2, mask):
        feature10 = self.relu(self.conv10(input1))
        feature8 = self.relu(self.conv8(feature10))
        feature6 = self.relu(self.conv6(feature8))
        feature4 = self.relu(self.conv4(feature6))
        y1 = self.avgpool1(feature4)

        feature9 = self.relu(self.conv9(input2))
        feature7 = self.relu(self.conv7(feature9))
        feature5 = self.relu(self.conv5(feature7))
        feature3 = self.relu(self.conv3(feature5))
        print(feature3.size())
        y2 = self.avgpool2(feature3)

        print(y1.shape, y2.shape)
        y1 = self.SharedMLP1(y1)
        y2 = self.SharedMLP2(y2)
        y = self.relu(y1) + self.relu(y2)
        print(y.size())
        return y

    #     x1, m1 = self.Pconv1(y, mask)
    #     x1 = F.relu(self.bn1(x1), inplace=True)
    #     x1, m1 = self.Pconv2(x1, m1)
    #     x1 = F.relu(self.bn20(x1), inplace=True)
    #     x2 = x1
    #     x2, m2 = x1, m1
    #     n, c, h, w = x2.size()
    #     c1 = m1.size()[1]
    #     feature_group = [x2.view(n, c, 1, h, w)]
    #     mask_group = [m2.view(n, c1, 1, h, w)]
    #     self.RFRModule.att.att.att_scores_prev = None
    #     self.RFRModule.att.att.masks_prev = None
    #
    #     for i in range(6):
    #         x2, m2 = self.Pconv21(x2, m2)
    #         x2, m2 = self.Pconv22(x2, m2)
    #         x2 = F.leaky_relu(self.bn2(x2), inplace=True)
    #
    #         x2 = self.RFRModule(x2, m2[:, 0:1, :, :])
    #         x2 = x2 * m2
    #         feature_group.append(x2.view(n, c, 1, h, w))
    #         mask_group.append(m2.view(n, c1, 1, h, w))
    #     x3 = torch.cat(feature_group, dim=2)
    #     m3 = torch.cat(mask_group, dim=2)
    #     amp_vec = m3.mean(dim=2)
    #     x3 = (x3 * m3).mean(dim=2) / (amp_vec + 1e-7)
    #     x3 = x3.view(n, c, h, w)
    #     m3 = m3[:, :, -1, :, :]
    #     x4 = self.Tconv(x3)
    #     # torch.Size([16, 64, 256, 256])
    #     x4 = F.leaky_relu(self.bn3(x4), inplace=True)
    #     m4 = F.interpolate(m3, scale_factor=2)
    #     # torch.Size([16, 67, 256, 256])
    #     x5 = torch.cat([y, x4], dim=1)
    #     m5 = torch.cat([mask, m4], dim=1)
    #     x5, _ = self.tail1(x5, m5)
    #     x5 = F.leaky_relu(x5, inplace=True)
    #     x6 = self.tail2(x5)
    #     x6 = torch.cat([x5, x6], dim=1)
    #     output = self.out(x6)
    #     return output, None
    #
    # def train(self, mode=True, finetune=False):
    #     super().train(mode)
    #     if finetune:
    #         for name, module in self.named_modules():
    #             if isinstance(module, nn.BatchNorm2d):
    #                 module.eval()


if __name__ == '__main__':
    model = RFRNet()
    image1 = torch.randn(16, 3, 160, 160)
    image2 = torch.randn(16, 4, 160, 160)
    mask = torch.randn(16, 3, 160, 160)
    with torch.no_grad():
        output1 = model.forward(image1, image2, mask)
