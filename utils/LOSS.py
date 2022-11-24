import torch
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


def get_g_loss(self):
    real_B = self.real_B
    fake_B = self.fake_B
    comp_B = self.comp_B
    self.lossNet = VGG16FeatureExtractor()
    if self.lossNet is not None:
        self.lossNet.cuda()
    real_B_feats = self.lossNet(real_B)
    fake_B_feats = self.lossNet(fake_B)
    comp_B_feats = self.lossNet(comp_B)

    tv_loss = self.TV_loss(comp_B * (1 - self.mask))
    style_loss = self.style_loss(real_B_feats, fake_B_feats) + self.style_loss(real_B_feats, comp_B_feats)
    preceptual_loss = self.preceptual_loss(real_B_feats, fake_B_feats) + self.preceptual_loss(real_B_feats,
                                                                                              comp_B_feats)
    valid_loss = self.l1_loss(real_B, fake_B, self.mask)
    hole_loss = self.l1_loss(real_B, fake_B, (1 - self.mask))

    loss_G = (tv_loss * 0.1
              + style_loss * 120
              + preceptual_loss * 0.05
              + valid_loss * 1
              + hole_loss * 6)

    self.l1_loss_val += valid_loss.detach() + hole_loss.detach()
    return loss_G


def l1_loss(self, f1, f2, mask=1):
    return torch.mean(torch.abs(f1 - f2) * mask)


def style_loss(self, A_feats, B_feats):
    assert len(A_feats) == len(B_feats), "the length of two input feature maps lists should be the same"
    loss_value = 0.0
    for i in range(len(A_feats)):
        A_feat = A_feats[i]
        B_feat = B_feats[i]
        _, c, w, h = A_feat.size()
        A_feat = A_feat.view(A_feat.size(0), A_feat.size(1), A_feat.size(2) * A_feat.size(3))
        B_feat = B_feat.view(B_feat.size(0), B_feat.size(1), B_feat.size(2) * B_feat.size(3))
        A_style = torch.matmul(A_feat, A_feat.transpose(2, 1))
        B_style = torch.matmul(B_feat, B_feat.transpose(2, 1))
        loss_value += torch.mean(torch.abs(A_style - B_style) / (c * w * h))
    return loss_value


def TV_loss(self, x):
    h_x = x.size(2)
    w_x = x.size(3)
    h_tv = torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :h_x - 1, :]))
    w_tv = torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :w_x - 1]))
    return h_tv + w_tv


def preceptual_loss(self, A_feats, B_feats):
    assert len(A_feats) == len(B_feats), "the length of two input feature maps lists should be the same"
    loss_value = 0.0
    for i in range(len(A_feats)):
        A_feat = A_feats[i]
        B_feat = B_feats[i]
        loss_value += torch.mean(torch.abs(A_feat - B_feat))
    return loss_value
