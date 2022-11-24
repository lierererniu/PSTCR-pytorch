import time

from modules.BasicBlock import *


class BasicConv_do(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, bias=False, norm=False, relu=True,
                 transpose=False,
                 relu_method=nn.ReLU, groups=1, norm_method=nn.BatchNorm2d):
        super(BasicConv_do, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 - 1
            layers.append(
                nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                DOConv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias,
                         groups=groups))
        if norm:
            layers.append(norm_method(out_channel))
        if relu:
            if relu_method == nn.ReLU:
                layers.append(nn.ReLU(inplace=True))
            elif relu_method == nn.LeakyReLU:
                layers.append(nn.LeakyReLU(inplace=True))
            else:
                layers.append(relu_method())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


# Supervised Attention Module
class SAM(nn.Module):
    def __init__(self, n_feat, kernel_size=3, bias=True):
        super(SAM, self).__init__()
        stride = kernel_size // 2
        self.conv1 = BasicConv(n_feat, n_feat, kernel_size, stride, bias=bias)
        self.conv2 = BasicConv(n_feat, 1, kernel_size, stride, bias=bias)
        self.conv3 = BasicConv(1, n_feat, kernel_size, stride, bias=bias)

    def forward(self, x, x_img):
        x1 = self.conv1(x)
        img = self.conv2(x) + x_img
        x2 = torch.sigmoid(self.conv3(img))
        x1 = x1 * x2
        x1 = x1 + x
        return x1, img

class ResBlock_do(nn.Module):
    def __init__(self, out_channel):
        super(ResBlock_do, self).__init__()
        self.main = nn.Sequential(
            BasicConv_do(out_channel, out_channel, kernel_size=3, stride=1, relu=True),
            BasicConv_do(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x):
        return self.main(x) + x


class ResBlock_do_fft_bench(nn.Module):
    def __init__(self, out_channel, norm='backward', att=False):
        super(ResBlock_do_fft_bench, self).__init__()
        self.main = nn.Sequential(
            BasicConv_do(out_channel, out_channel, kernel_size=3, stride=1, relu=True),
            BasicConv_do(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

        self.main_fft = nn.Sequential(
            BasicConv_do(out_channel * 2, out_channel * 2, kernel_size=1, stride=1, relu=True),
            BasicConv_do(out_channel * 2, out_channel * 2, kernel_size=1, stride=1, relu=False)
        )
        self.att = att
        if self.att:
            c2wh = dict([(64, 56), (128, 28), (256, 14), (512, 7), (32, 128)])
            # self.at = CALayer(out_channel, reduction=8)
            self.att = MultiSpectralAttentionLayer(out_channel, c2wh[out_channel], c2wh[out_channel], reduction=16,
                                                   freq_sel_method='top16')
        self.dim = out_channel
        self.norm = norm

    def forward(self, x):
        _, _, H, W = x.shape
        dim = 1
        y = torch.fft.rfft2(x, norm=self.norm)
        y_imag = y.imag
        y_real = y.real
        y_f = torch.cat([y_real, y_imag], dim=dim)
        y = self.main_fft(y_f)
        y_real, y_imag = torch.chunk(y, 2, dim=dim)
        y = torch.complex(y_real, y_imag)
        y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        if self.att:
            yy = self.main(x)
            ca = self.att(yy)
            out = ca + x + y
        else:
            out = self.main(x) + x + y
        return out




def window_partitions(x, window_size):
    """
    Args:
        x: (B, C, H, W)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, C, window_size, window_size)
    """
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    windows = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, C, window_size, window_size)
    return windows


def window_reverses(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, C, window_size, window_size)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, C, H, W)
    """
    # B = int(windows.shape[0] / (H * W / window_size / window_size))
    # print('B: ', B)
    # print(H // window_size)
    # print(W // window_size)
    C = windows.shape[1]
    # print('C: ', C)
    x = windows.view(-1, H // window_size, W // window_size, C, window_size, window_size)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(-1, C, H, W)
    return x


def window_partitionx(x, window_size):
    _, _, H, W = x.shape
    h, w = window_size * (H // window_size), window_size * (W // window_size)
    x_main = window_partitions(x[:, :, :h, :w], window_size)
    b_main = x_main.shape[0]
    if h == H and w == W:
        return x_main, [b_main]
    if h != H and w != W:
        x_r = window_partitions(x[:, :, :h, -window_size:], window_size)
        b_r = x_r.shape[0] + b_main
        x_d = window_partitions(x[:, :, -window_size:, :w], window_size)
        b_d = x_d.shape[0] + b_r
        x_dd = x[:, :, -window_size:, -window_size:]
        b_dd = x_dd.shape[0] + b_d
        # batch_list = [b_main, b_r, b_d, b_dd]
        return torch.cat([x_main, x_r, x_d, x_dd], dim=0), [b_main, b_r, b_d, b_dd]
    if h == H and w != W:
        x_r = window_partitions(x[:, :, :h, -window_size:], window_size)
        b_r = x_r.shape[0] + b_main
        return torch.cat([x_main, x_r], dim=0), [b_main, b_r]
    if h != H and w == W:
        x_d = window_partitions(x[:, :, -window_size:, :w], window_size)
        b_d = x_d.shape[0] + b_main
        return torch.cat([x_main, x_d], dim=0), [b_main, b_d]

class SCM(nn.Module):
    def __init__(self, out_plane):
        super(SCM, self).__init__()
        self.main = nn.Sequential(
            BasicConv(1, out_plane // 4, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 4, out_plane // 2, kernel_size=1, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane // 2, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane - 1, kernel_size=1, stride=1, relu=True)
        )

        self.conv = BasicConv(out_plane, out_plane, kernel_size=1, stride=1, relu=False)

    def forward(self, x):
        x = torch.cat([x, self.main(x)], dim=1)
        return self.conv(x)


class FAM(nn.Module):
    def __init__(self, channel):
        super(FAM, self).__init__()
        self.merge = BasicConv(channel, channel, kernel_size=3, stride=1, relu=False)

    def forward(self, x1, x2):
        x = x1 * x2
        out = x1 + self.merge(x)
        return out


def window_reversex(windows, window_size, H, W, batch_list):
    h, w = window_size * (H // window_size), window_size * (W // window_size)
    x_main = window_reverses(windows[:batch_list[0], ...], window_size, h, w)
    B, C, _, _ = x_main.shape
    # print('windows: ', windows.shape)
    # print('batch_list: ', batch_list)
    res = torch.zeros([B, C, H, W], device=windows.device)
    res[:, :, :h, :w] = x_main
    if h == H and w == W:
        return res
    if h != H and w != W and len(batch_list) == 4:
        x_dd = window_reverses(windows[batch_list[2]:, ...], window_size, window_size, window_size)
        res[:, :, h:, w:] = x_dd[:, :, h - H:, w - W:]
        x_r = window_reverses(windows[batch_list[0]:batch_list[1], ...], window_size, h, window_size)
        res[:, :, :h, w:] = x_r[:, :, :, w - W:]
        x_d = window_reverses(windows[batch_list[1]:batch_list[2], ...], window_size, window_size, w)
        res[:, :, h:, :w] = x_d[:, :, h - H:, :]
        return res
    if w != W and len(batch_list) == 2:
        x_r = window_reverses(windows[batch_list[0]:batch_list[1], ...], window_size, h, window_size)
        res[:, :, :h, w:] = x_r[:, :, :, w - W:]
    if h != H and len(batch_list) == 2:
        x_d = window_reverses(windows[batch_list[0]:batch_list[1], ...], window_size, window_size, w)
        res[:, :, h:, :w] = x_d[:, :, h - H:, :]
    return res


class ResBlock(nn.Module):
    def __init__(self, out_channel):
        super(ResBlock, self).__init__()
        self.main = nn.Sequential(
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=True, norm=False),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False, norm=False)
        )

    def forward(self, x):
        return self.main(x) + x


class AFF(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(AFF, self).__init__()
        self.conv = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=1, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x1, x2, x4):
        x = torch.cat([x1, x2, x4], dim=1)
        return self.conv(x)


class EBlock(nn.Module):
    def __init__(self, out_channel, num_res=8, ResBlock=ResBlock):
        super(EBlock, self).__init__()

        layers = [ResBlock(out_channel) for _ in range(num_res)]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DBlock(nn.Module):
    def __init__(self, channel, num_res=8, ResBlock=ResBlock):
        super(DBlock, self).__init__()

        layers = [ResBlock(channel) for _ in range(num_res)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class FCB(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(FCB, self).__init__()
        self.conv1 = BasicConv(in_channel, out_channel, kernel_size=1, stride=1, relu=True)
        self.conv2 = BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)

    def forward(self, prev, aft):
        y = torch.cat([prev, aft], dim=1)
        y1 = self.conv1(y)
        y2 = self.conv2(y1)
        return y2


class TSRnet(nn.Module):
    def __init__(self, num_res=1, ResBlock=ResBlock_do_fft_bench):
        super(TSRnet, self).__init__()
        base_channel = 32
        # ResBlock_do_fft_bench
        self.Encoder = nn.ModuleList([
            EBlock(base_channel, num_res, ResBlock=ResBlock),
            EBlock(base_channel * 2, num_res, ResBlock=ResBlock),
            EBlock(base_channel * 4, num_res, ResBlock=ResBlock),
            EBlock(base_channel, num_res, ResBlock=ResBlock),
            EBlock(base_channel * 2, num_res, ResBlock=ResBlock),
            EBlock(base_channel * 4, num_res, ResBlock=ResBlock)
        ])

        self.feat_extract = nn.ModuleList([
            BasicConv(1, base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel, base_channel * 2, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel * 4, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel, 1, kernel_size=3, relu=False, stride=1),
            BasicConv(1, base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel, base_channel * 2, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel * 2, base_channel * 4, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel * 2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel, 1, kernel_size=3, relu=False, stride=1)
        ])

        self.Decoder = nn.ModuleList([
            DBlock(base_channel * 4, num_res, ResBlock=ResBlock),
            DBlock(base_channel * 2, num_res, ResBlock=ResBlock),
            DBlock(base_channel, num_res, ResBlock=ResBlock),
            DBlock(base_channel * 4, num_res, ResBlock=ResBlock),
            DBlock(base_channel * 2, num_res, ResBlock=ResBlock),
            DBlock(base_channel, num_res, ResBlock=ResBlock)
        ])

        self.Convs = nn.ModuleList([
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1)

        ])

        self.FeatCombine = nn.ModuleList([
            FCB(base_channel * 2, base_channel),
            FCB(base_channel * 4, base_channel * 2)
        ])

        self.AFFs = nn.ModuleList([
            AFF(base_channel * 7, base_channel * 1),
            AFF(base_channel * 7, base_channel * 2)
        ])

        self.ConvsOut = nn.ModuleList(
            [
                BasicConv(base_channel * 4, 1, kernel_size=3, relu=False, stride=1),
                BasicConv(base_channel * 2, 1, kernel_size=3, relu=False, stride=1),
            ]
        )

        self.SAM = SAM(base_channel)

        self.FAM1 = FAM(base_channel * 4)
        self.SCM1 = SCM(base_channel * 4)
        self.FAM2 = FAM(base_channel * 2)
        self.SCM2 = SCM(base_channel * 2)


    def forward(self, cloud):
        outputs = list()
        z2 = self.SCM2(cloud[1])
        z4 = self.SCM1(cloud[2])

        x1 = self.feat_extract[0](cloud[0])
        res1 = self.Encoder[0](x1)
        x2 = self.feat_extract[1](res1)
        z = self.FAM2(x2, z2)
        res2 = self.Encoder[1](z)

        x3 = self.feat_extract[2](res2)
        z = self.FAM1(x3, z4)
        z = self.Encoder[2](z)
        aff1 = self.AFFs[1](res1, res2, z)
        aff2 = self.AFFs[0](res1, res2, z)
        z = self.Decoder[0](z)
        z_ = self.ConvsOut[0](z)
        outputs.append(z_ + z4)
        z = self.feat_extract[3](z)  # 128-64
        z = torch.cat([aff1, z], dim=1)
        z = self.Convs[0](z)  # 128 - 64
        z = self.Decoder[1](z)
        z_ = self.ConvsOut[1](z)
        outputs.append(z_ + z2)
        z = self.feat_extract[4](z)  # 64- 32
        z = torch.cat([aff2, z], dim=1)

        z = self.Convs[1](z)
        z = self.Decoder[2](z)
        z = self.feat_extract[5](z)
        outputs.append(z + cloud[0])

        return outputs


        # x = torch.cat([cloud, temp], dim=1)
        # x1 = self.feat_extract[0](x)
        # res1 = self.Encoder[0](x1)  # 32
        #
        # x2 = self.feat_extract[1](res1)  # 下采样 32-64
        # res2 = self.Encoder[1](x2)  # 64
        #
        # x3 = self.feat_extract[2](res2)  # 下采样 64- 128
        # z = self.Encoder[2](x3)
        #
        # z = self.Decoder[0](z)
        # z = self.feat_extract[3](z)  # 上采样 128-64
        #
        # z = torch.cat([z, res2], dim=1)  # 跳跃连接 64->128
        #
        # z = self.Convs[0](z)  # 128 ->64
        #
        # res3 = self.Decoder[1](z)
        #
        # z = self.feat_extract[4](res3)  # 上采样64->32
        #
        # z = torch.cat([z, res1], dim=1)  # 跳跃连接 32->64
        # z = self.Convs[1](z)
        # res4 = self.Decoder[2](z)  # 32
        #
        # sam, restore = self.SAM(res4, cloud)  # 32
        #
        # outputs.append(restore)
        #
        # # 二阶段纹理细化
        # z = self.feat_extract[5](restore)
        # z = torch.cat([z, sam], dim=1)  # 64
        # z = self.Convs[2](z)  # 64-32   1
        #
        # res5 = self.Encoder[3](z)  # 32
        #
        # z_ = self.FeatCombine[0](res1, res4)  # 32  跨阶段特征融合
        # z = torch.cat([res5, z_], dim=1)  # 64
        # z1 = self.Convs[3](z)  # 64-32  1
        # z = self.feat_extract[6](z1)  # 下采样 64
        # res6 = self.Encoder[4](z)
        # z_ = self.FeatCombine[1](res2, res3)  # 64  跨阶段特征融合
        # z = torch.cat([res6, z_], dim=1)  # 128
        # z2 = self.Convs[4](z)  # 128-64  0
        # z = self.feat_extract[7](z2)  # 下采样 128
        #
        # z = self.Encoder[5](z)
        # z = self.Decoder[3](z)
        # z = self.feat_extract[8](z)  # 上采样 128-64
        # z = torch.cat([z, z2], dim=1)
        # z = self.Convs[5](z)  # 128 ->64  0
        # z = self.Decoder[4](z)
        #
        # z = self.feat_extract[9](z)  # 上采样64->32
        #
        # z = torch.cat([z, z1], dim=1)  # 跳跃连接 32->64
        # z = self.Convs[6](z)  # 64 - 32
        # z = self.Decoder[5](z)  # 32
        # resfinal = self.feat_extract[10](z) + restore
        # outputs.append(resfinal)
        # return outputs


if __name__ == '__main__':
    model = TSRnet()
    # import os
    # import pytorch_ssim
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    #
    # # os.environ["TF_CPP_MIN_LOG_LEVEL"]='1' # 这是默认的显示等级，忽略所有信息
    # os.environ["TF_CPP_MIN_LOG_LEVEL"] = '1'  # 忽略 warning 和 Error
    image1 = torch.randn((1, 1, 256, 256))

    image2 = torch.randn((1, 1, 256, 256))
    image3 = torch.randn((1, 1, 256, 256))

    # ssim_loss = pytorch_ssim.SSIM()
    start_time = time.time()
    with torch.no_grad():
        output1 = model([image1, image2, image3])
    print(output1[2].shape)
    print('training time:', time.time() - start_time)
   #  print(ssim_loss(image1, image2).numpy())
