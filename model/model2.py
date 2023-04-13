import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import numpy as np
from model.net_utils import lf2sublfs, sub_spatial_flip


class ConvBn(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding='same', dilation=1):
        super(ConvBn, self).__init__()
        if padding == 'same':
            pad = int((kernel_size-1)/2)*dilation
        else:
            pad = padding
        self.bone = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, pad, dilation, bias=False),
            nn.BatchNorm2d(out_planes)
        )

    def forward(self, x):
        x = self.bone(x)
        return x


class ResBlock(nn.Module):
    """
    残差块
    """
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, dilation=1):
        super(ResBlock, self).__init__()
        self.bone = nn.Sequential(
            ConvBn(in_channel, out_channel, kernel_size, stride, dilation=dilation),
            nn.LeakyReLU(inplace=True),
            ConvBn(out_channel, out_channel, kernel_size, 1, dilation=dilation),
        )
        self.shoutcut = nn.Sequential()
        if stride != 1 or in_channel != out_channel:
            self.shoutcut = nn.Sequential(
                ConvBn(in_channel, out_channel, 1, stride, dilation=dilation)
            )
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        out = self.bone(x)
        out = out + self.shoutcut(x)
        out = self.relu(out)
        return out


class Layer(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, dilation=1, blocks=1):
        super(Layer, self).__init__()
        self.first = ResBlock(in_planes, out_planes, kernel_size, stride, dilation)
        self.second = nn.ModuleList([])
        for i in range(blocks-1):
            self.second.append(ResBlock(out_planes, out_planes, kernel_size, 1, dilation))
        self.blocks = blocks

    def forward(self, x):
        x = self.first(x)
        for i in range(self.blocks-1):
            x = self.second[i](x)
        return x


class Pool(nn.Module):
    def __init__(self, kernel_size):
        super(Pool, self).__init__()
        self.bone = nn.Sequential(
            nn.AvgPool2d(kernel_size),
            ConvBn(16, 4, kernel_size=1),
            nn.LeakyReLU()
        )
        self.scale = kernel_size

    def forward(self, x):
        x = self.bone(x)
        x = F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=True)
        return x


class FeatureExtraction(nn.Module):
    # input: batch_size, channels, patch_size, patch_size
    # output: batch_size, 4, patch_size, patch_size
    # 特征提取，对每张图片提取特征
    def __init__(self, in_planes):
        super(FeatureExtraction, self).__init__()
        self.first = nn.Sequential(
            ConvBn(in_planes, 4, 3),
            nn.LeakyReLU(),
            ConvBn(4, 4, 3),
            nn.LeakyReLU()
        )
        self.res_block = nn.Sequential(
            Layer(4, 4, blocks=2),
            Layer(4, 8, blocks=8),
        )
        self.res_block2 = nn.Sequential(
            Layer(8, 16, blocks=2),
            Layer(16, 16, blocks=2, dilation=2)
        )
        self.branch1 = Pool(2)
        self.branch2 = Pool(4)
        self.branch3 = Pool(8)
        self.branch4 = Pool(16)
        self.last = nn.Sequential(
            ConvBn(40, 16, 3),
            nn.LeakyReLU(),
            nn.Conv2d(16, 4, 1, bias=False)
        )

    def forward(self, x):
        x = self.first(x)
        x1 = self.res_block(x)
        x2 = self.res_block2(x1)
        b1 = self.branch1(x2)
        b2 = self.branch2(x2)
        b3 = self.branch3(x2)
        b4 = self.branch4(x2)
        x = torch.cat([x1, x2, b1, b2, b3, b4], dim=1)
        x = self.last(x)
        return x


class ModulateConv2d(nn.Module):
    def __init__(self, kernel_size, stride=1, dilation=1):
        super(ModulateConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation

    def forward(self, x, h, w):

        Unfold = nn.Unfold(kernel_size=self.kernel_size, stride=self.stride, dilation=self.dilation)
        x_unfold = Unfold(x)
        x_unfold_modulated = x_unfold
        Fold = nn.Fold(output_size=(h, w), kernel_size=1, stride=1)
        out = Fold(x_unfold_modulated)  
        out = rearrange(out, 'b (c n) h w -> b c n h w', n=self.kernel_size**2)
        return out


class BuildCost(nn.Module):
    '''input: b c n h w  --> output: b c n d h w'''
    def __init__(self, angRes, mindisp, maxdisp):
        super(BuildCost, self).__init__()
        self.angRes = angRes
        self.mindisp = mindisp
        self.maxdisp = maxdisp
        self.oacc = ModulateConv2d(kernel_size=angRes, stride=1)

    def forward(self, x):
        b, c, n, h, w = x.shape
        x = rearrange(x, 'b c (a1 a2) h w -> (b a1 a2) c h w', a1=self.angRes, a2=self.angRes)
        bdr = (self.angRes // 2) * self.maxdisp
        pad = nn.ZeroPad2d((bdr, bdr, bdr, bdr))
        x_pad = pad(x)
        x_pad = rearrange(x_pad, '(b a1 a2) c h w -> b c (a1 h) (a2 w)', a1=self.angRes, a2=self.angRes)
        h_pad, w_pad = h + 2 * bdr, w + 2 * bdr
        cost = []
        for d in range(self.mindisp, self.maxdisp + 1):
            dila = [h_pad - d, w_pad - d]
            self.oacc.dilation = dila
            crop = (self.angRes // 2) * (d - self.mindisp)
            if d == self.mindisp:
                feat = x_pad
            else:
                feat = x_pad[:, :, crop: -crop, crop: -crop]
            current_cost = self.oacc(feat, h, w)  # b c*n h w
            cost.append(current_cost)
        cost = torch.stack(cost, dim=3)
        return cost
    

class ConvBn3d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1):
        super(ConvBn3d, self).__init__()
        padding = int((kernel_size-1)/2+stride-1)
        self.bone = nn.Sequential(
            nn.Conv3d(in_planes, out_planes, kernel_size, stride, padding, bias=False),
            nn.BatchNorm3d(out_planes)
        )

    def forward(self, x):
        return self.bone(x)


class Basic(nn.Module):
    def __init__(self, in_planes):  # batch_size, c, d, h, w
        super(Basic, self).__init__()
        feature = 2*75
        self.res0 = nn.Sequential(
            ConvBn3d(in_planes, feature, 3, 1),
            nn.LeakyReLU(),
            ConvBn3d(feature, feature, 3, 1),
            nn.LeakyReLU()
        )
        self.res1 = nn.Sequential(
            ConvBn3d(feature, feature, 3, 1),
            nn.LeakyReLU(),
            ConvBn3d(feature, feature, 3, 1)
        )
        self.res2 = nn.Sequential(
            ConvBn3d(feature, feature, 3, 1),
            nn.LeakyReLU(),
            ConvBn3d(feature, feature, 3, 1)
        )
        self.last = nn.Sequential(
            ConvBn3d(feature, feature, 3, 1),
            nn.LeakyReLU(),
            nn.Conv3d(feature, 1, 3, 1, padding=1, bias=False)
        )

    def forward(self, cv):  # batch_size, c, d, h, w
        cost = self.res0(cv)
        res = self.res1(cost)
        cost = cost + res
        res = self.res2(cost)
        cost = cost + res
        cost = self.last(cost)
        return cost   # batch_size, 1, d, h, w


def disparity_regression(pred, device, dis_max):  # batch_size, d, h, w
    b, d, h, w = pred.shape
    disparity = torch.linspace(-dis_max, dis_max, 2*dis_max+1)
    disparity = disparity.to(device)
    disparity = disparity.reshape((1, -1, 1, 1))
    disparity = disparity.repeat(b, 1, h, w)
    disparity = disparity * pred
    disparity = torch.sum(disparity, dim=1)
    return disparity


class LightFeature(nn.Module):
    def __init__(self, in_planes) -> None:
        super(LightFeature, self).__init__()
        self.first = nn.Sequential(
            ConvBn(in_planes, 16, 3),
            nn.LeakyReLU(),
            ResBlock(16, 16),
            ResBlock(16, 16),
            ResBlock(16, 16),
            ResBlock(16, 16),
            ResBlock(16, 16),
            ResBlock(16, 16),
            ConvBn(16, 4, 3),
            nn.LeakyReLU(),
            nn.Conv2d(4, 4, 3, 1, 1, bias=False),
        )

    def forward(self, x):
        x = self.first(x)
        return x


class LightBasic(nn.Module):
    def __init__(self, in_planes):  # batch_size, c, d, h, w
        super(LightBasic, self).__init__()
        feature = 20
        self.res0 = nn.Sequential(
            ConvBn3d(in_planes, feature, 3, 1),
            nn.LeakyReLU(),
            ConvBn3d(feature, feature, 3, 1),
            nn.LeakyReLU()
        )
        self.res1 = nn.Sequential(
            ConvBn3d(feature, feature, 3, 1),
            nn.LeakyReLU(),
            ConvBn3d(feature, feature, 3, 1)
        )
        self.last = nn.Sequential(
            ConvBn3d(feature, feature, 3, 1),
            nn.LeakyReLU(),
            nn.Conv3d(feature, 1, 3, 1, padding=1, bias=False)
        )

    def forward(self, cv):  # batch_size, c, d, h, w
        cost = self.res0(cv)
        res = self.res1(cost)
        cost = cost + res
        cost = self.last(cost)
        return cost   # batch_size, 1, d, h, w


class CenterAttention(nn.Module):
    def __init__(self, device):
        super(CenterAttention, self).__init__()
        self.unfold1 = nn.Unfold(7, dilation=1, padding=3)
        self.unfold2 = nn.Unfold(7, dilation=2, padding=6)
        self.unfold3 = nn.Unfold(7, dilation=3, padding=9)
        self.unfold4 = nn.Unfold(7, dilation=4, padding=12)
        self.device = device
        self.bone = nn.Sequential(
            ConvBn3d(49, 98, 1),
            nn.LeakyReLU(),
            ConvBn3d(98, 98, 1),
            nn.LeakyReLU(),
            nn.Conv3d(98, 49, 1, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):  # b c h w
        h = x.shape[2]
        m1 = self.unfold1(x)
        m2 = self.unfold2(x)
        m3 = self.unfold3(x)
        m4 = self.unfold4(x)
        ones = torch.ones_like(m1)
        x = torch.stack((m4, m3, m2, m1, ones, m1, m2, m3, m4), dim=1)
        x = x - x[:, :, :, 512:513]
        x = rearrange(x, 'b d a (h w) -> b a d h w ', h=h)
        x = self.bone(x)
        return x


class Net1(nn.Module):   
    # 使用LFAttNet，没有att
    def __init__(self, device):
        super(Net1, self).__init__()
        self.feature = FeatureExtraction(3)
        self.basic_9 = Basic(64)
        self.device = device
        self.softmax = nn.Softmax(dim=1)
        self.bulidcost = BuildCost(7, -4, 4)
        self.index = [
            0, 1, 2, 3,
            7, 8, 9, 10,
            14, 15, 16, 17,
            21, 22, 23, 24
        ]
        self.conf = nn.Sequential(
            ConvBn(9, 8, 3),
            nn.LeakyReLU(),
            nn.Conv2d(8, 1, 3, padding=1, bias=False)
        )

    def forward(self, x):
        b, s, n, c, h, w = x.shape
        x = rearrange(x, 'b s n c h w -> (b s n) c h w', c=c)
        x = self.feature(x)  # bn, 4, h, w
        x = rearrange(x, '(b s n) c h w -> (b s) c n h w', b=b, n=n)
        x = rearrange(x, 'b c (n1 n2) h w -> b c h w n1 n2', n1=4, n2=4)
        pad = (0, 3, 0, 3)
        x = F.pad(x, pad, "constant", 0)
        x = rearrange(x, 'b c h w n1 n2 -> b c (n1 n2) h w')
        cost_volume_9 = self.bulidcost(x)  # b c n d h w
        cost_volume_9 = cost_volume_9[:, :, self.index, :, :]
        cost_volume_9 = rearrange(cost_volume_9, 'b c n d h w -> b (c n) d h w')
        cost_volume_9 = self.basic_9(cost_volume_9)  # batch_size, 1, d, h, w
        cost_volume_9 = torch.squeeze(cost_volume_9, dim=1)
        cost_volume_9 = self.softmax(cost_volume_9)  # b d h w
        disparity = disparity_regression(cost_volume_9, self.device, 4)
        conf = self.conf(cost_volume_9)
        disparity = disparity.unsqueeze(dim=1)
        out = [disparity, conf]
        out = [out]

        out_disp_sub = []
        out_conf_sub = []
        out_disp = []

        for i in range(1):
            disp_init_sub, conf_init = out[i]

            disp_init_sub = disp_init_sub.view(b, 4, h, w)
            conf_init = torch.sigmoid(conf_init.view(b, 4, h, w))

            disp_fliped = sub_spatial_flip(disp_init_sub)
            conf_fliped = sub_spatial_flip(conf_init)
            conf_fliped_norm = self.softmax(conf_fliped)

            disp_init = torch.sum(disp_fliped*conf_fliped_norm, dim=1).unsqueeze(1) #[N,1,h,w]

            h = h//2
            w = w//2

            out_disp_sub.append(disp_fliped)
            out_conf_sub.append(conf_fliped_norm)
            out_disp.append(disp_init)
        return out_disp_sub, out_conf_sub, out_disp
    

