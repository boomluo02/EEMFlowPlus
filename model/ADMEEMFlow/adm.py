import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 -1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            # layers.append(nn.BatchNorm2d(out_channel))
            layers.append(nn.InstanceNorm2d(out_channel))
        if relu:
            layers.append(nn.ReLU(inplace=True))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResBlock, self).__init__()
        self.main = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x):
        return self.main(x) + x


class EBlock(nn.Module):
    def __init__(self, out_channel, num_res=8):
        super(EBlock, self).__init__()

        layers = [ResBlock(out_channel, out_channel) for _ in range(num_res)]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DBlock(nn.Module):
    def __init__(self, channel, num_res=8):
        super(DBlock, self).__init__()

        layers = [ResBlock(channel, channel) for _ in range(num_res)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


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


class SCM(nn.Module):
    def __init__(self, in_channel, out_plane):
        super(SCM, self).__init__()
        self.main = nn.Sequential(
            BasicConv(in_channel, out_plane//4, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 4, out_plane // 2, kernel_size=1, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane // 2, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane-in_channel, kernel_size=1, stride=1, relu=True)
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


class MDC(nn.Module):
    def __init__(self, in_channel, out_channel, num_res=8):
        super(MDC, self).__init__()

        base_channel = 16

        self.Encoder = nn.ModuleList([
            EBlock(base_channel, num_res),
            EBlock(base_channel*2, num_res),
            EBlock(base_channel*4, num_res),
        ])

        self.feat_extract = nn.ModuleList([
            BasicConv(in_channel, base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel, base_channel*2, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*2, base_channel*4, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*4, base_channel*2, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel*2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel, out_channel, kernel_size=3, relu=False, stride=1)
        ])

        self.Decoder = nn.ModuleList([
            DBlock(base_channel * 4, num_res),
            DBlock(base_channel * 2, num_res),
            DBlock(base_channel, num_res)
        ])

        self.Convs = nn.ModuleList([
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1),
        ])

        self.ConvsOut = nn.ModuleList(
            [
                BasicConv(base_channel * 4, out_channel, kernel_size=3, relu=False, stride=1),
                BasicConv(base_channel * 2, out_channel, kernel_size=3, relu=False, stride=1),
            ]
        )

        self.AFFs = nn.ModuleList([
            AFF(base_channel * 7, base_channel*1),
            AFF(base_channel * 7, base_channel*2)
        ])

        self.FAM1 = FAM(base_channel * 4)
        self.SCM1 = SCM(in_channel, base_channel * 4)
        self.FAM2 = FAM(base_channel * 2)
        self.SCM2 = SCM(in_channel, base_channel * 2)
    
    def forward(self, x, with_res=True):
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        x_2 = F.interpolate(x, scale_factor=0.5,recompute_scale_factor=True)
        x_4 = F.interpolate(x_2, scale_factor=0.5,recompute_scale_factor=True)
        z2 = self.SCM2(x_2)
        z4 = self.SCM1(x_4)

        outputs = list()

        x_ = self.feat_extract[0](x)
        res1 = self.Encoder[0](x_)

        z = self.feat_extract[1](res1)
        z = self.FAM2(z, z2)
        res2 = self.Encoder[1](z)

        z = self.feat_extract[2](res2)
        z = self.FAM1(z, z4)
        z = self.Encoder[2](z)

        z12 = F.interpolate(res1, scale_factor=0.5,recompute_scale_factor=True)
        z21 = F.interpolate(res2, scale_factor=2,recompute_scale_factor=True)
        z42 = F.interpolate(z, scale_factor=2,recompute_scale_factor=True)
        z41 = F.interpolate(z42, scale_factor=2,recompute_scale_factor=True)

        res2 = self.AFFs[1](z12, res2, z42)
        res1 = self.AFFs[0](res1, z21, z41)

        z = self.Decoder[0](z)
        z_ = self.ConvsOut[0](z)
        z = self.feat_extract[3](z)
        if(with_res):
            out1 = z_+x_4
        else:
            out1 = z_
        if is_list:
            out1 = torch.split(out1, [batch_dim, batch_dim], dim=0)
        outputs.append(out1)

        z = torch.cat([z, res2], dim=1)
        z = self.Convs[0](z)
        z = self.Decoder[1](z)
        z_ = self.ConvsOut[1](z)
        z = self.feat_extract[4](z)
        if(with_res):
            out2 = z_+x_2
        else:
            out2 = z_
        if is_list:
            out2 = torch.split(out2, [batch_dim, batch_dim], dim=0)
        outputs.append(out2)

        z = torch.cat([z, res1], dim=1)
        z = self.Convs[1](z)
        z = self.Decoder[2](z)
        z = self.feat_extract[5](z)
        if(with_res):
            out3 = z+x
        else:
            out3 = z
        if is_list:
            out3 = torch.split(out3, [batch_dim, batch_dim], dim=0)
        outputs.append(out3)

        return outputs


class MDC_little(nn.Module):
    def __init__(self, in_channel, out_channel, num_res=1):
        super(MDC_little, self).__init__()

        base_channel = 4

        # self.Encoder = nn.ModuleList([
        #     EBlock(base_channel, num_res),
        #     EBlock(base_channel*2, num_res),
        #     EBlock(base_channel*4, num_res),
        # ])

        self.feat_extract = nn.ModuleList([
            BasicConv(in_channel, base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel, base_channel*2, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*2, base_channel*4, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*4, base_channel*2, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel*2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel, out_channel, kernel_size=3, relu=False, stride=1)
        ])

        # self.Decoder = nn.ModuleList([
        #     DBlock(base_channel * 4, num_res),
        #     DBlock(base_channel * 2, num_res),
        #     DBlock(base_channel, num_res)
        # ])

        self.Convs = nn.ModuleList([
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1),
        ])

        self.ConvsOut = nn.ModuleList(
            [
                BasicConv(base_channel * 4, out_channel, kernel_size=3, relu=False, stride=1),
                BasicConv(base_channel * 2, out_channel, kernel_size=3, relu=False, stride=1),
            ]
        )

        self.AFFs = nn.ModuleList([
            AFF(base_channel * 7, base_channel*1),
            AFF(base_channel * 7, base_channel*2)
        ])

        self.FAM1 = FAM(base_channel * 4)
        self.SCM1 = SCM(in_channel, base_channel * 4)
        self.FAM2 = FAM(base_channel * 2)
        self.SCM2 = SCM(in_channel, base_channel * 2)
    
    def forward(self, x, with_res=True):
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        x_2 = F.interpolate(x, scale_factor=0.5,recompute_scale_factor=True)
        x_4 = F.interpolate(x_2, scale_factor=0.5,recompute_scale_factor=True)
        z2 = self.SCM2(x_2)
        z4 = self.SCM1(x_4)

        outputs = list()

        res1 = self.feat_extract[0](x)
        # res1 = self.Encoder[0](x_)

        z = self.feat_extract[1](res1)
        res2 = self.FAM2(z, z2)
        # res2 = self.Encoder[1](z)

        z = self.feat_extract[2](res2)
        z = self.FAM1(z, z4)
        # z = self.Encoder[2](z)

        z12 = F.interpolate(res1, scale_factor=0.5,recompute_scale_factor=True)
        z21 = F.interpolate(res2, scale_factor=2,recompute_scale_factor=True)
        z42 = F.interpolate(z, scale_factor=2,recompute_scale_factor=True)
        z41 = F.interpolate(z42, scale_factor=2,recompute_scale_factor=True)

        res2 = self.AFFs[1](z12, res2, z42)
        res1 = self.AFFs[0](res1, z21, z41)

        # z = self.Decoder[0](z)
        z_ = self.ConvsOut[0](z)
        z = self.feat_extract[3](z)
        if(with_res):
            out1 = z_+x_4
        else:
            out1 = z_
        if is_list:
            out1 = torch.split(out1, [batch_dim, batch_dim], dim=0)
        outputs.append(out1)

        z = torch.cat([z, res2], dim=1)
        z = self.Convs[0](z)
        # z = self.Decoder[1](z)
        z_ = self.ConvsOut[1](z)
        z = self.feat_extract[4](z)
        if(with_res):
            out2 = z_+x_2
        else:
            out2 = z_
        if is_list:
            out2 = torch.split(out2, [batch_dim, batch_dim], dim=0)
        outputs.append(out2)

        z = torch.cat([z, res1], dim=1)
        z = self.Convs[1](z)
        # z = self.Decoder[2](z)
        z = self.feat_extract[5](z)
        if(with_res):
            out3 = z+x
        else:
            out3 = z
        if is_list:
            out3 = torch.split(out3, [batch_dim, batch_dim], dim=0)
        outputs.append(out3)

        return outputs

class MDS(nn.Module):
    def __init__(self, input_channels=15):

        super(MDS, self).__init__()

        self.fc = nn.Sequential(            
            nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(2),
            nn.ReLU(inplace=True))

        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, event, d_event):

        is_list = isinstance(event, tuple) or isinstance(event, list)
        if is_list:
            batch_dim = event[0].shape[0]
            event = torch.cat(event, dim=0)
            d_event = torch.cat(d_event, dim=0)

        feats_U = event + d_event
        feats_Z = self.fc(feats_U)

        attention_map = self.softmax(feats_Z)

        feats_V = attention_map[:,:1,:,:] * event + attention_map[:,1:,:,:] * d_event

        if is_list:
            feats_V = torch.split(feats_V, [batch_dim, batch_dim], dim=0)
        return feats_V



""" Parts of the U-Net model """
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

""" Full assembly of the parts to form the complete network """
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear

        factor = 2 if bilinear else 1

        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256 // factor)

        self.up3 = Up(256, 128 // factor , bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, out_channels)

    def forward(self, x):
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)


        # print(x1.shape)
        # print(x2.shape)
        # print(x3.shape)

        x2_ = self.up3(x3, x2)
        x1_ = self.up4(x2_, x1)
        out = self.outc(x1_)
        if is_list:
            out = torch.split(out, [batch_dim, batch_dim], dim=0)

        return out