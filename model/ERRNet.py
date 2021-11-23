import torch
import torch.nn as nn
import torch.nn.functional as F
from model.resnet import resnet50
from model.aspp import ASPP


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class ReverseRecalibrationUnit(nn.Module):
    def __init__(self, feature_channel=1024, intern_channel=32):
        super(ReverseRecalibrationUnit, self).__init__()
        self.feature_channel = feature_channel

        self.ra_conv1 = BasicConv2d(feature_channel+64, intern_channel, kernel_size=3, padding=1)
        self.ra_conv2 = BasicConv2d(intern_channel, intern_channel, kernel_size=3, padding=1)
        self.ra_conv3 = BasicConv2d(intern_channel, intern_channel, kernel_size=3, padding=1)
        self.ra_out = nn.Conv2d(intern_channel, 1, kernel_size=3, padding=1)

    def forward(self, features, cam_guidance, edge_guidance):
        crop_sal = F.interpolate(cam_guidance, size=features.size()[2:], mode='bilinear', align_corners=True)
        crop_edge = F.interpolate(edge_guidance, size=features.size()[2:], mode='bilinear', align_corners=True)

        x_sal = -1 * (torch.sigmoid(crop_sal)) + 1
        x_sal = x_sal.expand(-1, self.feature_channel, -1, -1).mul(features)

        x = self.ra_conv1(torch.cat((x_sal, crop_edge), dim=1))
        x = F.relu(self.ra_conv2(x))
        x = F.relu(self.ra_conv3(x))
        x = self.ra_out(x)

        x = x + crop_sal

        return x


class SEA(nn.Module):
    def __init__(self):
        super(SEA, self).__init__()
        self.conv1h = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv2h = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv3h = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv4h = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv1v = BasicConv2d(256, 64, kernel_size=3, stride=1, padding=1)
        self.conv2v = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv3v = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv4v = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv5 = BasicConv2d(64+64, 64, kernel_size=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(64, 1, 1)

    def forward(self, left, down):
        if down.size()[2:] != left.size()[2:]:
            down = F.interpolate(down, size=left.size()[2:], mode='bilinear')
        out1h = F.relu(self.conv1h(left), inplace=True)
        out2h = F.relu(self.conv2h(out1h), inplace=True)
        out1v = F.relu(self.conv1v(down), inplace=True)
        out2v = F.relu(self.conv2v(out1v), inplace=True)
        fuse = out2h*out2v
        out3h = F.relu(self.conv3h(fuse), inplace=True)+out1h
        out4h = F.relu(self.conv4h(out3h), inplace=True)
        out3v = F.relu(self.conv3v(fuse), inplace=True)+out1v
        out4v = F.relu(self.conv4v(out3v), inplace=True)

        edge_feature = self.conv5(torch.cat((out4h, out4v), dim=1))
        edge_out = self.conv_out(edge_feature)

        return edge_feature, edge_out


class ERRNet(nn.Module):
    def __init__(self, channel=32):
        super(ERRNet, self).__init__()
        
        self.resnet = resnet50(pretrained=True)
        self.ASPP_Global = ASPP()
        # -- edge global --
        self.sea = SEA()

        # reverse attention
        self.rru_4 = ReverseRecalibrationUnit(feature_channel=2048, intern_channel=256)
        self.rru_3 = ReverseRecalibrationUnit(feature_channel=1024, intern_channel=64)
        self.rru_2 = ReverseRecalibrationUnit(feature_channel=512, intern_channel=64)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x0 = self.resnet.maxpool(x)      # bs, 64, 88, 88
        # ---- low-level features ----
        x1 = self.resnet.layer1(x0)      # bs, 256, 88, 88
        x2 = self.resnet.layer2(x1)     # bs, 512, 44, 44

        x3 = self.resnet.layer3(x2)     # bs, 1024, 22, 22
        x4 = self.resnet.layer4(x3)     # bs, 2048, 11, 11

        s_g = self.ASPP_Global(x4)
        cam_out_g = F.interpolate(s_g, scale_factor=32, mode='bilinear', align_corners=True)    # Sup-1 (44 -> 352)

        # global edge guidance
        e_g, e_g_out = self.sea(x0, x1)
        e_g_out = F.interpolate(e_g_out, scale_factor=4, mode='bilinear', align_corners=True)

        # reverse attention 4
        s_4 = self.rru_4(x4,
                        s_g,
                        e_g)
        cam_out_4 = F.interpolate(s_4, scale_factor=32, mode='bilinear', align_corners=True)  # Sup-2 (11 -> 352)

        # reverse attention 3
        s_3 = self.rru_3(x3,
                                 s_4 + s_g,
                                 e_g)
        cam_out_3 = F.interpolate(s_3, scale_factor=16, mode='bilinear', align_corners=True)  # Sup-3 (22 -> 352)

        # reverse attention 2
        s_2 = self.rru_2(x2,
                                 s_3 + F.interpolate(s_g, scale_factor=2, mode='bilinear', align_corners=True),
                                 e_g)
        cam_out_2 = F.interpolate(s_2, scale_factor=8, mode='bilinear', align_corners=True)   # Sup-3 (44 -> 352)

        return cam_out_g, cam_out_4, cam_out_3, cam_out_2, e_g_out
