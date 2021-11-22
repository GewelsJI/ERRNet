import torch.nn as nn
affine_par = True
import torch
from torch.nn import Module, Sequential, Conv2d, ReLU,AdaptiveMaxPool2d, AdaptiveAvgPool2d, Softmax, Parameter
from torch.nn import functional as F


class PAM_Module(Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//4, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//4, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
               out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)
        out = self.gamma*out + x

        return out


class ASPP(nn.Module):
    """just borrowed from: https://github.com/carrierlxk/AGNN/blob/7d9f20a15b5cd33634b0239d13da29ac872d4d2d/deeplab/siamese_model_conf_gnn.py#L98"""
    def __init__(self, dilation_series=[6, 12, 18], padding_series=[6, 12, 18], depth=512):
        super(ASPP, self).__init__()
        self.mean = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Conv2d(2048, depth, 1, 1)
        self.bn_x = nn.BatchNorm2d(depth)
        self.conv2d_0 = nn.Conv2d(2048, depth, kernel_size=1, stride=1)
        self.bn_0 = nn.BatchNorm2d(depth)
        self.conv2d_1 = nn.Conv2d(2048, depth, kernel_size=3, stride=1, padding=padding_series[0],
                                  dilation=dilation_series[0])
        self.bn_1 = nn.BatchNorm2d(depth)
        self.conv2d_2 = nn.Conv2d(2048, depth, kernel_size=3, stride=1, padding=padding_series[1],
                                  dilation=dilation_series[1])
        self.bn_2 = nn.BatchNorm2d(depth)
        self.conv2d_3 = nn.Conv2d(2048, depth, kernel_size=3, stride=1, padding=padding_series[2],
                                  dilation=dilation_series[2])
        self.bn_3 = nn.BatchNorm2d(depth)
        self.relu = nn.ReLU(inplace=True)
        self.bottleneck = nn.Conv2d(depth * 5, 256, kernel_size=3, padding=1)  # 512 1x1Conv
        self.bn = nn.BatchNorm2d(256)
        self.prelu = nn.PReLU()
        self.sa = PAM_Module(256)
        self.conv51 = nn.Sequential(nn.Conv2d(256, 64, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(64, affine=affine_par),
                                    nn.ReLU())
        self.dropout = nn.Dropout2d(p=0.1)
        self.conv_out = nn.Conv2d(64, 1, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_stage_(self, dilation1, padding1):
        Conv = nn.Conv2d(2048, 256, kernel_size=3, stride=1, padding=padding1, dilation=dilation1, bias=True)  # classes
        Bn = nn.BatchNorm2d(256)
        Relu = nn.ReLU(inplace=True)
        return nn.Sequential(Conv, Bn, Relu)

    def forward(self, x):
        # out = self.conv2d_list[0](x)
        # mulBranches = [conv2d_l(x) for conv2d_l in self.conv2d_list]
        size = x.shape[2:]
        image_features = self.mean(x)
        image_features = self.conv(image_features)
        image_features = self.bn_x(image_features)
        image_features = self.relu(image_features)
        image_features = F.upsample(image_features, size=size, mode='bilinear', align_corners=True)
        out_0 = self.conv2d_0(x)
        out_0 = self.bn_0(out_0)
        out_0 = self.relu(out_0)
        out_1 = self.conv2d_1(x)
        out_1 = self.bn_1(out_1)
        out_1 = self.relu(out_1)
        out_2 = self.conv2d_2(x)
        out_2 = self.bn_2(out_2)
        out_2 = self.relu(out_2)
        out_3 = self.conv2d_3(x)
        out_3 = self.bn_3(out_3)
        out_3 = self.relu(out_3)
        out = torch.cat([image_features, out_0, out_1, out_2, out_3], 1)
        out = self.bottleneck(out)
        out = self.bn(out)
        out = self.prelu(out)
        out = self.sa(out)
        out = self.conv51(out)
        out = self.dropout(out)
        out = self.conv_out(out)

        return out


if __name__ == '__main__':
    i = torch.randn(4, 2048, 11, 11).cuda()
    aspp = ASPP().cuda()
    o = aspp(i)
    print(o.shape)