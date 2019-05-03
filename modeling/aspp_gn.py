import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class _ASPPModule_GN(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, groups):
        super(_ASPPModule_GN, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=dilation, bias=False)
        self.gn = nn.GroupNorm(groups, planes, affine=True)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.gn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)


class ASPP_GN(nn.Module):
    def __init__(self, backbone, output_stride, groups):
        super(ASPP_GN, self).__init__()
        if backbone == 'drn':
            inplanes = 512
        elif backbone == 'mobilenet':
            inplanes = 320
        else:
            inplanes = 2048
        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = _ASPPModule_GN(inplanes, 256, 1, padding=0, dilation=dilations[0], groups=groups)
        self.aspp2 = _ASPPModule_GN(inplanes, 256, 3, padding=dilations[1], dilation=dilations[1], groups=groups)
        self.aspp3 = _ASPPModule_GN(inplanes, 256, 3, padding=dilations[2], dilation=dilations[2], groups=groups)
        self.aspp4 = _ASPPModule_GN(inplanes, 256, 3, padding=dilations[3], dilation=dilations[3], groups=groups)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(inplanes, 256, 1, stride=1, bias=False),
                                             nn.GroupNorm(groups, 256, affine=True),
                                             nn.ReLU())
        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.gn1 = nn.GroupNorm(groups, 256, affine=True)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.gn1(x)
        x = self.relu(x)

        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)


def build_aspp_gn(backbone, output_stride, groups):
    return ASPP_GN(backbone, output_stride, groups)
