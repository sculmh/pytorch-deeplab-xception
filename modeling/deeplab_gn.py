# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.aspp_gn import build_aspp_gn
from modeling.decoder_gn import build_decoder_gn
from modeling.backbone import build_backbone_gn


class DeepLab_GN(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=21,
                 groups=32):
        super(DeepLab_GN, self).__init__()
        if backbone == 'drn':
            output_stride = 8

        self.backbone = build_backbone_gn(backbone, output_stride, groups=groups)
        self.aspp = build_aspp_gn(backbone, output_stride, groups)
        self.decoder = build_decoder_gn(num_classes, backbone, groups)

    def forward(self, input):
        x, low_level_feat = self.backbone(input)
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.GroupNorm):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.GroupNorm):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p


if __name__ == "__main__":
    model = DeepLab_GN(backbone='xception', output_stride=16, groups=32)
    model.eval()
    input = torch.rand(1, 3, 513, 513)
    output = model(input)
    print(output.size())
