from modeling.backbone import resnet, xception, xception_gn, drn, mobilenet


def build_backbone(backbone, output_stride, BatchNorm):
    if backbone == 'resnet':
        return resnet.ResNet101(output_stride, BatchNorm)
    elif backbone == 'xception':
        return xception.AlignedXception(output_stride, BatchNorm)
    elif backbone == 'drn':
        return drn.drn_d_54(BatchNorm)
    elif backbone == 'mobilenet':
        return mobilenet.MobileNetV2(output_stride, BatchNorm)
    else:
        raise NotImplementedError


def build_backbone_gn(backbone, output_stride, groups):
    if backbone == 'xception':
        return xception_gn.AlignedXception_GN(output_stride, groups)
    else:
        raise NotImplementedError
