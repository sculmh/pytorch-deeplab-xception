import math
import torch.nn as nn
import torch.nn.functional as F


def fixed_padding(inputs, kernel_size, dilation):
    kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded_inputs = F.pad(inputs, (pad_beg, pad_end, pad_beg, pad_end))
    return padded_inputs


class SeparableConv2d(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False, groups=32):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size, stride, 0, dilation,
                               groups=inplanes, bias=bias)
        self.gn = nn.GroupNorm(groups, inplanes, affine=True)
        self.pointwise = nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = fixed_padding(x, self.conv1.kernel_size[0], dilation=self.conv1.dilation[0])
        x = self.conv1(x)
        x = self.gn(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self, inplanes, planes, reps, stride=1, dilation=1, groups=32,
                 start_with_relu=True, grow_first=True, is_last=False):
        super(Block, self).__init__()

        if planes != inplanes or stride != 1:
            self.skip = nn.Conv2d(inplanes, planes, 1, stride=stride, bias=False)
            self.skipgn = nn.GroupNorm(groups, planes, affine=True)
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=True)
        rep = []

        filters = inplanes
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(inplanes, planes, 3, 1, dilation, groups=groups))
            rep.append(nn.GroupNorm(groups, planes, affine=True))
            filters = planes

        for i in range(reps - 1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters, filters, 3, 1, dilation, groups=groups))
            rep.append(nn.GroupNorm(groups, filters, affine=True))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(inplanes, planes, 3, 1, dilation, groups=groups))
            rep.append(nn.GroupNorm(groups, planes, affine=True))

        if stride != 1:
            rep.append(self.relu)
            rep.append(SeparableConv2d(planes, planes, 3, 2, groups=groups))
            rep.append(nn.GroupNorm(groups, planes, affine=True))

        if stride == 1 and is_last:
            rep.append(self.relu)
            rep.append(SeparableConv2d(planes, planes, 3, 1, groups=groups))
            rep.append(nn.GroupNorm(groups, planes, affine=True))

        if not start_with_relu:
            rep = rep[1:]

        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipgn(skip)
        else:
            skip = inp

        x = x + skip

        return x


class AlignedXception_GN(nn.Module):
    """
    Modified Alighed Xception
    """
    def __init__(self, output_stride, groups):
        super(AlignedXception_GN, self).__init__()

        if output_stride == 16:
            entry_block3_stride = 2
            middle_block_dilation = 1
            exit_block_dilations = (1, 2)
        elif output_stride == 8:
            entry_block3_stride = 1
            middle_block_dilation = 2
            exit_block_dilations = (2, 4)
        else:
            raise NotImplementedError

        # Entry flow
        self.conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(groups, 32, affine=True)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(groups, 64, affine=True)

        self.block1 = Block(64, 128, reps=2, stride=2, groups=groups, start_with_relu=False)
        self.block2 = Block(128, 256, reps=2, stride=2, groups=groups, start_with_relu=False,
                            grow_first=True)
        self.block3 = Block(256, 736, reps=2, stride=entry_block3_stride, groups=groups,
                            start_with_relu=True, grow_first=True, is_last=True)

        # Middle flow
        self.block4  = Block(736, 736, reps=3, stride=1, dilation=middle_block_dilation,
                             groups=groups, start_with_relu=True, grow_first=True)
        self.block5  = Block(736, 736, reps=3, stride=1, dilation=middle_block_dilation,
                             groups=groups, start_with_relu=True, grow_first=True)
        self.block6  = Block(736, 736, reps=3, stride=1, dilation=middle_block_dilation,
                             groups=groups, start_with_relu=True, grow_first=True)
        self.block7  = Block(736, 736, reps=3, stride=1, dilation=middle_block_dilation,
                             groups=groups, start_with_relu=True, grow_first=True)
        self.block8  = Block(736, 736, reps=3, stride=1, dilation=middle_block_dilation,
                             groups=groups, start_with_relu=True, grow_first=True)
        self.block9  = Block(736, 736, reps=3, stride=1, dilation=middle_block_dilation,
                             groups=groups, start_with_relu=True, grow_first=True)
        self.block10 = Block(736, 736, reps=3, stride=1, dilation=middle_block_dilation,
                             groups=groups, start_with_relu=True, grow_first=True)
        self.block11 = Block(736, 736, reps=3, stride=1, dilation=middle_block_dilation,
                             groups=groups, start_with_relu=True, grow_first=True)
        self.block12 = Block(736, 736, reps=3, stride=1, dilation=middle_block_dilation,
                             groups=groups, start_with_relu=True, grow_first=True)
        self.block13 = Block(736, 736, reps=3, stride=1, dilation=middle_block_dilation,
                             groups=groups, start_with_relu=True, grow_first=True)
        self.block14 = Block(736, 736, reps=3, stride=1, dilation=middle_block_dilation,
                             groups=groups, start_with_relu=True, grow_first=True)
        self.block15 = Block(736, 736, reps=3, stride=1, dilation=middle_block_dilation,
                             groups=groups, start_with_relu=True, grow_first=True)
        self.block16 = Block(736, 736, reps=3, stride=1, dilation=middle_block_dilation,
                             groups=groups, start_with_relu=True, grow_first=True)
        self.block17 = Block(736, 736, reps=3, stride=1, dilation=middle_block_dilation,
                             groups=groups, start_with_relu=True, grow_first=True)
        self.block18 = Block(736, 736, reps=3, stride=1, dilation=middle_block_dilation,
                             groups=groups, start_with_relu=True, grow_first=True)
        self.block19 = Block(736, 736, reps=3, stride=1, dilation=middle_block_dilation,
                             groups=groups, start_with_relu=True, grow_first=True)

        # Exit flow
        self.block20 = Block(736, 1024, reps=2, stride=1, dilation=exit_block_dilations[0],
                             groups=groups, start_with_relu=True, grow_first=False, is_last=True)

        self.conv3 = SeparableConv2d(1024, 1536, 3, stride=1, dilation=exit_block_dilations[1], groups=groups)
        self.gn3 = nn.GroupNorm(groups, 1536, affine=True)

        self.conv4 = SeparableConv2d(1536, 1536, 3, stride=1, dilation=exit_block_dilations[1], groups=groups)
        self.gn4 = nn.GroupNorm(groups, 1536, affine=True)

        self.conv5 = SeparableConv2d(1536, 2048, 3, stride=1, dilation=exit_block_dilations[1], groups=groups)
        self.gn5 = nn.GroupNorm(groups, 2048, affine=True)

        # Init weights
        self._init_weight()

    def forward(self, x):
        # Entry flow
        x = self.conv1(x)
        x = self.gn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.gn2(x)
        x = self.relu(x)

        x = self.block1(x)
        # add relu here
        x = self.relu(x)
        low_level_feat = x
        x = self.block2(x)
        x = self.block3(x)

        # Middle flow
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.block13(x)
        x = self.block14(x)
        x = self.block15(x)
        x = self.block16(x)
        x = self.block17(x)
        x = self.block18(x)
        x = self.block19(x)

        # Exit flow
        x = self.block20(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.gn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.gn4(x)
        x = self.relu(x)

        x = self.conv5(x)
        x = self.gn5(x)
        x = self.relu(x)

        return x, low_level_feat

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                nn.init.kaiming_normal_(m.weight)


if __name__ == "__main__":
    import torch
    model = AlignedXception_GN(groups=32, output_stride=16)
    input = torch.rand(1, 3, 512, 512)
    output, low_level_feat = model(input)
    print(output.size())
    print(low_level_feat.size())
