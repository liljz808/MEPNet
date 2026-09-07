# coding: utf-8
"""
Custom ResNet101 with spatial attention output (modified from PCRL-MRG).

Usage:
    from resnet import resnet101
    model = resnet101(num_classes=2)
"""

import torch
import torch.nn as nn
from torchvision.models.resnet import Bottleneck, ResNet


class ResNet101Features(ResNet):
    def forward(self, x, att_size=14):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        att_feat = x

        if att_size != att_feat.shape[-1]:
            att_feat = nn.functional.interpolate(
                att_feat, size=(att_size, att_size), mode="bilinear",
                align_corners=False
            )

        fc_feat = self.avgpool(x)
        fc_feat = torch.flatten(fc_feat, 1)

        return fc_feat, att_feat


def resnet101(num_classes=2, **kwargs):
    return ResNet101Features(Bottleneck, [3, 4, 23, 3],
                             num_classes=num_classes, **kwargs)
