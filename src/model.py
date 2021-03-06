import torch
import torch.nn as nn

from efficientnet import EfficientNet
from inceptionv4 import inceptionv4
from inceptionresnetv2 import inceptionresnetv2
from senet import se_resnext50_32x4d
from mish import Mish


class AdaptiveConcatPool2d(nn.Module):
    def __init__(self, output_size=None):
        super().__init__()
        self.output_size = output_size
        self.ap = nn.AdaptiveAvgPool2d(self.output_size)
        self.mp = nn.AdaptiveMaxPool2d(self.output_size)

    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], 1)


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class MelanomaNet(nn.Module):
    def __init__(self, arch, pretrained=True):
        super().__init__()

        # load EfficientNet
        if arch == 'se_resnext50_32x4d':
            if pretrained:
                self.base = se_resnext50_32x4d()
            else:
                self.base = se_resnext50_32x4d(pretrained=None)
            self.nc = self.base.last_linear.in_features
        elif arch == 'inceptionv4':
            if pretrained:
                self.base = inceptionv4()
            else:
                self.base = inceptionv4(pretrained=None)
            self.nc = self.base.last_linear.in_features
        elif arch == 'inceptionresnetv2':
            if pretrained:
                self.base = inceptionresnetv2()
            else:
                self.base = inceptionresnetv2(pretrained=None)
            self.nc = self.base.last_linear.in_features
        elif 'efficientnet' in arch:
            if pretrained:
                self.base = EfficientNet.from_pretrained(model_name=arch)
            else:
                self.base = EfficientNet.from_name(model_name=arch)

            self.nc = self.base._fc.in_features

        self.logit = nn.Sequential(AdaptiveConcatPool2d(1),
                                   Flatten(),
                                   nn.BatchNorm1d(2 * self.nc),
                                   nn.Dropout(0.5),
                                   nn.Linear(2 * self.nc, 512),
                                   Mish(),
                                   nn.BatchNorm1d(512),
                                   nn.Dropout(0.5),
                                   nn.Linear(512, 1))

    def forward(self, x):
        x = self.base.features(x)
        x = self.logit(x)

        return x
