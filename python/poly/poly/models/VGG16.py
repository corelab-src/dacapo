import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
from torchvision import datasets, transforms

import signal
import sys
from torch.autograd import Variable

__all__ = ['VGG16','vgg16']

from pathlib import Path

use_cuda = torch.cuda.is_available()
class BasicConv2d(nn.Module):
    def __init__(self, inCH, outCh, ksize, padding=0, stride=1):
        super(BasicConv2d, self).__init__()
        self.Conv2d = nn.Conv2d(kernel_size=ksize, in_channels=inCH, 
                    out_channels=outCh, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(outCh)
        self.mish = nn.SiLU() 
    def forward(self, x):
        x = self.Conv2d(x)
        x = self.bn(x)
        x = self.mish(x)
        return x


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()

        self.conv_1_1 = BasicConv2d(3, 64, 3, padding=(1, 1))
        self.conv_1_2 = BasicConv2d(64, 64, 3, padding=(1, 1))
        self.avgpool_1 = nn.AvgPool2d((2, 2), stride=2)

        self.conv_2_1 = BasicConv2d(64, 128, 3, padding=(1, 1))
        self.conv_2_2 = BasicConv2d(128, 128, 3, padding=(1, 1))
        self.avgpool_2 = nn.AvgPool2d((2, 2), stride=2)

        self.conv_3_1 = BasicConv2d(128, 256, 3, padding=(1, 1))
        self.conv_3_2 = BasicConv2d(256, 256, 3, padding=(1, 1))
        self.conv_3_3 = BasicConv2d(256, 256, 3, padding=(1, 1))
        self.avgpool_3 = nn.AvgPool2d((2, 2), stride=2)

        self.conv_4_1 = BasicConv2d(256, 512, 3, padding=(1, 1))
        self.conv_4_2 = BasicConv2d(512, 512, 3, padding=(1, 1))
        self.conv_4_3 = BasicConv2d(512, 512, 3, padding=(1, 1))
        self.avgpool_4 = nn.AvgPool2d((2, 2), stride=2)

        self.conv_5_1 = BasicConv2d(512, 512, 3, padding=(1, 1))
        self.conv_5_2 = BasicConv2d(512, 512, 3, padding=(1, 1))
        self.conv_5_3 = BasicConv2d(512, 512, 3, padding=(1, 1))
        self.avgpool_5 = nn.AvgPool2d((2, 2), stride=2)

        self.fc_1 = nn.Linear(512*1*1, 256)
        self.dp_1 = nn.Dropout()
        self.fc_2 = nn.Linear(256, 128)
        self.bn_1 = nn.BatchNorm1d(128)
        self.fc_3 = nn.Linear(128, 10)
        self.mish = nn.SiLU()

        if use_cuda :
            self.conv_1_1 = self.conv_1_1.cuda()
            self.conv_1_2 = self.conv_1_2.cuda()
            self.avgpool_1 = self.avgpool_1.cuda()
            self.conv_2_1 = self.conv_2_1.cuda()
            self.conv_2_2 = self.conv_2_2.cuda()
            self.avgpool_2 = self.avgpool_2.cuda()
            self.conv_3_1 = self.conv_3_1.cuda()
            self.conv_3_2 = self.conv_3_2.cuda()
            self.conv_3_3 = self.conv_3_3.cuda()
            self.avgpool_3 = self.avgpool_3.cuda()
            self.conv_4_1 = self.conv_4_1.cuda()
            self.conv_4_2 = self.conv_4_2.cuda()
            self.conv_4_3 = self.conv_4_3.cuda()
            self.avgpool_4 = self.avgpool_4.cuda()
            self.conv_5_1 = self.conv_5_1.cuda()
            self.conv_5_2 = self.conv_5_2.cuda()
            self.conv_5_3 = self.conv_5_3.cuda()
            self.avgpool_5 = self.avgpool_5.cuda()
            self.fc_1 = self.fc_1.cuda()
            self.dp_1 = self.dp_1.cuda()
            self.fc_2 = self.fc_2.cuda()
            self.bn_1 = self.bn_1.cuda()
            self.fc_3 = self.fc_3.cuda()
            self.mish = self.mish.cuda()
    def forward(self, x):
        x = self.conv_1_1(x)
        x = self.conv_1_2(x)
        x = self.avgpool_1(x)

        x = self.conv_2_1(x)
        x = self.conv_2_2(x)
        x = self.avgpool_2(x)

        x = self.conv_3_1(x)
        x = self.conv_3_2(x)
        x = self.conv_3_3(x)
        x = self.avgpool_3(x)

        x = self.conv_4_1(x)
        x = self.conv_4_2(x)
        x = self.conv_4_3(x)
        x = self.avgpool_4(x)

        x = self.conv_5_1(x)
        x = self.conv_5_2(x)
        x = self.conv_5_3(x)
        x = self.avgpool_5(x)
        x = x.view(-1, 512*1*1)
        x = self.fc_1(x)
        x = self.mish(x)
        # x = self.dp_1(x)
        x = self.fc_2(x)
        x = self.mish(x)
        x = self.bn_1(x)
        x = self.fc_3(x)
        return x


def vgg16():
    return VGG16()
