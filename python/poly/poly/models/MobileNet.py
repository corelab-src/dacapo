import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
from torchvision import datasets, transforms

import signal
import sys
from torch.autograd import Variable

__all__ = ['MobileNet', 'mobilenet']

from pathlib import Path

use_cuda = torch.cuda.is_available()
# use_cuda = False
FireBlockConfig = {
    'fire2':{'s1x1':16, 'e1x1':64, 'e3x3':64},
    'fire3':{'s1x1':16, 'e1x1':64, 'e3x3':64},
    'fire4':{'s1x1':32, 'e1x1':128, 'e3x3':128},
    'fire5':{'s1x1':32, 'e1x1':128, 'e3x3':128},
    'fire6':{'s1x1':48, 'e1x1':192, 'e3x3':192},
    'fire7':{'s1x1':48, 'e1x1':192, 'e3x3':192},
    'fire8':{'s1x1':64, 'e1x1':256, 'e3x3':256},
    'fire9':{'s1x1':64, 'e1x1':256, 'e3x3':256}
    }

class BasicConv2d(nn.Module):
    def __init__(self, ksize, inCH, outCH, padding=0, stride=1):
        super(BasicConv2d, self).__init__()
        self.Conv2d = nn.Conv2d(kernel_size=ksize, in_channels=inCH,out_channels=outCH, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(outCH)
        self.mish = nn.SiLU()

    def forward(self, x):
        x = self.Conv2d(x)
        x = self.bn(x)
        x = self.mish(x)
        return x

class DepthwiseConv2d(nn.Module):
    def __init__(self, ksize, inCH, outCH, padding=0, stride=1):
        super(DepthwiseConv2d, self).__init__()
        self.dwConv2d = nn.Conv2d(kernel_size=ksize, in_channels=inCH,out_channels=inCH, stride=stride, padding=padding, groups=inCH)
        self.bn = nn.BatchNorm2d(inCH)
        self.pointwiseConv2d = BasicConv2d(ksize=1, inCH=inCH, outCH=outCH)
        self.mish = nn.SiLU()

    def forward(self, x):
        x = self.dwConv2d(x)
        x = self.bn(x)
        x = self.mish(x)
        x = self.pointwiseConv2d(x)
        return x


class MobileNet(nn.Module):
    def __init__(self):
        super(MobileNet, self).__init__()
        self.pre_layer = BasicConv2d(ksize=3, inCH=3, outCH=32, padding=1)
        self.Depthwise = nn.Sequential(
            DepthwiseConv2d(ksize=3, inCH=32, outCH=64, padding=1),
            DepthwiseConv2d(ksize=3, inCH=64, outCH=128, stride=2, padding=1),
            DepthwiseConv2d(ksize=3, inCH=128, outCH=128, padding=1),
            DepthwiseConv2d(ksize=3, inCH=128, outCH=256, stride=2, padding=1),
            DepthwiseConv2d(ksize=3, inCH=256, outCH=256, padding=1),
            DepthwiseConv2d(ksize=3, inCH=256, outCH=512, stride=2, padding=1),
            DepthwiseConv2d(ksize=3, inCH=512, outCH=512, padding=1),
            DepthwiseConv2d(ksize=3, inCH=512, outCH=512, padding=1),
            DepthwiseConv2d(ksize=3, inCH=512, outCH=512, padding=1),
            DepthwiseConv2d(ksize=3, inCH=512, outCH=512, padding=1),
            DepthwiseConv2d(ksize=3, inCH=512, outCH=512, padding=1),
            DepthwiseConv2d(ksize=3, inCH=512, outCH=1024, stride=2, padding=1),
            DepthwiseConv2d(ksize=3, inCH=1024, outCH=1024, padding=1)
        )
        self.avgpool = nn.AvgPool2d((2, 2))
        self.linear = nn.Linear(1024*1*1, 10)

        if use_cuda : 
            self.pre_layer = self.pre_layer.cuda()
            self.Depthwise = self.Depthwise.cuda()
            self.avgpool = self.avgpool.cuda()
            self.linear = self.linear.cuda()

    def forward(self, x) :
        x = self.pre_layer(x)
        x = self.Depthwise(x)
        x = self.avgpool(x)
        x = x.view(-1, 1024)
        x = self.linear(x)

        return x


#
def mobilenet():
    return MobileNet()
