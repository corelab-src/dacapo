import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
from torchvision import datasets, transforms

import signal
import sys
from torch.autograd import Variable

__all__ = ['SqueezeNet', 'squeezenet']

from pathlib import Path

use_cuda = torch.cuda.is_available()
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
    def __init__(self, ksize, inCH, outCh, padding=0, stride=1):
        super(BasicConv2d, self).__init__()
        self.Conv2d = nn.Conv2d(kernel_size=ksize, in_channels=inCH,out_channels=outCh, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(outCh)
        self.mish = nn.SiLU()

    def forward(self, x):
        x = self.Conv2d(x)
        x = self.bn(x)
        x = self.mish(x)
        return x


class Fire(nn.Module) :
    def __init__(self, inCH, config, name):
        super(Fire, self).__init__()
        self.cfg = config[name]
        self.squeeze = BasicConv2d(1, inCH, self.cfg['s1x1'])
        self.expand1x1 = nn.Conv2d(in_channels=self.cfg['s1x1'], out_channels=self.cfg['e1x1'], kernel_size = 1)
        self.expand3x3 = nn.Conv2d(in_channels=self.cfg['s1x1'], out_channels=self.cfg['e3x3'], kernel_size = 3, padding=1)


    def forward(self, x) :
        x = self.squeeze(x)
        x_1 = self.expand1x1(x)
        x_2 = self.expand3x3(x)
        out = torch.cat([x_1, x_2], dim=1)
        return out

class SqueezeNet(nn.Module):
    def __init__(self):
        super(SqueezeNet, self).__init__()
        self.conv_1 = BasicConv2d(3, 3, 64, 1)
        self.avgpool_1 = nn.AvgPool2d((3, 3), stride=2, padding=1) 

        self.fire_2 = Fire(inCH=64, config=FireBlockConfig, name='fire2')
        self.fire_3 = Fire(inCH=128, config=FireBlockConfig, name='fire3')
        self.fire_4 = Fire(inCH=128, config=FireBlockConfig, name='fire4')
        self.avgpool_4 = nn.AvgPool2d((3, 3), stride=2, padding=1)

        self.fire_5 = Fire(inCH=256, config=FireBlockConfig, name='fire5')
        self.fire_6 = Fire(inCH=256, config=FireBlockConfig, name='fire6')
        self.fire_7 = Fire(inCH=384, config=FireBlockConfig, name='fire7')
        self.fire_8 = Fire(inCH=384, config=FireBlockConfig, name='fire8')
        self.avgpool_8 = nn.AvgPool2d((3, 3), stride=2, padding=1)

        self.fire_9 = Fire(inCH=512, config=FireBlockConfig, name='fire9')
        self.conv_10 = BasicConv2d(1, 512, 10)
        self.avgpool_10 = nn.AdaptiveAvgPool2d(1)

        if use_cuda : 
            self.conv_1 = self.conv_1.cuda()
            # self.conv_2 = self.conv_2.cuda()
            self.avgpool_1 = self.avgpool_1.cuda()

            self.fire_2 = self.fire_2.cuda()
            self.fire_3 = self.fire_3.cuda()
            self.fire_4 = self.fire_4.cuda()
            self.avgpool_4 = self.avgpool_4.cuda()

            self.fire_5 = self.fire_5.cuda()
            self.fire_6 = self.fire_6.cuda()
            self.fire_7 = self.fire_7.cuda()
            self.fire_8 = self.fire_8.cuda()
            self.avgpool_8 = self.avgpool_8.cuda()

            self.fire_9 = self.fire_9.cuda()
            self.conv_10 = self.conv_10.cuda()
            self.avgpool_10 = self.avgpool_10.cuda()

    def forward(self, x) :
        x = self.conv_1(x)
        # x = self.conv_2(x)
        x = self.avgpool_1(x)

        x = self.fire_2(x)
        x = self.fire_3(x)
        x = self.fire_4(x)
        x = self.avgpool_4(x)

        x = self.fire_5(x)
        x = self.fire_6(x)
        x = self.fire_7(x)
        x = self.fire_8(x)
        x = self.avgpool_8(x)

        x = self.fire_9(x)
        
        x = self.conv_10(x)
        x = self.avgpool_10(x)

        x = x.view(-1, 10)

        return x

def squeezenet() :
    return SqueezeNet()

