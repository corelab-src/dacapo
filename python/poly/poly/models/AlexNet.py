import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
from torchvision import datasets, transforms

import signal
import sys
from torch.autograd import Variable

__all__ = ['AlexNet',  'alexnet']

from pathlib import Path

use_cuda = torch.cuda.is_available()
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.Conv2d_1 = nn.Conv2d(kernel_size=3, in_channels=3, out_channels=96, padding=1)
        self.bn_1 = nn.BatchNorm2d(96)
        self.avgpool_1 = nn.AvgPool2d((3, 3), stride=2, padding=1) 

        self.Conv2d_2 = nn.Conv2d(kernel_size=5, in_channels=96, out_channels=256, padding=2)
        self.bn_2 = nn.BatchNorm2d(256)
        self.avgpool_2 = nn.AvgPool2d((3, 3), stride=2, padding=1)

        self.Conv2d_3 = nn.Conv2d(kernel_size=3, in_channels=256, out_channels=384, padding=1)
        self.bn_3 = nn.BatchNorm2d(384)
        self.Conv2d_4 = nn.Conv2d(kernel_size=3, in_channels=384, out_channels=384, padding=1)
        self.bn_4 = nn.BatchNorm2d(384)
        self.Conv2d_5 = nn.Conv2d(kernel_size=3, in_channels=384, out_channels=256, padding=1)
        self.bn_5 = nn.BatchNorm2d(256)
        self.avgpool_3 = nn.AvgPool2d((3, 3), stride=2, padding=1)

        self.fc_1 = nn.Linear(4*4*256, 2048)
        self.dp_1 = nn.Dropout()
        self.fc_2 = nn.Linear(2048, 1024)
        self.dp_2 = nn.Dropout()
        self.fc_3 = nn.Linear(1024, 10)
        self.mish = nn.SiLU()

        if use_cuda :
            self.Conv2d_1 = self.Conv2d_1.cuda() 
            self.bn_1 = self.bn_1.cuda() 
            self.avgpool_1 = self.avgpool_1.cuda() 

            self.Conv2d_2 = self.Conv2d_2.cuda() 
            self.bn_2 = self.bn_2.cuda()
            self.avgpool_2 = self.avgpool_2.cuda() 

            self.Conv2d_3 = self.Conv2d_3.cuda() 
            self.bn_3 = self.bn_3.cuda() 
            self.Conv2d_4 = self.Conv2d_4.cuda() 
            self.bn_4 = self.bn_4.cuda() 
            self.Conv2d_5 = self.Conv2d_5.cuda() 
            self.bn_5 = self.bn_5.cuda() 
            self.avgpool_3 = self.avgpool_3.cuda() 

            self.fc_1 = self.fc_1.cuda() 
            self.dp_1 = self.dp_1.cuda()
            self.fc_2 = self.fc_2.cuda() 
            self.dp_2 = self.dp_2.cuda() 
            self.fc_3 = self.fc_3.cuda() 
            self.mish = self.mish.cuda()

               

    def forward(self, x):
        x = self.Conv2d_1(x)
        x = self.bn_1(x)
        x = self.mish(x)
        x = self.avgpool_1(x)

        x = self.Conv2d_2(x)
        x = self.bn_2(x)
        x = self.mish(x)
        x = self.avgpool_2(x)

        x = self.Conv2d_3(x)
        x = self.bn_3(x)
        x = self.mish(x)
        x = self.Conv2d_4(x)
        x = self.bn_4(x)
        x = self.mish(x)
        x = self.Conv2d_5(x)
        x = self.bn_5(x)
        x = self.mish(x)
        x = self.avgpool_3(x)

        x = x.view(-1, 4*4*256)
        x = self.mish(self.fc_1(x))
        # x = self.dp_1(x)
        x = self.mish(self.fc_2(x))
        # x = self.dp_2(x)
        x = self.fc_3(x)
        return x


def alexnet() :
    return AlexNet()

