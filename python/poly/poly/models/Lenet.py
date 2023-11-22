import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np

use_cuda = False

#PreProcessing

from pathlib import Path

source_path = Path(__file__).resolve()
source_dir = source_path.parent


batch_size = 64

class Square(nn.Module):
    def __init__(self) : 
        super().__init__()
    def forward ( self, input) :
        return torch.square(input)

SquareAct = Square()
Pad2 = nn.ZeroPad2d(2)
Pad1 = nn.ZeroPad2d(1)

class CNNClassifier(nn.Module):
    
    def __init__(self):
        # 항상 torch.nn.Module을 상속받고 시작
        super(CNNClassifier, self).__init__()
        # ZeroPad2d
        self.pad1 = nn.ZeroPad2d(2) # 6@32*32
        self.conv1 = nn.Conv2d(1, 6, 5, 1, bias=False) # 6@28*28
        self.bn1 = nn.BatchNorm2d(6) #6@28,28
        self.act1 = Square()
        self.pool1 = nn.AvgPool2d(2, 2) # 6@14*14
        self.pad2 = nn.ZeroPad2d(1) # 6@16*16
        self.conv2 = nn.Conv2d(6, 16, 5, 1, bias=False) # 16@12*12
        self.bn2 = nn.BatchNorm2d(16) #16@12*12
        self.act2 = Square()
        self.pool2 = nn.AvgPool2d(2, 2) # 16@6*6
        self.pad3 = nn.ZeroPad2d(1) # 16@8*8
        self.fc1 = nn.Linear(16*8*8, 128, bias = False) #  2048 -> 128
        self.bn3 = nn.BatchNorm1d(128) #128
        self.act3 = Square()
        self.fc2 = nn.Linear(128, 64, bias = False)
        self.bn4 = nn.BatchNorm1d(64) #128
        self.act4 = Square()
        self.fc3 = nn.Linear(64,16, bias = False)
        self.bn5 = nn.BatchNorm1d(16) #128

        
        # gpu로 할당
        if use_cuda:
            self.pad1 = self.pad1.cuda()
            self.conv1 = self.conv1.cuda()
            self.bn1 = self.bn1.cuda()
            self.act1 = self.act1.cuda()
            self.pool1 = self.pool1.cuda()
            self.pad2 = self.pad2.cuda()
            self.conv2 = self.conv2.cuda()
            self.bn2 = self.bn2.cuda()
            self.act2 = self.act2.cuda()
            self.pool2 = self.pool2.cuda()
            self.pad3 = self.pad3.cuda()
            self.fc1 = self.fc1.cuda()
            self.bn3 = self.bn3.cuda()
            self.act3 = self.act3.cuda()
            self.fc2 = self.fc2.cuda()
            self.bn4 = self.bn4.cuda()
            self.act4 = self.act4.cuda()
            self.fc3 = self.fc3.cuda()
            self.bn5 = self.bn5.cuda()
        
    def forward(self, x):
        batch = x.size (0)
        x = self.pad1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.pool1(x)
        x = self.pad2(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.pool2(x)
        x = self.pad3(x)
        x = x.view(batch, 16*8*8)
        x = self.fc1(x)
        x = self.bn3(x)
        x = self.act3(x)
        x = self.fc2(x)
        x = self.bn4(x)
        x = self.act4(x)
        x = self.fc3(x)
        x = self.bn5(x)
        x = x[:,:10]
        #omit softmax for eval
        return x
