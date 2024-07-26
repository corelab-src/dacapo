#!/usr/bin/env python

import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
#import resnet

import numpy as np

import poly
from poly.models.VGG16 import *
from poly.MPCB import *
from poly.Func import *

import hecate as hc
import sys

def getModel():
    from pathlib import Path
    source_path = Path(__file__).resolve()
    source_dir = source_path.parent
    model = torch.nn.DataParallel(vgg16())
    model_dict = torch.load(str(source_dir)+"/../data/vgg16_silu_avgpool_model", map_location=torch.device('cpu'))
    # model_dict = torch.load(str(source_dir)+"/../data/vgg16_relu_maxpool_model", map_location=torch.device('cpu'))
    model.module.load_state_dict(model_dict)
    model = model.eval()
    return model


@hc.func("c")
def VGG16 (ctxt) :

    model = getModel()
    model = model.type(torch.double)
    model = model.cpu()
    #input_var = input_var.type(torch.double)
    input_var = np.empty((1), dtype= object)
    input_var[0] = ctxt


    def act(x) :
        return HE_SiLU(x)
        # return HE_ReLU(x)
    def pooling(close, x) :
        return HE_Avg(close, x) 
        # return HE_Max(close, x) 
    initial_shapes = {
        # Constant
        # "nt" : 2**14,
        "nt" : 2**16,
        "bb" : 32,
        # Input Characteristics (Cascaded)
        "ko" : 1,
        "ho" : 32,
        "wo" : 32
    }
    ## manual bootstrapping placement based on silu and average pooling
    print("Conv_1_1")        
    conv_1_1_shapes = CascadeConv(initial_shapes, model.module.conv_1_1.Conv2d)
    close = shapeClosure(**conv_1_1_shapes)
    out = HE_ConvBN(close, input_var, model.module.conv_1_1.Conv2d, model.module.conv_1_1.bn)
    out = hc.bootstrap(out)
    out = act(out)
    block_in = conv_1_1_shapes
    print("Conv_1_2")
    conv_1_2_shapes = CascadeConv(block_in, model.module.conv_1_2.Conv2d)
    close = shapeClosure(**conv_1_2_shapes)
    out = HE_ConvBN(close, out, model.module.conv_1_2.Conv2d, model.module.conv_1_2.bn)
    out = hc.bootstrap(out)
    out = act(out)
    block_in = conv_1_2_shapes
    print("avgpool_1")
    avgpool_1_shapes = CascadeMax (block_in, model.module.avgpool_1)
    close = shapeClosure(**avgpool_1_shapes)
    out = pooling(close, out)
    out = hc.bootstrap(out)
    block_in = avgpool_1_shapes
    conv_2_1_shapes = CascadeConv(block_in, model.module.conv_2_1.Conv2d)
    close = shapeClosure(**conv_2_1_shapes)
    out = HE_ConvBN(close, out, model.module.conv_2_1.Conv2d, model.module.conv_2_1.bn)
    out = hc.bootstrap(out)
    out = act(out)
    block_in = conv_2_1_shapes
    
    print("Conv_2_2")
    conv_2_2_shapes = CascadeConv(block_in, model.module.conv_2_2.Conv2d)
    close = shapeClosure(**conv_2_2_shapes)
    out = HE_ConvBN(close, out, model.module.conv_2_2.Conv2d, model.module.conv_2_2.bn)
    out = hc.bootstrap(out)
    out = act(out)
    block_in = conv_2_2_shapes
    
    print("avgpool_2")
    avgpool_2_shapes = CascadeMax (block_in, model.module.avgpool_2)
    close = shapeClosure(**avgpool_2_shapes)
    out = pooling(close, out)
    out = hc.bootstrap(out)
    block_in = avgpool_2_shapes
    
    print("Conv_3_1")        
    conv_3_1_shapes = CascadeConv(block_in, model.module.conv_3_1.Conv2d)
    close = shapeClosure(**conv_3_1_shapes)
    out = HE_ConvBN(close, out, model.module.conv_3_1.Conv2d, model.module.conv_3_1.bn)
    out = hc.bootstrap(out)
    out = act(out)
    block_in = conv_3_1_shapes
    
    print("Conv_3_2")
    conv_3_2_shapes = CascadeConv(block_in, model.module.conv_3_2.Conv2d)
    close = shapeClosure(**conv_3_2_shapes)
    out = HE_ConvBN(close, out, model.module.conv_3_2.Conv2d, model.module.conv_3_2.bn)
    out = hc.bootstrap(out)
    out = act(out)
    block_in = conv_3_2_shapes
    
    print("Conv_3_3")
    conv_3_3_shapes = CascadeConv(block_in, model.module.conv_3_3.Conv2d)
    close = shapeClosure(**conv_3_3_shapes)
    out = HE_ConvBN(close, out, model.module.conv_3_3.Conv2d, model.module.conv_3_3.bn)
    out = hc.bootstrap(out)
    out = act(out)
    block_in = conv_3_3_shapes
    
    print("avgpool_3")
    avgpool_3_shapes = CascadeMax (block_in, model.module.avgpool_3)
    close = shapeClosure(**avgpool_3_shapes)
    out = pooling(close, out)
    out = hc.bootstrap(out)
    block_in = avgpool_3_shapes
    print("Conv_4_1")        
    conv_4_1_shapes = CascadeConv(block_in, model.module.conv_4_1.Conv2d)
    close = shapeClosure(**conv_4_1_shapes)
    out = HE_ConvBN(close, out, model.module.conv_4_1.Conv2d, model.module.conv_4_1.bn)
    out = hc.bootstrap(out)
    out = act(out)
    block_in = conv_4_1_shapes
    
    print("Conv_4_2")
    conv_4_2_shapes = CascadeConv(block_in, model.module.conv_4_2.Conv2d)
    close = shapeClosure(**conv_4_2_shapes)
    out = HE_ConvBN(close, out, model.module.conv_4_2.Conv2d, model.module.conv_4_2.bn)
    out = hc.bootstrap(out)
    out = act(out)
    block_in = conv_4_2_shapes
    
    print("Conv_4_3")
    conv_4_3_shapes = CascadeConv(block_in, model.module.conv_4_3.Conv2d)
    close = shapeClosure(**conv_4_3_shapes)
    out = HE_ConvBN(close, out, model.module.conv_4_3.Conv2d, model.module.conv_4_3.bn)
    out = hc.bootstrap(out)
    out = act(out)
    block_in = conv_4_3_shapes
    
    print("avgpool_4")
    avgpool_4_shapes = CascadeMax (block_in, model.module.avgpool_4)
    close = shapeClosure(**avgpool_4_shapes)
    out = pooling(close, out)
    out = hc.bootstrap(out)
    block_in = avgpool_4_shapes

    print("Conv_5_1")        
    conv_5_1_shapes = CascadeConv(block_in, model.module.conv_5_1.Conv2d)
    close = shapeClosure(**conv_5_1_shapes)
    out = HE_ConvBN(close, out, model.module.conv_5_1.Conv2d, model.module.conv_5_1.bn)
    out = hc.bootstrap(out)
    out = act(out)
    block_in = conv_5_1_shapes
    
    print("Conv_5_2")
    conv_5_2_shapes = CascadeConv(block_in, model.module.conv_5_2.Conv2d)
    close = shapeClosure(**conv_5_2_shapes)
    out = HE_ConvBN(close, out, model.module.conv_5_2.Conv2d, model.module.conv_5_2.bn)
    out = hc.bootstrap(out)
    out = act(out)
    block_in = conv_5_2_shapes
    
    print("Conv_5_3")
    conv_5_3_shapes = CascadeConv(block_in, model.module.conv_5_3.Conv2d)
    close = shapeClosure(**conv_5_3_shapes)
    out = HE_ConvBN(close, out, model.module.conv_5_3.Conv2d, model.module.conv_5_3.bn)
    out = hc.bootstrap(out)
    out = act(out)
    block_in = conv_5_3_shapes
    
    print("avgpool_5")
    avgpool_5_shapes = CascadeMax (block_in, model.module.avgpool_5)
    close = shapeClosure(**avgpool_5_shapes)
    out = pooling(close, out)
    out = hc.bootstrap(out)
    block_in = avgpool_5_shapes
    
    print("fc_1")
    out = HE_Linear(close["OP"], out, model.module.fc_1, scale = 32.0)
    
    out = hc.bootstrap(out)
    out = act(out)
    print("dp_1 & fc_2")
    out = HE_Linear(close["OP"], out, model.module.fc_2, scale=32.0)
    
    out = hc.bootstrap(out)
    out = act(out)
    print("bn_1")
    #ori, out = debugBN(ori, out, model.module.bn_1, scale=32.0)
    out = HE_MPBN(out, model.module.bn_1, scale=32.0)
    
    print("fc_3")
    #ori, out = debugLinear(close["OP"], ori, out, model.module.fc_3, scale=32.0)
    out = HE_Linear(close["OP"], out, model.module.fc_3, scale=32.0)
    return out

modName = hc.save("traced", "traced")
print (modName)

