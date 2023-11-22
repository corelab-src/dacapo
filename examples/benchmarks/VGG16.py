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

import hecate as hc
import sys

mish = nn.SiLU()

# def maximum(a,b) : 
#     return Poly.maxx (a,b)
def poly2(x) :
    out = models.MPCB.GenPoly(Poly.treeStr2,Poly.coeffStr2,  4, scale = 1.7)(x)
    out[0] = hc.bootstrap(out[0])
    return out 


def getModel():
    from pathlib import Path
    source_path = Path(__file__).resolve()
    source_dir = source_path.parent
    model = torch.nn.DataParallel(vgg16())
    model_dict = torch.load(str(source_dir)+"/../data/vgg16_silu_avgpool_model", map_location=torch.device('cpu'))
    # There is no state_dict with checkpoint
    #model.load_state_dict(model_dict['state_dict'])
    model.module.load_state_dict(model_dict)
    model = model.eval()
    return model
eps = 0.001


def HE_BN (mpp, bn, scale=1.0) :
    G, H = abstractBN(bn)
    mpcb = BN(mpp, G, H/scale, 2**16)
    return mpcb

def HE_Conv (close, mpp, conv, bn) :
    mpcb =  close["MPCB"] (mpp, conv.weight, *abstractBN(bn))
    return mpcb

def HE_Avg (close, mpp) :
    mpcb =  close["MA"] (mpp)
    return mpcb

def HE_Max (close, mpp) :
    mpcb =  close["MP"] (mpp)
    return mpcb

def HE_Pool (close, mpp) :
    #mpcb = close["AP"](mpp)[:, :origin.shape[1]]
    return close["AP"](mpp)

def HE_Linear(close, mpp, linear, p = 1.0, scale = 1.0) :
    mpcb = Linear(mpp, linear.weight * p , linear.bias / scale, 2**16)
    return mpcb



@hc.func("c")
def VGG16 (ctxt) :

    model = getModel()
    model = model.type(torch.double)
    #input_var = input_var.type(torch.double)
    input_var = np.empty((1), dtype= object)
    input_var[0] = ctxt


    calculation = poly.GenPoly()
    def mish_s (A) :
        return A * (calculation(A)+0.5)
    def relu_s (x) : 
        return Poly.relu(x) 
    def act(x) :
        return mish_s(x)
        # return relu_s(x)
    initial_shapes = {
        # Constant
        #"nt" : 2**16,
        "nt" : 2**16,
        "bb" : 32,
        # Input Characteristics (Cascaded)
        "ko" : 1,
        "ho" : 32,
        "wo" : 32
    }
    print("Conv_1_1")        
    conv_1_1_shapes = CascadeConv(initial_shapes, model.module.conv_1_1.Conv2d)
    close = shapeClosure(**conv_1_1_shapes)
    out = HE_Conv(close, input_var, model.module.conv_1_1.Conv2d, model.module.conv_1_1.bn)
    out[0] = hc.bootstrap(out[0]) # 1 - 390
    out = act(out)
    block_in = conv_1_1_shapes
    print("Conv_1_2")
    conv_1_2_shapes = CascadeConv(block_in, model.module.conv_1_2.Conv2d)
    close = shapeClosure(**conv_1_2_shapes)
    out = HE_Conv(close, out, model.module.conv_1_2.Conv2d, model.module.conv_1_2.bn)
    out[0] = hc.bootstrap(out[0]) # 2 - 3351
    out = act(out)
    #out[0] = hc.bootstrap(out[0]) # 3 - 3
    block_in = conv_1_2_shapes
    print("avgpool_1")
    avgpool_1_shapes = CascadeMax (block_in, model.module.avgpool_1)
    close = shapeClosure(**avgpool_1_shapes)
    out = HE_Avg(close, out)
    out[0] = hc.bootstrap(out[0]) # 3 - 3761
    block_in = avgpool_1_shapes
    conv_2_1_shapes = CascadeConv(block_in, model.module.conv_2_1.Conv2d)
    close = shapeClosure(**conv_2_1_shapes)
    out = HE_Conv(close, out, model.module.conv_2_1.Conv2d, model.module.conv_2_1.bn)
    out[0] = hc.bootstrap(out[0]) # 4 - 5533
    out = act(out)
    block_in = conv_2_1_shapes
    
    print("Conv_2_2")
    conv_2_2_shapes = CascadeConv(block_in, model.module.conv_2_2.Conv2d)
    close = shapeClosure(**conv_2_2_shapes)
    out = HE_Conv(close, out, model.module.conv_2_2.Conv2d, model.module.conv_2_2.bn)
    out[0] = hc.bootstrap(out[0]) # 5 - 8828
    out = act(out)
    #out[0] = hc.bootstrap(out[0]) # 6 - 
    block_in = conv_2_2_shapes
    
    print("avgpool_2")
    avgpool_2_shapes = CascadeMax (block_in, model.module.avgpool_2)
    close = shapeClosure(**avgpool_2_shapes)
    out = HE_Avg(close, out)
    out[0] = hc.bootstrap(out[0]) # 6 - 9240 
    block_in = avgpool_2_shapes
    
    print("Conv_3_1")        
    conv_3_1_shapes = CascadeConv(block_in, model.module.conv_3_1.Conv2d)
    close = shapeClosure(**conv_3_1_shapes)
    out = HE_Conv(close, out, model.module.conv_3_1.Conv2d, model.module.conv_3_1.bn)
    out[0] = hc.bootstrap(out[0]) # 7 - 11590
    out = act(out)
    block_in = conv_3_1_shapes
    
    print("Conv_3_2")
    conv_3_2_shapes = CascadeConv(block_in, model.module.conv_3_2.Conv2d)
    close = shapeClosure(**conv_3_2_shapes)
    out = HE_Conv(close, out, model.module.conv_3_2.Conv2d, model.module.conv_3_2.bn)
    out[0] = hc.bootstrap(out[0]) # 8 - 15527
    out = act(out)
    block_in = conv_3_2_shapes
    
    print("Conv_3_3")
    conv_3_3_shapes = CascadeConv(block_in, model.module.conv_3_3.Conv2d)
    close = shapeClosure(**conv_3_3_shapes)
    out = HE_Conv(close, out, model.module.conv_3_3.Conv2d, model.module.conv_3_3.bn)
    out[0] = hc.bootstrap(out[0]) # 9 - 19464
    out = act(out)
    #out[0] = hc.bootstrap(out[0]) # 10
    block_in = conv_3_3_shapes
    
    print("avgpool_3")
    avgpool_3_shapes = CascadeMax (block_in, model.module.avgpool_3)
    close = shapeClosure(**avgpool_3_shapes)
    out = HE_Avg(close, out)
    out[0] = hc.bootstrap(out[0]) # 10 - 19878
    block_in = avgpool_3_shapes
    print("Conv_4_1")        
    conv_4_1_shapes = CascadeConv(block_in, model.module.conv_4_1.Conv2d)
    close = shapeClosure(**conv_4_1_shapes)
    out = HE_Conv(close, out, model.module.conv_4_1.Conv2d, model.module.conv_4_1.bn)
    out[0] = hc.bootstrap(out[0]) # 11 - 23318
    out = act(out)
    block_in = conv_4_1_shapes
    
    print("Conv_4_2")
    conv_4_2_shapes = CascadeConv(block_in, model.module.conv_4_2.Conv2d)
    close = shapeClosure(**conv_4_2_shapes)
    out = HE_Conv(close, out, model.module.conv_4_2.Conv2d, model.module.conv_4_2.bn)
    out[0] = hc.bootstrap(out[0]) # 12 - 28409
    out = act(out)
    block_in = conv_4_2_shapes
    
    print("Conv_4_3")
    conv_4_3_shapes = CascadeConv(block_in, model.module.conv_4_3.Conv2d)
    close = shapeClosure(**conv_4_3_shapes)
    out = HE_Conv(close, out, model.module.conv_4_3.Conv2d, model.module.conv_4_3.bn)
    out[0] = hc.bootstrap(out[0]) # 13 - 33500
    out = act(out)
    #out[0] = hc.bootstrap(out[0]) # 14
    block_in = conv_4_3_shapes
    
    print("avgpool_4")
    avgpool_4_shapes = CascadeMax (block_in, model.module.avgpool_4)
    close = shapeClosure(**avgpool_4_shapes)
    out = HE_Avg(close, out)
    out[0] = hc.bootstrap(out[0]) # 14 - 33916
    block_in = avgpool_4_shapes

    print("Conv_5_1")        
    conv_5_1_shapes = CascadeConv(block_in, model.module.conv_5_1.Conv2d)
    close = shapeClosure(**conv_5_1_shapes)
    out = HE_Conv(close, out, model.module.conv_5_1.Conv2d, model.module.conv_5_1.bn)
    out[0] = hc.bootstrap(out[0]) # 15 - 36704
    out = act(out)
    block_in = conv_5_1_shapes
    
    print("Conv_5_2")
    conv_5_2_shapes = CascadeConv(block_in, model.module.conv_5_2.Conv2d)
    close = shapeClosure(**conv_5_2_shapes)
    out = HE_Conv(close, out, model.module.conv_5_2.Conv2d, model.module.conv_5_2.bn)
    out[0] = hc.bootstrap(out[0]) # 16 - 39639
    out = act(out)
    block_in = conv_5_2_shapes
    
    print("Conv_5_3")
    conv_5_3_shapes = CascadeConv(block_in, model.module.conv_5_3.Conv2d)
    close = shapeClosure(**conv_5_3_shapes)
    out = HE_Conv(close, out, model.module.conv_5_3.Conv2d, model.module.conv_5_3.bn)
    out[0] = hc.bootstrap(out[0]) # 17 - 42574 
    out = act(out)
    #out[0] = hc.bootstrap(out[0]) # 
    block_in = conv_5_3_shapes
    
    print("avgpool_5")
    avgpool_5_shapes = CascadeMax (block_in, model.module.avgpool_5)
    close = shapeClosure(**avgpool_5_shapes)
    out = HE_Avg(close, out)
    out[0] = hc.bootstrap(out[0]) # 18 - 42864
    block_in = avgpool_5_shapes
    
    print("fc_1")
    out = HE_Linear(close["OP"], out, model.module.fc_1, scale = 32.0)
    
    out[0] = hc.bootstrap(out[0]) # 19 - 43895
    out = act(out)
    print("dp_1 & fc_2")
    out = HE_Linear(close["OP"], out, model.module.fc_2, scale=32.0)
    
    out[0] = hc.bootstrap(out[0]) # 20 - 44561
    out = act(out)
    print("bn_1")
    #ori, out = debugBN(ori, out, model.module.bn_1, scale=32.0)
    out = HE_BN(out, model.module.bn_1, scale=32.0)
    
    print("fc_3")
    #ori, out = debugLinear(close["OP"], ori, out, model.module.fc_3, scale=32.0)
    out = HE_Linear(close["OP"], out, model.module.fc_3, scale=32.0)
    return out

modName = hc.save("traced", "traced")
print (modName)

