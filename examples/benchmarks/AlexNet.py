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
from poly.models.AlexNet import *
from poly.MPCB import *

import sys

def roll(A, i) :
    return A.rotate(-i)

def poly2(x) :
    out = models.MPCB.GenPoly(Poly.treeStr2,Poly.coeffStr2,  4, scale = 1.7)(x)
    # out[0] = hc.bootstrap(out[0])
    return out 

def nprelu(x) : 
    return np.array([ np.maximum (xx, 0) for xx in x], dtype = object)

def getModel():
    from pathlib import Path
    source_path = Path(__file__).resolve()
    source_dir = source_path.parent
    model = torch.nn.DataParallel(alexnet())
    model_dict = torch.load(str(source_dir)+"/../data/alexNet_silu_avgpool_model", map_location=torch.device('cpu'))
    model.module.load_state_dict(model_dict)
    model = model.eval()
    return model



eps = 0.001


def HE_BN (close, mpp, bn, scale=1.0) :
    G, H = abstractBN(bn)
    mpcb = close["BN"](mpp, G, H)
    return mpcb

def HE_Conv (close, mpp, conv) :
    mpcb =  close["MPC"] (mpp, conv.weight)
    return mpcb

def HE_ConvBN (close, mpp, conv, bn) :
    mpcb =  close["MPCB"] (mpp, conv.weight, *abstractBN(bn))
    return mpcb

def HE_Max (close, mpp) :
    mpcb =  close["MPD"] (mpp)
    return mpcb

def HE_Avg (close, mpp) :
    mpcb =  close["MA"] (mpp)
    return mpcb

def HE_DS (close, mpp) :
    mpcb = close["DS"](mpp)
    return mpcb

def HE_Pool (close, mpp) :
    return close["AP"](mpp)
def HE_Linear(close, mpp, linear, p = 1.0, scale = 1.0) :
    mpcb = Linear(mpp, linear.weight * p , linear.bias.cpu() / scale, 2**16)
    return mpcb

def HE_ReshapeLinear(close, mpp, linear, p = 1.0, scale = 1.0, reshape = {}) :
    weight = Reshape (linear.weight, reshape)
    mpcb = Linear(mpp, weight * p , linear.bias.cpu() / scale, 2**16)
    return mpcb



@hc.func("c")
def AlexNet (ctxt) :

    model = getModel()
    model = model.type(torch.double)
    for p in model.parameters():
        p.requires_grad = False
    input_var = np.empty((1), dtype=object)
    input_var[0] = ctxt

    calculation = poly.GenPoly()
    def mish_s (A) :
        return A * (calculation(A)+0.5)
    def act (x) : 
        # x[0] = hc.bootstrap(x[0])
        return mish_s(x)

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
    
    print("Conv_1_BN")        
    conv_1_shapes = CascadeConv(initial_shapes, model.module.Conv2d_1)
    close = shapeClosure(**conv_1_shapes)
    out = HE_ConvBN(close, input_var,model.module.Conv2d_1, model.module.bn_1)
    out[0] = hc.bootstrap(out[0]) # 1 - 571
    out[1] = hc.bootstrap(out[1]) # 2 - 581
    out = act(out)
    out[0] = hc.bootstrap(out[0]) # 3-1
    out[1] = hc.bootstrap(out[1]) # 3-2
    block_in = conv_1_shapes
    
    print("avgpool_1")
    avgpool_1_shapes = CascadeMax (block_in, model.module.avgpool_1)
    close = shapeClosure(**avgpool_1_shapes)
    out = HE_Avg(close, out)
    #out[0] = hc.bootstrap(out[0]) # 3 - 1341
    block_in = avgpool_1_shapes
    
    print("Conv_2_BN")        
    conv_2_shapes = CascadeConv(block_in, model.module.Conv2d_2)
    close = shapeClosure(**conv_2_shapes)
    out = HE_ConvBN(close, out,model.module.Conv2d_2, model.module.bn_2)
    out[0] = hc.bootstrap(out[0]) # 4 - 13783
    out = act(out)
    block_in = conv_2_shapes
 
    print("avgpool_2")
    avgpool_2_shapes = CascadeMax (block_in, model.module.avgpool_2)
    close = shapeClosure(**avgpool_2_shapes)
    out = HE_Avg(close, out)
    out[0] = hc.bootstrap(out[0]) # 5 - 14479
    block_in = avgpool_2_shapes

    print("Conv_3")        
    conv_3_shapes = CascadeConv(block_in, model.module.Conv2d_3)
    close = shapeClosure(**conv_3_shapes)
    out = HE_ConvBN(close, out ,model.module.Conv2d_3, model.module.bn_3)
    out[0] = hc.bootstrap(out[0]) # 6 - 20155
    out = act (out)
    block_in = conv_3_shapes
 
    print("Conv_4")        
    conv_4_shapes = CascadeConv(block_in, model.module.Conv2d_4)
    close = shapeClosure(**conv_4_shapes)
    out = HE_ConvBN(close, out ,model.module.Conv2d_4, model.module.bn_4)
    out[0] = hc.bootstrap(out[0]) # 7 - 30490
    # out[1] = hc.bootstrap(out[1])
    out = act (out)
    block_in = conv_4_shapes
 
    print("Conv_5")        
    conv_5_shapes = CascadeConv(block_in, model.module.Conv2d_5)
    close = shapeClosure(**conv_5_shapes)
    out = HE_ConvBN(close, out ,model.module.Conv2d_5, model.module.bn_5)
    
    out[0] = hc.bootstrap(out[0]) # 8 - 37435
    # out[1] = hc.bootstrap(out[1])
    out = act (out)
    block_in = conv_5_shapes
    print("avgpool_3")
    avgpool_3_shapes = CascadeMax (block_in, model.module.avgpool_3)
    close = shapeClosure(**avgpool_3_shapes)
    out = HE_Avg(close, out)
    out[0] = hc.bootstrap(out[0]) # 9 - 37879
    block_in = avgpool_3_shapes
    
    print("fc_1")
    out = HE_ReshapeLinear(close["OP"], out, model.module.fc_1, scale = 32.0, reshape = block_in)
    out[0] = hc.bootstrap(out[0]) # 10 - 46078
    out = act(out)
    print("fc_2")
    out = HE_Linear(close["OP"], out, model.module.fc_2, scale = 32.0)
    out[0] = hc.bootstrap(out[0]) # 11 - 50328
    out = act(out)
    print("fc_3")
    out = HE_Linear(close["OP"], out, model.module.fc_3, scale = 32.0)

    return out

modName = hc.save("traced", "traced")
print (modName)
