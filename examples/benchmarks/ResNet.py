#!/usr/bin/env python

import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import numpy as np
import hecate as hc
import sys

import poly
from poly.models.ResNet import *
from poly.MPCB import *

mish = nn.SiLU()

def getModel():
    # model_dict = torch.load("../data/resnet20.silu.model", map_location=torch.device('cpu'))
    from pathlib import Path
    source_path = Path(__file__).resolve()
    source_dir = source_path.parent
    model = torch.nn.DataParallel(resnet20())
    model_dict = torch.load(str(source_dir)+"/../data/resnet20.silu.model", map_location=torch.device('cpu'))
    model.load_state_dict(model_dict['state_dict'])
    model = model.eval()
    return model

def HE_Conv (close, mpp, conv, bn) :
    mpcb =  close["MPCB"] (mpp, conv.weight, *abstractBN(bn))
    return mpcb
def HE_DS (close, mpp) :
    mpcb = close["DS"](mpp)
    return mpcb

def HE_Pool (close, mpp) :
    return close["AP"](mpp)

def HE_Linear(close, mpp, linear, p = 1.0, scale = 1.0) :
    # mpcb = close["OP"](mpp)
    # mpcb = Linear(mpp, linear.weight * p , linear.bias / scale, 2**16)
    mpcb = Linear(mpp, linear.weight * p , linear.bias / scale, 2**15)
    return mpcb

@hc.func("c")
def ResNet (ctxt) :

    model = getModel()
    model = model.type(torch.double)
    model = model.cpu()
    # input_var = input_var.type(torch.double)
    input_var = np.empty((1), dtype=object)
    input_var[0] = ctxt

    calculation = poly.GenPoly()
    def mish_s (A) :
        return A * (calculation(A)+0.5)
    def relu_s (x) : 
        return Poly.relu(x) 
    def act(x) :
        return mish_s(x)
    initial_shapes = {
        # Constant
        "nt" : 2**16,
        "bb" : 32,
        # Input Characteristics (Cascaded)
        "ko" : 1,
        "ho" : 32,
        "wo" : 32
    }
    conv1_shapes = CascadeConv(initial_shapes, model.module.conv1)
    close = shapeClosure(**conv1_shapes)
    # out = HE_Conv(MPCB, MPP(input_var),model.module.conv1, model.module.bn1)
    out = HE_Conv(close, input_var,model.module.conv1, model.module.bn1)
    out[0] = hc.bootstrap(out[0])
    out = act(out)
    block_in = conv1_shapes
    print ("layer1")
    for i in range(0, len(model.module.layer1)) :
        print (i)
        dsout = out
        inconv1_shapes = CascadeConv (block_in, model.module.layer1[i].conv1)
        close = shapeClosure(**inconv1_shapes)
        out = HE_Conv(close, out, model.module.layer1[i].conv1, model.module.layer1[i].bn1)
        out[0] = hc.bootstrap(out[0])
        out = act (out)
        inconv2_shapes = CascadeConv (inconv1_shapes, model.module.layer1[i].conv2)
        close = shapeClosure(**inconv2_shapes)
        out = HE_Conv(close, out,model.module.layer1[i].conv2, model.module.layer1[i].bn2)
        out = out +dsout
        out[0] = hc.bootstrap(out[0])
        out = act (out)
        block_in = inconv2_shapes

    print ("layer2")
    ds1_shapes = CascadeDS (block_in)
    close = shapeClosure(**ds1_shapes)
    dsout = HE_DS(close, out)
    for i in range(0, len(model.module.layer2)) :
        print (i)
        if not (i == 0) :
            dsout = out
        inconv1_shapes = CascadeConv (block_in, model.module.layer2[i].conv1)
        close = shapeClosure(**inconv1_shapes)
        out = HE_Conv(close, out, model.module.layer2[i].conv1, model.module.layer2[i].bn1)
        out[0] = hc.bootstrap(out[0])
        out = act (out)
        inconv2_shapes = CascadeConv (inconv1_shapes, model.module.layer2[i].conv2)
        close = shapeClosure(**inconv2_shapes)
        out = HE_Conv(close,  out, model.module.layer2[i].conv2, model.module.layer2[i].bn2)
        out = out +dsout
        out[0] = hc.bootstrap(out[0])
        out = act (out)
        block_in = inconv2_shapes
        
    print ("layer3")
    ds2_shapes = CascadeDS (block_in)
    close = shapeClosure(**ds2_shapes)
    dsout = HE_DS(close, out)
    for i in range(0, len(model.module.layer3)) :
        print (i)
        if not (i == 0) : 
            dsout = out
        inconv1_shapes = CascadeConv (block_in, model.module.layer3[i].conv1)
        close = shapeClosure(**inconv1_shapes)
        out = HE_Conv(close, out, model.module.layer3[i].conv1, model.module.layer3[i].bn1)
        out[0] = hc.bootstrap(out[0])
        out = act (out)
        inconv2_shapes = CascadeConv (inconv1_shapes, model.module.layer3[i].conv2)
        close = shapeClosure(**inconv2_shapes)
        out = HE_Conv(close, out, model.module.layer3[i].conv2, model.module.layer3[i].bn2)
        out= out + dsout
        out[0] = hc.bootstrap(out[0])
        out = act (out)
        block_in = inconv2_shapes
        
    pool_shapes = CascadePool (block_in)

    close = shapeClosure(**pool_shapes)
    out = HE_Pool(close,  out)
    out = HE_Linear(close["OP"], out, model.module.linear, scale=32.0)
    return out

modName = hc.save("traced", "traced")
print (modName)
