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
from poly.models.MobileNet import *
from poly.MPCB import *
from poly.Func import *


import hecate as hc
import sys

def getModel():
    from pathlib import Path
    source_path = Path(__file__).resolve()
    source_dir = source_path.parent
 
    model = torch.nn.DataParallel(mobilenet())

    model_dict = torch.load(str(source_dir)+"/../data/mobileNet_silu_model", map_location=torch.device('cpu'))
    # There is no state_dict with checkpoint
    #model.load_state_dict(model_dict['state_dict'])
    model.module.load_state_dict(model_dict)
    model = model.eval()
    return model

@hc.func("c")
def MobileNet (ctxt) :
    model = getModel()
    model = model.type(torch.double)
    # input_var = input_var.type(torch.double)
    input_var = np.empty((1), dtype=object)
    input_var[0] = ctxt

    def act(x) :
        return HE_SiLU(x)
        # return HE_ReLU(x)
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
    print("pre_layer")
    conv1_shapes = CascadeConv(initial_shapes, model.module.pre_layer.Conv2d)
    close = shapeClosure(**conv1_shapes)
    out = HE_ConvBN(close, input_var, model.module.pre_layer.Conv2d, model.module.pre_layer.bn)
    out[0] = hc.bootstrap(out[0]) # 1
    out = act(out)
    block_in = conv1_shapes
    for i in range(0, len(model.module.Depthwise)):
        print(i,"layer")
        inconv_0_shapes = CascadeConv(block_in, model.module.Depthwise[i].dwConv2d)
        close = shapeClosure(**inconv_0_shapes)
        out = HE_DwConv(close, out, model.module.Depthwise[i].dwConv2d, model.module.Depthwise[i].bn)
        out[0] = hc.bootstrap(out[0])
        out = act(out)
        
        inconv_0_pointwise_shapes = CascadeConv(inconv_0_shapes, model.module.Depthwise[i].pointwiseConv2d.Conv2d)
        close = shapeClosure(**inconv_0_pointwise_shapes)
        out = HE_ConvBN(close, out, model.module.Depthwise[i].pointwiseConv2d.Conv2d, model.module.Depthwise[i].pointwiseConv2d.bn)
        out[0] = hc.bootstrap(out[0])
        out = act(out)
        block_in = inconv_0_pointwise_shapes

    # avgpool_1_shapes = CascadeMax (block_in, model.module.avgpool)
    avgpool_1_shapes = CascadePool (block_in)
    close = shapeClosure(**avgpool_1_shapes)
    # out = HE_Avg(close, out)
    out = HE_Pool(close, out)
    block_in = avgpool_1_shapes
    
    out = HE_Linear(close["OP"], out, model.module.linear, scale = 32.0)
    
    print("end")
    return out

modName = hc.save("traced", "traced")
print (modName)

