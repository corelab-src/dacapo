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
from poly.Func import *

import sys

def getModel():
    from pathlib import Path
    source_path = Path(__file__).resolve()
    source_dir = source_path.parent
    model = torch.nn.DataParallel(alexnet())
    model_dict = torch.load(str(source_dir)+"/../data/alexNet_silu_avgpool_model", map_location=torch.device('cpu'))
    model.module.load_state_dict(model_dict)
    model = model.eval()
    return model


@hc.func("c")
def AlexNet (ctxt) :

    model = getModel()
    model = model.type(torch.double)
    for p in model.parameters():
        p.requires_grad = False
    input_var = np.empty((1), dtype=object)
    input_var[0] = ctxt

    def act (x) : 
        return HE_SiLU(x) 
        # return HE_ReLU(x)
    def pooling(close, x):
        return HE_Avg(close, x)
        # return HE_MaxPad(close, x)

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
    out = hc.bootstrap(out)
    out = act(out)
    out = hc.bootstrap(out)
    block_in = conv_1_shapes
    
    print("avgpool_1")
    avgpool_1_shapes = CascadeMax (block_in, model.module.avgpool_1)
    close = shapeClosure(**avgpool_1_shapes)
    out = pooling(close, out)
    block_in = avgpool_1_shapes
    
    print("Conv_2_BN")        
    conv_2_shapes = CascadeConv(block_in, model.module.Conv2d_2)
    close = shapeClosure(**conv_2_shapes)
    out = HE_ConvBN(close, out,model.module.Conv2d_2, model.module.bn_2)
    out = hc.bootstrap(out)
    out = act(out)
    block_in = conv_2_shapes
 
    print("avgpool_2")
    avgpool_2_shapes = CascadeMax (block_in, model.module.avgpool_2)
    close = shapeClosure(**avgpool_2_shapes)
    out = pooling(close, out)
    out = hc.bootstrap(out)
    block_in = avgpool_2_shapes

    print("Conv_3")        
    conv_3_shapes = CascadeConv(block_in, model.module.Conv2d_3)
    close = shapeClosure(**conv_3_shapes)
    out = HE_ConvBN(close, out ,model.module.Conv2d_3, model.module.bn_3)
    out = hc.bootstrap(out)
    out = act (out)
    block_in = conv_3_shapes
 
    print("Conv_4")        
    conv_4_shapes = CascadeConv(block_in, model.module.Conv2d_4)
    close = shapeClosure(**conv_4_shapes)
    out = HE_ConvBN(close, out ,model.module.Conv2d_4, model.module.bn_4)
    out = hc.bootstrap(out)
    out = act (out)
    block_in = conv_4_shapes
 
    print("Conv_5")        
    conv_5_shapes = CascadeConv(block_in, model.module.Conv2d_5)
    close = shapeClosure(**conv_5_shapes)
    out = HE_ConvBN(close, out ,model.module.Conv2d_5, model.module.bn_5)
    
    out = hc.bootstrap(out)
    out = act (out)
    block_in = conv_5_shapes
    print("avgpool_3")
    avgpool_3_shapes = CascadeMax (block_in, model.module.avgpool_3)
    close = shapeClosure(**avgpool_3_shapes)
    out = pooling(close, out)
    out = hc.bootstrap(out)
    block_in = avgpool_3_shapes
    
    print("fc_1")
    out = HE_ReshapeLinear(close["OP"], out, model.module.fc_1, scale = 32.0, reshape = block_in)
    out = hc.bootstrap(out)
    out = act(out)
    print("fc_2")
    out = HE_Linear(close["OP"], out, model.module.fc_2, scale = 32.0)
    out = hc.bootstrap(out)
    out = act(out)
    print("fc_3")
    out = HE_Linear(close["OP"], out, model.module.fc_3, scale = 32.0)

    return out

modName = hc.save("traced", "traced")
print (modName)
