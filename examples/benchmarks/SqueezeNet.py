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
from poly.models.SqueezeNet import *
from poly.MPCB import *
from poly.Func import *

import sys

def getModel():
    from pathlib import Path
    source_path = Path(__file__).resolve()
    source_dir = source_path.parent
    model = torch.nn.DataParallel(squeezenet())

    model_dict = torch.load(str(source_dir)+"/../data/squeezeNet_silu_avgpool_model", map_location=torch.device('cpu'))
    # There is no state_dict with checkpoint
    #model.load_state_dict(model_dict['state_dict'])
    model.module.load_state_dict(model_dict)
    model = model.eval()
    return model

@hc.func("c")
def SqueezeNet (ctxt) :
    model = getModel()
    model = model.type(torch.double)
    for p in model.parameters():
        p.requires_grad = False
    input_var = np.empty((1), dtype= object)
    input_var[0] = ctxt

    def act (x) : 
        return HE_SiLU(x)
        # return HE_ReLU(x)
    def pooling (close, x):
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
    print("Conv_1")        
    conv_1_shapes = CascadeConv(initial_shapes, model.module.conv_1.Conv2d)
    close = shapeClosure(**conv_1_shapes)
    out = HE_ConvBN(close, input_var,model.module.conv_1.Conv2d, model.module.conv_1.bn)
    out = act (out)
    out = hc.bootstrap(out)
    block_in = conv_1_shapes
 
    print("avgpool_1")
    avgpool_1_shapes = CascadeMax (block_in, model.module.avgpool_1)
    close = shapeClosure(**avgpool_1_shapes)
    out = pooling(close, out)
    block_in = avgpool_1_shapes

    print("fire_2")
    fire_2_squeeze_shapes = CascadeConv(block_in, model.module.fire_2.squeeze.Conv2d)
    close = shapeClosure(**fire_2_squeeze_shapes)
    out = hc.bootstrap(out)
    out = HE_ConvBN(close, out, model.module.fire_2.squeeze.Conv2d, model.module.fire_2.squeeze.bn)
    out = act (out)
    out = hc.bootstrap(out)
    block_in = fire_2_squeeze_shapes

    fire_2_expand1x1_shapes = CascadeConv(block_in, model.module.fire_2.expand1x1)
    close = shapeClosure(**fire_2_expand1x1_shapes)
    out1 = HE_Conv(close, out, model.module.fire_2.expand1x1)

    fire_2_expand3x3_shapes = CascadeConv(block_in, model.module.fire_2.expand3x3)
    close = shapeClosure(**fire_2_expand3x3_shapes)
    out2 = HE_Conv(close, out, model.module.fire_2.expand3x3)
    
    ##############concat################
#     ori = torch.cat([], dim=1)
    block_in = CascadeConcat(fire_2_expand1x1_shapes, fire_2_expand3x3_shapes)
    close = shapeClosure(**block_in)
    out = HE_Concat(close, out1, out2)
    ###################################

    print("fire_3")
    fire_3_squeeze_shapes = CascadeConv(block_in, model.module.fire_3.squeeze.Conv2d)
    close = shapeClosure(**fire_3_squeeze_shapes)
    out = HE_ConvBN(close, out, model.module.fire_3.squeeze.Conv2d, model.module.fire_3.squeeze.bn)
    out = hc.bootstrap(out)
    out = act (out)
    out = hc.bootstrap(out)
    block_in = fire_3_squeeze_shapes

    fire_3_expand1x1_shapes = CascadeConv(block_in, model.module.fire_3.expand1x1)
    close = shapeClosure(**fire_3_expand1x1_shapes)
    out1 = HE_Conv(close, out, model.module.fire_3.expand1x1)

    fire_3_expand3x3_shapes = CascadeConv(block_in, model.module.fire_3.expand3x3)
    close = shapeClosure(**fire_3_expand3x3_shapes)
    out2 = HE_Conv(close, out, model.module.fire_3.expand3x3)
    ##############concat################
    block_in = CascadeConcat(fire_3_expand1x1_shapes, fire_3_expand3x3_shapes)
    close = shapeClosure(**block_in)
    out = HE_Concat(close, out1, out2)
    ###################################

    print("fire_4")
    fire_4_squeeze_shapes = CascadeConv(block_in, model.module.fire_4.squeeze.Conv2d)
    close = shapeClosure(**fire_4_squeeze_shapes)
    out = HE_ConvBN(close, out, model.module.fire_4.squeeze.Conv2d, model.module.fire_4.squeeze.bn)
    out = hc.bootstrap(out)
    out = act(out)
    out = hc.bootstrap(out)
    print ("additional")
    block_in = fire_4_squeeze_shapes
    

    fire_4_expand1x1_shapes = CascadeConv(block_in, model.module.fire_4.expand1x1)
    close = shapeClosure(**fire_4_expand1x1_shapes)
    out1 = HE_Conv(close, out, model.module.fire_4.expand1x1)

    fire_4_expand3x3_shapes = CascadeConv(block_in, model.module.fire_4.expand3x3)
    close = shapeClosure(**fire_4_expand3x3_shapes)
    out2 = HE_Conv(close, out, model.module.fire_4.expand3x3)
    ##############concat################
    block_in = CascadeConcat(fire_4_expand1x1_shapes, fire_4_expand3x3_shapes)
    close = shapeClosure(**block_in)
    out = HE_Concat(close, out1, out2)
    ##################################    
    
 
    print("avgpool_4")
    avgpool_4_shapes = CascadeMax (block_in, model.module.avgpool_4)
    close = shapeClosure(**avgpool_4_shapes)
    out = pooling(close, out)
    out = hc.bootstrap(out)
    block_in = avgpool_4_shapes

    print("fire_5")
    fire_5_squeeze_shapes = CascadeConv(block_in, model.module.fire_5.squeeze.Conv2d)
    close = shapeClosure(**fire_5_squeeze_shapes)
    out = HE_ConvBN(close, out, model.module.fire_5.squeeze.Conv2d, model.module.fire_5.squeeze.bn)
    out = act(out)
    out = hc.bootstrap(out)
    block_in = fire_5_squeeze_shapes

    fire_5_expand1x1_shapes = CascadeConv(block_in, model.module.fire_5.expand1x1)
    close = shapeClosure(**fire_5_expand1x1_shapes)
    out1 = HE_Conv(close, out, model.module.fire_5.expand1x1)

    fire_5_expand3x3_shapes = CascadeConv(block_in, model.module.fire_5.expand3x3)
    close = shapeClosure(**fire_5_expand3x3_shapes)
    out2 = HE_Conv(close, out, model.module.fire_5.expand3x3)
    ##############concat################
    block_in = CascadeConcat(fire_5_expand1x1_shapes, fire_5_expand3x3_shapes)
    close = shapeClosure(**block_in)
    out = HE_Concat(close, out1, out2)
    ###################################
    print("fire_6")
    fire_6_squeeze_shapes = CascadeConv(block_in, model.module.fire_6.squeeze.Conv2d)
    close = shapeClosure(**fire_6_squeeze_shapes)
    out = HE_ConvBN(close, out, model.module.fire_6.squeeze.Conv2d, model.module.fire_6.squeeze.bn)
    out = hc.bootstrap(out)
    out = act(out)
    out = hc.bootstrap(out)
    block_in = fire_6_squeeze_shapes

    fire_6_expand1x1_shapes = CascadeConv(block_in, model.module.fire_6.expand1x1)
    close = shapeClosure(**fire_6_expand1x1_shapes)
    out1 = HE_Conv(close, out, model.module.fire_6.expand1x1)

    fire_6_expand3x3_shapes = CascadeConv(block_in, model.module.fire_6.expand3x3)
    close = shapeClosure(**fire_6_expand3x3_shapes)
    out2 = HE_Conv(close, out, model.module.fire_6.expand3x3)
    ##############concat################

    block_in = CascadeConcat(fire_6_expand1x1_shapes, fire_6_expand3x3_shapes)
    close = shapeClosure(**block_in)
    out = HE_Concat(close, out1, out2)
    ###################################
    
    
    print("fire_7")
    fire_7_squeeze_shapes = CascadeConv(block_in, model.module.fire_7.squeeze.Conv2d)
    close = shapeClosure(**fire_7_squeeze_shapes)
    out = HE_ConvBN(close, out, model.module.fire_7.squeeze.Conv2d, model.module.fire_7.squeeze.bn)
    out = hc.bootstrap(out)
    out = act(out)
    out = hc.bootstrap(out)
    block_in = fire_7_squeeze_shapes

    fire_7_expand1x1_shapes = CascadeConv(block_in, model.module.fire_7.expand1x1)
    close = shapeClosure(**fire_7_expand1x1_shapes)
    out1 = HE_Conv(close, out, model.module.fire_7.expand1x1)

    fire_7_expand3x3_shapes = CascadeConv(block_in, model.module.fire_7.expand3x3)
    close = shapeClosure(**fire_7_expand3x3_shapes)
    out2 = HE_Conv(close, out, model.module.fire_7.expand3x3)
    ##############concat################
    block_in = CascadeConcat(fire_7_expand1x1_shapes, fire_7_expand3x3_shapes)
    close = shapeClosure(**block_in)
    out = HE_Concat(close, out1, out2)
    ###################################
    
    print("fire_8")
    fire_8_squeeze_shapes = CascadeConv(block_in, model.module.fire_8.squeeze.Conv2d)
    close = shapeClosure(**fire_8_squeeze_shapes)
    out = HE_ConvBN(close, out, model.module.fire_8.squeeze.Conv2d, model.module.fire_8.squeeze.bn)
    out = hc.bootstrap(out)
    out = act(out)
    out = hc.bootstrap(out)
    block_in = fire_8_squeeze_shapes

    fire_8_expand1x1_shapes = CascadeConv(block_in, model.module.fire_8.expand1x1)
    close = shapeClosure(**fire_8_expand1x1_shapes)
    out1 = HE_Conv(close, out, model.module.fire_8.expand1x1)

    fire_8_expand3x3_shapes = CascadeConv(block_in, model.module.fire_8.expand3x3)
    close = shapeClosure(**fire_8_expand3x3_shapes)
    out2 = HE_Conv(close, out, model.module.fire_8.expand3x3)
    ##############concat################
    block_in = CascadeConcat(fire_8_expand1x1_shapes, fire_8_expand3x3_shapes)
    close = shapeClosure(**block_in)
    out = HE_Concat(close, out1, out2)
    ###################################
    
    print("avgpool_8")
    avgpool_8_shapes = CascadeMax (block_in, model.module.avgpool_8)
    close = shapeClosure(**avgpool_8_shapes)
    out = pooling(close, out)
    out = hc.bootstrap(out)
    block_in = avgpool_8_shapes

    print("fire_9")
    fire_9_squeeze_shapes = CascadeConv(block_in, model.module.fire_9.squeeze.Conv2d)
    close = shapeClosure(**fire_9_squeeze_shapes)
    out = HE_ConvBN(close, out, model.module.fire_9.squeeze.Conv2d, model.module.fire_9.squeeze.bn)
    out = act(out)
    out = hc.bootstrap(out)
    block_in = fire_9_squeeze_shapes

    fire_9_expand1x1_shapes = CascadeConv(block_in, model.module.fire_9.expand1x1)
    close = shapeClosure(**fire_9_expand1x1_shapes)
    out1 = HE_Conv(close, out, model.module.fire_9.expand1x1)

    fire_9_expand3x3_shapes = CascadeConv(block_in, model.module.fire_9.expand3x3)
    close = shapeClosure(**fire_9_expand3x3_shapes)
    out2 = HE_Conv(close, out, model.module.fire_9.expand3x3)
    ##############concat################
    block_in = CascadeConcat(fire_9_expand1x1_shapes, fire_9_expand3x3_shapes)
    close = shapeClosure(**block_in)
    out = HE_Concat(close, out1, out2)
    ###################################
 
   
    print("Conv_10")
    conv_10_shapes = CascadeConv(block_in, model.module.conv_10.Conv2d)
    close = shapeClosure(**conv_10_shapes)
    out = HE_ConvBN(close, out, model.module.conv_10.Conv2d, model.module.conv_10.bn)
    out = hc.bootstrap(out)
    out = act(out)
    block_in = conv_10_shapes
    
    print("avgpool_10")
    avgpool_10_shapes = CascadePool (block_in)
    close = shapeClosure(**avgpool_10_shapes)
    out = HE_Pool(close, out)
    block_in = avgpool_10_shapes
 
    print("end")
    return out
    

modName = hc.save("traced", "traced")
print (modName)

