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

import sys

def poly2(x) :
    out = models.MPCB.GenPoly(Poly.treeStr2,Poly.coeffStr2,  4, scale = 1.7)(x)
    out[0] = hc.bootstrap(out[0])
    return out 

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

eps = 0.001


def nprelu(x) : 
    return np.array([ np.maximum (xx, 0) for xx in x], dtype = object)

def checkError (M, mpcb) :
    mpcb = torch.DoubleTensor(np.concatenate(mpcb).reshape(M.shape))
    maxarg = torch.max(torch.abs((M.cuda()-mpcb.cuda())))
    s = torch.argmax( (torch.abs(M.cuda()-mpcb.cuda())>eps).to(dtype=torch.int))
    ssum = torch.sum(torch.abs(M.cuda() - mpcb.cuda()))
    nsum = torch.sum(torch.abs(M.cuda()-mpcb.cuda())>eps)
    if (maxarg > eps) :
        print ("ERROR!!")
    else :
        print ("PASS")
    print (f"num : {nsum}, errsum : {ssum}, first : {s}, max : {maxarg}, valmax : {torch.max(M)}, valmin : {torch.min(M)}")

def HE_BN (close, mpp, bn, scale=1.0) :
    G, H = abstractBN(bn)
    mpcb = close["BN"](mpp, G, H)
    return mpcb
def HE_Conv (close, mpp, conv) :
    mpcb =  close["MPC"] (mpp, conv.weight, conv.bias)
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

def HE_Concat (close, mpp_1, mpp_2) :
    mpcb =  close["CC"] (mpp_1, mpp_2)
    return mpcb

def HE_DS (close, mpp) :
    mpcb = close["DS"](mpp)
    return mpcb

def HE_Pool (close , mpp) :
    return close["AP"](mpp)

def HE_Linear(close, mpp, linear, p = 1.0, scale = 1.0) :
    mpcb = Linear(mpp, linear.weight * p , linear.bias.cpu() / scale, 2**16)
    return mpcb


@hc.func("c")
def SqueezeNet (ctxt) :
    model = getModel()
    model = model.type(torch.double)
    # model = model.cpu()
    for p in model.parameters():
        p.requires_grad = False
    # input_var = input_var.type(torch.double)
    input_var = np.empty((1), dtype= object)
    input_var[0] = ctxt

    calculation = poly.GenPoly()
    def mish_s (A) :
        return A * (calculation(A)+0.5)
    def act (x) : 
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
    print("Conv_1")        
    conv_1_shapes = CascadeConv(initial_shapes, model.module.conv_1.Conv2d)
    close = shapeClosure(**conv_1_shapes)
    out = HE_ConvBN(close, input_var,model.module.conv_1.Conv2d, model.module.conv_1.bn)
    out[0] = hc.bootstrap(out[0]) # 1
    out = act (out)
    block_in = conv_1_shapes
 
    print("avgpool_1")
    avgpool_1_shapes = CascadeMax (block_in, model.module.avgpool_1)
    close = shapeClosure(**avgpool_1_shapes)
    out = HE_Avg(close, out)
    block_in = avgpool_1_shapes

    print("fire_2")
    fire_2_squeeze_shapes = CascadeConv(block_in, model.module.fire_2.squeeze.Conv2d)
    close = shapeClosure(**fire_2_squeeze_shapes)
    out[0] = hc.bootstrap(out[0]) # 2
    out = HE_ConvBN(close, out, model.module.fire_2.squeeze.Conv2d, model.module.fire_2.squeeze.bn)
    out = act (out)
    out[0] = hc.bootstrap(out[0]) # 3
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
    out[0] = hc.bootstrap(out[0]) # 4
    out = act (out)
    out[0] = hc.bootstrap(out[0]) # 5
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
    out[0] = hc.bootstrap(out[0]) # 6 - 3177
    out = act(out)
    out[0] = hc.bootstrap(out[0]) # 7
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
    out = HE_Avg(close, out)
    out[0] = hc.bootstrap(out[0]) # 8 - 5526
    #ori = ori.view(ori.size(0), -1)
    block_in = avgpool_4_shapes

    print("fire_5")
    fire_5_squeeze_shapes = CascadeConv(block_in, model.module.fire_5.squeeze.Conv2d)
    close = shapeClosure(**fire_5_squeeze_shapes)
    out = HE_ConvBN(close, out, model.module.fire_5.squeeze.Conv2d, model.module.fire_5.squeeze.bn)
    #out[0] = hc.bootstrap(out[0]) # ??
    out = act(out)
    out[0] = hc.bootstrap(out[0]) # 9 - 5965
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
    out[0] = hc.bootstrap(out[0]) # 10 - 7521
    out = act(out)
    out[0] = hc.bootstrap(out[0]) # 11
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
    out[0] = hc.bootstrap(out[0]) # 12 - 10262
    out = act(out)
    out[0] = hc.bootstrap(out[0]) # 13 - 12233
    block_in = fire_7_squeeze_shapes

    fire_7_expand1x1_shapes = CascadeConv(block_in, model.module.fire_7.expand1x1)
    close = shapeClosure(**fire_7_expand1x1_shapes)
    out1 = HE_Conv(close, out, model.module.fire_7.expand1x1)
    #out1[0] = hc.bootstrap(out1[0])

    fire_7_expand3x3_shapes = CascadeConv(block_in, model.module.fire_7.expand3x3)
    close = shapeClosure(**fire_7_expand3x3_shapes)
    out2 = HE_Conv(close, out, model.module.fire_7.expand3x3)
    #out2[0] = hc.bootstrap(out2[0])
    ##############concat################
    block_in = CascadeConcat(fire_7_expand1x1_shapes, fire_7_expand3x3_shapes)
    close = shapeClosure(**block_in)
    out = HE_Concat(close, out1, out2)
    ###################################
    
    print("fire_8")
    fire_8_squeeze_shapes = CascadeConv(block_in, model.module.fire_8.squeeze.Conv2d)
    close = shapeClosure(**fire_8_squeeze_shapes)
    out = HE_ConvBN(close, out, model.module.fire_8.squeeze.Conv2d, model.module.fire_8.squeeze.bn)
    out[0] = hc.bootstrap(out[0]) # 14 - 13171
    out = act(out)
    out[0] = hc.bootstrap(out[0]) # 15
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
    #out[0] = hc.bootstrap(out[0])
    out = HE_Avg(close, out)
    out[0] = hc.bootstrap(out[0]) # 16 - 16355
    #ori = ori.view(ori.size(0), -1)
    block_in = avgpool_8_shapes

    print("fire_9")
    fire_9_squeeze_shapes = CascadeConv(block_in, model.module.fire_9.squeeze.Conv2d)
    close = shapeClosure(**fire_9_squeeze_shapes)
    out = HE_ConvBN(close, out, model.module.fire_9.squeeze.Conv2d, model.module.fire_9.squeeze.bn)
    #out[0] = hc.bootstrap(out[0]) # 16794 <- Dacapo bootstrap here
    out = act(out)
    out[0] = hc.bootstrap(out[0]) # 17 - 16940
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
    #out[0] = hc.bootstrap(out[0]) # 18982
    out = HE_ConvBN(close, out, model.module.conv_10.Conv2d, model.module.conv_10.bn)
    out[0] = hc.bootstrap(out[0]) # 18 - 19077
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
