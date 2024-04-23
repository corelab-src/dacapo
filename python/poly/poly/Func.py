import poly.MPCB as MPCB
import poly.Poly as Poly
import numpy as np
import hecate as hc
import pathlib
import os

def HE_BN (close, mpp, bn, scale=1.0) :
    G, H = MPCB.abstractBN(bn)
    mpcb = close["BN"](mpp, G, H)
    return mpcb

def HE_MPBN (mpp, bn, scale=1.0) :
    G, H = MPCB.abstractBN(bn)
    mpcb = MPCB.BN(mpp, G, H/scale, 2**16)
    return mpcb

def HE_Conv (close, mpp, conv) :
    mpcb =  close["MPC"] (mpp, conv.weight, conv.bias)
    return mpcb

def HE_ConvBN (close, mpp, conv, bn) :
    mpcb =  close["MPCB"] (mpp, conv.weight, *MPCB.abstractBN(bn))
    return mpcb

def HE_MaxPad (close, mpp) :
    def maximum (a, b):
        out = Poly.maxx(a, b)
        out = hc.bootstrap(out)
        return out
    MPCB.maximum = maximum
    mpcb =  close["MPD"] (mpp)
    return mpcb

def HE_Max (close, mpp) :
    def maximum (a, b):
        out = Poly.maxx(a, b)
        out = hc.bootstrap(out)
        return out
    MPCB.maximum = maximum
    mpcb =  close["MP"] (mpp)
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
    mpcb = MPCB.Linear(mpp, linear.weight * p , linear.bias.cpu() / scale, 2**16)
    return mpcb

def HE_ReshapeLinear(close, mpp, linear, p = 1.0, scale = 1.0, reshape = {}) :
    weight = MPCB.Reshape (linear.weight, reshape)
    mpcb = MPCB.Linear(mpp, weight * p , linear.bias.cpu() / scale, 2**16)
    return mpcb

def HE_DwConv (close, mpp, conv, bn) :
    G, H = MPCB.abstractBN(bn)
    mpcb =  close["DW"] (mpp, conv.weight, G, H+conv.bias)
    return mpcb


def HE_Concat (close, mpp_1, mpp_2) :
    mpcb =  close["CC"] (mpp_1, mpp_2)
    return mpcb

def HE_ReLU (x) : 
    def sign (x) :
        out = Poly.poly2(Poly.poly1(x))
        out[0] = hc.bootstrap(out[0])
        out = Poly.poly3(out)
        return out 
    return (0.5 + sign (x)) *  x

def HE_SiLU (x) :
    calculation = Poly.GenPoly()
    return x * (calculation(x)+0.5)



