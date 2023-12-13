import poly.MPCB as MPCB
import numpy as np
import pathlib
import os


package_path = os.path.abspath(os.path.dirname(__file__))

with open(package_path+"/data/treeStr.txt") as f :
    tree_var =f.readlines() 
with open(package_path+"/data/coeffStr.txt") as f :
    coeff_var = f.readlines() 
path = str(pathlib.Path(__file__).parent.resolve())+"/"
with open(package_path+"/data/sgn151527.txt") as f :
    coeffStr = f.readlines()
coeffStr1 = coeffStr[0:16]
coeffStr2 = coeffStr[16:32]
coeffStr3 = coeffStr[32:60]
with open(package_path+"/data/tree15.txt") as f : 
    treeStr1 = f.readlines()
treeStr2 = treeStr1
with open(package_path+"/data/tree27.txt") as f :
    treeStr3 = f.readlines()
treeStr3

poly1 = MPCB.GenPoly(treeStr1,coeffStr1, 4, scale = 2.0)
poly2 = MPCB.GenPoly(treeStr2,coeffStr2, 4, scale = 1.7)
poly3 = MPCB.GenPoly(treeStr3,coeffStr3, 8, scale = 2.0)

def GenPoly (degree = 16) :
    return MPCB.GenPoly(tree_var, coeff_var, degree)
def sign (x) : 
    return poly3(poly2(poly1(x)))
def relu (x) : 
    return (0.5 + sign (x)) *  x
def genRelu6 (B) : 
    return lambda x : relua(x, 6/B)

def relua (x, a) : 
    return (sign (x) *  x) + (sign (x - a) * (a-x )) + (a/2)
def maxx (a,b) :
    input_var=np.empty((1),dtype=object)
    input_var[0]=a-b
    sign_var=sign(input_var)
    return (0.5 * (a+b)) + (a-b) * sign_var[0]

def ReLU(z) :
    return np.maximum(0, z)
def rms (z) :
    return np.sqrt(np.mean(np.square(z)))


