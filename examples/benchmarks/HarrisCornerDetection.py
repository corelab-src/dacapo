import hecate as hc 
import sys


def sqrt(x) : 
    term1 = x * hc.Plain([2.214]) 
    x2 = x*x
    term2 = x2 * hc.Plain([-1.098])
    term3 = term1 + term2
    x2_1 = x*x
    x3 = x2_1 *x
    term4 = x3 * hc.Plain([0.173])
    term5 = term3+term4
    return  term5

def sum_elements(data):
    i = 4096
    for i in range(12):
        rot = data.rotate(1<<i)
        data += rot

    return data

def SobelFilter (image) :
    F = [[-1, 0.00001, 1], [-2, 0.00001, 2], [-1, 0.00001, 1]]
    for i in range(3) : 
        for j in range(3) : 
            rot = image.rotate(i*64+j)
            h = rot * hc.Plain([F[i][j]])
            v = rot * hc.Plain([F[j][i]])
            first = i == 0 and j == 0
            Ix = h if first else Ix + h
            Iy = v if first else Iy + v
    Ix2 = Ix*Ix
    Iy2 = Iy*Iy
    IxIy =Ix*Iy
    return Ix2, Iy2, IxIy;

@hc.func("c")
def HarrisCornerDetection (image) :
    IxIx, IyIy, IxIy = SobelFilter(image)
    for i in range(3):
        for j in range(3):
            rot_Ix2 = IxIx.rotate(i*64+j)
            rot_Iy2 = IyIy.rotate(i*64+j)
            rot_IxIy = IxIy.rotate(i*64+j)
            first = i == 0 and j == 0
            Mxx = rot_Ix2 if first else Mxx + rot_Ix2
            Myy = rot_Iy2 if first else Myy + rot_Iy2
            Mxy = rot_IxIy if first else Mxy + rot_IxIy
           
    trace = Mxx + Myy 
    det = Mxx * Myy
    mMxy = - Mxy
    mMxy = mMxy * Mxy
    det = det + mMxy
    trace2 = trace * trace
    lamda = hc.Plain([-0.1]) * trace2
    r = det + lamda
    
    return r

            
modName = hc.save("traced", "traced")
print (modName)
