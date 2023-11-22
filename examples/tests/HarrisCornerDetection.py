
import hecate as hc
import numpy as np
import simfhe as sf


def roll (a, i) :
    return np.roll(a, -i)

def preprocess():

    lena = Image.open(f'{hc.hecate_dir}/examples//data/cornertest.jpg').convert('L')
    lena = lena.resize((64,64))
    lena_array = np.asarray(lena.getdata(), dtype=np.float64) / 256
    lena_array = lena_array.reshape([64*64])

    return lena_array.reshape([1, 4096]);

def process(lena_array) :
    lena_array = lena_array[0]
    F = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    Ix = 0
    Iy = 0
    for i in range(3) : 
        for j in range(3) : 
            # rot = lena_array[i*64+j:] + lena_array[:(i*64+j)]
            rot = roll (lena_array, i*64 +j)
            h = rot * F[i][j]
            v = rot * F[j][i]
            Ix = Ix + h
            Iy = Iy + v
    Ix2 = Ix * Ix 
    Iy2 = Iy * Iy 
    IxIy = Ix * Iy
    # d = [] 
    offset = 2

    Mxx = 0
    Myy = 0
    Mxy = 0
    for i in range (3) : 
        for j in range(3):
            rot_Ix2 = roll(Ix2, i*64+j)
            rot_Iy2 = roll(Iy2, i*64+j)
            rot_IxIy = roll(IxIy, i*64+j)
            Mxx = Mxx + rot_Ix2 
            Myy = Myy + rot_Iy2 
            Mxy = Mxy + rot_IxIy 
            
    det = Mxx * Myy - Mxy * Mxy 
    trace = Mxx + Myy
            
    d = det - 0.1 * trace * trace

    return d.reshape([1, 4096])

def postprocess (result) : 
    # return (result *256) [:, :4096]
    return (result) [:, :4096]


## EVAL

if __name__ == "__main__" :

    from random import *
    import sys
    from pathlib import Path
    import time 
    from PIL import Image

    a_compile_type = sys.argv[1]
    a_compile_opt = int(sys.argv[2])
    stem = Path(__file__).stem
    print(sf.simulate(f"optimized/{a_compile_type}/{stem}.{a_compile_opt}._hecate_{stem}.hevm"))
    hevm = hc.HEVM()
    stem = Path(__file__).stem
    hevm.load (f"traced/_hecate_{stem}.cst", f"optimized/{a_compile_type}/{stem}.{a_compile_opt}._hecate_{stem}.hevm")

    input_dat = preprocess()
    reference = postprocess(process(input_dat))
    [hevm.setInput(i, dat) for i, dat in enumerate(input_dat)]
    timer = time.perf_counter_ns()
    hevm.run()
    timer = time.perf_counter_ns() -timer
    res = hevm.getOutput()
    res = postprocess(res)
    err = res - reference 
    rms = np.sqrt( np.sum(err*err) / res.shape[-1])
    # print (timer/ (pow(10, 9)))
    # print (rms)
    hevm.printer(timer/pow(10, 9), rms)



