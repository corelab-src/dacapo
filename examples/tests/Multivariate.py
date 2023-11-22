
import hecate as hc
from random import *
import numpy as np
import simfhe as sf
import sys
from pathlib import Path
import time
import math

seed(100)
a_compile_type = sys.argv[1]
a_compile_opt = int(sys.argv[2])

stem = Path(__file__).stem
print(sf.simulate(f"optimized/{a_compile_type}/{stem}.{a_compile_opt}._hecate_{stem}.hevm"))

hevm = hc.HEVM()
stem = Path(__file__).stem
hevm.load (f"traced/_hecate_{stem}.cst", f"optimized/{a_compile_type}/{stem}.{a_compile_opt}._hecate_{stem}.hevm")

x0 = [ uniform (-1, 1) for a in range(4096)]
x1 = [ uniform (-1, 1) for a in range(4096)]
x2 = [ uniform (-1, 1) for a in range(4096)]
a0 = [0.5, 0.08, 0.04]
a1 = [0.6, 0.07, 0.05]
a2 = [0.7, 0.06, 0.06]
y0 = [ a0[0]*x0[i]+a0[1]*x1[i]+a0[2]*x2[i] + uniform (-0.01, 0.01) for i in range(4096)]
y1 = [ a1[0]*x0[i]+a1[1]*x1[i]+a1[2]*x2[i] + uniform (-0.01, 0.01) for i in range(4096)]
y2 = [ a2[0]*x0[i]+a2[1]*x1[i]+a2[2]*x2[i] + uniform (-0.01, 0.01) for i in range(4096)]
X = [x0, x1, x2]
Y = [y0, y1, y2]
w0 = [1.0,1.0,1.0]
w1 = [1.5,1.5,1.5]
w2 = [2.0,2.0,2.0]
W = [w0, w1, w2]

epochs = 2
learning_rate = -0.01
itr = [ 0, 1, 2]
for k in range(epochs):
    for j in range(3):
        wX0 = [X[0][i] * W[j][0] for i in range(4096)]
        wX1 = [X[1][i] * W[j][1] for i in range(4096)]
        wX2 = [X[2][i] * W[j][2] for i in range(4096)]
        error = [wX0[i]+wX1[i]+wX2[i]-Y[j][i] for i in range(4096)] 
        err0 = [error[i] * X[0][i] for i in range(4096)]
        err1 = [error[i] * X[1][i] for i in range(4096)]
        err2 = [error[i] * X[2][i] for i in range(4096)]
        gradW = [ sum(err0)/2048, sum(err1)/2048, sum(err2)/2048]
        Wup = [learning_rate * gradW[i] for i in range(3)]
        for i in range(3):
            W[j][i] +=  Wup[i] 

hevm.setInput(0, x0)
hevm.setInput(1, x1)
hevm.setInput(2, x2)
hevm.setInput(3, y0)
hevm.setInput(4, y1)
hevm.setInput(5, y2)

timer = time.perf_counter_ns()
hevm.run()
timer = time.perf_counter_ns() -timer
# print (timer / pow(10,9))
res = hevm.getOutput()
rms = 0
for i in range(3):
    for j in range(3) : 
        rms = rms + pow(res[3*i+j] - W[i][j], 2)
rms = math.sqrt(np.mean(rms))
# print (rms)

hevm.printer(timer/pow(10, 9), rms)

