
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
print("sim:", sf.simulate(f"optimized/{a_compile_type}/{stem}.{a_compile_opt}._hecate_{stem}.hevm"))

hevm = hc.HEVM()
stem = Path(__file__).stem
hevm.load (f"traced/_hecate_{stem}.cst", f"optimized/{a_compile_type}/{stem}.{a_compile_opt}._hecate_{stem}.hevm")

       
x = [ uniform (-1, 1) for a in range(4096)]
a = [0.5, 0.08, 0.004]
y = [ a[0]+a[1]*point+a[2]*point*point + uniform (-0.01, 0.01) for point in x]
W = [1.0, 1.0, 1.0]

epochs = 2
learning_rate = -0.0001

for i in range(epochs):
    error = [W[0]+W[1]*x[i]+W[2]*x[i]*x[i]-y[i] for i in range(4096)] 
    err0 = error
    err1 = [error[i] * x[i] for i in range(4096)]
    err2 = [err1[i] * x[i] for i in range(4096)]
    gradW = [ sum(err0)/2048, sum(err1)/2048, sum(err2)/2048]
    Wup = [learning_rate * gradW[i] for i in range(3)]
    W = [W[i] + Wup[i] for i in range(3)]


hevm.setInput(0, x)
hevm.setInput(1, y)
timer = time.perf_counter_ns()
hevm.run()
timer = time.perf_counter_ns() -timer
# print (timer / pow(10,9))
res = hevm.getOutput()
rms = 0
for i in range(3):
    rms = rms + pow(res[i] - W[i], 2)
rms = math.sqrt(np.mean(rms))
# print (rms)

hevm.printer(timer/pow(10, 9), rms)



