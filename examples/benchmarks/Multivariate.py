import hecate as hc 
import sys

def sum_elements(data):
    for i in range(12):
        rot = data.rotate(1<<i)
        data += rot

    return data

def poly_y_predict(x0, x1, x2, weight):
    y_predict = x0 * weight[0]
    y_predict += x1 * weight[1]
    y_predict += x2 * weight[2]
    return y_predict

@hc.func("c,c,c,c,c,c")
def Multivariate (x0_data, x1_data, x2_data, y0_data, y1_data, y2_data) :
    W0 = [hc.Plain([1.0]) for i in range(3)]
    W1 = [hc.Plain([1.5]) for i in range(3)]
    W2 = [hc.Plain([2.0]) for i in range(3)]
    W = [W0, W1, W2]
    X = [x0_data, x1_data, x2_data]
    Y = [y0_data, y1_data, y2_data]
    epochs = 2
    learning_rate = hc.Plain([-0.01])

    for k in range(epochs):
        for j in range(3):
            wX = [ X[i] * W[j][i] for i in range(3)]

            y_predict = wX[0] + wX[1] + wX[2]
            mY = -Y[j]
            error0 = y_predict + mY
            error = [ error0 * X[i] for i in range(3)]
            sumerror = [sum_elements(error[i]) for i in range(3)]

            gradW = [sumerror[i] * hc.Plain([1/2048]) for i in range(3)]
            Wup = [gradW[i] * learning_rate for i in range(3)]
            for i in range(3) :
                W[j][i] += Wup[i]
    
    return W[0][0], W[0][1], W[0][2],W[1][0], W[1][1], W[1][2],W[2][0], W[2][1], W[2][2],

modName = hc.save("traced", "traced")
print (modName)
