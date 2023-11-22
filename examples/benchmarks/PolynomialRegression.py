import hecate as hc 
import sys

def sum_elements(data):
    for i in range(12):
        rot = data.rotate(1<<i)
        data += rot

    return data

def poly_y_predict(x, weight):
    y_predict = weight[0]
    y_predict += x * weight[1]
    y_predict += x * x * weight[2]
    return y_predict

@hc.func("c,c")
def PolynomialRegression(x_data, y_data) :
    W = [hc.Plain([1.0]) , hc.Plain([1.0]), hc.Plain([1.0])]
    
    epochs = 2
    learning_rate = hc.Plain([-0.0001])

    for i in range(epochs):
     
        y_predict = poly_y_predict(x_data, W)
        mY = -y_data
        error0 = y_predict + mY
        error1 = error0 * x_data
        error2 = error0 * x_data * x_data
        error = [error0, error1, error2]
        error = [error[i] * hc.Plain([1/2048]) for i in range(3)]
        gradW = [sum_elements(error[i]) for i in range(3)]
        Wup = [gradW[i] * learning_rate for i in range(3)]
        W = [W[i]+Wup[i] for i in range(3)]
    
    return W[0], W[1], W[2]

modName = hc.save("traced", "traced")
print (modName)
