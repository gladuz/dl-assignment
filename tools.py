import numpy as np

def accuracy(ypred, y):
    pred = np.argmax(ypred, axis=1)
    correct = 0
    for i in range(pred.size):
        if y[i, pred[i]] == 1:
            correct += 1
    return correct / pred.size * 100

def check_accuracy(X, y, w0, w1, w2, forward):
    #Calculate the prediction using the forward function and the accuracy
    _, _, out = forward(X, w0, w1, w2)
    return accuracy(out, y)

def ReLU(z):
    return z * (z > 0)

#Derivative of ReLU
def dReLU(z):
    return 1. * (z > 0)

#Cost function derivative
def cost(ypred, y):
    return (ypred - y)

#MSE Loss function
def mse(ypred, ytrue):
    return np.sum(((ypred - ytrue)**2).mean(axis=1))