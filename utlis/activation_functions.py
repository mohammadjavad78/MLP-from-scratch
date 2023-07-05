import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(np.array(-x)))

def grad_sigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))

def tanh(x):
    return np.tanh(np.array(x))

def grad_tanh(x):
    return 1-(tanh(x)**2)

def relu(x):
    return np.maximum(np.zeros_like(x),np.array(x))

def grad_relu(x):
    return (np.array(x)>0)*1+0

def leakyrelu(x):
    return np.maximum(np.array(np.array(0.1*x)),np.array(x))

def grad_leakyrelu(x):
    return (np.array(x)>=0)*1+(np.array(x)<0)*0.1

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x),axis=1,keepdims=True)

def grad_softmax(x):
    return np.ones_like(x)

def linear(x):
    return np.array(x)

def grad_linear(x):
    return np.ones_like(x)