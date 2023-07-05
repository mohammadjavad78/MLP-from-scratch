import numpy as np
from utlis import activation_functions

def Cross_entropy(yhat:np.array,y:np.array):
    yhat=activation_functions.softmax(yhat.astype(np.float64))
    return -(1/len(y))*np.sum(np.multiply(y,np.log(yhat)))

def grad_Cross_entropy(yhat:np.array,y:np.array):
    yhat=activation_functions.softmax(yhat)
    return (yhat-y)/len(y)

def MSE(yhat:np.array,y:np.array):
    return (1/len(y))*np.sum(0.5*(yhat-y)**2)

def grad_MSE(yhat:np.array,y:np.array):
    return (yhat-y)/len(y)


if __name__=="__main__":
    MSE(np.array([1,2,3]),np.array([1,2,4]))
    