import numpy as np 

def sigmoid(Z):
    activation_cache = Z
    A= 1/(1+np.exp(-Z))
    return A,activation_cache

def relu(Z):
    activation_cache = Z
    A = np.maximum(0,Z)
    return A,activation_cache

def tanh(Z):
    activation_cache=Z
    A=np.tanh(Z)
    return A,activation_cache

def sigmoid_backwards(dA,activation_cache):
    Z=activation_cache
    A = 1/(1+np.exp(-Z))
    dZ = dA*A*(1-A)
    return dZ

def relu_backwards(dA,activation_cache):
    Z = activation_cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    assert (dZ.shape == Z.shape)
    return dZ

def tanh_backwards(dA,activation_cache):
    Z=activation_cache
    A=np.tanh(Z)
    dZ=dA*(1-A**2)
    return dZ

def softmax(Z):
    t = np.exp(Z-np.max(Z,axis=0,keepdims=True))
    A = t/(np.sum(t,axis=0,keepdims=True))
    activation_cache = Z
    return A,activation_cache


def softmax_backwards(dA,activation_cache):
    Z=activation_cache
    A,cache = softmax(Z)
    dZ=dA*np.dot(A,(1-A).T)
    return dZ

def linear(Z):
    activation_cache = Z
    A=Z
    return A,activation_cache

def linear_back(dA,activation_cache):
    dZ = dA
    return dZ

    

