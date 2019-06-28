import numpy as np
import StaticFunction.activations as act


def linear_backward(dZ,linear_cache):
    A_prev,W,b=linear_cache
    m=A_prev.shape[1]
    dA_prev = np.dot(W.T,dZ)
    dW=np.dot(dZ,A_prev.T)/m
    db=np.sum(dZ,axis=1,keepdims=True)/m
    return dA_prev,dW,db

def linear_activation_backward(dA,caches,activation,zeta):
    linear_cache,activation_cache = caches
    if activation.lower()=="sigmoid":
        dZ=act.sigmoid_backwards(dA,activation_cache)
         
    elif activation.lower()=="relu":
        dZ=act.relu_backwards(dA,activation_cache)
       
    elif activation.lower()=="tanh":
        dZ=act.tanh_backwards(dA,activation_cache)
         
    elif activation.lower()=="softmax":
        dZ = zeta

    elif activation.lower()=="linear":
        dZ = act.linear_back(dA,activation_cache)

    dA_prev,dW,db=linear_backward(dZ,linear_cache)
    return dA_prev,dW,db


def regularization_term(lambd,W,m,Reg_type):
    if Reg_type == "l2":
        reg_term = lambd*W/m

    if Reg_type == "l1":
        reg_term = lambd/m

    if Reg_type == "":
        reg_term = 0

    return reg_term

def regularization_cost(lambd,parameters,m,Reg_type) -> float:
    L=len(parameters)//2
    reg_cost=0
    if Reg_type.lower() == "l2":
        for l in range(L):
            reg_cost =  reg_cost + np.sum(np.square(parameters["W"+str(l+1)]))
        reg_cost=lambd*reg_cost/(2*m) 

    if Reg_type.lower() == "l1":
        for l in range(L):
            reg_cost =  reg_cost + np.sum(parameters["W"+str(l+1)])
        reg_cost=lambd*reg_cost/(m)   

    if Reg_type == "":
        reg_cost=0 

    return reg_cost 

        

