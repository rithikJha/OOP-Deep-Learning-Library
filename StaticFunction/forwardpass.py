import numpy as np
import StaticFunction.activations as act


def linear_forward(A_prev,W,b):
    m=A_prev.shape[1]
    Z=np.dot(W,A_prev)+b
    linear_cache =(A_prev,W,b)   
    return Z,linear_cache


def linear_activation_forward(A_prev,W ,b,activation):

    Z,linear_cache = linear_forward(A_prev,W,b)

    if activation.lower()=="sigmoid":
        A,activation_cache = act.sigmoid(Z)

    elif activation.lower()=="relu":
        A,activation_cache = act.relu(Z)

    elif activation.lower()=="tanh":
        A,activation_cache = act.tanh(Z)

    elif activation.lower()=="softmax":
        A,activation_cache = act.softmax(Z)

    elif activation.lower()=="linear":
        A,activation_cache = act.linear(Z)

    cache = (linear_cache,activation_cache)
    return A,cache