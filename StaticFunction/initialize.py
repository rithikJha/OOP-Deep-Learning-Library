import numpy as np 

def velocity_initializer(parameters):
    v={}
    L=len(parameters)//2

    for l in range(L):
        v["dW"+str(l+1)]=np.zeros_like(parameters["W"+str(l+1)])
        v["db"+str(l+1)]=np.zeros_like(parameters["b"+str(l+1)])
    return v


def speed_initializer(parameters):
    s={}
    L=len(parameters)//2

    for l in range(L):
        s["dW"+str(l+1)]=np.zeros_like(parameters["W"+str(l+1)])
        s["db"+str(l+1)]=np.zeros_like(parameters["b"+str(l+1)])
    return s    


def update_parameters_momentum(grads,v,beta,t):
    v = beta*v+(1-beta)*grads
    
    return v

def update_parameters_RMSprop(grads,s,beta,t):
    eps = 1e-8
    s = beta*s+(1-beta)*np.square(grads)
      
    return s


def update_parameters(parameters,direction,learning_rate):
    parameters = parameters- learning_rate * direction
    return parameters






    