import numpy as np  
import StaticFunction.forwardpass as fwd 
import StaticFunction.backpass as bck 

class Neural_Architecture:
    def __init__(self, layers_dims, activation_list,cost_type = "cel",A_type = "",Reg_type = "",lambd = 0.1):
        self.layers_dims = layers_dims
        self.activation_list = activation_list
        self.activation_list_cleanse()
        self.cost_type = cost_type
        self.A_type = A_type
        self.Reg_type = Reg_type
        self.lambd = lambd
        self.initialize_parameters()
        self.caches = [] 
        self.grads = {} 
                                                       # [(A_prev,W,b),Z]

    def activation_list_cleanse(self):
        unnamed =len(self.layers_dims)-len(self.activation_list)-1
        for i in range(unnamed):
            self.activation_list.append("linear")


    def initialize_parameters(self):
        np.random.seed(3)
        L =len(self.layers_dims)
        self.parameters={}
        for l in range(1,L):
            
            if self.A_type.lower()=="relu":
                self.parameters["W"+str(l)]=np.random.randn(self.layers_dims[l],self.layers_dims[l-1])*np.sqrt(2/self.layers_dims[l-1])
            
            elif self.A_type.lower()=="xavier":
                self.parameters["W"+str(l)]=np.random.randn(self.layers_dims[l],self.layers_dims[l-1])*np.sqrt(2/(self.layers_dims[l-1]*self.layers_dims[l]))
                     
            elif self.A_type.lower()=="":
                self.parameters["W"+str(l)]=np.random.randn(self.layers_dims[l],self.layers_dims[l-1])*np.sqrt(1/self.layers_dims[l-1])

            elif self.A_type.lower()=="linear":
                self.parameters["W"+str(l)]=np.zeros((self.layers_dims[l],self.layers_dims[l-1]))
            
                
            self.parameters["b"+str(l)]=np.zeros((self.layers_dims[l],1))


    def feed_forward(self,X):
        self.caches = []
        L = len(self.parameters)//2
        A_prev = X
        for l in range(L):
            A,cache = fwd.linear_activation_forward(A_prev,self.parameters["W"+str(l+1)],self.parameters["b"+str(l+1)],self.activation_list[l])
            self.caches.append(cache)
            A_prev = A 

        self.prediction = A_prev
        return self.prediction

    def compute_cost(self,Y):
        m=Y.shape[1]
        eps=1e-8
        reg_cost=bck.regularization_cost(self.lambd,self.parameters,m,self.Reg_type)
        if self.cost_type.lower()=="cel":
            cost = (-1/m)*((np.dot(Y,np.log(self.prediction+eps).T)+np.dot(1-Y,np.log(1-self.prediction+eps).T))) + reg_cost
        elif self.cost_type.lower()=="msel":
            cost = np.square(self.prediction-Y).mean()/2 +reg_cost
        elif self.cost_type.lower() == "softmax":
            cost = -1*np.sum(np.dot(Y,np.log(self.prediction).T)+eps)/m + reg_cost

        cost = np.squeeze(cost)
        return cost

    def feed_backward(self,Y):
        L=len(self.caches)
        self.grads={}
        m=Y.shape[1]
        eps=1e-8
        zeta = self.prediction-Y
        Y=Y.reshape(self.prediction.shape)
        if self.cost_type.lower()=="cel":
            dAL = -np.divide(Y,self.prediction+eps)+np.divide(1-Y,1-self.prediction+eps)
        elif self.cost_type.lower()=="msel":
            dAL = zeta
        elif self.cost_type.lower()=="softmax":
            dAL=-np.divide(Y,self.prediction+eps)
        
        dA=dAL
        for l in reversed(range(L)):
            current_cache = self.caches[l]
            reg_term =bck.regularization_term(self.lambd,self.parameters["W"+str(l+1)],m,self.Reg_type)
            dA_prev,dW_temp,db_temp = bck.linear_activation_backward(dA,current_cache,self.activation_list[l],zeta)
            self.grads["dA"+str(l)]   = dA_prev
            self.grads["dW"+str(l+1)] = dW_temp + reg_term
            self.grads["db"+str(l+1)] = db_temp
            dA=self.grads["dA"+str(l)]

        return self.grads

    @staticmethod
    def predict(X, Y,rax="test"):
        """
        This function is used to predict the results of a  n-layer neural network.
        
        Arguments:
        X -- data set of examples you would like to label
        parameters -- parameters of the trained model
        
        Returns:
        p -- predictions for the given dataset X
        """
        
        m = X.shape[1]
        p = np.zeros((1,m), dtype = np.int)

        
        # Forward propagation
        a3 = X
        
        # convert probas to 0/1 predictions
        for i in range(0, a3.shape[1]):
            if a3[0,i] > 0.5:
                p[0,i] = 1
            else:
                p[0,i] = 0

        # print results
        print("{0} Accuracy: {1}".format(rax,str(np.mean((p[0,:] == Y[0,:])))))
        
        return p