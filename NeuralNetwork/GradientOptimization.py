import StaticFunction.initialize as ini 
import numpy as np 
import StaticFunction.mini_batch_generator as mbg 
import matplotlib.pyplot as plt

class Set_Optimization_Attribute:
    def __init__(self,X,Y, nn_model,mini_batch_size = 0,beta1=0.9,beta2=0.999,lr=0.0007,optimizer="",num_epochs=10000 ):
        
        self.input_layer = X
        self.output_layer = Y
        self.nn_model = nn_model
        self.mini_batch_size = mini_batch_size
        self.beta1 = beta1
        self.beta2 = beta2
        self.learning_rate = lr
        self.optimizer = optimizer
        self.num_epochs = num_epochs

    
    def update_parameters_with_optimizer(self,v={},s={},t=0):
        L=len(self.nn_model.parameters)//2
        direction={}
        eps=1e-10
        v_corrected = {}                         # Initializing first moment estimate, python dictionary
        s_corrected = {} 
        for l in range(L):
            
            if self.optimizer.lower()=="momentum":
                v["dW"+str(l+1)] = ini.update_parameters_momentum(self.nn_model.grads["dW"+str(l+1)],v["dW"+str(l+1)],self.beta1,t)
                v["db"+str(l+1)] = ini.update_parameters_momentum(self.nn_model.grads["db"+str(l+1)],v["db"+str(l+1)],self.beta1,t)
                direction = v

            elif self.optimizer.lower()=="":
                direction=self.nn_model.grads

            elif self.optimizer.lower() == "adam":
                v["dW" + str(l+1)] = ini.update_parameters_momentum(self.nn_model.grads["dW"+str(l+1)],v["dW"+str(l+1)],self.beta1,t)
                v["db" + str(l+1)] = ini.update_parameters_momentum(self.nn_model.grads["db"+str(l+1)],v["db"+str(l+1)],self.beta1,t)
                v_corrected["dW" + str(l+1)] = v["dW" + str(l+1)]/(1-self.beta1**t)
                v_corrected["db" + str(l+1)] = v["db" + str(l+1)]/(1-self.beta1**t)
                s["dW" + str(l+1)] = ini.update_parameters_RMSprop(self.nn_model.grads["dW"+str(l+1)],s["dW"+str(l+1)],self.beta2,t)
                s["db" + str(l+1)] = ini.update_parameters_RMSprop(self.nn_model.grads["db"+str(l+1)],s["db"+str(l+1)],self.beta2,t)
                s_corrected["dW" + str(l+1)] = s["dW" + str(l+1)]/(1-self.beta2**t)
                s_corrected["db" + str(l+1)] = s["db" + str(l+1)]/(1-self.beta2**t)
                direction["dW"+str(l+1)]=(v_corrected["dW" + str(l+1)]/(np.sqrt(s_corrected["dW" + str(l+1)])+eps))
                direction["db"+str(l+1)] =(v_corrected["db" + str(l+1)]/(np.sqrt(s_corrected["db" + str(l+1)])+eps))

            elif self.optimizer.lower() == "rmsprop":
                s["dW" + str(l+1)] = ini.update_parameters_RMSprop(self.nn_model.grads["dW"+str(l+1)],s["dW"+str(l+1)],self.beta2,t)
                s["db" + str(l+1)] = ini.update_parameters_RMSprop(self.nn_model.grads["db"+str(l+1)],s["db"+str(l+1)],self.beta2,t)
                direction["dW"+str(l+1)]=(self.nn_model.grads["dW" + str(l+1)]/(np.sqrt(s["dW" + str(l+1)])+eps))
                direction["db"+str(l+1)] =(self.nn_model.grads["db" + str(l+1)]/(np.sqrt(s["db" + str(l+1)])+eps))          

            self.nn_model.parameters["W"+str(l+1)]=ini.update_parameters(self.nn_model.parameters["W"+str(l+1)],direction["dW"+str(l+1)],self.learning_rate)
            self.nn_model.parameters["b"+str(l+1)]=ini.update_parameters(self.nn_model.parameters["b"+str(l+1)],direction["db"+str(l+1)],self.learning_rate)
            
        return self.nn_model.parameters,v,s



    def optimize(self,print_cost_at = 0,plot_cost = False):
        L = len(self.nn_model.layers_dims)             # number of layers in the neural networks
        costs = []                       # to keep track of the cost
        t = 0                            # initializing the counter required for Adam update
        seed = 10                        # For grading purposes, so that your "random" minibatches are the same as ours

        if self.mini_batch_size == 0:
            self.mini_batch_size=self.input_layer.shape[1]
        
        
        # Initialize parameters
        v={}
        s={}

        if self.optimizer.lower()=="adam":
            v=ini.velocity_initializer(self.nn_model.parameters)
            s=ini.speed_initializer(self.nn_model.parameters)   

        if self.optimizer.lower()=="rmsprop":
            s=ini.speed_initializer(self.nn_model.parameters)

        if self.optimizer.lower()=="momentum":
            v=ini.velocity_initializer(self.nn_model.parameters)

      

        if self.mini_batch_size != self.input_layer.shape[1]:
            for i in range(self.num_epochs):                                                                                                         # Optimization loop
                # Define the random minibatches. We increment the seed to reshuffle differently the dataset after each epoch
                seed = seed + 1
                minibatches = mbg.random_mini_batches(self.input_layer, self.output_layer, self.mini_batch_size, seed)
                for minibatch in minibatches:
                    (minibatch_X, minibatch_Y) = minibatch                                                                                      # Select a minibatch
                    self.nn_model.prediction = self.nn_model.feed_forward(minibatch_X)                                               # Forward propagation
                    cost = self.nn_model.compute_cost(minibatch_Y)                                               # Compute cost
                    self.nn_model.grads = self.nn_model.feed_backward(minibatch_Y)              # Backward propagation
                    t = t + 1                                                                                                                   # Adam counter
                    self.nn_model.parameters, v, s = self.update_parameters_with_optimizer( v, s, t)                                  #Update Parameter

                # Print the cost every 1000 epoch
                if  i % print_cost_at == 0:
                    print("Cost after epoch %i: %f" % (i, cost))
                if  i % 10 == 0:
                    costs.append(cost)


        elif self.mini_batch_size == self.input_layer.shape[1]:
            for i in range(self.num_epochs):
                self.nn_model.prediction = self.nn_model.feed_forward(self.input_layer)                                              # Forward Propagation
                cost = self.nn_model.compute_cost(self.output_layer)                                            # Compute Cost
                self.nn_model.grads = self.nn_model.feed_backward(self.output_layer)              # Backward Propagation
                t = t + 1
                self.nn_model.parameters, v, s = self.update_parameters_with_optimizer( v, s, t)               # Update Parameters
                if print_cost_at != 0:
                    if  i % print_cost_at == 0:
                        print("Cost after epoch %i: %f" % (i, cost))
                if  i % 10 == 0:
                    costs.append(cost)


        # plot the cost
        if plot_cost:
            plt.figure(2)
            plt.plot(costs)
            plt.ylabel('cost')
            plt.xlabel('epochs (per %i)'%(print_cost_at/10))
            plt.title("Learning rate = " + str(self.learning_rate))
            

        return self.nn_model.parameters   