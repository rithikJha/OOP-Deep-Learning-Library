from NeuralArchitecture import *
from GradientOptimization import *
from datasets import *

np.random.seed(5)   
X = np.random.randn(1,60000)
Y = X**2 

layers_dims = [1,10,1] 
activation_list = ["relu","linear"]

nn_model = Neural_Architecture(layers_dims,activation_list,
                              cost_type="msel",
                              A_type="xavier",
                              Reg_type="l2",
                              lambd =0.05)

optim = Set_Optimization_Attribute(X, Y, nn_model,
                                  optimizer="adam",
                                  lr=0.3,
                                  num_epochs = 800,
                                  mini_batch_size = 0,
                                  beta1=0.9,
                                  beta2=0.99)

optim.optimize(print_cost_at=100,plot_cost=True)


Y_predicted = Neural_Architecture.feed_forward(nn_model,np.array([[5,4,0.2]]))
print(Y_predicted)
plt.show()



