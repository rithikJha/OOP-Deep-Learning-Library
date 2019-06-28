from NeuralArchitecture import *
from GradientOptimization import *
from datasets import *

import matplotlib.pyplot as plt


train_x, train_y, test_x, test_y, classes = load_cat_data()
'''
index = 14
plt.imshow(train_x_orig[index])
plt.show()
print ("y = " + str(train_y[0,index]) + ". It's a " + classes[train_y[0,index]].decode("utf-8") +  " picture.")
'''

'''
m_train = train_x.shape[0]
num_px = train_x.shape[1]
m_test = test_x.shape[0]

print ("Number of training examples: " + str(m_train))
print ("Number of testing examples: " + str(m_test))
print ("train_x shape: " + str(train_x.shape))
print ("train_y shape: " + str(train_y.shape))
print ("test_x shape: " + str(test_x.shape))
print ("test_y shape: " + str(test_y.shape))
'''
np.random.seed(3)
layers_dims = [train_x.shape[0],20,7,5,1]
activation_list = ["relu","relu","relu","sigmoid"]
nn_model = Neural_Architecture(layers_dims,activation_list,
                              cost_type="cel",
                              A_type="relu",
                              Reg_type="l2",
                              lambd = 0.05)

optim = Set_Optimization_Attribute(train_x, train_y,nn_model,
                                  optimizer="",
                                  lr=0.0075,
                                  num_epochs = 2500,
                                  mini_batch_size = 0,
                                  beta1=0.9,
                                  beta2=0.99)


optim.optimize(print_cost_at=1000,plot_cost=True)

Y_predicted = Neural_Architecture.feed_forward(nn_model,test_x)
p = Neural_Architecture.predict(Y_predicted,test_y)
plt.show()
