
from NeuralArchitecture import *
from GradientOptimization import *
from datasets import *

np.random.seed(3)

train_X, train_Y, test_X, test_Y = load_dataset(2)

print(train_X.shape)
plt.figure(1)
plt.scatter(train_X[0, :], train_X[1, :], c=train_Y.ravel(), s=40, cmap=plt.cm.Spectral)


layers_dims = [train_X.shape[0], 20,10, 1]
activation_list = ["relu","relu","sigmoid"]



nn_model = Neural_Architecture(layers_dims,activation_list,
                              cost_type="cel",
                              A_type="relu",
                              Reg_type="l2",
                              lambd =0.6)

optim = Set_Optimization_Attribute(train_X, train_Y,nn_model,
                                  optimizer="momentum",
                                  lr=0.8,
                                  num_epochs = 800,
                                  mini_batch_size = 0,
                                  beta1=0.91,
                                  beta2=0.99)


optim.optimize(print_cost_at=100,plot_cost=True)

Y_predicted = Neural_Architecture.feed_forward(nn_model,test_X)>0.5
p = Neural_Architecture.predict(Y_predicted,test_Y,"test")
Y_predicted2 = Neural_Architecture.feed_forward(nn_model,train_X)>0.5
p = Neural_Architecture.predict(Y_predicted2,train_Y,"train")


plot_decision_boundary(lambda x: predict_dec(nn_model,x.T), train_X, train_Y)
plt.show()


