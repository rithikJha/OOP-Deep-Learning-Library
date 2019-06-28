
from NeuralNetwork.NeuralArchitecture import *
from NeuralNetwork.GradientOptimization import *
from NeuralNetwork.datasets import *

np.random.seed(3)
k=5
train_X, train_Y, test_X, test_Y = load_dataset(3,k)
mx = train_Y
tx = test_Y


plt.figure(1)
plt.scatter(test_X[0, :], test_X[1, :], c=test_Y.ravel(), s=40, cmap=plt.cm.Spectral)

def indices_to_one_hot(data, nb_classes=5):
    """Convert an iterable of indices to one-hot encoded labels."""
    targets = np.array(data,dtype=int).reshape(-1)
    return np.eye(nb_classes)[targets]

train_Y = indices_to_one_hot(train_Y,k)
test_Y = indices_to_one_hot(test_Y,k)
train_Y=train_Y.T
test_Y=test_Y.T

layers_dims = [train_X.shape[0], 6, 10, k]
activation_list = ["relu","sigmoid","softmax"]

nn_model = Neural_Architecture(layers_dims,activation_list,
                              cost_type="softmax",
                              A_type="relu",
                              Reg_type="",
                              lambd = 0.1)

optim = Set_Optimization_Attribute(train_X, train_Y,nn_model,
                                  optimizer="rmsprop",
                                  lr=0.001,
                                  num_epochs = 5000,
                                  mini_batch_size = 514,
                                  beta1=0.909,
                                  beta2=0.999)

optim.optimize(print_cost_at=1000,plot_cost=True)

Y_predicted = Neural_Architecture.feed_forward(nn_model,test_X)>0.5
Y_labels = np.argmax(Y_predicted, axis=0).reshape(1,100)
p = Neural_Architecture.predict(Y_predicted,test_Y,"test")
Y_predicted1 = Neural_Architecture.feed_forward(nn_model,train_X)>0.5
Y_labels1 = np.argmax(Y_predicted1, axis=0).reshape(1,1000)
p1 = Neural_Architecture.predict(Y_predicted1,train_Y,"train")


plot_decision_boundary(lambda x: predict_dec(nn_model,x.T,True), train_X, mx)
plt.show()
