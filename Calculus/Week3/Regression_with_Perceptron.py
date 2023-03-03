import numpy as np
import pandas as pd
"""
One can describe a simple linear regression model as

𝑦̂ =𝑤𝑥+𝑏,(1)

where 𝑦̂ 
is a prediction of dependent variable 𝑦 based on independent variable 𝑥 using a line equation with the slope 𝑤 and intercept 𝑏

.

Given a set of training data points (𝑥1,𝑦1), ..., (𝑥𝑚,𝑦𝑚), you will find the "best" fitting line - such 
parameters 𝑤 and 𝑏 that the differences between original  values 𝑦𝑖 and predicted values 𝑦̂ 𝑖=𝑤𝑥𝑖+𝑏 are minimum.

Neural Network Model with a Single Perceptron and One Input Node

The simplest neural network model that describes the above problem can be realized by using one perceptron. 
The input and output layers will have one node each (𝑥 for input and 𝑦̂ =𝑧 for output)

Weight (𝑤) and bias (𝑏

) are the parameters that will get updated when you train the model. They are initialized to some random values or set to 0 and updated as the training progresses.

For each training example 𝑥(𝑖)
, the prediction 𝑦̂ (𝑖)

can be calculated as:

𝑧(𝑖)𝑦̂ (𝑖)=𝑤𝑥(𝑖)+𝑏,=𝑧(𝑖),(2)

where 𝑖=1,…,𝑚
You can organise all training examples as a vector 𝑋
of size (1×𝑚) and perform scalar multiplication of 𝑋 (1×𝑚) by a scalar 𝑤, adding 𝑏, which will be broadcasted to a vector of size (1×𝑚):

𝑍𝑌̂ =𝑤𝑋+𝑏,=𝑍,(3)

This set of calculations is called forward propagation.
"""

relative_path = "Calculus/Week3/data/house_prices_train.csv"

df=pd.read_csv(relative_path)

def layer_size(X,Y):
    """
    Arguments:
    X -- input dataset of shape (input size, number of examples)
    Y -- labels of shape (output size, number of examples)

    Returns:
    n_x -- the size of the input layer
    n_y -- the size of the output layer
    """
    n_x=X.shape[0]
    n_y=Y.shape[0]
    return (n_x,n_y)

def initialize_params(n_x,n_y):
    """
    Returns:
    params -- python dictionary containing your parameters:
                    W -- weight matrix of shape (n_y, n_x)
                    b -- bias (error) value set as a vector of shape (n_y, 1)
    """


    W = np.random.randn(n_y,n_x)*0.01
    b = np.zeros((n_y,1))
    params = {'W':W,
              'b':b}

    return params
def forward_propogation(X,params):


    W = params['W']
    b = params['b']

    Z = np.matmul(W,X) + b
    Y_hat = Z

    return Y_hat
def compute_cost(Y_hat,Y):

    m = X.shape[0]
    # Compute the cost function.
    cost = np.sum((Y_hat-Y)**2) /(2*m)
    return cost

def backward_propagation(Y_hat,X,Y):

    m = X.shape[1]

    dZ = Y_hat - Y
    dW = 1/m * np.dot(dZ,X.T)
    db= 1/m * np.sum(dZ, axis=1, keepdims=True)

    grads={"dW": dW,
           "db": db}
    return grads

def nn_model(X,Y,num_iter=100,learning_rate=0.5,print_cost=False):

    n_x, n_y = layer_size(X,Y)

    parameters = initialize_params(n_x,n_y)

    for i in range(0,num_iter):

        Y_hat = forward_propogation(X, parameters)

        cost = compute_cost(Y_hat, Y)

        grades = backward_propagation(Y_hat,X,Y)

        parameters = update_params(parameters, grades, learning_rate)

        if print_cost:
            print ("Cost after iteration", i, " is: " , cost)
    return parameters



def update_params(param,grads, learning_rate=0.5):

    W = param["W"]
    b = param["b"]
    dW = grads["dW"]
    db = grads["db"]

    W = W - learning_rate*dW
    b= b- learning_rate * db

    parameters = {'W':W,
                  'b':b}
    return parameters

X_multi = df[['GrLivArea', 'OverallQual']]
Y_multi = df['SalePrice']

X_multi_norm = (X_multi - np.mean(X_multi))/np.std(X_multi)
Y_multi_norm = (Y_multi - np.mean(Y_multi))/np.std(Y_multi)

#Convert results to the NumPy arrays, transpose X_multi_norm to get an array of a shape (2×𝑚)
# and reshape Y_multi_norm to bring it to the shape (1×𝑚):
X_multi_norm = np.array(X_multi_norm).T
Y_multi_norm = np.array(Y_multi_norm).reshape((1, len(Y_multi_norm)))


parameters_multi = nn_model(X_multi_norm, Y_multi_norm, num_iter=100, print_cost=True)

