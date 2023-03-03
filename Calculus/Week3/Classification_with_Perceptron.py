import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors


def layer_sizes(X,Y):
    """
    Arguments:
    X -- input dataset of shape (input size, number of examples)
    Y -- labels of shape (output size, number of examples)

    Returns:
    n_x -- the size of the input layer
    n_y -- the size of the output layer
    """
    n_x = X.shape[0]
    n_y = Y.shape[0]

    return (n_x,n_y)

def initialize_params(n_x,n_y):
    """
    Returns:
    params -- python dictionary containing your parameters:
                    W -- weight matrix of shape (n_y, n_x)
                    b -- bias value set as a vector of shape (n_y, 1)
    """
    W = np.random.randn(n_y,n_x)*0.01
    b = np.zeros((n_y,1))
    params = {'W' : W,
            'b' : b}
    return params

def sigmoid(z):
    return 1/(1+np.exp(-z))

def update_params (params, grad, learning_rate= 0.5):


    # Retrieve each parameter from the dictionary "params".
    W = params[ "W" ]
    b = params[ "b" ]

    # Retrieve each gradient from the dictionary "grads".
    dW = grad[ "dW" ]
    db = grad[ "db" ]

    # Update rule for each parameter.
    W = W - learning_rate * dW
    b = b - learning_rate * db

    parameters = {"W": W,
                  "b": b}

    return parameters

def forward_propogation(X,grad):
    W = grad['W']
    b = grad['b']

    Z = np.matmul(W,X) + b
    A = sigmoid(Z)
    return A

def backward_propogation(A,Y,X):

    m = Y.shape[1]
    dZ = A - Y
    dW = 1/m * np.dot(dZ,X.T)
    dB = 1/m * np.sum(dZ,keepdims=True)
    grad = {'dW': dW,
            'db': dB}
    return grad

def compute_cost(A,Y):

    m = Y.shape[1]

    erg = np.multiply(Y,np.log(A)) + np.multiply(np.log(1 - A),1-Y)

    cost = -1/m*np.sum(erg)

    return cost

def nn_perceptron (X,Y, learning_rate = 0.5, num_iter = 100, print_cost = True):

    n_x, n_y = layer_sizes(X,Y)

    params = initialize_params(n_x, n_y)


    for i in range(0, num_iter):


        A = forward_propogation(X, params)
        grad = backward_propogation(A,Y,X)
        cost = compute_cost(A, Y)

        params = update_params (params, grad, learning_rate= 0.5)

        if print_cost:
            print("Cost after iteration %i: %f" %(i, cost))
    return params


def predict(X,parameters):
    """
    Using the learned parameters, predicts a class for each example in X

    Arguments:
    parameters -- python dictionary containing your parameters
    X -- input data of size (n_x, m)

    Returns
    predictions -- vector of predictions of our model (blue: False / red: True)
    """

    # Computes probabilities using forward propagation, and classifies to 0/1 using 0.5 as the threshold.
    A = forward_propogation (X,parameters )
    predictions = A > 0.5

    return predictions


def plot_decision_boundary(X,Y,parameters):
    W = parameters[ "W" ]
    b = parameters[ "b" ]

    fig,ax = plt.subplots ()
    plt.scatter ( X[ 0,: ],X[ 1,: ],c=Y,cmap=colors.ListedColormap ( [ 'blue','red' ] ) );

    x_line = np.arange ( np.min ( X[ 0,: ] ),np.max ( X[ 0,: ] ) * 1.1,0.1 )
    ax.plot ( x_line,- W[ 0,0 ] / W[ 0,1 ] * x_line + -b[ 0,0 ] / W[ 0,1 ],color="black" )
    plt.plot ()
    plt.show ()




##### Simple Classification Example with Perceptron Algorithm #####

"""
n=30
np.random.seed(3)
X_ = np.random.randint(0,2, (2,n))
Y_ = np.logical_and(X[0]==0, X[1]==1).astype(int).reshape((1, n))

print('Training dataset X containing (x1, x2) coordinates in the columns:')
print(X)
print('Training dataset Y containing labels of two classes (0: blue, 1: red)')
print(Y)

print ('The shape of X is: ' + str(X.shape))
print ('The shape of Y is: ' + str(Y.shape))
print ('I have n = %d training examples!' % (X.shape[1]))
"""

