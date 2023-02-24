import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

relative_path = "Calculus/Week 2/data/tvmarketing.csv"

df=pd.read_csv(relative_path)

df.plot(x='TV',y='Sales', kind = 'scatter', c='red')

X=df['TV']
Y=df['Sales']

m_numpy, b_numpy = np.polyfit(X,Y,1)

print(f'Linear regression with Numpy. Slope: {m_numpy}. Intercept: {b_numpy}')

# This is organised as a function only for grading purposes.
def pred_numpy(m,b,X):

    Y=m*X+b

    return Y

X_pred = np.array([50, 120, 280])
Y_pred_numpy = pred_numpy(m_numpy, b_numpy, X_pred)

print(f"TV marketing expenses:\n{X_pred}")
print(f"Predictions of sales using NumPy linear regression:\n{Y_pred_numpy}")

lr_sklearn = LinearRegression()

print(f"Shape of X array: {X.shape}")
print(f"Shape of Y array: {Y.shape}")

#The estimator can learn from data calling the fit function. However, trying to run the following code you
# will get an error, as the data needs to be reshaped into 2D array
try:
    lr_sklearn.fit(X,Y)
except ValueError as err:
    print(err)
#You can increase the dimension of the array by one with reshape function, or there is another another way to do it:

X_sklearn = X[:, np.newaxis]
Y_sklearn = Y[:, np.newaxis]

print(f"Shape of new X array: {X_sklearn.shape}")
print(f"Shape of new Y array: {Y_sklearn.shape}")

lr_sklearn.fit(X_sklearn, Y_sklearn)

m_sklearn = lr_sklearn.coef_
b_sklearn = lr_sklearn.intercept_

print(f"Linear regression using Scikit-Learn. Slope: {m_sklearn}. Intercept: {b_sklearn}")

def pred_sklearn(X, lr_sklearn):
 """Increase the dimension of the 𝑋 array using the function np.newaxis (see an example above) and pass the result to the lr_sklearn.predict
 function to make predictions."""
    X_2D= X[;, np.newaxis]
    Y= lr_sklearn.predict(X_2D)

    return  Y
Y_pred_sklearn = pred_sklearn(X_pred, lr_sklearn)
print(f"TV marketing expenses:\n{X_pred}")
print(f"Predictions of sales using Scikit_Learn linear regression:\n{Y_pred_sklearn.T}")

#You can plot the linear regression line and the predictions by running the following code.
# The regression line is red and the predicted points are blue.

fig, ax = plt.subplot(1,1 , figsize=(8,5))
ax.plot(X,Y, 'o', color='black')
ax.set_xlabel('TV')
ax.set_ylabel('Sales')

ax.plot(X, m_sklearn[0][0]*X+b_sklearn[0],color='red')
ax.plot(X_pred, Y_pred_sklearn, 'o', color='blue')

######Linear Regression using Gradient Descent #######

#Original arrays X and Y have different units. To make gradient descent algorithm efficient, you need to bring them to the same units.
# A common approach to it is called normalization
X_norm = (X - np.mean(X))/np.std(X)
Y_norm = (Y - np.mean(Y))/np.std(Y)

#Define cost function. Which is simple one ind variable (Adv) and one depd variable (Sales)

def E(m, b, X, Y):
    return 1/(2*len(Y))*np.sum((m*X + b - Y)**2)

def dEmd(m,b,X,Y)
    """
    Partial Derivate of the Cost function  to m
    :param m: 
    :param b: constant
    :param X: independent variable
    :param Y: dependent variable
    :return: float
    """
    res= 1/len(X)*np.dot(m*X + b - Y, X)
    return res

def dEmb(m,b,X,Y):
    """
    Partial Derivate of the Cost function to b
    :param m:
    :param b: constant
    :param X: independent variable
    :param Y: dependent variable
    :return: float
    """
    res= 1/len(X)*np.sum(m*X + b -Y)
    return res