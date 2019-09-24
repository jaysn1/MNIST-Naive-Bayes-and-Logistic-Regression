# Importing Libraries        
from scipy.io import loadmat
import numpy as np
import scipy.optimize as opt
import pandas as pd

# Sigmoid function for outputing either 0 or 1
def sigmoid(t):
    return 1/(1+np.exp(-t))

# Calculating the cost for prediction
def calculate_cost(theta, X, y, lmbda):
    m = len(y)
    return ((((y*np.log(sigmoid(np.dot(X,theta)))) + ((1 - y)*np.log(1 - sigmoid(np.dot(X,theta)))))/(-1)).mean() 
                + np.sum(theta[1:]**2 ) * lmbda / (2*m))

# Applying Gradient Descent on the sigmoid function
def grad_descent(theta, X, y, lmbda):
    m = len(y)
    temp = (np.dot(X.T, (sigmoid(np.dot(X, theta)) - y)))/m 
    temp = temp - (lmbda*theta)/m
    return temp

# Main method to load the data and calculate the final parameter theta
def run():
    # Get data and seprate training and testing dataset
    data = loadmat('mnist_data.mat')
    X_train = data['trX']   
    Y_train = data['trY'][0]                       
    X_test = data['tsX']
    Y_test = data['tsY'][0] 

    # Seperate Images basedon class
    x_test = np.zeros((2002,2))
    x_test[:,0] = np.mean(X_test , axis = 1)
    x_test[:,1] = np.var(X_test , axis = 1)                                                                  
    
    # Extract the mean and variance as features
    X = np.zeros((12116,2))
    X[:,0] = np.mean(X_train , axis = 1)
    X[:,1] = np.var(X_train , axis = 1)
    rows = 2
    lmbda = 0.1
    columns = 2
    theta = np.zeros((rows,columns)) #inital parameters 
    
    #Optimize the costfunction
    for i in range(columns):
        digit_class = i
        theta[i] = opt.fmin_cg(f = calculate_cost, x0 = theta[i],  fprime = grad_descent, args = (X, (Y_train == digit_class).flatten(), lmbda), maxiter = 50, disp = False)
    
    #Predict the class from testing data based on parameter theta
    pred = np.argmax(x_test @ theta.T, axis = 1)
    y_actu = pd.Series(Y_test)
    y_pred = pd.Series(pred)
    df_confusion = pd.crosstab(y_actu, y_pred)
    return ((df_confusion[0][0]/(df_confusion[0][0] + df_confusion[0][1]))*100, (df_confusion[1][1]/(df_confusion[1][0] + df_confusion[1][1]))*100)
