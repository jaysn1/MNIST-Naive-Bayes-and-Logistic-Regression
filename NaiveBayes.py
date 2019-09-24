# Import Libraries
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.stats import multivariate_normal as mvn

# We need to seprate images by class for calculating the mean and variance for each class.
def separate_images_by_class(X,Y):                                  
    X_7 = []                                                        
    X_8 = []                                                        
    labels = Y[0]                                                   
    i = 1                                                           
    j = 1
    for l in labels:                            # Count the number ofimages in each class
        if l == 0:                              # or just the values 6265 and 5851
            i += 1                              
        if l == 1:                             
            j += 1                                              
    X_7.append(X[:i,:])                        # Append image pixel list per class and 
    X_8.append(X[i:,:])                        # return them
    return X_7[0], X_8[0]

# Get mean and variance of each image
def get_mean_var(X):
    mean =  np.zeros(2)
    var = np.zeros(2)
    mean = [np.mean(np.mean(X , axis = 0)), np.mean(np.var(X , axis = 0))]
    var = [np.var(np.mean(X , axis = 0)), np.var(np.var(X , axis = 0))]
    return mean, var

# The prior probability of each class is calculated and returned
def get_prior_probability(Y):                                       
    prior_probability_7 = float(len(Y[Y == 0])/len(Y))
    prior_probability_8 = float(len(Y[Y == 1])/len(Y))
    return prior_probability_7, prior_probability_8

# Predicts the probability of an image being in a class using the multivariate formula.
def predict(X, X_7_mean, X_7_var, prior_probability_7, X_8_mean, X_8_var, prior_probability_8):                                                      
    rows = len(X)                                                   
    columns = 2                                                     
    pred_prob = np.zeros((rows,columns))                            
    for i in range(len(X)):
        pred_prob[i,0] = (mvn.logpdf(np.array(np.mean(X[i]), np.var(X[i])), mean = X_7_mean, 
                             cov = X_7_var) + np.log(prior_probability_7))
        pred_prob[i,1] = (mvn.logpdf(np.array(np.mean(X[i]), np.var(X[i])), mean = X_8_mean, 
                             cov = X_8_var) + np.log(prior_probability_8))
    return np.argmax(pred_prob, axis = 1)

# Main function
def run():
    # Get data and seprate training and testing dataset
    data = loadmat('mnist_data.mat')                                    
    X_train = data['trX']                                               
    Y_train = data['trY']                                               
    X_test = data['tsX']
    Y_test = data['tsY'][0]
    
    # Seperate Images
    X_train_7, X_train_8 = separate_images_by_class(X_train,Y_train)    
    
    # Get mean and variance
    X_7_mean, X_7_var = get_mean_var(X_train_7)
    X_8_mean, X_8_var = get_mean_var(X_train_8)
    
    # Get prior probability
    prior_probability_7, prior_probability_8 = get_prior_probability(Y_train[0])
    
    #Predict the probabilities    
    pred_prob = predict(X_test, X_7_mean, X_7_var, prior_probability_7, X_8_mean, X_8_var, prior_probability_8)
    
    y_actu = pd.Series(Y_test)
    y_pred = pd.Series(pred_prob)
    df_confusion = pd.crosstab(y_actu, y_pred)
    return (((df_confusion[0][0]/(df_confusion[0][0] + df_confusion[0][1]))*100), (df_confusion[1][1]/(df_confusion[1][0] + df_confusion[1][1]))*100)

