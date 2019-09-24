# MNIST-Naive-Bayes-and-Logistic-Regression
Self coded the Naive Bayes Classifier and Logistic Regressor for two digits with an accuracy of 71% and 83% respectively using mean and variance as features rather than using each pixel as input array.

### Introduction
MNIST dataset consists of 70,000 images of handwritten digits with almost 85 – 25% split in training and test dataset. But we are considering images only for digits 7 and 8.
Our first task is to extract the features out of these images. So each image contains 784 pixels and we need to get 2 features: mean and standard variance of each image. Then we are supposed to classify the test images using Naïve Bayes and Logistic Regression and getting an acceptable accuracy for it.

### Installation
There are 3 python file and 1 matlab file which contains the data. To run the program, you need to save all these files in the same folder so that the main.py file could import the NaiveBayes.py, LinearRegression.py and mnist.mat
