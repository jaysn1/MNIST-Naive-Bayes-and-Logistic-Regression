# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 02:35:18 2019

@author: ADMIN
"""
#Import NaiveBayes.py and LogisticRegression.py
import NaiveBayes as nb
import LogisticRegression as lr
ans = ""
while(ans != "3"):
    print("---------------------------------")
    print("| Enter your Choice:            |")
    print("|                               |")
    print("| 1. Naive Bayes                |")
    print("| 2. Logistic Regression        |")
    print("| 3. Exit                       |")
    print("---------------------------------")
    
    ans=input("""What would you like to do?
    """) 
    if ans=="1": 
        accuracy_nb_7, accuracy_nb_8 = nb.run()
        print("The accuracy for class 7 Naive Bayes is:{0}" .format(accuracy_nb_7))
        print("The accuracy for class 8 Naive Bayes is: {0}" .format(accuracy_nb_8))
        print("")
    elif ans=="2":
        accuracy_lr_7, accuracy_lr_8  = lr.run()
        print("The accuracy for class 7 Logistic Regression is: {0}" .format(accuracy_lr_7))
        print("The accuracy for class 8 Logistic Regression is: {0}" .format(accuracy_lr_8))
        print("")
