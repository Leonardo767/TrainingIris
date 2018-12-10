# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 15:10:41 2018

@author: Leonardo
"""

import moduleIris as mI

k = 6
filename = 'iris.txt'
accuracyHist = []

for kval in range(k):
    xTrain, yTrain, xTest, yTest = mI.CrossValidateLoad(filename, k, kval)
    # Model
    # ----------------
    # initialize
    features = xTrain.shape[0]
    c = yTrain.shape[0]
    dims = [features, 5, c]
    parameters = mI.initialize(dims)  # creates W1 and b1
    
    # feed forward
    epochs = 10000
    costHistory = []
    
    for i in range(epochs):
        Z1 = mI.forward(xTrain, parameters["W1"], parameters["b1"])
        A1 = mI.activation(Z1, 'sigmoid')
        Z2 = mI.forward(A1, parameters["W2"], parameters["b2"])
        A2 = mI.softmax(Z2)
        
        # cost fcn
        J = mI.cost(A2, yTrain)
        costHistory.append(J)
        
        # backprop
        cache = {"Z1":Z1, "A1":A1, "Z2":Z2, "A2":A2}  # storing all the layer outputs for grad calcs
        grads = mI.backprop(parameters, cache, 'sigmoid', xTrain, yTrain)
        
        # update
        parameters = mI.update(parameters, grads, learn_rate=0.05)
    
    
#    plt.plot(costHistory)
    
#    accuracyTrain = mI.predict(parameters, 'sigmoid', xTrain, yTrain, c)
    accuracyTest = mI.predict(parameters, 'sigmoid', xTest, yTest, c)
    accuracyHist.append(accuracyTest)
    
print(sum(accuracyHist)/len(accuracyHist))








