# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 20:30:16 2018

@author: Leonardo
"""

"""
The following module will take the Iris dataset and convert it to a usable 
shuffled input matrix for a neural network. This module will be used later 
to construct the customizable Iris neural network model.

Output training matrix of size:
    rows = features (Sepal Width, Sepal Length, Petal Width, Petal Length)
    columns = size of data set (m)
    
Output label matrix of size:
    rows = label
    columns = size of data set (m)
"""

"""
x1, x2, x3, x4, y = np.genfromtxt('iris.txt', delimiter = ',', unpack=True)
    # assigning labels (see plot data for something more neatly packaged)
    y[0:50] = 0
    y[50:100] = 1
    y[100:150] = 2

"""


# Import packages
# ----------------
import numpy as np
import pandas as pd



# Import data
# ----------------
def convertLabelToSoftmax(labelVector, c):
    # this is analogous to tf.one_hot from tensorflow
    m = labelVector.shape[1]
    y = np.zeros((c, m))
    labelVector = np.ndarray.tolist(labelVector)
    for i in range(m):
        y[int(labelVector[0][i]), i] = 1  # 2nd index to prevent broadcasting
        
    return y


def loadIrisData(filename, testSetSize):
    headers = ['LenSepal', 'WidSepal', 'LenPetal', 'WidPetal', 'Class']
    df = pd.read_csv('iris.txt', names=headers)
    
    LenSepal = df['LenSepal']
    WidSepal = df['WidSepal']
    LenPetal = df['LenPetal']
    WidPetal = df['WidPetal']
    Class = df['Class']
    labels = [0]*len(Class)
    
    for i in range(len(labels)):
        if Class[i] == 'Iris-setosa':
            labels[i] = 0
        elif Class[i] == 'Iris-versicolor':
            labels[i] = 1
        else:
            labels[i] = 2
    
    c = max(labels) + 1
    
    # build feature string as input for np.matrix()
    xStr = ''
    for i in range(len(LenSepal)):    
        xStr += str(LenSepal[i]) + ' '
    xStr += ';'
    for i in range(len(LenSepal)):    
        xStr += str(WidSepal[i]) + ' '
    xStr += ';'
    for i in range(len(LenSepal)):    
        xStr += str(LenPetal[i]) + ' '
    xStr += ';'
    for i in range(len(LenSepal)):    
        xStr += str(WidPetal[i]) + ' '
    
    # shuffle input matrices
    xy = np.matrix(xStr + ';' + str(labels))
    np.random.shuffle(xy.T)
    
    # seperate into train and test sets according to testSetSize
    xTest = xy[0:4, 0:testSetSize]
    yTestTemp = xy[4, 0:testSetSize]
    xTrain = xy[0:4, testSetSize:xy.shape[1]]
    yTrainTemp = xy[4, testSetSize:xy.shape[1]]
    
    # rebuild labels to prepare for softmax
    yTest = convertLabelToSoftmax(yTestTemp, c)
    yTrain = convertLabelToSoftmax(yTrainTemp, c)

    return xTrain, yTrain, xTest, yTest


#testSetSize = 20
#filename = 'iris.txt'

#xTrain, yTrain, xTest, yTest = loadIrisData(filename, testSetSize)



# Initialize Parameters
# ----------------
def initialize(dims):
    # dims is a list stating the number of neurons per layer
    # W and b are dictionaries containing the parameters
    parameters = {}
    
    for i in range(len(dims)):
        if i > 0:
            parameters["W" + str(i)] = np.random.randn(dims[i], dims[i - 1])*0.01
            parameters["b" + str(i)] = np.zeros((dims[i], 1))

    return parameters

#parameters = initialize([4, 3])



# Forward Prop
# ----------------
def forward(A_prev, W, b):
    Z = np.dot(W, A_prev) + b
#    cache = {"A_prev": A_prev, "W": W, "b": b}  # used to compute derivatives
    
    return Z#, cache



# Activation
# ----------------
def activation(Z, activateFcn):
    if activateFcn == 'sigmoid':
        A = 1/(1 + np.exp(-Z))
    elif activateFcn == 'tanh':
        A = 2/(1 + np.exp(-2*Z)) - 1
    elif activateFcn == 'leakyRelu':
        A = max(0.01*Z, Z)
    elif activateFcn == 'unity':
        A = Z
    else:
        # assumed Rectifier Linear Unit (ReLU)
        A = max(0, Z)
        
    return A
    
    
    
    # Softmax
# ----------------
def softmax(Z):
    # AL is a vector output of the last layer
    t = np.exp(Z)
    tot = np.sum(np.asarray(t), axis=0, keepdims=True)
    A = np.multiply(t, 1/tot)
    
    return A



# Cost Fcn
# ----------------
def cost(A, y):
    # A is the output vector of the softmax fcn (y_hat)
    Loss = np.sum(np.multiply(y, np.log(A)))
    # compute cost from loss function
    m = y.shape[1]
    J = -1/m*Loss
    # gradient of cost, needed for backprop
#    dJ = np.multiply(y, -1/A) + np.multiply(1 - y, 1/(1 - A))  # dJ/dA
#    dZL = A - y  # dA/dZL for softmax eqn
    
    return J #, dJ, dZL



# Backprop
# ----------------    
def backprop(parameters, cache, activateFcn, X, Y):
    
    # nested helper fcn to find derivative of activation
    def fcnDerivative(Z, Fcn):
        if activateFcn == 'sigmoid':
            dgZ = np.multiply(activation(Z, 'sigmoid'),(1 - activation(Z, 'sigmoid')))
        elif activateFcn == 'tanh':
            dgZ = np.power((1 - activation(Z, 'tanh')), 2)
        elif activateFcn == 'leakyRelu':
            if Z > 0:
                dgZ = 1
            else:
                dgZ = 0.01
        elif activateFcn == 'unity':
            dgZ = 1
        else:
            # assumed Rectifier Linear Unit (ReLU)
            if Z > 0:
                dgZ = 1
            else:
                dgZ = 0
            
        return dgZ  # g'(Z)
    
    
    # grab grad calc inputs from dictionaries
#    W1 = parameters["W1"]
#    b1 = parameters["b1"]
    W2 = parameters["W2"]
#    b2 = parameters["b2"]
    
    Z1 = cache["Z1"]
    A1 = cache["A1"]  # from sigmoid
#    Z2 = cache["Z2"]
    A2 = cache["A2"]  # from softmax
    
    m = X.shape[1]
        
    # grads from cost function (softmax)
    dZ2 = A2 - Y
    
    # 2nd layer grads
    dW2 = (1/m)*np.dot(dZ2, A1.T)
    db2 = (1/m)*np.sum(np.asarray(dZ2), axis=1, keepdims=True)
    
    # grads for activation function
    activateFcnPRIME = fcnDerivative(Z1, activateFcn)

    # 1st layer grads
    dZ1 = np.multiply(np.dot(W2.T, dZ2),activateFcnPRIME)
    dW1 = (1/m)*np.dot(dZ1, X.T)
    db1 = (1/m)*np.sum(np.asarray(dZ1), axis=1, keepdims=True)
    
    # grads are stored in dict
    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
    
    return grads

    

# Update Params
# ----------------
def update(parameters, grads, learn_rate):
    
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]

    W1 = W1 - learn_rate*dW1
    b1 = b1 - learn_rate*db1
    W2 = W2 - learn_rate*dW2
    b2 = b2 - learn_rate*db2
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters



# Test Performance
# ----------------
def predict(parameters, activateFcn, X, Y, c):
    m = X.shape[1]
    
    Z1 = forward(X, parameters["W1"], parameters["b1"])
    A1 = activation(Z1, 'sigmoid')
    Z2 = forward(A1, parameters["W2"], parameters["b2"])
    A2 = softmax(Z2)
    A2 = (A2 > 1/c)
    diff = abs(A2 - Y)
    wrong = 0
    for i in range(diff.shape[1]):
        if np.sum(diff[0:(c - 1), i]) != 0:
            wrong += 1

    accuracy = (m - wrong)/m*100
    
    return accuracy
    
    
    
    
    
    
    
    
    
    
    
    


