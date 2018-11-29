# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 16:44:34 2018

@author: Leonardo
"""


# Import packages
# ----------------
import pandas as pd



# Import Data
# ----------------
headers = ['LenSepal', 'WidSepal', 'LenPetal', 'WidPetal', 'Class']
df = pd.read_csv('iris.txt', names=headers)


# Process Data:
# ----------------
def attributeAssign(flowerData):
    LenS = [0]*len(flowerData)
    WidS = [0]*len(flowerData)
    LenP = [0]*len(flowerData)
    WidP = [0]*len(flowerData)
    for i in range(len(flowerData)):
        LenS[i] = flowerData[i][0]
        WidS[i] = flowerData[i][1]
        LenP[i] = flowerData[i][2]
        WidP[i] = flowerData[i][3]
    return LenS, WidS, LenP, WidP

LenSepal = df['LenSepal']
WidSepal = df['WidSepal']
LenPetal = df['LenPetal']
WidPetal = df['WidPetal']
Class = df['Class']

colorlabels = [0]*len(Class)
setosa = []
versi = []
virginica = []
for i in range(len(colorlabels)):
    if Class[i] == 'Iris-setosa':
        colorlabels[i] = 0
        setosa.append([df['LenSepal'][i], df['WidSepal'][i], df['LenPetal'][i], df['WidPetal'][i]])
    elif Class[i] == 'Iris-versicolor':
        colorlabels[i] = 1
        versi.append([df['LenSepal'][i], df['WidSepal'][i], df['LenPetal'][i], df['WidPetal'][i]])
    else:
        colorlabels[i] = 2
        virginica.append([df['LenSepal'][i], df['WidSepal'][i], df['LenPetal'][i], df['WidPetal'][i]])

"""
# for sorting out by class:
setosaLenS, setosaWidS, setosaLenP, setosaWidP = attributeAssign(setosa)
versiLenS, versiWidS, versiLenP, versiWidP = attributeAssign(versi)
virginicaLenS, virginicaWidS, virginicaLenP, virginicaWidP = attributeAssign(virginica)
"""














