import math
import numpy as np
import matplotlib.pyplot as plt
import random

def loadDataSet():
    dataMat = []
    labelMat = []
    with open('LogisticRegression/TestSet.txt') as fr:
        for line in fr.readlines():
            lineArr = line.strip().split()
            dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
            labelMat.append(int(lineArr[2]))
    return dataMat, labelMat

dataArr, labelMat = loadDataSet()

def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))

def gradAscent(dataMatIn, classLabels):
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()
    m, n = np.shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = np.ones((n, 1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)
        error = labelMat - h
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights

# to test gradAscent
# weights = gradAscent(dataArr, labelMat)
# print(weights)

def plotBestFit(wei):
    dataMat, labelMat = loadDataSet()
    dataArr = np.array(dataMat)
    n = np.shape(dataArr)[0] 
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-wei[0] - wei[1] * x) / wei[2]
    ax.plot(x, y.ravel())  # Use .ravel() to flatten y
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

def stochGradAscent0(dataMatrix, classLabels):
    m, n = np.shape(dataMatrix)
    alpha = 0.1
    weights = np.ones(n)
    for i in range(m):
        h = sigmoid(np.sum(dataMatrix[i] * weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights

def stochGradAscent1(dataMatrix, classLabels, numIter=150):
    m, n = np.shape(dataMatrix)
    weights=np.ones(n)
    for j in range(numIter):
        dataIndex=list(range(m))
        for i in range(m):
            alpha=4/(1.0+j+i)+0.01
            randIndex=int(random.uniform(0,len(dataIndex)))
            h=sigmoid(np.sum(dataMatrix[randIndex] * weights))
            error=classLabels[randIndex]-h
            weights+=alpha*error*dataMatrix[randIndex]
            del(dataIndex[randIndex])
        return weights

dataArr, labelMat = loadDataSet()
weights = stochGradAscent1(np.array(dataArr), labelMat)
print(weights)
plotBestFit(weights)
