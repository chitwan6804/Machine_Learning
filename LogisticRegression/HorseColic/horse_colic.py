import math
import numpy
import random

def sigmoid(inX):
    return 1.0 / (1 + numpy.exp(-numpy.clip(inX, -709, 709)))

def stochGradAscent1(dataMatrix, classLabels, numIter=150):
    m, n = numpy.shape(dataMatrix)
    weights=numpy.ones(n)
    for j in range(numIter):
        dataIndex=list(range(m))
        for i in range(m):
            alpha=4/(1.0+j+i)+0.01
            randIndex=int(random.uniform(0,len(dataIndex)))
            h=sigmoid(numpy.sum(dataMatrix[randIndex] * weights))
            error=classLabels[randIndex]-h
            weights+=alpha*error*dataMatrix[randIndex]
            del(dataIndex[randIndex])
        return weights

def classifyVector(inX,weights):
    prob=sigmoid((numpy.sum(inX*weights)))
    if prob>0.5:
        return 1.0
    else:
        return 0.0

def colicTest():
    frTrain=open(r'LogisticRegression\HorseColicTraining.txt')
    frtest=open(r'LogisticRegression\HorseColicTest.txt')
    trainingSet=[]
    trainingLabels=[]
    for line in frTrain.readlines():
        currLine=line.strip().split('\t')
        if len(currLine) < 22 or '?' in currLine:  # Check if the line has enough elements
            continue  # Skip lines with insufficient elements
        lineArr=[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights=stochGradAscent1(numpy.array(trainingSet),trainingLabels,500)
    errorCount=0; numTestVec=0.0
    for line in frtest.readlines():
        numTestVec+=1.0
        currLine=line.strip().split('\t')
        lineArr=[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(numpy.array(lineArr),trainWeights))!=int(currLine[21]):
            int(currLine[21])
            errorCount+=1
    errorRate=float(errorCount)/numTestVec
    print("The error rate of this test is :", errorRate)
    return errorRate

def multiTest():
    numTests=10; errorSum=0.0
    for k in range(numTests):
        errorSum+=colicTest()
    print("After %d iterations the average rate is: %f" %(numTests,errorSum/float(numTests)))

multiTest()
    