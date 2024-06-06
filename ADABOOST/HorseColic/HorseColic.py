import numpy as np
import math
import matplotlib.pyplot as plt

def loadDataSet(filename):
    DataMat = []
    LabelMat = []
    with open(filename) as fr:
        for line in fr.readlines():
            lineArr = []
            currLine = line.strip().split('\t')
            for i in range(len(currLine) - 1):
                lineArr.append(float(currLine[i]))
            DataMat.append(lineArr)
            LabelMat.append(float(currLine[-1]))
    return DataMat, LabelMat

def stumpClassify(dataMatrix, dimen, thresVal, threshIneq):
    retArr = np.ones((np.shape(dataMatrix)[0], 1))
    if threshIneq == 'lt':  
        retArr[dataMatrix[:, dimen] <= thresVal] = -1.0
    else:
        retArr[dataMatrix[:, dimen] > thresVal] = 1.0
    return retArr

def buildStump(dataArr, classLabels, D):
    dataMatrix = np.mat(dataArr)
    labelMat = np.mat(classLabels).T
    m, n = np.shape(dataMatrix)
    numSteps = 10.0
    bestStump = {}
    bestClassEst = np.mat(np.zeros((m, 1)))
    minError = np.inf

    for i in range(n):
        rangeMin = dataMatrix[:, i].min()
        rangeMax = dataMatrix[:, i].max()
        stepSize = (rangeMax - rangeMin) / numSteps
        for j in range(-1, int(numSteps) + 1):
            for inequal in ['lt', 'gt']:
                thresVal = (rangeMin + float(j) * stepSize)
                predictedVal = stumpClassify(dataMatrix, i, thresVal, inequal)
                errArr = np.mat(np.ones((m, 1)))
                errArr[predictedVal == labelMat] = 0
                weightedError = D.T @ errArr
                weightedError = weightedError[0, 0]

                if weightedError < minError:
                    minError = weightedError
                    bestClassEst = predictedVal.copy()
                    bestStump['dimen'] = i
                    bestStump['thresh'] = thresVal
                    bestStump['ineq'] = inequal

    return bestStump, minError, bestClassEst

def adaBoostTrainDS(dataArr, classLabels, numIt=40):
    weakClassArr = []
    m = np.shape(dataArr)[0]
    D = np.mat(np.ones((m, 1)) / m)
    aggClassEst = np.mat(np.zeros((m, 1)))

    for i in range(numIt):
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)

        alpha = float(0.5 * np.log((1.0 - error) / max(error, 1e-16)))
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)

        expon = np.multiply(-1 * alpha * np.mat(classLabels).T, classEst)
        D = np.multiply(D, np.exp(expon))
        D = D / D.sum()

        aggClassEst += alpha * classEst

        aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(classLabels).T, np.ones((m, 1)))
        errorRate = aggErrors.sum() / m

        if errorRate == 0.0:
            break

    return weakClassArr

def adaClassify(datToClass, classifierArr):
    dataMatrix = np.mat(datToClass)
    m = np.shape(dataMatrix)[0]
    aggClassEst = np.mat(np.zeros((m, 1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix, classifierArr[i]['dimen'], classifierArr[i]['thresh'], classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha'] * classEst
    return np.sign(aggClassEst)

def calculateErrorRate(dataArr, classLabels, classifierArr):
    predictions = adaClassify(dataArr, classifierArr)
    errArr = np.mat(np.ones((len(dataArr), 1)))
    errorCount = errArr[predictions != np.mat(classLabels).T].sum()
    errorRate = errorCount / len(dataArr)
    return errorRate

# Load training and test data
trainData, trainLabels = loadDataSet(r'ADABOOST/horseColicTraining2.txt')
testData, testLabels = loadDataSet(r'ADABOOST/horseColicTest2.txt')

# Initialize the list of iteration values to check
numClassifiersList = [1, 10, 50, 100, 500, 1000]

print(f"{'Number of Classifiers':<20} {'Training Error':<15} {'Test Error':<15}")

for numClassifiers in numClassifiersList:
    classifierArray = adaBoostTrainDS(trainData, trainLabels, numClassifiers)
    trainError = calculateErrorRate(trainData, trainLabels, classifierArray)
    testError = calculateErrorRate(testData, testLabels, classifierArray)
    print(f"{numClassifiers:<20} {trainError:<15.2f} {testError:<15.2f}")


