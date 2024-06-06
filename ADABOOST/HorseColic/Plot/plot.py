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

    return weakClassArr, aggClassEst



def plotROC(predStrengths, classLabels):
    cur = (1.0, 1.0)  # Cursor
    ySum = 0.0  # Variable to calculate AUC
    numPosClas = sum(np.array(classLabels) == 1.0)
    yStep = 1 / float(numPosClas)
    xStep = 1 / float(len(classLabels) - numPosClas)
    sortedIndicies = predStrengths.argsort()
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    for index in sortedIndicies.tolist()[0]:
        if classLabels[index] == 1.0:
            delX = 0
            delY = yStep
        else:
            delX = xStep
            delY = 0
            ySum += cur[1]
        ax.plot([cur[0], cur[0] - delX], [cur[1], cur[1] - delY], c='b')
        cur = (cur[0] - delX, cur[1] - delY)
    ax.plot([0, 1], [0, 1], 'b--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve for AdaBoost Horse Colic Detection System')
    ax.axis([0, 1, 0, 1])
    plt.show()
    print("The Area Under the Curve is: ", ySum * xStep)

dataArr,labelArr = loadDataSet(r'ADABOOST/horseColicTraining2.txt')
classifierArray, aggClassEst = adaBoostTrainDS(dataArr, labelArr, 10)

# Plot the ROC curve
plotROC(aggClassEst, labelArr)