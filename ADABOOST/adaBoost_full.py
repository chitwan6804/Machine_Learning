import numpy as np
import math

def loadDataSet():
    dataMat = np.matrix([[1.0, 2.1],
                         [2.0, 1.1],
                         [1.3, 1.0],
                         [1.0, 1.0],
                         [2.0, 1.0]])
    classLabel = [1.0, 1.0, -1.0, -1.0, 1.0]
    return dataMat, classLabel

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
                weightedError = weightedError[0, 0]  # Extract scalar

                print(f"split: dim {i}, thresh {thresVal:.2f}, thresh ineqal: {inequal}, the weighted error is {weightedError:.3f}")

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
        print("D:", D.T)

        alpha = float(0.5 * np.log((1.0 - error) / max(error, 1e-16)))
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        print("classEst: ", classEst.T)

        expon = np.multiply(-1 * alpha * np.mat(classLabels).T, classEst)
        D = np.multiply(D, np.exp(expon))
        D = D / D.sum()

        aggClassEst += alpha * classEst
        print("aggClassEst: ", aggClassEst.T)

        aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(classLabels).T, np.ones((m, 1)))
        errorRate = aggErrors.sum() / m
        print("total error: ", errorRate, "\n")

        if errorRate == 0.0:
            break

    return weakClassArr


# Load data
Features, Labels = loadDataSet()

# # Train AdaBoost classifier
# classifierArray = adaBoostTrainDS(Features, Labels, 9)
# print("classifierArray:", classifierArray)


def adaClassify(datToClass,classifierArr):
    dataMatrix=np.mat(datToClass)
    m=np.shape(dataMatrix)[0]
    aggClassEst=np.mat(np.zeros((m,1)))
    for i in range(len(classifierArr)):
        classEst=stumpClassify(dataMatrix,classifierArr[i]['dimen'],classifierArr[i]['thresh'],classifierArr[i]['ineq'])
        aggClassEst+=classifierArr[i]['alpha']*classEst
        print(aggClassEst)
    return np.sign(aggClassEst)

classifierArray = adaBoostTrainDS(Features,Labels,30)
classified=adaClassify([0,0],classifierArray)
print("[0,0] is classified to class: ",classified)


classified=adaClassify([[5, 5],[0,0]],classifierArray)
print("[0,0] is classified to class: ",classified)