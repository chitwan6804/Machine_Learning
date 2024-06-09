import numpy as np
import random

def loadDataSet(filename):
    dataMat = []
    labelMat = []
    with open(filename) as fr:
        for line in fr.readlines():
            lineArr = line.strip().split('\t')
            dataMat.append([float(lineArr[0]), float(lineArr[1])])
            labelMat.append(float(lineArr[2]))
    return dataMat, labelMat

def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj

class optStruct:
    def __init__(self, dataMatIn, classLabels, C, toler):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = np.shape(dataMatIn)[0]
        self.alphas = np.mat(np.zeros((self.m, 1)))
        self.b = 0
        self.eCache = [(0, 0) for _ in range(self.m)]  # Initialize as list of tuples

def calcEk(os, k):
    fXk = float(np.multiply(os.alphas, os.labelMat).T * (os.X * os.X[k, :].T)) + os.b
    Ek = fXk - float(os.labelMat[k, 0])  # Accessing a single element of labelMat
    return Ek

def updateEk(os, k):
    Ek = calcEk(os, k)
    os.eCache[k] = [1, Ek]

def selectJrand(i, m):
    j = i
    while (j == i):
        j = int(np.random.uniform(0, m))
    return j

def selectJ(i, os, Ei):
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    os.eCache[i] = (1, Ei)  # Use tuple instead of list for eCache
    validEcacheList = [k for k, (isValid, _) in enumerate(os.eCache) if isValid]
    if len(validEcacheList) > 1:
        for k in validEcacheList:
            if k == i:
                continue
            Ek = calcEk(os, k)
            deltaE = abs(Ei - Ek)
            if deltaE > maxDeltaE:
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej
    else:
        j = selectJrand(i, os.m)
        Ej = calcEk(os, j)
    return j, Ej

def innerL(i, os):
    Ei = calcEk(os, i)
    if ((os.labelMat[i] * Ei < -os.tol) and (os.alphas[i] < os.C)) or ((os.labelMat[i] * Ei > os.tol) and (os.alphas[i] > 0)):
        j, Ej = selectJ(i, os, Ei)
        alphaIold = os.alphas[i].copy()
        alphaJold = os.alphas[j].copy()
        if (os.labelMat[i] != os.labelMat[j]):
            L = max(0, os.alphas[j] - os.alphas[i])
            H = min(os.C, os.C + os.alphas[j] - os.alphas[i])
        else:
            L = max(0, os.alphas[j] + os.alphas[i] - os.C)
            H = min(os.C, os.alphas[j] + os.alphas[i])
        if L == H:
            print("L == H")
            return 0
        eta = 2.0 * os.X[i, :] * os.X[j, :].T - os.X[i, :] * os.X[i, :].T - os.X[j, :] * os.X[j, :].T
        if eta >= 0:
            print("eta >= 0")
            return 0
        os.alphas[j] -= os.labelMat[j] * (Ei - Ej) / eta
        os.alphas[j] = clipAlpha(os.alphas[j], H, L)
        updateEk(os, j)
        if abs(os.alphas[j] - alphaJold) < 0.00001:
            print("j not moving enough")
            return 0
        os.alphas[i] += os.labelMat[j] * os.labelMat[i] * (alphaJold - os.alphas[j])
        updateEk(os, i)
        b1 = os.b - Ei - os.labelMat[i] * (os.alphas[i] - alphaIold) * os.X[i, :] * os.X[i, :].T - os.labelMat[j] * (os.alphas[j] - alphaJold) * os.X[i, :] * os.X[j, :].T
        b2 = os.b - Ej - os.labelMat[i] * (os.alphas[i] - alphaIold) * os.X[i, :] * os.X[j, :].T - os.labelMat[j] * (os.alphas[j] - alphaJold) * os.X[j, :] * os.X[j, :].T
        if 0 < os.alphas[i] < os.C:
            os.b = b1
        elif 0 < os.alphas[j] < os.C:
            os.b = b2
        else:
            os.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0

def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup=('lin', 0)):
    os = optStruct(np.mat(dataMatIn), np.mat(classLabels).transpose(), C, toler)
    iter = 0
    entireSet = True
    alphaPairsChanged = 0
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:
            for i in range(os.m):
                alphaPairsChanged += innerL(i, os)
                print("fullSet, iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
            iter += 1
        else:
            nonBoundIs = np.nonzero((os.alphas.A > 0) * (os.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i, os)
                print("non-bound, iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
            iter += 1
        if entireSet:
            entireSet = False
        elif alphaPairsChanged == 0:
            entireSet = True
        print("iteration number: %d" % iter)
    return os.b, os.alphas

dataMat, Label = loadDataSet('SVM/testSet.txt')
b, alphas = smoP(dataMat, Label, 0.6, 0.001, 40)

def calcWs(alphas, dataArr, classLabels):
    X = np.mat(dataArr)
    labelMat = np.mat(classLabels).transpose()
    m, n = np.shape(X)
    w = np.zeros((n, 1))
    for i in range(m):
        w += np.multiply(alphas[i] * labelMat[i], X[i, :].T)
    return w

ws=calcWs(alphas,dataMat, Label)
print(ws)

print(dataMat[2]*np.mat(ws)+b , "      " , Label[2])   # if fisrst value greater than 0 then class 1
print(dataMat[1]*np.mat(ws)+b , "      " , Label[1])   # if fisrst value lesser than 0 then class -1