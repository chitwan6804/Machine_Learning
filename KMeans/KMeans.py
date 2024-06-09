import numpy as np
import matplotlib.pyplot as plt

def loadDataSet(filename):
    dataMat = []
    with open(filename) as fr:
        for line in fr.readlines():
            currLine = line.strip().split('\t')
            fltLine = list(map(float, currLine))  # Convert each element to float and create a list
            dataMat.append(fltLine)
    return np.array(dataMat)

def EcludDis(vecA, vecB):
    return np.sqrt(np.sum(np.power(vecA - vecB, 2)))  

def randCent(dataSet, k):
    n = np.shape(dataSet)[1]
    centroids = np.mat(np.zeros((k, n)))
    for j in range(n):
        minJ = np.min(dataSet[:, j])
        rangeJ = float(np.max(dataSet[:, j]) - minJ)
        centroids[:, j] = minJ + rangeJ * np.random.rand(k, 1)
    return centroids

def kMeans(dataSet, k, disMeas=EcludDis, createCent=randCent):
    m = np.shape(dataSet)[0]
    clusterAssment = np.mat(np.zeros((m, 2)))
    centroids = createCent(dataSet, k)
    clusterChanged = True

    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = float('inf')
            minIndex = -1
            for j in range(k):
                distJI = disMeas(centroids[j, :], dataSet[i, :])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j

            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
                clusterAssment[i, :] = minIndex, minDist**2

        for cent in range(k):
            ptsInClust = dataSet[np.nonzero(clusterAssment[:, 0].A == cent)[0]]
            if len(ptsInClust) != 0:
                centroids[cent, :] = np.mean(ptsInClust, axis=0)
    
    return centroids, clusterAssment

def biKmeans(dataSet, k, disMeas=EcludDis):
    m = np.shape(dataSet)[0]
    clusterAssment = np.mat(np.zeros((m, 2)))
    centroid0 = np.mean(dataSet, axis=0).tolist()[0]
    centList = [centroid0]

    for j in range(m):
        clusterAssment[j, 1] = disMeas(np.mat(centroid0), dataSet[j, :])**2

    while len(centList) < k:
        lowestSSE = float('inf')
        for i in range(len(centList)):
            ptsInCurrCluster = dataSet[np.nonzero(clusterAssment[:, 0].A == i)[0], :]
            if len(ptsInCurrCluster) == 0:
                continue
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, disMeas)
            sseSplit = np.sum(splitClustAss[:, 1])
            sseNotSplit = np.sum(clusterAssment[np.nonzero(clusterAssment[:, 0].A != i)[0], 1])
            print("SSE split and SSE not split: ", sseSplit, "   ", sseNotSplit)
            if (sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit

        bestClustAss[np.nonzero(bestClustAss[:, 0].A == 1)[0], 0] = len(centList)
        bestClustAss[np.nonzero(bestClustAss[:, 0].A == 0)[0], 0] = bestCentToSplit
        print("the bestCentToSplit is: ", bestCentToSplit)
        print("the len of bestClusterAss is: ", len(bestClustAss))
        centList[bestCentToSplit] = bestNewCents[0, :].tolist()[0]
        centList.append(bestNewCents[1, :].tolist()[0])
        clusterAssment[np.nonzero(clusterAssment[:, 0].A == bestCentToSplit)[0], :] = bestClustAss

    return np.mat(centList), clusterAssment

# Plotting clusters and centroids
def plotClusters(dataMat, centList, clusterAssment):
    dataMat = np.array(dataMat)
    numSamples, dim = dataMat.shape
    marks = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']

    for i in range(numSamples):
        markIndex = int(clusterAssment[i, 0])
        plt.plot(dataMat[i, 0], dataMat[i, 1], marks[markIndex])

    for i in range(len(centList)):
        plt.plot(centList[i, 0], centList[i, 1], '+k', markersize=12, markeredgewidth=2)

    plt.show()



# Load the dataset
dataMat = loadDataSet('KMeans/testSet.txt')

# Apply kMeans clustering
centroids, clusterAssment = kMeans(dataMat, 2)
print("Centroids:\n", centroids)
print("Cluster Assignment:\n", clusterAssment)
plotClusters(dataMat, centroids, clusterAssment)

#Load different dataSet

datMat=loadDataSet('KMeans/testSet2.txt')

# Apply biKmeans clustering
centList, mynewAssments = biKmeans(datMat, 3)
print("Centroids:\n", centList)
print("Cluster Assignments:\n", mynewAssments)
print(centList)
plotClusters(datMat, centList, mynewAssments)
