import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def loadDataSet(filename):
    dataMat = []
    with open(filename) as fr:
        # Skip the header line
        next(fr)
        for line in fr.readlines():
            currLine = line.strip().split(',')
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
    centroid0 = np.mean(dataSet, axis=0).tolist()
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

def distSLC(vecA, vecB): 
    vecA = np.asarray(vecA).flatten()
    vecB = np.asarray(vecB).flatten()
    a = np.sin(vecA[1] * np.pi / 180) * np.sin(vecB[1] * np.pi / 180)
    b = np.cos(vecA[1] * np.pi / 180) * np.cos(vecB[1] * np.pi / 180) * np.cos(np.pi * (vecB[0] - vecA[0]) / 180)
    return np.arccos(a + b) * 6371.0

def clusterClubs(dataMat, numClust=5):
    myCentroids, clustAssing = biKmeans(dataMat, numClust, disMeas=distSLC)
    
    fig = plt.figure()
    rect = [0.1, 0.1, 0.8, 0.8]
    scatterMarkers = ['s', 'o', '^', '8', 'p', 'd', 'v', 'h', '>', '<']
    axprops = dict(xticks=[], yticks=[])
    
    ax0 = fig.add_axes(rect, label='ax0', **axprops)
    ax1 = fig.add_axes(rect, label='ax1', frameon=False)
    for i in range(numClust):
        ptsInCurrCluster = dataMat[np.nonzero(clustAssing[:,0].A==i)[0],:]
        markerStyle = scatterMarkers[i % len(scatterMarkers)]
        ax1.scatter(ptsInCurrCluster[:,0], ptsInCurrCluster[:,1], marker=markerStyle, s=90)
    
    ax1.scatter(myCentroids[:,0].A, myCentroids[:,1].A, marker='+', s=300)
    plt.show()

# Fetching data from the URL
urlData = requests.get('https://api.data.gov.in/resource/bccc6a91-cde0-4d1a-b255-6aab90a9e303?api-key=579b464db66ec23bdd0000015b00e97c644748c54cfed3dabac89c96&format=json&limit=1779')
data = urlData.json()

# Creating DataFrame
df = pd.DataFrame(data['records'])

# Selecting specific columns (latitude and longitude)
specific_column = df[['latitude___n', 'longitude___e']]

# Saving the data to a CSV file
specific_column.to_csv(r'KMeans\lat_lon_data.csv', index=False)

datMat = loadDataSet(r'KMeans\lat_lon_data.csv')

# Apply biKmeans clustering
centList, mynewAssments = biKmeans(datMat, 5)

# Visualize clusters
clusterClubs(datMat, numClust=5)