import numpy as np

def loadDataSet():
    dataMat=np.matrix([[1.0,2.1],
                      [2.0,1.1],
                      [1.3,1.0],
                      [1.0,1.0],
                      [2.0,1.0]])
    classLabel=[1.0,1.0,-1.0,-1.0,1.0]
    return dataMat,classLabel

# Features,Labels=loadDataSet()
# # print(Features)
# # print(Labels)

def stumpClassify(dataMatrix,dimen,thresVal,threshIneq):
    retArr=np.ones((np.shape(dataMatrix)[0],1))
    if threshIneq=='lt':  
        retArr[dataMatrix[:,dimen]<=thresVal]=-1.0
    else:
        retArr[dataMatrix[:,dimen]>thresVal]=1.0
    return retArr

def buildStump(dataArr,classLabels,D):
    dataMatrix=np.mat(dataArr); labelMat=np.mat(classLabels).T
    m,n=np.shape(dataMatrix)
    numSteps=10.0; bestStump={}; bestClassEst=np.mat(np.zeros((m,1)))
    minError=1000000
    for i in range(n):
        rangeMin=dataMatrix[:,i].min()
        rangeMax=dataMatrix[:,i].max()
        stepSize=(rangeMax-rangeMin)/numSteps
        for j in range(-1,int(numSteps)+1):
            for inequal in ['lt','gt']:
                thresVal=(rangeMin+float(j)*stepSize)
                predictedVal=stumpClassify(dataMatrix,i,thresVal,inequal)
                errArr=np.mat(np.ones((m,1)))
                errArr[predictedVal==labelMat]=0
                weightedError=D.T*errArr
                print ("split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (i, thresVal, inequal, weightedError))
                if weightedError<minError:
                    minError=weightedError
                    bestClassEst=predictedVal.copy()
                    bestStump['dimen']=i
                    bestStump['thresh']=thresVal
                    bestStump['ineq']=inequal
    return bestStump,minError,bestClassEst

Features, Labels = loadDataSet()
D = np.mat(np.ones((5, 1)) / 5)  # Initialize weights for each data point
bestStump, minError, bestClassEst = buildStump(Features, Labels, D)
print("Best Stump:", bestStump)
print("Minimum Error:", minError)
print("Best Class Estimates:", bestClassEst)

