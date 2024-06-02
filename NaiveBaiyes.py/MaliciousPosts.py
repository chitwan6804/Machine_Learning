import numpy
import math

def loadDataSet():
    postingList=[['my','dog','has','flea','problems','help','please'],
                 ['maybe','not','take','him','to','dog','park','stupid'],
                 ['my','dalmation','is','so','cute','I','love','him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how','to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]  # 1 for abusive, 0 not
    return postingList,classVec

def createVocabList(dataSet):
    vocabSet=set([])
    for document in dataSet:
        vocabSet= vocabSet | set(document)
    return list(vocabSet)

def setofwords2vec(vocabList,inputSet):
    returnVec=[0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)]=1
        else:
            print("the word: %s is not in my vocabulary!" %word)
    return returnVec

listofPosts,listClasses=loadDataSet()
myvocblist=createVocabList(listofPosts)
print(myvocblist)
Veclist=setofwords2vec(myvocblist, listofPosts[0])
print("vector list is as follows: ",Veclist)

def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    p0Num = numpy.ones(numWords)  # Initialize counts to 1 for Laplace smoothing
    p1Num = numpy.ones(numWords)  # Initialize counts to 1 for Laplace smoothing
    p0Denom = 2.0  # Initialize denominator to 2 for Laplace smoothing
    p1Denom = 2.0  # Initialize denominator to 2 for Laplace smoothing
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1vect = numpy.log(p1Num / p1Denom)
    p0vect = numpy.log(p0Num / p0Denom)
    return p0vect, p1vect, pAbusive


trainMat=[]
for PostinDoc in listofPosts:
    trainMat.append(setofwords2vec(myvocblist,PostinDoc))

p0V,p1V,pAbuse=trainNB0(trainMat,listClasses)
print("Probability of abusive sentence:",pAbuse)
print("P0 vector:\n",p0V)
print("P1 vector:\n",p1V)

def classifyNB(vec2classify, p0vec, p1vec, pclass1):
    p1 = sum(vec2classify * p1vec) + math.log(pclass1)
    p0 = sum(vec2classify * p0vec) + math.log(1.0 - pclass1)
    if p1 > p0:
        return 1
    else:
        return 0

def testingNB():
    listofPosts,listClasses=loadDataSet()
    myvocablist=createVocabList(listofPosts)
    trainMat=[]
    for postinDoc in listofPosts:
        trainMat.append(setofwords2vec(myvocablist,postinDoc))
    p0V,p1V,pAb = trainNB0(numpy.array(trainMat),numpy.array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = numpy.array(setofwords2vec(myvocablist, testEntry))
    print (testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = numpy.array(setofwords2vec(myvocablist, testEntry))
    print (testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb))

def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec


testingNB()