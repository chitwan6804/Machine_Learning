import numpy as np
import math
import random
import re

def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

def setofwords2vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print(f"The word '{word}' is not in my vocabulary!")
    return returnVec

def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    p0Num = np.ones(numWords)  # Initialize counts to 1 for Laplace smoothing
    p1Num = np.ones(numWords)  # Initialize counts to 1 for Laplace smoothing
    p0Denom = 2.0  # Initialize denominator to 2 for Laplace smoothing
    p1Denom = 2.0  # Initialize denominator to 2 for Laplace smoothing
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = np.log(p1Num / p1Denom)
    p0Vect = np.log(p0Num / p0Denom)
    return p0Vect, p1Vect, pAbusive

def textParse(bigString):
    listOfTokens = re.split(r'\W+', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

def classifyNB(vec2classify, p0vec, p1vec, pclass1):
    p1 = sum(vec2classify * p1vec) + math.log(pclass1)
    p0 = sum(vec2classify * p0vec) + math.log(1.0 - pclass1)
    if p1 > p0:
        return 1
    else:
        return 0

def spamTest():
    docList = []
    classList = []
    fullText = []
    for i in range(1, 26):
            wordList = textParse(open(f'NaiveBaiyes.py/spam/{i}.txt').read()) 
            docList.append(wordList)
            fullText.extend(wordList)
            classList.append(1)
            wordList = textParse(open(f'NaiveBaiyes.py/ham/{i}.txt').read())
            docList.append(wordList)
            fullText.extend(wordList)
            classList.append(0)
    
    vocabList = createVocabList(docList)
    trainingSet = list(range(50))
    testSet = [] 
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(setofwords2vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    
    p0V, p1V, pSpam = trainNB0(np.array(trainMat), np.array(trainClasses))
    
    errorCount = 0
    for docIndex in testSet:
        wordVector = setofwords2vec(vocabList, docList[docIndex])
        if classifyNB(np.array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    
    print('The error rate is: ', float(errorCount) / len(testSet))

spamTest()
