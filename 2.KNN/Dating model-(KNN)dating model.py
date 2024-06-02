import numpy as np
import operator
import matplotlib.pyplot as plt

# Function to load dataset from a file
def file2matrix(filename):
    fr = open(filename)
    numberoflines = len(fr.readlines())  # Count the number of lines in the file
    returnMat = np.zeros((numberoflines, 3))  # Initialize a matrix to hold the dataset
    classLabelVector = []  # Initialize a list to hold the labels
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()  # Remove leading and trailing whitespace
        listfromline = line.split('\t')  # Split the line by tabs
        returnMat[index, :] = listfromline[0:3]  # Extract the features
        classLabelVector.append(int(listfromline[-1]))  # Extract the label and append to the list
        index += 1
    return returnMat, classLabelVector

# Function to normalize the dataset
def autoNorm(dataSet):
    minVals = dataSet.min(0)  # Minimum values of features
    maxVals = dataSet.max(0)  # Maximum values of features
    ranges = maxVals - minVals  # Range of features
    normDataSet = np.zeros(np.shape(dataSet))  # Initialize normalized dataset
    m = dataSet.shape[0]  # Number of data points
    normDataSet = dataSet - np.tile(minVals, (m, 1))  # Subtract the minimum values
    normDataSet = normDataSet / np.tile(ranges, (m, 1))  # Divide by the range
    return normDataSet, ranges, minVals

# K-Nearest Neighbors classifier function
def classify(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet  # Calculate the difference matrix
    sqDiffMat = diffMat ** 2  # Square the differences
    sqDistances = sqDiffMat.sum(axis=1)  # Sum the squared differences
    distances = sqDistances ** 0.5  # Take the square root to get the Euclidean distances
    sortedDisIndices = distances.argsort()  # Sort distances and get the sorted indices
    classCount = {}  # Dictionary to count the occurrences of each label
    for i in range(k):
        voteIlabel = labels[sortedDisIndices[i]]  # Get the label of the i-th nearest neighbor
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1  # Count the label
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)  # Sort by count
    return sortedClassCount[0][0]  # Return the label with the highest count

# Function to test the KNN classifier
def datingClassTest():
    hoRatio = 0.10  # Hold-out ratio: proportion of the dataset to be used as the test set
    datingDataMat, datingLabels = file2matrix('K-nearest-neighbor.py/datingTestSet2.txt') # Load the dataset and corresponding labels
    
    normMat, ranges, minVals = autoNorm(datingDataMat) # Normalize the dataset
    
    m = normMat.shape[0] # Determine the number of data points
    numTestVecs = int(m * hoRatio) # Calculate the number of test vectors
    
    errorCount = 0.0 # Initialize the error count
    
    # Loop over the test set
    for i in range(numTestVecs):
        # Classify the i-th test vector using the KNN algorithm
        classifierResult = classify(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]):
            errorCount += 1.0
    print("the total error rate is: %f" % (errorCount / float(numTestVecs)))

# datingClassTest()

def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    
    # Collect input from the user
    percentTats = float(input("percentage of time spent playing video games? "))
    ffMiles = float(input("frequent flier miles earned per year? "))
    iceCream = float(input("liters of ice cream consumed per year? "))
    
    # Load and normalize the dataset
    datingDataMat, datingLabels = file2matrix('K-nearest-neighbor.py/datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    
    # Create the input array and normalize it
    inArr = np.array([ffMiles, percentTats, iceCream])
    normalizedInArr = (inArr - minVals) / ranges
    
    # Classify the input
    classifierResult = classify(normalizedInArr, normMat, datingLabels, 3)
    
    # Print the result
    print("You will probably like this person:", resultList[classifierResult - 1])

# Example usage
classifyPerson()
