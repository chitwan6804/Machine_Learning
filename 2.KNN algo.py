# Importing the necessary modules
import numpy as np  #in order to create dataset(make arrays)
import operator   #sort in KNN

# Creating Dataset
def createDataSet():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

group,labels=createDataSet()
print("\nFeatures of DataSet created:\n",group)
print("\nLabels for DataSet created:\n",labels)


# Function to classify an input vector using the k-Nearest Neighbors algorithm

def classify(inX,dataSet,labels,k):
    dataSetSize=dataSet.shape[0]
    diffMat = np.tile(inX,(dataSetSize,1)) - dataSet  # Creating an array where each row is a copy of inX, and subtracting the dataset from this array
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5   # calculating distances using Euclidian distance
    sortedDisIndices = distances.argsort()
    classCount={} # Initializing a dictionary to count the occurrences of each label among the k nearest neighbors
    for i in range(k):
        # Getting the label of the i-th nearest neighbor
        voteIlabel = labels[sortedDisIndices[i]]
        # Counting the occurrences of the label
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)  # Sorting the labels by their count in descending order  
    return sortedClassCount[0][0]

result = classify([0,1.2], group, labels, 3)  # Classifying an input vector [0, 1.2] using the k-NN algorithm with k=3
print("The input vector [0,1.2] is classified as:", result)
