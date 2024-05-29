import numpy as np

# Array creation: Conversion from other Python structures
listarray = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("Array is as follows:\n", listarray)
print("Size of array: ", listarray.size)
print("Shape of array: ", listarray.shape)
print("Data type of array: ", listarray.dtype)

# Intrinsic NumPy array creation objects (arange, ones, zeros)
zeros = np.zeros((2, 5)) # Creating array with that size having 0 element at each place
print("\nArray of size (2, 5) all with zero as data is as follows:\n", zeros)

rng = np.arange(15) # Creating array by defining its range
print("\nArray of range (0 to 14) using arange function:\n", rng)

lspace = np.linspace(1, 50, 10) # Creates 10 elements between 1 to 50 having equal space between them
print("\nArray made using linspace(1, 50, 10)\n", lspace)

emp = np.empty((4, 6)) # Creates an empty array of size given with random values in it
print("\nArray made using np.empty((4, 6))\n", emp)

emp_like = np.empty_like(lspace)
print("\nArray made using np.empty_like(lspace) where lspace was defined using linspace function\n", emp_like)

ide = np.identity(45) # Creates an identity matrix of size defined
print("\nIdentity array:\n", ide)
print("\nShape of Identity array: ", ide.shape)

arr = np.arange(99)
arr = arr.reshape(3, 33) # Changes 1D array to 2D using reshape
print("\nReshape the array to 2D:\n", arr)
arr = arr.ravel() # Undo the changes made by reshape function
print("\nAfter undoing the changes made using reshape by using ravel:\n", arr)

# NumPy Axis
X = [[1, 2, 3], [4, 5, 6], [7, 1, 0]] # Define a 2D list
ar = np.array(X) # Convert the list to a NumPy array
print("Array:\n", ar)

sum_axis_0 = ar.sum(axis=0) # Sum of elements along axis 0 (column-wise sum)
print("\nSum along axis 0 (columns):\n", sum_axis_0)

sum_axis_1 = ar.sum(axis=1) # Sum of elements along axis 1 (row-wise sum)
print("\nSum along axis 1 (rows):\n", sum_axis_1)

print("\nTranspose of the array:\n", ar.T) # Transpose of the array

print("\nElements of the array:")
for i in ar.flat: # Iterate over each element in the array
    print(i)

print("\nNumber of dimensions:", ar.ndim) # Number of dimensions of the array
print("Total number of elements:", ar.size) # Total number of elements in the array
print("Total number of bytes consumed by the elements of the array:", ar.nbytes) # Total number of bytes consumed by the elements of the array

# Additional array operations
one = np.array([1, 3, 4, 634, 2])
print("\nArray one:\n", one)

# Finding the indices of the maximum and minimum values
print("\nIndex of the maximum value in 'one':", one.argmax())
print("Index of the minimum value in 'one':", one.argmin())

# Getting the sorted indices of the array
print("\nIndices that would sort the array 'one':\n", one.argsort())

# Applying argmin, argmax, and argsort to the 2D array 'ar'
print("\nIndex of the minimum value in 'ar':", ar.argmin())
print("Index of the maximum value in 'ar':", ar.argmax())
print("\nIndices that would sort the array 'ar' along axis 0:\n", ar.argsort(axis=0))
print("Indices that would sort the array 'ar' along axis 1:\n", ar.argsort(axis=1))

# Element-wise operations
arr1 = np.array([[1, 2, 3], [3, 4, 5], [6, 7, 8]])
arr2 = np.array([[0, 1, 0], [2, 2, 2], [1, 1, 1]])

print("\nElement-wise addition of arr1 and arr2:\n", arr1 + arr2)
print("Element-wise multiplication of arr1 and arr2:\n", arr1 * arr2)

# Checking difference in space taken in memory by Python list and NumPy array
import sys
py_ar = [0, 4, 55, 2]
np_ar = np.array(py_ar)

print("\nSize of each element in the Python list:", sys.getsizeof(1))
print("Total size of Python list:", sys.getsizeof(1) * len(py_ar))

print("Size of each element in the NumPy array:", np_ar.itemsize)
print("Total size of NumPy array:", np_ar.itemsize * np_ar.size)
