#!/usr/bin/env python
# coding: utf-8

# # Introduction to Pandas and Numpy

# ## Part 1: Introduction to Numpy

# In[1]:


#Installation and Setup
get_ipython().system('pip install numpy')
import numpy as np


# In[2]:


#Create an Array From a List
arr1 = np.array([1, 2, 3, 4, 5])
print(arr1)

#Create an Array of Zeros
arr2 = np.zeros(5)
print(arr2)

#Create an Array of Ones
arr3 = np.ones((3, 3))
print(arr3)

#Create an Array of Evenly Spaced Values
arr4 = np.arange(0, 10, 2)
print(arr4)

#Create an Array of Random Values
arr5 = np.random.rand(3, 3)
print(arr5)


# In[3]:


#Array Attributes: Shape, Size, Dtype

#Shape of the Array
print(arr1.shape)

#Size of the Array
print(arr1.size)

#Data Type of the Array
print(arr1.dtype)


# In[4]:


#Indexing and Slicing Arrays

#Accessing Elements
print(arr1[0])

#Slicing
print(arr1[1:4])


# In[5]:


#Array Operations: Arithmetic, Aggregation, Broadcasting

#Arithmetic Operations
arr6 = arr1 + arr2
print(arr6)

#Aggregation Functions
print(np.sum(arr1))

#Broadcasting
arr7 = arr1 * 2
print(arr7)


# In[6]:


#Reshaping Arrays
arr8 = np.arange(9).reshape(3, 3)
print(arr8)


# In[7]:


#Stacking and Splitting Arrays

#Stacking Arrays (Vertically)
arr9 = np.vstack((arr8, arr8))
print(arr9)

#Splitting Arrays
arr10, arr11 = np.split(arr9, 2)
print(arr10, arr11)


# In[8]:


#Transposing Arrays
arr12 = arr8.T
print(arr12)


# In[9]:


#Universal Functions (ufuncs)/Advanced Numpy Functions

#Sin
arr13 = np.sin(arr1)
print(arr13)

#Fancy Indexing
indices = np.array([0, 2, 4])
print(arr1[indices])

#Boolean Indexing
bool_arr = arr1 > 3
print(arr1[bool_arr])

#Vectorized Operations
arr14 = arr1 + 10
print(arr14)

#Broadcasting Pt.2
arr15 = arr1 + np.array([[10], [20], [30], [40], [50]])
print(arr15)


# In[10]:


#Implementing Matrix Operations: Matrix Multiplication

#Create two matrices
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

#Multiply Matrices
C = np.dot(A, B)
print(C)

#Matrix Inversion
A_inv = np.linalg.inv(A)
print(A_inv)


# ## Part 2: Introduction to Pandas

# In[11]:


#Installation and Setup
get_ipython().system('pip install pandas')
import pandas as pd


# In[12]:


#Create a Series From a List
s1 = pd.Series([1, 2, 3, 4, 5])
print(s1)

#Create a Series From a Array
s2 = pd.Series(np.array([1, 2, 3, 4, 5]))
print(s2)

#Create a Series From a Dictionary
s3 = pd.Series({'a': 1, 'b': 2, 'c': 3})
print(s3)


# In[13]:


#Indexing and Slicing Series

#Accessing Elememts by Label
print(s3['a'])

#Accessing Elememts by Position
print(s3[0])

#Slicing
print(s3[:2])


# In[14]:


#Operations on Series

#Arithmetic Operations
s4 = s1 + s2
print(s4)

#Element-wise Operations
s5 = s1 * 2
print(s5)

#Aggregation Function
print(s1.sum())


# In[15]:


#Handling Missing Data

#Drop Missing Values
s6 = s1.dropna()

#Fill Missing Values
s7 = s1.fillna(0)

#Check for Missing Values
print(s1.isnull())


# In[16]:


#Creating DataFrames

#Create DataFrame From a Dictionary
data = {'Name': ['Alice', 'Bob', 'Charlie'], 'Age': [25, 30, 35]}
df1 = pd.DataFrame(data)
print(df1)

#Create DataFrame From a List
data = [['Alice', 25], ['Bob', 30], ['Charlie', 35]]
df2 = pd.DataFrame(data, columns = ['Name', 'Age'])
print(df2)


# In[17]:


#Indexing and Slicing DataFrames

#Label-based Indexing
print(df1.loc[0, 'Name'])

#Position-based Indexing
print(df1.iloc[0, 0])

#Slicing
print(df1[:2])


# In[19]:


#Basic Operations

#Sorting
df1_sorted = df1.sort_values(by = 'Age')
print(df1_sorted)

#Filtering
df1_filtered = df1[df1['Age'] > 30]
print(df1_filtered)

#Selecting Columns
names = df1['Name']
print(names)


# In[22]:


#Data Manipulation

#Adding a Column
df1['Gender'] = ['Female', 'Male', 'Male']
print(df1)

#Deleting a Column
del df1['Gender']
print(df1)

#Updating a Column
df1['Age'] = df1['Age'] + 1
print(df1)


# In[23]:


#Handling Missing Data

#Drop Missing Values
df1_cleaned = df1.dropna()

#Fill Missing Values
df1_filled = df1.fillna(0)

#Check for Missing Values
print(df1.isnull().any())


# In[28]:


#Concatenating DataFrames

#Create Simple DataFrames
df1 = pd.DataFrame({'A': ['A0', 'A1', 'A2'], 'B': ['B0', 'B1', 'B2']})
df2 = pd.DataFrame({'A': ['A3', 'A4', 'A5'], 'B': ['B3', 'B4', 'B5']})

#Concatenate Along Rows
result_row = pd.concat([df1, df2])
print(result_row)

#Concatenate Along Columns
result_col = pd.concat([df1, df2], axis = 1)
print(result_col)


# In[35]:


#Merging DataFrames

#Create Simple DataFrames
left = pd.DataFrame({'Key' : ['K0', 'K1', 'K2'], 'Value' : ['V0', 'V1', 'V2']})
right = pd.DataFrame({'Key' : ['K3', 'K4', 'K5'], 'Value' : ['V3', 'V4', 'V5']})

#Inner Join
inner_join = pd.merge(left, right, on = 'Key', how = 'inner')
print(inner_join)

#Left Join
left_join = pd.merge(left, right, on = 'Key', how = 'left')
print(left_join)

#Right Join
right_join = pd.merge(left, right, on = 'Key', how = 'right')
print(right_join)

#Outer Join
outer_join = pd.merge(left, right, on = 'Key', how = 'outer')
print(outer_join)


# In[42]:


#Joining DataFrames

##Create Simple DataFrames
left = pd.DataFrame({'Value1' : [1, 2, 3]}, index = ['A', 'B', 'C'])
right = pd.DataFrame({'Value2' : [4, 5, 6]}, index = ['D', 'E', 'F'])

#Join Based on Index
join_df = left.join(right, how = 'inner')
print(join_df)

