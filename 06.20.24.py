#!/usr/bin/env python
# coding: utf-8

# # Introduction to Functions

# ## A function is a block or organized, reusable code that is used to perform a single, relation action. Functions provide better modularity for your application and a high degree of code reusability.

# In[1]:


'''
def function_name(parameters):

    """
    Docstring explaining the function
    """
    
    Function Body
    return result
'''


# ## Key Concepts

# 1. Function Definition and Calling:
# - Define a function using a def keyword
# - Call a function using its name followed by parentheses
# 
# 2. Parameters and Arguments:
# - Parameters: Variables listed inside the parantheses in the function definition
# - Arguments: Values passed to the function 
# 
# 3. Return Statement:
# - The return statement is used to exit a function and go back to the place from where it was called.
# 
# 4. Default Arguments:
# - Function arguments can have default values.
# 
# 5. Variable Scope:
# - Variables defined inside a function are local to that function.

# In[2]:


#Basic Function
def greet(name):
    '''
    Function to greet a person.
    '''
    
    return f"Hello, {name}!"

#Calling the Function
print(greet('Alice'))


# In[3]:


#Function with Default Arguments
def greet(name = 'World'):
    '''
    Function to greet a person with a default name.
    '''
    
    return f"Hello, {name}!"

#Calling the Function
print(greet())
print(greet('Alice'))


# In[4]:


#Function Returning Multiple Values
def arithmetic_operations(a, b):
    '''
    Function to perform arithmetic operations.
    '''
    
    addition = a + b
    subtraction = a - b
    multiplication = a * b
    division = a / b if b != 0 else none
    
    return addition, subtraction, multiplication, division

#Calling the Function
add, sub, mul, div = arithmetic_operations(10, 5)
print(f'Add: {add}, Subtract: {sub}, Multiply: {mul}, Divide: {div}')


# ## Functions in Data Science

# In[5]:


#Reading Data
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

def read_data(file_path):
    '''
    Function to read CSV data.
    '''
    
    return pd.read_csv(file_path, header = None, names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'])

#Example Usage
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
df = read_data(url)
print(df.head())


# In[6]:


#Data Preprocessing
def preprocess_data(df):
    '''
    Function to preprocess data.
    '''
    
    #Encode the Target Variable
    label_encoder = LabelEncoder()
    df['species'] = label_encoder.fit_transform(df['species'])
    
    return df

#Preprocess the Data
df_clean = preprocess_data(df)
print(df_clean.head())


# In[7]:


#Model Training
def train_model(x, y):
    '''
    Function to train a logistic regression model
    '''
    
    x_train, x_test, y_train, y_test, = train_test_split(x, y, test_size = 0.2, random_state = 42)
    model = LogisticRegression(max_iter = 200)
    model.fit(x_train, y_train)
    
    return model, x_test, y_test

#Example Usage
x = df_clean.drop('species', axis = 1)
y = df_clean['species']
model, x_test, y_test = train_model(x, y)
print(f'Model Accuracy: {model.score(x_test, y_test)}')


# ## Real-World Questions on Functions

# In[8]:


#Question 1: Calculate the Monthly Sales Average
""" 
You are given a dictionary where the keys are product names and the values are lists of monthly 
sales figures for each product. Write a function to calculate the average monthly sales for each 
product. 
"""

import numpy as np

#Sample Data
sales_data = {
    'Product A': [150, 200, 250, 300],
    'Product B': [400, 500, 600, 700],
    'Product C': [100, 150, 200, 250]

}

def calculate_monthly_averages(sales_data):
    '''
    Function to calculate the average monthly sales for each product.
    '''
    
    averages = {}
    
    for product, sales in sales_data.items():
        averages[product]  = np.mean(sales)
        
    return averages

#Example Usage
averages = calculate_monthly_averages(sales_data)
print(averages)


# In[9]:


#Question 2: Count Unique Values in a Dataframe Column
'''
Given a DataFrame, write a function to count the number of unique values in a specified column.
'''

import pandas as pd

#Sample data
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Alice'],
    'Age': [25, 30, 35, 40, 25]
}
df = pd.DataFrame(data)

def count_unique_values(df, column):
    '''
    Function to count the number of unique values in a specified column.
    '''
    
    return df[column].nunique()

#Example Usage
unique_names_count = count_unique_values(df, 'Name')
print(unique_names_count)

unique_ages_count = count_unique_values(df, 'Age')
print(unique_ages_count)


# In[10]:


#Question 3: Normalize Data in a DataFrame
'''
Write a function to normalize the data in a DataFrame such that each value 
in a column is scaled to a range of 0 to 1.
'''

import pandas as pd

#Sample data
data = {
    'A': [1, 2, 3, 4, 5],
    'B': [10, 20, 30, 40, 50]
}
df = pd.DataFrame(data)

def normalize_column(df):
    '''
    Function to normalize the data in the DataFrame.
    '''
    
    result = df.copy()
    
    for column in result.columns:
        max_value = result[column].max()
        min_value = result[column].min()
        result[column] = (result[column] - min_value) / (max_value - min_value)
        
    return result

#Example usage
normalized_df = normalize_column(df)
print(normalized_df)


# In[11]:


#Question 4: Find Missing Values
'''
Given a DataFrame, write a function to find and return the number of missing values in each column.
'''

import pandas as pd

#Sample data
data = {
 'A': [1, 2, None, 4, 5],
 'B': [None, 2, 3, 4, None],
 'C': [1, None, 3, None, 5]
}
df = pd.DataFrame(data)

def find_missing_values(df):
 '''
 Function to find the number of missing values in each column.
 '''

 return df.isnull().sum()

#Example usage
missing_values = find_missing_values(df)
print(missing_values)


# In[12]:


#Question 6: Grouping and Aggregation
'''
How do you group a DataFrame by a column and compute the sum of another column?
'''

import pandas as pd

#Sample data
data = {
 'Department': ['HR', 'IT', 'HR', 'IT', 'Finance'],
 'Salary': [50000, 60000, 45000, 65000, 70000]
}
df = pd.DataFrame(data)

def group_and_aggregate(df):
 '''
 Function to group by 'Department' and compute the sum of 'Salary'.
 '''
 return df.groupby('Department')['Salary'].sum().reset_index()

#Example usage
result = group_and_aggregate(df)
print(result)


# ## Pandas Questions

# In[13]:


#Question 1: Filter DataFrame by Multiple Conditions: Given a DataFrame with columns Name, Age, and Score, write a function to filter rows where Age is greater than 25 and Score is greater than 80.

import pandas as pd

#Sample data
data = {
 'Name': ['Zeik', 'Sarah', 'Joey', 'Victor', 'Mark'],
 'Age': [26, 21, 19, 29, 27],
 'Score': [82, 80, 75, 97, 88]
}

df = pd.DataFrame(data)
def filter_df(df):

    return df[(df['Age'] > 25) & (df['Score'] > 80)]

print(filter_df(df))


# In[14]:


#Question 2: Pivot Table Creation: Given a DataFrame with columns Department, Employee, and Salary,write a function to create a pivot table that shows the average salary for each department.

import pandas as pd

data = {
 'Department': ['HR', 'IT', 'HR', 'IT', 'Finance'],
 'Employee': ['Kartherine', 'Matt', 'Ethan', 'Mia', 'Amanda'],
 'Salary': [80000, 110000, 85000, 122000, 90000]
}

df = pd.DataFrame(data)
def pivot_table(df):
    pivot_table = df.pivot_table(index = 'Department', values = 'Salary', aggfunc = 'mean')

    return pivot_table

print(pivot_table(df))


# In[15]:


#Question 3: Merge DataFrames: Given two DataFrames, df1 and df2, which both have a common column ID, write a function to merge these DataFrames on the ID column and return the merged DataFrame.

import pandas as pd

df1 = pd.DataFrame({'ID': [1, 2, 3], 'Name': ['Alice', 'Bob', 'Charlie']})
df2 = pd.DataFrame({'ID': [2, 3, 4], 'Age': [25, 30, 35]})

def merge_dataframes(df1, df2, id_column):
    merged_df = pd.merge(df1, df2, on = id_column)

    return merged_df

print(merge_dataframes(df1, df2, 'ID'))


# In[16]:


#Question 4: Handle Missing Values: Write a function to fill missing values in a DataFrame. If the column is numerical, fill with the mean of the column. If the column is categorical, fill with the mode of the column.

import pandas as pd

#Sample data
data = {
 'A': [1, 2, None, 4, 5],
 'B': ['Apple', 'Banana', 'Banana', 'Orange', None],
 'C': [14, None, 38, None, 23]
 }

df = pd.DataFrame(data)

def fill_missing_values(df):
    for col in df.columns:

        if df[col].dtype == 'float64':
            df[col] = df[col].fillna(df[col].mean())

        else:
            df[col] = df[col].fillna(df[col].mode()[0])

    return df

print(fill_missing_values(df))


# ## Lists Questions

# In[1]:


#Question 1: Chunk a List: Write a function to split a list into chunks of a given size. For example, given the list [1, 2, 3, 4, 5, 6, 7, 8, 9] and chunk size 3, the function should return [[1, 2, 3], [4, 5, 6], [7, 8, 9]].

data = [1, 2, 3, 4, 5, 6, 7, 8, 9]
chunk_size = 3

def chunk_list(data, chunk_size):
    return [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]

print(chunk_list(data, chunk_size))


# In[2]:


#Question 2: List Intersection: Write a function to find the intersection of two lists, returning a list of elements that are present in both lists.

list1 = [1, 2, 3, 4, 5]
list2 = [3, 4, 5, 6, 7]

def list_intersection(list1, list2):
    intersection = []
    for item in list1:

        if item in list2:
            intersection.append(item)

    return intersection

print(list_intersection(list1, list2))


# In[3]:


#Question 3: Rotate List: Write a function to rotate a list n positions to the left. For example, rotating [1, 2, 3, 4, 5] by 2 positions should result in [3, 4, 5, 1, 2].

data = [1, 2, 3, 4, 5]
n = 2

def rotate_list(data, n):
    return data[n:] + data[:n]

print(rotate_list(data, 2))


# In[4]:


#Question 4: Find Duplicates: Write a function to find all duplicate elements in a list. The function should return a list of duplicates.

data = [1, 2, 3, 4, 2, 5, 3, 6]

def find_duplicates(data):
    duplicates = []
    seen = set()

    for item in data:

        if item in seen:
            duplicates.append(item)

        else:
            seen.add(item)

    return duplicates

print(find_duplicates(data))


# In[5]:


#Question 5: Cumulative Sum: Write a function to compute the cumulative sum of a list. For example, given the list [1, 2, 3, 4], the function should return [1, 3, 6, 10].

mylist = [1, 2, 3, 4]

def cumulative_sum(mylist):
    cumulative_sum = []
    sum = 0

    for num in mylist:
        sum += num
        cumulative_sum.append(sum)

    return cumulative_sum

print(cumulative_sum(mylist))

