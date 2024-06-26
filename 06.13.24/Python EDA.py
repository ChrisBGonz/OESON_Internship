# -*- coding: utf-8 -*-
"""06.13.24/Python EDA.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1oncTbqsZzTr0lE17V5w972mKkICvmhTC

# Fastag Frauds Records

## About Dataset

### The dataset comprises transaction records from the Fastag electronic toll collection system in India. It includes various features such as transaction details, vehicle information, geographical location, and transaction amounts. The dataset is labeled with a binary indicator for fraudulent activity, making it suitable for developing a fraud detection system.
"""

#Import Necessary Libraries
import pandas as pd

#Adjust the file name if necessary based
filename = 'FastagFraudDetection.csv'
data = pd.read_csv(filename)

#Display the First Few Rows of the Dataset
print(data.head())

#Questions: How many rows and columns does this dataset contain?

"""#Basic Information"""

#Display all the basic information
print(data.info())

#Dataset Summary
print(data.describe())

#Check for Missing Values
print(data.isnull().sum())

"""###From the data, we can say there are 549 missing Fastag ID's from the dataset."""

#Replace Missing Values From Mean and Median for Numerical Columns Only
data.fillna(data.mean(numeric_only = True), inplace = True)

#Replace Missing Values From Mode for Numerical Columns Only
data['FastagID'].fillna(data['FastagID'].mode()[0], inplace = True)

#Verify Previous Missing Values
print(data.isnull().sum())

"""##Data Visualization"""

#What does histogram tell you
#Outliers baed on boxplot
#What insights can you draw from

#Import Necessary Libraries
import matplotlib.pyplot as plt
import seaborn as sns

#Histogram for Numerical Columns
data.hist(bins = 20, figsize = (12, 9))
plt.show()

#Boxplot for Numerical Values
plt.figure(figsize = (15, 10))
sns.boxplot(data = data)
plt.xticks(rotation = 90)
plt.show()

#Count Plots for Categorical Columns
categorical = ['Timestamp', 'Vehicle_Type', 'FastagID', 'TollBoothID', 'Lane_Type',
               'Vehicle_Dimensions', 'Geographical_Location', 'Vehicle_Plate_Number', 'Fraud_indicator']

for col in categorical:
  plt.figure(figsize = (15, 10))
  sns.countplot(x = col, data = data)
  plt.show()
