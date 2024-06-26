# -*- coding: utf-8 -*-
"""Python EDA (Continued).ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1pSgXNYzac7TCnGJ6H2IrtI6ZhdpxL4ow
"""

#Code from 06/13/24

"""# Fastag Frauds Records

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

#Code from 06/18/24 (Continued)

#Distribution of Vehicle Speed
plt.figure(figsize = (10, 6))
sns.histplot(data = data, x = 'Vehicle_Speed', bins = 20, kde = True)
plt.title('Distribution of Vehicle Speed')
plt.show()

#Pair Plot for Numerical Features
plt.figure(figsize = (12, 10))
sns.pairplot(data, diag_kind = 'kde', hue = 'Fraud_indicator')
plt.suptitle('Pair Plot of Numerical Features', y = 1.02)
plt.show()

#Violin Plot for Transaction Amount by Vehicle Type and Fraud Indicator
plt.figure(figsize = (12, 8))
sns.violinplot(data = data, x = 'Vehicle_Type', y = 'Transaction_Amount', hue = 'Fraud_indicator', split = True)
plt.title('Transaction Amount by Vehicle Type and Fraud Indicator')
plt.show()

"""## Detecting Outliers"""

#Boxplot for Detecting Outliers for Transaction Amount
plt.figure(figsize = (10, 6))
sns.boxplot(data = data, x = 'Transaction_Amount')
plt.title('Boxplot for Transaction Amount')
plt.show()

"""## Correlation Analysis"""

#Correlation Matrix
corr_matrix = data.corr(numeric_only = True)

#Heatmap of Correlation Matrix
plt.figure(figsize = (10, 7))
sns.heatmap(corr_matrix, annot = True, cmap = 'coolwarm', linewidths = 0.5)
plt.title('Correlation Matrix')
plt.show()

#How do transaction amounts vary by vehicle type (e.g., Car, Bus, Truck)?

#Transaction Amounts by Vehicle Type
plt.figure(figsize = (10, 8))
sns.boxplot(data = data, x = 'Vehicle_Type', y = 'Transaction_Amount')
plt.title('Transaction Amounts by Vehicle Type')
plt.show()

#Is there a correlation between vehicle speed and the likelihood of a transaction being fraudulent?

correlation = data[['Vehicle_Speed', 'Transaction_Amount']].corr()
print(correlation)
