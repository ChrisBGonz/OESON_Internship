#!/usr/bin/env python
# coding: utf-8

# # Air Quality Index (AQI)

# In[8]:


#Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


# In[12]:


#Load Dataset
data = pd.read_csv('AQI.csv')
data.head(10)


# In[14]:


#Check for Missing Values
print(data.isnull().sum())


# In[16]:


#Feature Engineering
data['Total Pollutant Days'] = data['Days CO'] + data['Days NO2'] + data['Days Ozone'] + data['Days PM2.5'] + data['Days PM10']


# In[20]:


#Data Normalization
scaler = StandardScaler()

features = ['Days with AQI', 'Good Days', 'Moderate Days', 'Unhealthy for Sensitive Groups Days', 'Unhealthy Days', 'Very Unhealthy Days', 
'Hazardous Days', 'Max AQI', '90th Percentile AQI', 'Median AQI', 'Days CO', 'Days NO2', 'Days Ozone', 'Days PM2.5', 'Days PM10',
'Total Pollutant Days']

data[features] = scaler.fit_transform(data[features])


# In[40]:


#Columns to consider for histograms
columns = ['Max AQI', 'Days with AQI', 'Good Days', 'Moderate Days', 'Unhealthy for Sensitive Groups Days', 'Unhealthy Days', 
           'Very Unhealthy Days', 'Hazardous Days']

#Creating histograms
for column in columns:
    plt.figure(figsize = (8, 6))
    plt.hist(data[column], bins = 30, alpha = 0.7, color = 'blue')
    plt.title(f'Histogram of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()


# In[64]:


#Compute the Correlation Matrix
numeric_data = data.select_dtypes(include = ['float64', 'int64'])
corr_matrix = numeric_data.corr()

#Heatmap of Correlations
plt.figure(figsize = (8, 5))
sns.heatmap(corr_matrix, annot = True, cmap = 'coolwarm', fmt = '.2f')
plt.title('Correlation Matrix')
plt.show()

