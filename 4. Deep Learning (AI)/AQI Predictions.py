#!/usr/bin/env python
# coding: utf-8

# # Air Quality Index (AQI) (Continued)

# In[2]:


#From 07/26/24


# In[4]:


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


# In[6]:


#Load Dataset
data = pd.read_csv('AQI.csv')
data.head(10)


# In[8]:


#Check for Missing Values
print(data.isnull().sum())


# In[10]:


#Feature Engineering
data['Total Pollutant Days'] = data['Days CO'] + data['Days NO2'] + data['Days Ozone'] + data['Days PM2.5'] + data['Days PM10']


# In[12]:


#Data Normalization
scaler = StandardScaler()

features = ['Days with AQI', 'Good Days', 'Moderate Days', 'Unhealthy for Sensitive Groups Days', 'Unhealthy Days', 'Very Unhealthy Days', 
'Hazardous Days', 'Max AQI', '90th Percentile AQI', 'Median AQI', 'Days CO', 'Days NO2', 'Days Ozone', 'Days PM2.5', 'Days PM10',
'Total Pollutant Days']

data[features] = scaler.fit_transform(data[features])


# In[14]:


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


# In[16]:


#Compute the Correlation Matrix
numeric_data = data.select_dtypes(include = ['float64', 'int64'])
corr_matrix = numeric_data.corr()

#Heatmap of Correlations
plt.figure(figsize = (8, 5))
sns.heatmap(corr_matrix, annot = True, cmap = 'coolwarm', fmt = '.2f')
plt.title('Correlation Matrix')
plt.show()


# In[18]:


############################################################################################################################################


# ## Data Modeling

# In[23]:


#Drop Non-Numeric Columns
x = data.drop(columns = ['Max AQI', 'State', 'County'])
y = pd.to_numeric(data['Max AQI'], errors = 'coerce')

#Drop NAN Values
x = x.dropna()
y = y[x.index]

#Split the Dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)


# ## Train and Evaluate

# In[34]:


#Initialize
lr_model = LinearRegression()
lr_model.fit(x_train, y_train)

#Make Predictions
lr_predict = lr_model.predict(x_test)

#Evaluate
print('Linear Regression Model: ')
print(f'MAE : {mean_absolute_error(y_test, lr_predict)}')
print(f'MSE : {mean_squared_error(y_test, lr_predict)}')
print(f'R2 : {r2_score(y_test, lr_predict)}')


# In[32]:


#Initialize
rfr_model = RandomForestRegressor()
rfr_model.fit(x_train, y_train)

#Make Predictions
rfr_predict = rfr_model.predict(x_test)

#Evaluate
print('Random Forest Regressor Model: ')
print(f'MAE : {mean_absolute_error(y_test, rfr_predict)}')
print(f'MSE : {mean_squared_error(y_test, rfr_predict)}')
print(f'R2 : {r2_score(y_test, rfr_predict)}')


# In[38]:


#Initialize
svr_model = SVR(kernel = 'rbf')
svr_model.fit(x_train, y_train)

#Make Predictions
svr_predict = svr_model.predict(x_test)

#Evaluate
print('Suppor Vector Regression Model: ')
print(f'MAE : {mean_absolute_error(y_test, svr_predict)}')
print(f'MSE : {mean_squared_error(y_test, svr_predict)}')
print(f'R2 : {r2_score(y_test, svr_predict)}')

