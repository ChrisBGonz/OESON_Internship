#!/usr/bin/env python
# coding: utf-8

# # Medicine Details Analysis

# ## Your task is to analyze a dataset containing detailed information about over 11,000 medicines, including their salt compositions, uses, side effects, manufacturers, and user reviews. The goal is to uncover patterns and insights that can help improve decision-making in the healthcare industry and enhance patient outcomes.

# In[103]:


#Import Necessary Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score)

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam


# In[105]:


#Load the dataset
data = pd.read_csv('Medicine_Details.csv')
data.head(10)


# ## Data Preprocessing

# In[108]:


#Check for Missing Values
print(data.isnull().sum())


# In[110]:


#Drop Irrelevant Columns
data = data.drop(columns = ['Image URL'])
data.head()


# In[112]:


#Separate Features and Target Variable
x = data.drop('Excellent Review %', axis = 1)
y = data['Excellent Review %']

#Identify Categorical and Numerical Columns
categorical_cols = x.select_dtypes(include = ['object']).columns
numerical_cols = x.select_dtypes(include = ['float64', 'int64']).columns

#Data Preprocessing for Numerical Data
numerical_transformer = Pipeline(steps = [('scaler', StandardScaler())])

#Data Preprocessing for Categorical Data
categorical_transformer = Pipeline(steps = [('onehot', OneHotEncoder(handle_unknown = 'ignore'))])

#Combine Numerical and Categorical Data
preprocessor = ColumnTransformer(
    transformers = [('num', numerical_transformer, numerical_cols), ('cat', categorical_transformer, categorical_cols)])

#Apply Transformations
x = preprocessor.fit_transform(x)


# In[114]:


#Splitting the Dataset Into Training and Test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)


# ## EDA

# In[117]:


#General Statistics of Data
data.describe()


# ## Machine Learning Models

# In[120]:


#Linear Regression
lr_model = LinearRegression()
lr_model.fit(x_train, y_train)

#Make Predictions on the Test Set
lr_predict = lr_model.predict(x_test)

#Evaluate
print('Linear Regression Model: ')
print(f'Mean Absolute Error : {mean_absolute_error(y_test, lr_predict)}')
print(f'Mean Squared Error : {mean_squared_error(y_test, lr_predict)}')
print(f'R-Squared : {r2_score(y_test, lr_predict)}')


# In[122]:


#Decision Tree
dt_model = DecisionTreeRegressor()
dt_model.fit(x_train, y_train)

#Make Predictions
dt_predict = dt_model.predict(x_test)

#Evaluate
print('Decision Tree Model: ')
print(f'Mean Absolute Error : {mean_absolute_error(y_test, dt_predict)}')
print(f'Mean Squared Error : {mean_squared_error(y_test, dt_predict)}')
print(f'R-Squared : {r2_score(y_test, dt_predict)}')


# In[124]:


#Random Forest
rfr_model = RandomForestRegressor()
rfr_model.fit(x_train, y_train)

#Make Predictions
rfr_predict = rfr_model.predict(x_test)

#Evaluate
print('Random Forest Regressor Model: ')
print(f'Mean Absolute Error : {mean_absolute_error(y_test, rfr_predict)}')
print(f'Mean Squared Error : {mean_squared_error(y_test, rfr_predict)}')
print(f'R-Squared : {r2_score(y_test, rfr_predict)}')


# ## Deep Learning Models

# In[127]:


#General info
data.info()


# In[129]:


#Data Preprocessing
features = data[['Excellent Review %', 'Average Review %', 'Poor Review %']]

#Scale data
scaler = MinMaxScaler(feature_range = (0, 1))
scaled_data = scaler.fit_transform(features)


# In[131]:


#Prepare Training Data
def create_sequences(data, seq_length):
    xs, ys = [], []
    
    for i in range(len(data) - seq_length):
        x = data[i:i + seq_length]
        y = data[i + seq_length][2] 
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

seq_length = 60
x, y = create_sequences(scaled_data, seq_length)


# In[133]:


#Split Data Into Training and Test Sets
split = int(0.8 * len(x))
x_train, x_test = x[:split], x[split:]
y_train, y_test = y[:split], y[split:]


# In[135]:


#Building the Model
model = Sequential()
model.add(LSTM(units = 50, return_sequences = True, input_shape = (x_train.shape[1], x_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units = 50, return_sequences = False))
model.add(Dropout(0.2))
model.add(Dense(units = 1))

model.compile(optimizer = 'adam', loss = 'mean_squared_error')


# In[137]:


#Training the Model
history = model.fit(x_train, y_train, epochs = 100, batch_size = 32, validation_split = 0.1)


# In[139]:


#Evaluating the Model
predicted_review = model.predict(x_test)
predicted_review = scaler.inverse_transform(np.concatenate((np.zeros((predicted_review.shape[0], 2)), predicted_review), axis = 1))[:, 2]

#Inverse transform the actual medals
actual_review = scaler.inverse_transform(np.concatenate((np.zeros((y_test.shape[0], 2)), y_test.reshape(-1, 1)), axis = 1))[:, 2]


# In[141]:


#Calculate Performance Metrics
mae = mean_absolute_error(actual_review, predicted_review)
mse = mean_squared_error(actual_review, predicted_review)
r2 = r2_score(actual_review, predicted_review)

print(f'MSE: {mse}')
print(f'MAE: {mae}')
print(f'R2: {r2}')

