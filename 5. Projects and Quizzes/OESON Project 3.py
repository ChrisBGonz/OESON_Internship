#!/usr/bin/env python
# coding: utf-8

# # Machine Learning/Deep Learning

# ## In this project, you will explore basic machine learning (ML) and deep learning (DL) techniques to predict the number of Olympic medals a country will win. The dataset provided includes features such as GDP, population, and sports index, along with the actual number of medals won. You will build and evaluate different models to understand which factors are most influential in predicting Olympic success.

# In[3]:


#Import Necessary Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
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

from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV


# In[5]:


#Load the dataset
data = pd.read_csv('OlympicMedals.csv')
data.head(10)


# In[7]:


x = data.iloc[:, :-1].values
y = data.iloc[:, -1].values


# ## Step 1: Data Preprocessing

# In[10]:


#Check for Missing Values
print(data.isnull().sum())


# ### Note: Missing values were handled manually.

# In[13]:


#Drop Irrelevant/Duplicate Columns
data = data.drop(columns = ['iso', 'ioc', 'name', 'continent', 'olympicsIndex', 'sportsIndex'])
data.head()


# In[15]:


#Extract features and target
x = data.drop(columns = ['total'])  
y = data['total'] 

#Standardizing Data
scaler = StandardScaler()
x = scaler.fit_transform(x)


# In[17]:


#Splitting the Dataset Into Training and Test sets
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)


# ## Step 2: Explatory Data Analysis (EDA)

# In[20]:


#General Statistics of Data
data.describe()


# In[22]:


sns.scatterplot(x = data['population'], y = data['total'])
plt.xlabel('population')
plt.ylabel('Total Medals')
plt.title('Population vs Total Medals')
plt.show()


# In[24]:


sns.scatterplot(x = data['gdp'], y = data['total'])
plt.xlabel('GDP')
plt.ylabel('Total Medals')
plt.title('GDP vs Total Medals')
plt.show()


# In[26]:


sns.scatterplot(x = data['olympics_index'], y = data['total'])
plt.xlabel('Olympics Index')
plt.ylabel('Total Medals')
plt.title('Olympics Index vs Total Medals')
plt.show()


# In[28]:


sns.scatterplot(x = data['sports_index'], y = data['total'])
plt.xlabel('Sports Index')
plt.ylabel('Total Medals')
plt.title('Sports Index vs Total Medals')
plt.show()


# In[30]:


#Visualizing the Data (Correlation Matrix)
corr_matrix = data.corr(numeric_only = True)

#Heatmap of Correlation Matrix
plt.figure(figsize = (8, 6))
sns.heatmap(corr_matrix, annot = True, cmap = 'coolwarm', linewidths = 0.5)
plt.title('Correlation Matrix')
plt.show()


# ## Step 3: Machine Learning Models

# In[33]:


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


# In[35]:


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


# In[37]:


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


# ## Step 4: Deep Learning Models

# In[40]:


#General info
data.info()


# In[42]:


#Data Preprocessing
features = data[['population', 'gdp', 'olympics_index', 'sports_index', 'total', 'gold', 'silver', 'bronze']]

#Scale data
scaler = MinMaxScaler(feature_range = (0, 1))
scaled_data = scaler.fit_transform(features)


# In[44]:


#Prepare Training Data
def create_sequences(data, seq_length):
    xs, ys = [], []
    
    for i in range(len(data) - seq_length):
        x = data[i:i + seq_length]
        y = data[i + seq_length][4] 
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

seq_length = 60
x, y = create_sequences(scaled_data, seq_length)


# In[46]:


#Split Data Into Training and Test Sets
split = int(0.8 * len(x))
x_train, x_test = x[:split], x[split:]
y_train, y_test = y[:split], y[split:]


# In[48]:


#Building the Model
model = Sequential()
model.add(LSTM(units = 50, return_sequences = True, input_shape = (x_train.shape[1], x_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units = 50, return_sequences = False))
model.add(Dropout(0.2))
model.add(Dense(units = 1))

model.compile(optimizer = 'adam', loss = 'mean_squared_error')


# In[50]:


#Training the Model
history = model.fit(x_train, y_train, epochs = 100, batch_size = 32, validation_split = 0.1)


# In[52]:


#Evaluating the Model
predicted_medals = model.predict(x_test)
predicted_medals = scaler.inverse_transform(np.concatenate((np.zeros((predicted_medals.shape[0], 7)), predicted_medals), axis = 1))[:, 7]

#Inverse transform the actual medals
actual_medals = scaler.inverse_transform(np.concatenate((np.zeros((y_test.shape[0], 7)), y_test.reshape(-1, 1)), axis = 1))[:, 7]


# In[54]:


#Calculate Performance Metrics
mae = mean_absolute_error(actual_medals, predicted_medals)
mse = mean_squared_error(actual_medals, predicted_medals)
r2 = r2_score(actual_medals, predicted_medals)

print(f'MSE: {mse}')
print(f'MAE: {mae}')
print(f'R2: {r2}')

