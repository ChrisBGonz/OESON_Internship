#!/usr/bin/env python
# coding: utf-8

# # Model Evaluation

# 1. Data Preparation
# 2. Train Test Split
# 3. Model Training
# 4. Model Evaluation Using Metrics
# 5. Cross - Validation
# 6. Confusion Matrix
# 7. Precision - Recall Curve
# 8. ROC Curve

# ## 1. Data Preparation

# In[1]:


#Import Necessary Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score,f1_score, confusion_matrix, precision_recall_curve,
                             roc_curve, roc_auc_score, mean_squared_error, mean_absolute_error, r2_score)

from sklearn.datasets import load_iris, fetch_california_housing
from sklearn.linear_model import LogisticRegression, LinearRegression


# ### Regression Metrics:
# 1. MSE (Mean Squared Error)
# 2. MAE (Mean Absolute Error)
# 3. RMSE (Root Mean Squared Error)
# 4. R2 Score (R Squared Score): Value ranges between 0 and 1. Higher score = more accurate.

# ### Classification Metrics:
# 1. Accuracy: Values are correctly classified.
# 2. Precision: How many values are relevent from selected data.
# 3. Recall (Sensitivity): How many relevant items are selected.
# 4. F1 Score: For imbalanced dataset, make balance precision and recall.
# 5. ROC (Receiver Operating Characteristics) and AUC (Area Under Curve): AUC ranges from 0.5 to 1, where 0.5 represents random distribution and 1 represent perfect distribution.

# ## 2. Train Test Split

# In[2]:


#Iris Dataset

#Train Test Split
iris = load_iris()
x_iris = iris.data
y_iris = iris.target

#Splitting Into the Training Set and Test Set
x_train_iris, x_test_iris, y_train_iris, y_test_iris = train_test_split(x_iris, y_iris, 
                                                                        test_size = 0.2, random_state = 42)


# In[3]:


#House Dataset

#Train Test Split
house = fetch_california_housing()
x_house = house.data
y_house = house.target

#Splitting Into the Training Set and Test Set
x_train_house, x_test_house, y_train_house, y_test_house = train_test_split(x_house, y_house, 
                                                                        test_size = 0.2, random_state = 42)


# ## 3. Model Training

# In[4]:


#Train Logistic Regression Model
clf = LogisticRegression(max_iter = 200)
clf.fit(x_train_iris, y_train_iris)

#Predict on Test Set
y_pred_iris = clf.predict(x_test_iris)


# In[5]:


#Train Linear Regression Model 
reg = LinearRegression()
reg.fit(x_train_house, y_train_house)

#Predict on Test Set
y_pred_house = reg.predict(x_test_house)

