#!/usr/bin/env python
# coding: utf-8

# In[1]:


#From 07/09/24 ####################################################################################################


# # Model Evaluation

# 1. Data Preparation
# 2. Train Test Split
# 3. Model Training
# 4. Model Evaluation Using Metrics
# 5. Cross Validation
# 6. Confusion Matrix
# 7. Precision-Recall Curve

# ## 1. Data Preparation

# In[6]:


#Import Necessary Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score,f1_score, confusion_matrix, precision_recall_curve,
                             recall_score, roc_curve, auc, roc_auc_score, mean_squared_error, mean_absolute_error, r2_score)

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

# In[11]:


#Iris Dataset

#Train Test Split
iris = load_iris()
x_iris = iris.data
y_iris = iris.target

#Splitting Into the Training Set and Test Set
x_train_iris, x_test_iris, y_train_iris, y_test_iris = train_test_split(x_iris, y_iris, 
                                                                        test_size = 0.2, random_state = 42)


# In[13]:


#House Dataset

#Train Test Split
house = fetch_california_housing()
x_house = house.data
y_house = house.target

#Splitting Into the Training Set and Test Set
x_train_house, x_test_house, y_train_house, y_test_house = train_test_split(x_house, y_house, 
                                                                        test_size = 0.2, random_state = 42)


# ## 3. Model Training

# In[16]:


#Train Logistic Regression Model
clf = LogisticRegression(max_iter = 200)
clf.fit(x_train_iris, y_train_iris)

#Predict on Test Set
y_pred_iris = clf.predict(x_test_iris)


# In[18]:


#Train Linear Regression Model 
reg = LinearRegression()
reg.fit(x_train_house, y_train_house)

#Predict on Test Set
y_pred_house = reg.predict(x_test_house)


# In[20]:


###################################################################################################################


# ## 4. Model Evaluation Using Metrics

# In[23]:


#Classification Metrics
accuracy = accuracy_score(y_test_iris, y_pred_iris)
precision = precision_score(y_test_iris, y_pred_iris, average = 'weighted')
recall = recall_score(y_test_iris, y_pred_iris, average = 'weighted')
f1 = f1_score(y_test_iris, y_pred_iris, average = 'weighted')

print(f'Accuracy: {accuracy}')
print(f'Precision: {accuracy}')
print(f'Recall: {accuracy}')
print(f'F1: {accuracy}')


# In[25]:


#Regression Metrics
mse = mean_squared_error(y_test_house, y_pred_house)
rmse = mean_squared_error(y_test_house, y_pred_house, squared = False)
mae = mean_absolute_error(y_test_house, y_pred_house)
r2 = r2_score(y_test_house, y_pred_house)

print(f'MSE: {mse}')
print(f'RMSE: {rmse}')
print(f'MAE: {mae}')
print(f'R2: {r2}')


# ## 5. Cross Validation

# In[28]:


#Classification CV
cv_score = cross_val_score(clf, x_iris, y_iris, cv = 5)

print(f'Cross Validation Score: {cv_score}')
print(f'Mean CV Score: {cv_score.mean()}')


# In[30]:


#Regression CV
cv_scores = cross_val_score(reg, x_house, y_house, cv = 5, scoring = 'neg_mean_squared_error')

print(f'Cross Validation Regression Score: {cv_scores}')
print(f'Mean Regression CV Score: {cv_scores.mean()}')


# ## 6. Confusion Matrix
# - True Positive (TP): Correctly predicted positive instances.
# - True Negative (TN): Correctly predicted negative instances.
# - False Positive (FP): Incorrectly predicted positive instances (Type I error).
# - False Negative (FN): Incorrectly predicted negative instances (Type II error).

# In[33]:


#Classification
cm = confusion_matrix(y_test_iris, y_pred_iris)
sns.heatmap(cm, annot = True, fmt = 'd', cmap = 'Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# ## 7. Precision Recall Curve

# In[36]:


#Predict the Probabilities for Positive Class
y_scores = clf.predict_proba(x_test_iris)[:, 1]

#P-R Pair for Different Threshold
pre, recall, thresholds = precision_recall_curve(y_test_iris, y_scores, pos_label = 1)

plt.plot(thresholds, pre[:-1], 'b--', label = 'Precision')
plt.plot(thresholds, recall[:-1], 'g-', label = 'Recall')
plt.xlabel('Threshold')
plt.legend(loc = 'best')
plt.title('Precision Recall Curve')
plt.show()

