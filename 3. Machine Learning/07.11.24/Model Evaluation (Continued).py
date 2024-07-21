#!/usr/bin/env python
# coding: utf-8

# In[1]:


#From 07/09/24 ####################################################################################################


# # Model Evaluation
# 1. Data Preparation
# Data preparation is like getting all the ingredients ready before cooking. It involves cleaning the data (removing or fixing any errors or missing values), normalizing or standardizing it (so all features have similar scales), and sometimes augmenting it (adding more data through transformations). For the MVTec AD dataset, this might include resizing images, converting them to grayscale or color, and ensuring that each image is labeled correctly.
# 
# 2. Train - Test Split
# This step involves dividing your dataset into two parts: one for training your model (train set) and one for testing it (test set). It's like studying for an exam using a portion of the material (training) and then testing your knowledge on a different set of questions (testing). Typically, 70-80% of the data is used for training, and the rest for testing. This helps in evaluating how well your model will perform on unseen data.
# 
# 3. Model Training
# Model training is where the magic happens. The training data is fed into the machine learning algorithm, which learns patterns and relationships within the data. This is similar to learning from examples and improving your skills over time. For anomaly detection, the model learns what normal (non-anomalous) data looks like, so it can identify what doesn’t fit that pattern.
# 
# 4. Model Evaluation using Metrics
# After training, you need to evaluate how well your model performs. This involves using metrics like accuracy, precision, recall, F1-score, etc. It’s like getting a report card that tells you how well you did. Each metric provides different insights:
# 
# - Accuracy: How many predictions were correct overall.
# - Precision: Out of all the predicted anomalies, how many were actual anomalies.
# - Recall: Out of all actual anomalies, how many were correctly predicted.
# - F1-score: A balance between precision and recall.
# 
# 5. Cross-Validation
# Cross-validation is a technique to make sure your model's performance is robust and not dependent on a particular train-test split. It involves dividing the data into multiple folds (subsets) and training/testing the model multiple times, each time with a different fold as the test set and the remaining folds as the training set. This is like taking multiple mock exams to ensure you're consistently good and not just lucky once.
# 
# 6. Confusion Matrix
# A confusion matrix is a table that summarizes the performance of a classification model. It shows the counts of true positive (correctly identified anomalies), true negative (correctly identified normal data), false positive (normal data incorrectly labeled as anomalies), and false negative (anomalies incorrectly labeled as normal). It’s a detailed report card that shows where the model is getting things right and where it’s making mistakes.
# 
# 7. Precision-Recall Curve
# The precision-recall curve is a graph that shows the trade-off between precision and recall for different threshold settings. Higher precision means fewer false positives, and higher recall means fewer false negatives. The curve helps in understanding the balance between precision and recall and finding an optimal threshold for your specific application.
# 
# 8. ROC Curve
# The ROC (Receiver Operating Characteristic) curve is another graph that shows the trade-off between true positive rate (recall) and false positive rate. The area under the ROC curve (AUC) provides a single measure of the model’s performance. A model with an AUC close to 1 indicates a good model, while an AUC close to 0.5 suggests a model no better than random guessing. It’s like a comprehensive score that tells you how well your model distinguishes between anomalies and normal data.

# 1. Data Preparation
# 2. Train Test Split
# 3. Model Training
# 4. Model Evaluation Using Metrics
# 5. Cross Validation
# 6. Confusion Matrix
# 7. Precision-Recall Curve

# ## 1. Data Preparation

# In[10]:


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


# ### Classification Metrics:
# 1. Accuracy
# - Definition: The ratio of correctly predicted instances to the total number of instances.
# - Use: Good for balanced datasets but can be misleading if classes are imbalanced.
# 
# 2. Precision
# - Definition: The ratio of true positive predictions to the total number of positive predictions (true positives + false positives).
# -Use: Important when the cost of false positives is high, such as in spam detection.
# 
# 3. Recall (Sensitivity)
# - Definition: The ratio of true positive predictions to the total number of actual positives (true positives + false negatives).
# - Use: Crucial when missing positive instances is costly, such as in medical diagnoses.
# 
# 4. F1-Score
# - Definition: The harmonic mean of precision and recall. It balances both metrics and provides a single score.
# - Use: Useful when you need a balance between precision and recall.
# 
# 5. Confusion Matrix
# - Definition: A table showing the counts of true positives, true negatives, false positives, and false negatives.
# - Use: Helps visualize performance and understand where the model makes errors.
# 
# 6. ROC Curve and AUC
# - ROC Curve: A graph plotting the true positive rate (recall) against the false positive rate at various threshold settings.
# - AUC (Area Under Curve): The area under the ROC curve, which summarizes the model’s ability to distinguish between classes. An AUC close to 1 indicates a good model.
# 
# 7. Precision-Recall Curve
# - PR Curve: A graph showing the trade-off between precision and recall for different threshold settings.
# - Use: Especially useful for imbalanced datasets to evaluate the model’s performance across different levels of precision and recall.

# ### Regression Metrics
# - Regression metrics are used to evaluate models that predict continuous outcomes (e.g., predicting house prices). Here are some common metrics:
# 
# 1. Mean Absolute Error (MAE)
# - Definition: The average absolute differences between predicted and actual values.
# - Use: Gives a straightforward measure of prediction accuracy in the same units as the target variable.
# 
# 2. Mean Squared Error (MSE)
# - Definition: The average of the squared differences between predicted and actual values.
# - Use: Penalizes larger errors more heavily due to squaring, making it sensitive to outliers.
# 
# 3. Root Mean Squared Error (RMSE)
# - Definition: The square root of the mean squared error. It brings the error measure back to the same units as the target variable.
# - Use: Useful when you need to understand the magnitude of prediction errors in the same units as the target variable.
# 
# 4. R-Squared (R²)
# - Definition: The proportion of variance in the target variable that is predictable from the features.
# - Use: Indicates how well the model explains the variability of the target variable. R² values range from 0 to 1, with higher values indicating better model performance.
# 
# 5. Adjusted R-Squared
# - Definition: A modified version of R² that adjusts for the number of predictors in the model.
# - Use: Useful for comparing models with different numbers of predictors.

# ## 2. Train Test Split

# In[14]:


#Iris Dataset

#Train Test Split
iris = load_iris()
x_iris = iris.data
y_iris = iris.target

#Splitting Into the Training Set and Test Set
x_train_iris, x_test_iris, y_train_iris, y_test_iris = train_test_split(x_iris, y_iris, 
                                                                        test_size = 0.2, random_state = 42)


# In[17]:


#House Dataset

#Train Test Split
house = fetch_california_housing()
x_house = house.data
y_house = house.target

#Splitting Into the Training Set and Test Set
x_train_house, x_test_house, y_train_house, y_test_house = train_test_split(x_house, y_house, 
                                                                        test_size = 0.2, random_state = 42)


# ## 3. Model Training

# In[20]:


#Train Logistic Regression Model
clf = LogisticRegression(max_iter = 200)
clf.fit(x_train_iris, y_train_iris)

#Predict on Test Set
y_pred_iris = clf.predict(x_test_iris)


# In[22]:


#Train Linear Regression Model 
reg = LinearRegression()
reg.fit(x_train_house, y_train_house)

#Predict on Test Set
y_pred_house = reg.predict(x_test_house)


# In[24]:


###################################################################################################################


# ## 4. Model Evaluation Using Metrics

# In[27]:


#Classification Metrics
accuracy = accuracy_score(y_test_iris, y_pred_iris)
precision = precision_score(y_test_iris, y_pred_iris, average = 'weighted')
recall = recall_score(y_test_iris, y_pred_iris, average = 'weighted')
f1 = f1_score(y_test_iris, y_pred_iris, average = 'weighted')

print(f'Accuracy: {accuracy}')
print(f'Precision: {accuracy}')
print(f'Recall: {accuracy}')
print(f'F1: {accuracy}')


# In[29]:


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

# In[32]:


#Classification CV
cv_score = cross_val_score(clf, x_iris, y_iris, cv = 5)

print(f'Cross Validation Score: {cv_score}')
print(f'Mean CV Score: {cv_score.mean()}')


# In[34]:


#Regression CV
cv_scores = cross_val_score(reg, x_house, y_house, cv = 5, scoring = 'neg_mean_squared_error')

print(f'Cross Validation Regression Score: {cv_scores}')
print(f'Mean Regression CV Score: {cv_scores.mean()}')


# ## 6. Confusion Matrix
# - True Positive (TP): Correctly predicted positive instances.
# - True Negative (TN): Correctly predicted negative instances.
# - False Positive (FP): Incorrectly predicted positive instances (Type I error).
# - False Negative (FN): Incorrectly predicted negative instances (Type II error).

# In[37]:


#Classification
cm = confusion_matrix(y_test_iris, y_pred_iris)
sns.heatmap(cm, annot = True, fmt = 'd', cmap = 'Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# ## 7. Precision Recall Curve

# In[40]:


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

