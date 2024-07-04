# -*- coding: utf-8 -*-
"""Data Preprocessing Tools

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1MWZgoggSFJjR_mgyqP9xEmwr0JJM4Nqf

#Data Preprocessing Tools
"""

#Import Necessary Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Import the Dataset
dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#Print the data
print(x)
print('\n', y)

#Taking Care of Missing Data
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])

print(x)

"""##Encoding Categorical Data"""

#Encoding the Independent Variable
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder = 'passthrough')
x = np.array(ct.fit_transform(x))

print(x)

#Encoding the Dependent Variable
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y = le.fit_transform(y)

print(y)

#Splitting the Dataset into the Training Set and Test Set
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)

print(x_train)
print('\n', x_test)
print('\n', y_train)
print('\n', y_test)

"""##Feature Scaling"""

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
x_train[:, 3:] = sc.fit_transform(x_train[:, 3:])
x_test[:, 3:] = sc.transform(x_test[:, 3:])

print(x_train)
print('\n', x_test)