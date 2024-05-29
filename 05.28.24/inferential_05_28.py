# -*- coding: utf-8 -*-
"""Inferential_05/28.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1k5-fN6gy6D5XfSdHt4PBkS4xg0NGT2MR

# About the dataset:

kidney_stone_data

Source: https://www.kaggle.com/datasets/utkarshxy/kidney-stone-data/data

## Introduction
In 1986, a group of urologists in London published a research paper in The British Medical Journal that compared the effectiveness of two different methods to remove kidney stones. Treatment A was open surgery (invasive), and treatment B was percutaneous nephrolithotomy (less invasive).

When they looked at the results from 700 patients, treatment B had a higher success rate. However, when they only looked at the subgroup of patients different kidney stone sizes, treatment A had a better success rate.

Simpon's paradox occurs when trends appear in subgroups but disappear or reverse when subgroups are combined.
In this project -> medical data published in 1986 in "The British Medical Journal" where the effectiveness of two types of kidney stone removal treatments (A - open surgery and B - percutaneous nephrolithotomy) were compared.

Using multiple logistic regression and visualize model output to help the doctors determine if there is a difference between the two treatments. While not required, it will also help to have some knowledge of inferential statistics.

## Content
The data contains three columns: treatment (A or B), stone_size (large or small) and success (0 = Failure or 1 = Success).
"""

#Install the library
!pip install scipy

#Import all the libraries
import pandas as pd
from scipy.stats import ttest_ind, f_oneway, chi2_contingency

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Load the dataset
data = pd.read_csv("kidney_stone_data.csv")
df = pd.DataFrame(data)

df.head(10)

df.tail(10)

df.shape

df.dtypes

df.describe()

"""# T-Test

- H0 (Null Hypothesis): There is no significant difference in the effectiveness/success rate of Treatment A and Treatment B.

- H1 (Alternate Hypothesis): There is a significant difference in the effectiveness/success rate of Treatment A and Treatment B.

Note: If the p-value is less than 0.05, reject H0. Otherwise, it is accepted.
"""

#Success rate of Treatment A and Treatment B:
success_A = df[df["treatment"] == "A"]["success"]
success_B = df[df["treatment"] == "B"]["success"]

#Perform Independent Sample T-Test
t_stat = ttest_ind(success_A, success_B)
print(t_stat)

"""# Anova
- H0: There is no significant difference among the group means.
- H1: At least one group has a significantly different mean than the other(s).
"""

#Success rate of Treatment A and Treatment B:
success_small = df[df["stone_size"] == "small"]["success"]
success_large = df[df["stone_size"] == "large"]["success"]

#Perform One Way Sample T-Test
t_stat = f_oneway(success_small, success_large)
print(t_stat)

"""# Chi Square
- H0: There is no significant effect between the treatment type and stone size on the overall success rate.
- H1: There is a signfiicant effect between the treatment type and stone size on the overall success rate.
"""

chi = pd.crosstab([df["treatment"], df["stone_size"]], [df["success"]])

result = chi2_contingency(chi)
print(result)