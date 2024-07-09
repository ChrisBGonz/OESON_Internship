#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis

# ## Your task is to perform an Exploratory Data Analysis (EDA) on a dataset collected from these smartwatches. The dataset includes various physiological parameters along with a 'drowsiness' label, which indicates the level of sleepiness based on an adapted Karolinska Sleepiness Scale (KSS).

# In[1]:


#Import Necessary Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


#Load the dataset
data = pd.read_csv('drowsiness_dataset.csv')
data.head(10)


# In[3]:


#Check for Missing Values
print(data.isnull().sum())


# In[4]:


#General Statistics of Data
data.describe()


# In[5]:


#Measure of Central Tendency
print("Mean: ")
print(data.mean())

print("\nMedian: ")
print(data.median())

print("\nMode: ")
print(data.mode())


# In[6]:


#Graph Analysis: Histogram Chart
plt.figure(figsize = (8, 4))
plt.hist(data['heartRate'], bins = 100, color = 'orange', label = 'Heart Rate')
plt.xlabel('Heart Rate')
plt.ylabel('Occurrences')
plt.title('Heart Rate Across Dataset')
plt.legend()
plt.show()


# In[7]:


#Graph Analysis: Histogram Chart
plt.figure(figsize = (8, 4))
plt.hist(data['ppgGreen'], bins = 100, color = 'green', label = 'ppgGreen')
plt.xlabel('ppgGreen')
plt.ylabel('Occurrences')
plt.title('ppgGreen Levels Across Dataset')
plt.legend()
plt.show()


# In[8]:


#Graph Analysis: Histogram Chart
plt.figure(figsize = (8, 4))
plt.hist(data['ppgRed'], bins = 100, color = 'red', label = 'ppgRed')
plt.xlabel('ppgRed')
plt.ylabel('Occurrences')
plt.title('ppgRed Levels Across Dataset')
plt.legend()
plt.show()


# In[9]:


#Graph Analysis: Histogram Chart
plt.figure(figsize = (8, 4))
plt.hist(data['ppgIR'], bins = 100, color = 'purple', label = 'ppgIR')
plt.xlabel('ppgIR')
plt.ylabel('Occurrences')
plt.title('ppgIR Levels Across Dataset')
plt.legend()
plt.show()


# In[10]:


#Graph Analysis: Histogram Chart
plt.figure(figsize = (8, 4))
plt.hist(data['drowsiness'], bins = 5, color = 'blue', label = 'Drowsiness')
plt.xlabel('Drowsiness')
plt.ylabel('Occurrences')
plt.title('Drowsiness Levels Across Dataset')
plt.legend()
plt.show()


# In[11]:


#Heart Rate Accross Different Levels of Drowsiness
plt.figure(figsize = (8, 6))
sns.boxplot(x = 'drowsiness', y = 'heartRate', color = 'orange', data = data)
plt.title('Heart Rate vs Drowsiness')


# In[12]:


#PPG Levels Accross Different Levels of Drowsiness
plt.figure(figsize = (8, 6))
sns.boxplot(x = 'drowsiness', y = 'ppgGreen', color = 'green', data = data)
plt.title('ppgGreen vs Drowsiness')


# In[13]:


#PPG Levels Accross Different Levels of Drowsiness
plt.figure(figsize = (8, 6))
sns.boxplot(x = 'drowsiness', y = 'ppgRed', color = 'red', data = data)
plt.title('ppgRed vs Drowsiness')


# In[14]:


#PPG Levels Accross Different Levels of Drowsiness
plt.figure(figsize = (8, 6))
sns.boxplot(x = 'drowsiness', y = 'ppgIR', color = 'purple', data = data)
plt.title('ppgIR vs Drowsiness')


# In[15]:


#Correlation Matrix
corr_matrix = data.corr(numeric_only = True)

#Heatmap of Correlation Matrix
plt.figure(figsize = (10, 7))
sns.heatmap(corr_matrix, annot = True, cmap = 'coolwarm', linewidths = 0.5)
plt.title('Correlation Matrix')
plt.show()


# In[16]:


#Define the number of periods
num_periods = 4

#Calculate the size of each period
period_size = len(data) // num_periods

#Create a period column
period_labels = ['Morning', 'Afternoon', 'Evening', 'Night']
data['period'] = pd.cut(data.index, bins = num_periods, labels = period_labels)

#Check the distribution of periods
print(data['period'].value_counts())

#Segment the data by period
morning_data = data[data['period'] == 'Morning']
afternoon_data = data[data['period'] == 'Afternoon']
evening_data = data[data['period'] == 'Evening']
night_data = data[data['period'] == 'Night']


# In[17]:


import matplotlib.pyplot as plt

def calculate_and_plot_correlations(data, period_name):
  correlation_heart_rate = data['drowsiness'].corr(data['heartRate'])
  correlation_ppg_green = data['drowsiness'].corr(data['ppgGreen'])
  correlation_ppg_red = data['drowsiness'].corr(data['ppgRed'])
  correlation_ppg_ir = data['drowsiness'].corr(data['ppgIR'])

  print (f'Correlation between drowsiness and heart rate ({period_name}): {correlation_heart_rate}')
  print (f'Correlation between drowsiness and PPG Green ({period_name}): {correlation_ppg_green}')
  print (f'Correlation between drowsiness and PPG Red ({period_name}): {correlation_ppg_red}')
  print(f'Correlation between drowsiness and PPG IR ({period_name}): {correlation_ppg_ir}')

  plt.scatter(data['heartRate'], data['drowsiness'], alpha = 0.5, label = 'Heart Rate')
  plt.scatter (data['ppgGreen'], data['drowsiness'], alpha = 0.5, label='PPG Green', color = 'green')
  plt.scatter(data['ppgRed'], data['drowsiness'], alpha = 0.5, label = 'PPG Red', color = 'red')
  plt.scatter(data['ppgIR'], data['drowsiness'], alpha = 0.5, label = 'PPG IR', color = 'purple')

  plt.title(f'Drowsiness Levels vs. Heart Rate and PPG Readings ({period_name})')
  plt.xlabel('Activity Level')
  plt.ylabel('Drowsiness Level')
  plt.legend()
  plt.show()


# In[18]:


calculate_and_plot_correlations(morning_data, 'Morning')


# In[19]:


calculate_and_plot_correlations(afternoon_data, 'Afternoon')


# In[20]:


calculate_and_plot_correlations(evening_data, 'Evening')


# In[21]:


calculate_and_plot_correlations(night_data, 'Night')

