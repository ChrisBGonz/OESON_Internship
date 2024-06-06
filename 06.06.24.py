#!/usr/bin/env python
# coding: utf-8

# # Matplotlib
# 
# - Static
# - Animated
# - Interactive
# - 2D Graphs

# ## Basic Plotting With Mathplotlib

# ### Line Plot

# In[9]:


#Question: Create a line plot for the function y = sin(x)

#Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt

#Generate the data
x = np.linspace(0, 10, 100)
y = np.sin(x)

#Create the plot
plt.plot(x, y)
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.title('Simple Line Plot')
plt.show()


# ### Scatterplot

# In[10]:


#Question: Create a scatterplot using random values/data

#Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt

#Generate the data
x = np.random.rand(100)
y = np.random.rand(100)

#Create the plot
plt.scatter(x, y)
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.title('Simple Scatter Plot')
plt.show()


# ### Bar Chart

# In[11]:


#Question: Create a bar chart on the basis of given categories and values

#Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt

#Generate data
categories = ['A', 'B', 'C']
values = [10, 25, 32]

#Create the chart
plt.bar(categories, values)
plt.xlabel('Categories')
plt.ylabel('Values')
plt.title('Simple Bar Chart')
plt.show()


# ### Line Plot (Revisited)

# In[3]:


#Question: Create a line plot for the function y = sin(x) and y = cos(x) with Legends

#Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt

#Generate the data
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

#Create the plot
plt.plot(x, y1, label = 'sin(x)')
plt.plot(x, y2, label = 'cos(x)')
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.title('Line Plot With Legends')
plt.legend()
plt.show()


# ### Subplots

# In[18]:


#Question: Create a subplot for the function y = sin(x) and y = cos(x)

#Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt

#Generate the data
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

#Create the plot
fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.plot(x, y1)
ax1.set_title('Sin(x)')
ax2.plot(x, y2)
ax2.set_title('Cos(x)')
plt.tight_layout
plt.show()


# # Seaborn
# - Statistical data visualization
# - High level interface
# - Informative statistical graph

# ## Basic Plotting With Seaborn

# ### Distribution Plots

# In[21]:


#Question: Create a histogram with KDE (Kernel Density Estimation) overlay using random data

#Import necessary libraries
import seaborn as sns
import numpy as np

#Define the data
data = np.random.rand(1000)
sns.histplot(data, kde = True)
plt.title('Histogram with KDE')
plt.show()


# ### Exersices

# In[2]:


#Question 1: Create a line plot for the functions y = x and y = x^2 on the same graph. Add a legend to differentiate between the two lines.

#Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt

#Generate the data
x = np.linspace(0, 10, 100)
y1 = x
y2 = x * x

#Create the plot
plt.plot(x, y1, label = 'X')
plt.plot(x, y2, label = 'X^2')
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.title('Question 1')
plt.legend()
plt.show()


# In[71]:


#Question 2: Create a scatterplot using random values for x and y. Color the points based on their value and add a color bar.

#Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt

#Generate the data
x = np.random.rand(50)
y = np.random.rand(50)
colors = np.random.rand(50)

#Create the plot
plt.scatter(x, y, c = colors, cmap = 'viridis')
plt.colorbar(label = 'Color Values')
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.title('Simple Scatter Plot')
plt.show()


# In[40]:


#Question 3: Create a 2x2 grid of subplots. Each subplot should contain a different plot type with some random data.

#Import necessary libraries
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

#Generate the data
x1 = np.linspace(0, 10, 100)
y1 = np.tan(x)
x2 = np.random.rand(50)
y2 = np.random.rand(50)
x3 = ['0-10', '11-20', '20-30']
y3 = [35, 52, 13]

#Create the plot
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 15))
ax1.plot(x1, y1)
ax1.set_title('Line Plot (Tan(x))')
ax2.scatter(x2, y2)
ax2.set_title('Random Scatterplot')
ax3.bar(x3, y3)
ax3.set_title('Randon Bar Graph (Age of People at a Baseball Game')
ax4.hist(np.random.rand(200), bins = 10)
ax4.set_title('Random Histogram')
#plt.tight_layout
plt.show()


# In[49]:


#Question 4: Create a bar plot showing the average monthly temperatures of a city. Add error bars to represent the standard deviation of the temperatures.

#Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt

#Generate data
month = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
avgTemp = [30, 39, 43, 52, 65, 77, 90, 91, 80, 65, 43, 20]
stdTemp = [3, 2, 4, 3, 2, 3, 4, 5, 3, 2, 4, 3]

#Create the chart
plt.bar(month, avgTemp, yerr = stdTemp, capsize = 10)
plt.xlabel('Month')
plt.ylabel('Average Temperature')
plt.title('Simple Bar Chart')
plt.show()


# In[12]:


#Question 5: Create a histogram of 1000 data points drawn from a normal distribution. Add a KDE overlay to the histogram. 

#Import necessary libraries
import seaborn as sns
import numpy as np

#Define the data
data = np.random.rand(500)
sns.histplot(data, kde = True)
plt.title('Histogram with KDE')
plt.show()


# In[89]:


#Create a scatterplot with a regression line using the 'tips' dataset

tips = sns.load_dataset('tips')
tips.head()


# In[95]:


sns.lmplot(x = 'total_bill', y = 'tip', data = tips)
plt.title('Scatterplot With Regression Line')
plt.show()


# In[96]:


#Create a box plot of total bill by day using the 'tips' dataset

sns.boxplot(x = 'day', y = 'total_bill', data = tips)
plt.title("Boxplot by total bill by day")
plt.show()

