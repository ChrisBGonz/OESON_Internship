#!/usr/bin/env python
# coding: utf-8

# # "seaborn.color_palette"

# ## The "seaborn.color_palette" returns a list of colors or continuous colormap defining a palette.

# ## Possible palette values include:
# 
# - Name of a seaborn palette (deep, muted, bright, pastel, dark, colorblind)
# - Name of matplotlib colormap
# - husl or hls
# - light: (color), dark: (color), blend: (color), (color)
# - etc.

# ### "seaborn.color_palette" commands

# In[1]:


#Installation and Setup

import seaborn as sns


# In[2]:


#Default Color Palette

sns.color_palette()


# In[3]:


#Specifying a Color Palette by Name

sns.color_palette("pastel")


# In[4]:


#Return a Specified Number of Evenly Spaced Hues in the “HUSL” System

sns.color_palette("husl", 9)


# In[5]:


#Return all unique colors in a categorical Color Brewer palette

sns.color_palette("Set2")


# In[6]:


#Return a Diverging Color Brewer Palette as a Continuous Colormap

sns.color_palette("Spectral", as_cmap = True)


# In[7]:


#Return One of the Perceptually-Uniform Palettes Included in Seaborn as a Discrete Palette

sns.color_palette("flare")


# In[8]:


#Return One of the Perceptually-Uniform Palettes Included in Seaborn as a Continuous Colormap

sns.color_palette("flare", as_cmap = True)


# In[9]:


#Return a Customized Cubehelix Color Palette

sns.color_palette("ch:s = .25,rot = -.25", as_cmap = True)


# In[10]:


#Return a Light Sequential Gradient

sns.color_palette("light:#5A9", as_cmap = True)


# In[11]:


#Return a reversed dark sequential gradient:

sns.color_palette("dark:#5A9_r", as_cmap = True)


# In[12]:


#Return a blend gradient between two endpoints:

sns.color_palette("blend:#7AB,#EDA", as_cmap = True)


# In[13]:


#See the underlying color values as hex codes:

print(sns.color_palette("pastel6").as_hex())

