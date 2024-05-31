#!/usr/bin/env python
# coding: utf-8

# # Introduction to Python Coding

# ### Basic Operators

# In[2]:


#Addition
x = 5 + 3
print(x)


# In[3]:


#Subtraction
x = 8 - 5
print(x)


# In[4]:


#Multiplication
x = 4 * 10
print(x)


# In[7]:


#Modulus
x = 4 % 2
print(x)


# In[8]:


#Floor division
x = 5 / 2
print(x)


# In[ ]:


#Comparison Operator


# In[9]:


#Equal to (==)
5 == 4


# In[10]:


#Not Equal to (!=)
4 != 3


# In[11]:


#Greater Than (>)
5 > 2


# In[12]:


#Less Than (<)
4 < 7


# In[13]:


#Greater Than or Equal To
5 >= 3


# In[14]:


#Less Than or Equal To
8 <= 10


# ### Assignment Operators

# In[30]:


#Assign (=)
x = 10
print(x)


# In[31]:


#Add and Assign
x += 5
print(x)


# In[32]:


#Subtract and Assign
x -= 7
print(x)


# In[33]:


#Multiply and Assign
x *= 6
print(x)


# In[34]:


#Divide and Assign
x /= 4
print(x)


# In[35]:


#Floor Division and Assign
x /= 3
print(x)


# In[36]:


#Modulus and Assign
x %= 2
print(x)


# ### Logical Operators

# In[37]:


#and
(4 > 5) and (3 < 6)


# In[38]:


#or
(4 > 5) or (3 < 6)


# In[39]:


#not
not (4 > 5)


# ### Data Structures

# In[49]:


#List
myList = [1, 2, 3, 'a', 'b', 'c']
print(myList)

print(myList[0])
print(myList[3])

myList[2] = 5
print(myList)

myList.append('cat')
print(myList)

myList.remove('a')
print(myList)

myList.append('new_element')
print(myList)


# In[53]:


#Tuple
myTuple = (1, 2, 3, 'a', 'b', 'c')
print(myTuple)

print(myTuple[0])
print(myTuple[3])

#The following will give you errors because tuple is not mutable (can't be changed once created):
myTuple[2] = 5
print(myTuple)

myTuple.append('cat')
print(myTuple)

myTuple.remove('a')
print(myTuple)

myTuple.append('new_element')
print(myTuple)


# In[60]:


#Dictionary
myDict = {'Name': 'Christopher', 'Age': 21, 'Gender': 'Male'}
print(myDict)

print(myDict['Name'])

myDict['Age'] = 25
print(myDict)


# In[65]:


#Sets
mySet = {1, 2, 3, 'a', 'b', 'c'}
print(mySet)

mySet.add('cat')
print(mySet)

mySet.remove('a')
print(mySet)


# In[70]:


#String
myString = 'Hello'
print(myString)

print(myString[0:5])
print(myString[0])
print(myString[1:3])

