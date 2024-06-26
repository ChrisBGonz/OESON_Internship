#!/usr/bin/env python
# coding: utf-8

# # Plotly

# - Graphing Library
# - Interactive
# - Publication-quality graphs online
# - Create complex visualuzations and dashboards

# ## Basic Plotting with Plotly

# ### Line Plots

# In[1]:


#Question: Create an interactive line plot for y = sin(x)

import plotly.graph_objs as go
import numpy as np

x = np.linspace(0, 10, 100)
y = np.sin(x)

fig = go.Figure(data = go.Scatter(x = x, y = y, mode = 'lines', name = 'Sin(x)'))
fig.update_layout(title = 'Simple Line Plot', xaxis_title = 'x-axis', yaxis_title = 'y-axis')
fig.show()


# In[2]:


#Question: Create an interactive scatterplot using random data

x = np.random.rand(100)
y = np.random.rand(100)

fig = go.Figure(data = go.Scatter(x = x, y = y, mode = 'markers', name = 'Random Scatter'))
fig.update_layout(title = 'Simple Scatter Plot', xaxis_title = 'x-axis', yaxis_title = 'y-axis')
fig.show()


# In[3]:


#Question: Create an interactive bar graph for the given categories and values

categories = ['A', 'B', 'C']
values = [10, 20, 15]

fig = go.Figure(data = go.Bar(x = categories, y = values, name = 'Values'))
fig.update_layout(title = 'Simple Bar Graph', xaxis_title = 'x-axis', yaxis_title = 'y-axis')
fig.show()


# In[4]:


#Question: Add titles and labels for an interactive line plot of y = sin(x)

x = np.linspace(0, 10, 100)
y = np.sin(x)

fig = go.Figure(data = go.Scatter(x = x, y = y, mode = 'lines', name = 'Sin(x)'))
fig.update_layout(title = 'Line Plot for y = sin(x)', xaxis_title = 'x-axis', yaxis_title = 'y-axis')
fig.show()


# In[5]:


#Question: Create subplots for y = sin(x) and y = cos(x) using Plotly

import plotly.subplots as sp

x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

fig = sp.make_subplots(rows = 2, cols = 1)
fig.add_trace(go.Scatter(x = x, y = y1, mode = 'lines', name = 'Sin(x)'), row = 1, col = 1)
fig.add_trace(go.Scatter(x = x, y = y2, mode = 'lines', name = 'Cos(x)'), row = 2, col = 1)
fig.update_layout(title = 'Subplots for y = sin(x) and y = cos(x)')
fig.show()


# In[6]:


###################################################################################################################


# # Control Flow in Python
# ## Introduction:
# ### Control flow in programming refers to the order in which individual statements, instructions, or function calls are executed or evaluated. Understanding control flow is essential for creating algorithms and solving complex problems.
# 
# ## Conditional statements
# ### Conditional statements allow you to execute certain pieces of code based on whether a condition is true or false.

# ## "if" Statement
# ### The if statement executes a block of code if a specified condition is true.

# In[7]:


x = 10

if x > 5:
    print('x is greater than 5')


# ## "if-else" Statement
# ### The if-else statement executes one block of code if the condition is true and another block if the condition is false.

# In[8]:


x = 3

if x > 5:
    print('x is greater than 5')
    
else:
    print('x is not greater than 5')


# ## "if-elif-else" Statement
# ### The if-elif-else statement executes one block of code of whichever condition is true.

# In[9]:


x = 10

if x > 15:
    print('x is greatr than 15')
    
elif x > 5:
    print('x is greater than 5')
    
else:
    print('x is 5 or less')


# ## Looping Statements
# ### Looping statements allow you to execute a block of code repeatedly.

# ## "for" loop
# ### The for loop iterates over a sequence (such as a list, tuple, dictionary, set, or string).

# In[10]:


for i in range(5):
    print(i)


# ## "while" loop
# ### The while loop executes a block of code as long as a specified condition is true.

# In[11]:


count = 0

while count < 5:
    print(count)
    count += 1


# ## Control Flow 
# ### Python provides several tools to control the flow of loops and conditionals.

# ## "break" Statement
# ### The break statement terminates the loop prematurely.

# In[12]:


for i in range(10):
    if i == 5:
        break
        
    print(i)


# ## ""continue" Statement
# ### The continue statement skips the rest of the code inside the loop of the current iteration and moves on to the next iteration.

# In[13]:


for i in range(10):
    if i % 2 == 0:
        continue
        
    print(i)


# ## ""pass" Statement
# ### The pass statement does nothing and is used as a placeholder.

# In[14]:


for i in range(10):
    if i % 2 == 0:
        pass
    
    else:
        print(i)


# In[15]:


#Filter out the even numbers from a list

data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
filtered_data = [x for x in data if x % 2 != 0]

print(filtered_data)


# In[16]:


#Calculate the sum of all elements in a list

data = [1, 2, 3, 4, 5]
total = 0

for num in data:
    total += num

print('Sum:', total)


# In[17]:


#Write a program that prints the numbers from 1 - 50, but for multiples of three, print "Frizz" instead.
#For for multiples of five, print "Buzz".
#For numbers that multiples of both three and five, print "FizzBuzz"

for i in range(1, 51):
    if i % 3 == 0 and i % 5 == 0:
        print('FizzBuzz')
        
    elif i % 3 == 0:
        print('Fizz')
        
    elif i % 5 == 0:
        print('Buzz')
    
    else:
        print(i)


# In[18]:


#Write a program that prints all print numbers between 1 and 100

for num in range(2, 101):
    is_prime = True
    
    for i in range(2, int(num ** 0.5) + 1):
        
        if num % i == 0:
            is_prime = False
            break
            
    if is_prime:
        print(num)


# In[19]:


#Write a program to find the sum of the digits of a number

number = 12345
total = 0

while number > 0:
    digit = number % 10
    total += digit
    number //= 10
    
print('Sum of digits:', total)


# In[20]:


#Write a program to print the Fibbonacci sequence up to 'n' terms

n = 10
a, b = 0, 1

for i in range(n):
    print(a)
    a, b = b, a + b


# In[21]:


#Write a program to calculate the factoral of a given number

num = int(input('Enter the number: '))
factoral = 1

for i in range(1, num + 1):
    factoral *= i
    
print('The factoral of', num, 'is:', factoral)


# In[23]:


#Write a program to reverse a given string

string = 'data science'
reversed_string = ''

for char in string:
    reversed_string = char + reversed_string
    
print('Reversed string:', reversed_string)


# In[24]:


#Write a program to check if a given string is a palindrome (is the same if read forwards or backwards)

string = str(input('Enter the string: '))
is_palindrome = string == string[:: -1]

print('Is palindrome:', is_palindrome)


# In[46]:


#Write a program to count the number of vowels and consonants in a string

string = str(input('Enter the string in lowercase: ')) 
vowels = 'aeiou'
vowel_count = 0
consonant_count = 0

for char in string.lower():
    
    if char in vowels:
        vowel_count += 1
    
    elif char.isalpha():
        consonant_count += 1
        
print('Vowel count:', vowel_count)
print('Consonant count:', consonant_count)


# In[26]:


#Write a program to print a multiplication table (from 1 to 10)

for i in range(1, 11):
    
    for j in range(1, 11):
        print(i * j, end = '\t')
        
    print()


# In[27]:


#Write a program to find the common elements in two lists

list1 = [1, 2, 3, 4, 5]
list2 = [4, 5, 6, 7, 8]

common_elements = [element for element in list1 if element in list2]

print('Common elements:', common_elements)


# In[28]:


#Write a program to calculate the average of numbers in a list

numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
average = sum(numbers) / len(numbers)

print('Average:', average)


# In[29]:


#Write a program to remove duplicates from a list

numbers = [1, 2, 3, 4, 4, 5, 5, 6, 7, 7, 8, 9, 10, 10]
unique_numbers = []

for num in numbers:
    
    if num not in unique_numbers:
        unique_numbers.append(num)
        
print('Unique numbers:', unique_numbers)


# In[30]:


#Write a program to flatten a nested list

nested_list = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
flattened_list = [item for sublist in nested_list for item in sublist]

print("Flattened list:", flattened_list)


# In[31]:


#Write a program tp generate a dictionary from two lists, one for keys and one for values

keys = ['a', 'b', 'c', 'd']
values = [1, 2, 3, 4]

dictionary = dict(zip(keys, values))

print('Dictionary:', dictionary)


# In[32]:


#Write a program to count the occurrences of each element in a list

numbers = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]
count_dict = {} 

for num in numbers:
    
    if num in count_dict:
        count_dict[num] += 1
        
    else:
        count_dict[num] = 1
        
print('Occurrences:', count_dict)


# In[33]:


#Write a program to transpose a given matrix

matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

transposed_matrix = [[row[i] for row in matrix] for i in range(len(matrix[0]))]

print('Trabsposed matrix:', transposed_matrix)


# In[34]:


#Write a program to check if two strings are anagrams (if two strings have the same characters but rearranged)

str1 = 'listen'
str2 = 'silent'

is_anagram = sorted(str1) == sorted(str2)

print('Is anagram:', is_anagram)


# In[35]:


###################################################################################################################


# # Practice

# In[94]:


#Write a program to generate the first n prime numbers

count = 0
n = int(input('Enter a value for n: '))

for num in range(2, 1000):
    is_prime = True
    
    for i in range(2, int(num ** 0.5) + 1):
        
        if num % i == 0:
            is_prime = False
            break
            
    if is_prime:
        print(num)
        count += 1
        
        if count == n:
            break


# In[1]:


#Write a program to convert a decimal number into its binary representation   

decimal_num = int(input('Enter a decimal number: '))
binary_num = bin(decimal_num)

print(decimal_num, 'in binary is', binary_num)


# In[49]:


#Write a program to generate a list of the first n Fibonacci numbers

n = int(input('Enter a number: '))

a, b = 0, 1

for i in range(n):
    print(a)
    a, b = b, a + b


# In[57]:


#Write a program to merge two sorted lists into a single sorted list

list1 = [3, 4, 1, 5, 2]
list2 = [8, 6, 7, 9]

sorted_list1 = sorted(list1)
sorted_list2 = sorted(list2)

merged_sorted_lists = sorted_list1 + sorted_list2

print('Merged sorted lists:', merged_sorted_lists)


# In[64]:


#Write a program to implement a simple calculator that can perform simple arithmetic operations

num1 = int(input('Enter an integer: '))
num2 = int(input('Enter another integer: '))

sum_nums = num1 + num2
diff_nums = num1 - num2
prod_nums = num1 * num2
quot_nums = num1 / num2

print('Sum:',sum_nums)
print('Difference:', diff_nums)
print('Product:', prod_nums)
print('Quotient:', quot_nums)

