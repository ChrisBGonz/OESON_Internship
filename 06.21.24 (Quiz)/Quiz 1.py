# -*- coding: utf-8 -*-
"""Quiz 1.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1ovP7E1vyqrlG_JMIRZv8PW-mICdzg8nr

#Name: Christopher Gonzalez

#Email: christopherg0514@gmail.com
"""

#Question 1: Setup

#Import the necessary libraries.
import pandas as pd
import numpy as np

#Create a DataFrame using the given data.
data =  {
 'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eva'],
 'Age': [24, 27, 22, 32, 29],
 'City': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix']
}

df = pd.DataFrame(data)

#Display the DataFrame.
print(df)

#Question 2: Pandas Dataframe Operations

#Select the 'Name' and 'City' columns and display them.
names = df['Name']
cities = df['City']

print('Names:\n', names)
print('\nCities:\n', cities)

#Filter the DataFrame to only include rows where 'Age' is greater than 25 and display the result.
filtered_df = df[df['Age'] > 25]
print('\nFiltered DataFrame:\n', filtered_df)

#Sort the DataFrame by 'Age' in ascending order and display the sorted DataFrame.
sorted_df = df.sort_values(by = 'Age')
print('\nSorted DataFrame:\n', sorted_df)

#Add a new column 'Age_Group' that categorizes ages into 'Young' (less than 30) and 'Adult' (30 and above), and display the updated DataFrame.
df['Age_Group'] = np.where(df['Age'] < 30, 'Young', 'Adult')
print('\nUpdated DataFrame:\n', df)

#Question 3: Data Structures

#Create a list of new ages: [25, 28, 23, 33, 30]. Update the 'Age' column in the DataFrame with these new ages and display the updated DataFrame.
new_ages = [25, 28, 23, 33, 30]
df['Age'] = new_ages

print('\nUpdated DataFrame:\n', df)

#Create a dictionary mapping names to new cities: {'Alice': 'San Francisco', 'Bob': 'Seattle', 'Charlie': 'Austin', 'David': 'Dallas', 'Eva': 'Miami'}.
#Update the 'City' column using this dictionary and display the updated DataFrame.
new_cities = {'Alice': 'San Francisco', 'Bob': 'Seattle', 'Charlie': 'Austin', 'David': 'Dallas', 'Eva': 'Miami'}
df['City'] = df['Name'].map(new_cities)

print('\nUpdated DataFrame:\n', df)

#Question 4:Control Flow

#Use a for loop to print the name and age of each person in the DataFrame.
for index, row in df.iterrows():
  print(row['Name'], row['Age'])

#Write a function to categorize ages into 'Youth' (less than 30) and 'Adult' (30 and above).
#Apply this function to create a new column 'Category' in the DataFrame and display the updated DataFrame.
def categorize_age(age):
  if age < 30:
    return 'Youth'

  else:
    return 'Adult'

df['Category'] = df['Age'].apply(categorize_age)

print('\nUpdated DataFrame:\n', df)
