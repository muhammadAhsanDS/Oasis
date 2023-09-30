#!/usr/bin/env python
# coding: utf-8

# # Unemployment Analysis with Python
# # Name: Muhammad Ahsan
# # Task: 2

# In[1]:


import pandas as pd
unemployment = pd.read_csv("Unemployment_Rate_upto_11_2020.csv")
unemployment_india = pd.read_csv("Unemployment in India.csv")


# In[2]:


# missing values
print("Missing Values:\n", unemployment.isnull().sum())
unemployment[' Date'] = pd.to_datetime(unemployment[' Date'])
unemployment_india[' Date'] = pd.to_datetime(unemployment_india[' Date'])
# Check for duplicates
duplicates = unemployment.duplicated().sum()
print("Number of Duplicates:", duplicates)


# In[3]:


import matplotlib.pyplot as plt
import seaborn as sns

#  Unemployment rate over time
plt.figure(figsize=(12, 6))
sns.lineplot(x=' Date', y=' Estimated Unemployment Rate (%)', data=unemployment)
plt.title('Unemployment Rate Over Time')
plt.xlabel('Date')
plt.ylabel('Unemployment Rate (%)')
plt.xticks(rotation=45)
plt.show()


# In[4]:


# Distribution of unemployment rates
plt.figure(figsize=(8, 6))
sns.histplot(data=unemployment, x=' Estimated Unemployment Rate (%)', bins=20, kde=True)
plt.title('Distribution of Unemployment Rates')
plt.xlabel('Unemployment Rate (%)')
plt.ylabel('Frequency')
plt.show()


# In[5]:


# Unemployment rates among regions
region_unemployment = unemployment.groupby('Region')[' Estimated Unemployment Rate (%)'].mean().reset_index()
plt.figure(figsize=(10, 6))
sns.barplot(x='Region', y=' Estimated Unemployment Rate (%)', data=region_unemployment)
plt.title('Average Unemployment Rate by Region')
plt.xlabel('Region')
plt.ylabel('Average Unemployment Rate (%)')
plt.xticks(rotation=90)
plt.show()


# In[6]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

X = unemployment[['Region', 'Region.1']]  
y = unemployment[' Estimated Unemployment Rate (%)']
X = pd.get_dummies(X, columns=['Region', 'Region.1'], drop_first=True)

# Split the data 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression model 
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)


# In[ ]:





# In[ ]:




