#!/usr/bin/env python
# coding: utf-8

# # Sales Prediction using Python
# # Name: Muhammad Ahsan
# # Task: 5

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
#Load dataset 
df = pd.read_csv('Advertising.csv')


# In[2]:


#Visualize Data
sns.pairplot(df, x_vars=['TV', 'Radio', 'Newspaper'], y_vars='Sales', height=5, aspect=0.7, kind='scatter')
plt.show()


# In[3]:


# Model Train
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(x_train, y_train)


# In[4]:


#Model Evaluation
y_pred = model.predict(x_test)
meanAbsoluteError = mean_absolute_error(y_test, y_pred)
rootMeanSquaredError = np.sqrt(meanAbsoluteError)
print("Mean Absolute Error:", meanAbsoluteError)
print("Root Mean Squared Error:", rootMeanSquaredError)


# In[5]:


# Prediction
new_campaign = [[100, 25, 10]]
predicted_sales = model.predict(new_campaign)
print("Predicted Sales:", predicted_sales[0])


# In[ ]:




