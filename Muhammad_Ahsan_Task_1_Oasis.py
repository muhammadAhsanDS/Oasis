#!/usr/bin/env python
# coding: utf-8

# # Iris Flower Classification
# # Name : Muhammad Ahsan
# # Task : 1

# In[2]:


#import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,precision_score, f1_score

#read data
df = pd.read_csv('iris.csv')
# df.drop('Id', axis=1,inplace=True)
df


# In[3]:


#checking is their any null values
df.isnull().sum()


# In[4]:


# Colors for plot
colors = {'Iris-setosa': 'blue', 'Iris-versicolor': 'green', 'Iris-virginica': 'yellow'}

# Box plot 
plt.figure(figsize=(10, 6))
sns.boxplot(x='Species', y='SepalLengthCm', data=df, palette=colors)
plt.title('Distribution of Sepal Length by Species')
plt.xlabel('Iris Species')
plt.ylabel('Length of Sepal')
plt.show()


# In[5]:


# Features and Labels
X = df.drop('Species', axis=1)
y = df['Species']
# Here, i Split 80% data in training and 20% in testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[6]:


# Now we Train Decision Tree Model on our trainig data
DTC = DecisionTreeClassifier()
DTC.fit(X_train, y_train)


# In[7]:


#Now predict Test data and check model performance
y_pred = DTC.predict(X_test)
Acc = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
print("Accuracy: ",Acc)
print("Precision : ",precision)
print("F1-Score : ",f1)


# In[8]:


# Predict Value
df1 = pd.DataFrame({'Id': [200], 'SepalLengthCm': [5.8], 'SepalWidthCm': [5.5], 'PetalLengthCm': [2.4], 'PetalWidthCm': [1.2]})
PredictIt = DTC.predict(df1)
print("Predicted Class:", PredictIt)

