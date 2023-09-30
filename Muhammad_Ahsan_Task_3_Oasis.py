#!/usr/bin/env python
# coding: utf-8

# # Car Price Prediction with Machine Learning
# # Name : Muhammad Ahsan
# # Task : 3

# In[1]:


import pandas as pd

# Load datasets
DataCar = pd.read_csv('CarPrice_Assignment.csv')
print(DataCar.info())


# In[2]:


# Handle any missing value and duplicates
DataCar.dropna(inplace=True)
DataCar.reset_index(drop=True, inplace=True)
DataCar.drop_duplicates(inplace=True)

# Convert categorical to numerical 
DataCar = pd.get_dummies(DataCar, columns=['fueltype', 'aspiration', 'carbody', 'drivewheel', 'enginelocation', 'enginetype', 'cylindernumber', 'fuelsystem'])


# In[3]:


# Visualize data
import matplotlib.pyplot as plt
import seaborn as sns

corr_matrix = DataCar.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='viridis')
plt.title('Correlation Matrix')
plt.show()


# In[4]:


# Selecting relevant features 
selected_features = ['horsepower', 'enginesize', 'curbweight', 'carwidth', 'highwaympg']

# selected features
X = DataCar[selected_features]
y = DataCar['price']


# In[5]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[6]:


from sklearn.linear_model import LinearRegression

#  train model
model = LinearRegression()
model.fit(X_train, y_train)


# In[7]:


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Make predictions 
y_pred = model.predict(X_test)

# Evaluate  model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'MSE: {mse}')
print(f'R-squared (R2): {r2}')


# In[8]:


# Making Prediction 
new_car_features = pd.DataFrame({
    'horsepower': [150],
    'enginesize': [200],
    'curbweight': [2800],
    'carwidth': [68],
    'highwaympg': [30]
})
predicted_price = model.predict(new_car_features)
print(f'Predicted Price: ${predicted_price[0]:.2f}')


# In[ ]:





# In[ ]:




