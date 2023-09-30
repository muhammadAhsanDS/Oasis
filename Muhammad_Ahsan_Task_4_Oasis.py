#!/usr/bin/env python
# coding: utf-8

# # Email spam Detection with Machine Learning
# # Name: Muhammad Ahsan
# # Task: 4

# In[1]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# load dataset
df = pd.read_csv('spam.csv', encoding='ISO-8859-1')



# In[2]:


df = df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)
df


# In[3]:


#data preprocessing
data['v1'] = data['v1'].map({'ham': 0, 'spam': 1})
x = data['v2']
y = data['v1']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# In[ ]:


#feature extraction
vectorizer = TfidfVectorizer()
x_train_vectorizer = vectorizer.fit_transform(x_train)
x_test_vectorizer = vectorizer.transform(x_test)


# In[ ]:


#Model Training
model = MultinomialNB()
model.fit(x_train_vectorizer, y_train)
y_pred = model.predict(x_test_vectorizer)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
print("Model Accuracy:", accuracy*100)
print("Model Classification Report:\n", report)


# In[ ]:


#Prediction
text = ["Congratulations! You've won a free gift. Click here to claim your prize."]
#TF-IDF vectorizer
text_vectorizer = tfidf_vectorizer.transform(text) 
pre = model.predict(text_vectorizer)
if pre[0] == 0:
    print("Message Is Ham")
else:
    print("Message Is Spam")


# In[ ]:




