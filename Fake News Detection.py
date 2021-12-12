#!/usr/bin/env python
# coding: utf-8

# In[27]:


import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


# In[6]:


#Read the data
df= pd.read_csv("C:/Users/DELL/Desktop/Project/news.csv")
df.head()


# In[8]:


df.isnull().sum() ## there are no null items in the dataframe


# In[9]:


df.info()


# In[12]:


df.shape
df.describe()


# In[13]:


labels=df.label
labels.head()


# In[17]:


## split the dataset

x_train,x_test,y_train,y_test=train_test_split(df['text'], labels, test_size=0.2)


# In[18]:


###TfidfVectorizer with stop words from the English language and a maximum document frequency of 0.7 


# In[19]:


tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)


# In[20]:


tfidf_train=tfidf_vectorizer.fit_transform(x_train) 
tfidf_test=tfidf_vectorizer.transform(x_test)


# ###It is one of the few ‘online-learning algorithms‘. In online machine learning algorithms, the input data comes in sequential order and the machine learning model is updated step-by-step, as opposed to batch learning, where the entire training dataset is used at once. This is very useful in situations where there is a huge amount of data and it is computationally infeasible to train the entire dataset because of the sheer size of the data. We can simply say that an online-learning algorithm will get a training example, update the classifier, and then throw away the example.
# ###A very good example of this would be to detect fake news on a social media website like Twitter, where new data is being added every second. To dynamically read data from Twitter continuously, the data would be huge, and using an online-learning algorithm would be ideal.  

# In[21]:


pac=PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train,y_train)
####he test set and calculate accuracy
y_pred=pac.predict(tfidf_test)
score=accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')


# In[23]:


####he test set and calculate accuracy
y_pred=pac.predict(tfidf_test)
y_pred


# In[24]:


score=accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')


# In[33]:


##confusion matrix
confusion_matrix(y_test,y_pred, labels = ["FAKE", "REAL"])

### 610- True positives, 591- True negatives 37 false positives,29- false negatives 

