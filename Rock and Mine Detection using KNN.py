#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[6]:


df = pd.read_csv('Documents/Python for Data Science and Machine Learning/DATA/sonar.all-data.csv')


# In[14]:


df.head()


# In[15]:


df['Target'] = df['Label'].map({'R':0,'M':1})


# In[16]:


X = df.drop(['Target','Label'],axis=1)
y = df['Target']


# In[17]:


from sklearn.model_selection import train_test_split


# In[18]:


X_cv, X_test, y_cv, y_test = train_test_split(X, y,test_size=0.3,random_state=101)


# In[19]:


from sklearn.preprocessing import StandardScaler


# In[23]:


from sklearn.neighbors import KNeighborsClassifier


# In[24]:


scaler = StandardScaler()


# In[28]:


knn = KNeighborsClassifier()


# In[29]:


from sklearn.pipeline import Pipeline


# In[30]:


operations = [('scaler',scaler),('knn',knn)]


# In[31]:


pipe = Pipeline(operations)


# In[32]:


from sklearn.model_selection import GridSearchCV


# In[36]:


k_values = list(range(1,30))


# In[37]:


print(k_values)


# In[38]:


param_grid = {'knn__n_neighbors': k_values}


# In[39]:


full_cv_classifier = GridSearchCV(pipe,param_grid,cv=5,scoring='accuracy')


# In[40]:


full_cv_classifier.fit(X_cv,y_cv)


# In[43]:


full_cv_classifier.best_estimator_.get_params()


# In[44]:


predictions = full_cv_classifier.predict(X_test)


# In[49]:


predictions


# In[45]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# In[48]:


print(classification_report(y_test,predictions))


# In[ ]:




