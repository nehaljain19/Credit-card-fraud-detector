
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score,cross_val_predict


# In[4]:


data=pd.read_excel('mymachinedata.xlsx')


# In[5]:


arr=data.values
x=arr[:,0:8]
y=arr[:,8]


# In[6]:


random_state = np.random.RandomState(0)


# In[7]:


x


# In[8]:


le = preprocessing.LabelEncoder()


# In[9]:


y=le.fit_transform(y)


# In[10]:


y


# In[11]:


x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.40,random_state=random_state)


# In[12]:


clf=tree.DecisionTreeClassifier(criterion='gini',min_samples_split=30,class_weight={0:10,1:90},max_features='auto')


# In[13]:


clf = clf.fit(x_train,y_train)


# In[14]:


y_pred = clf.predict(x_test)


# In[15]:


np.unique(y_pred)


# In[16]:



accuracy = accuracy_score(y_test,y_pred)


# In[17]:


accuracy


# In[18]:


print(clf.score(x_test,y_test))


# In[19]:


scores = cross_val_score(clf,x,y,cv=10)
print(scores)


# In[20]:


y_pred.shape


# In[21]:


import pickle


# In[22]:


data.head()


# In[23]:


x_new1=[[0.3,8,500,1,3,15,10,0]]


# In[24]:


y_new=clf.predict_proba(x_new1)[:,1]


# In[25]:


print(y_new)


# In[26]:


x_new2=[[-0.3,1,1500,5,3,5,0,1]]


# In[27]:


y_new2=clf.predict_proba(x_new2)[:,1]


# In[28]:


print(y_new2)


# In[86]:


x_new3=[[200,4,10000,1,1,2,7,1]]


# In[87]:


y_new3=clf.predict_proba(x_new3)[:,1]


# In[88]:


print(y_new3)

