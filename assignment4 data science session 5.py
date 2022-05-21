#!/usr/bin/env python
# coding: utf-8

# ### Load the dataset

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
data= pd.read_csv('train.csv',sep=',')
data


# ### Dimensions of this data

# In[58]:


shape=data.shape
print("dataset shape:",shape)


# ###  Num of features

# In[59]:



print("num of features:",len(data.columns))


# ### Output data type

# In[60]:


data.dtypes


# ### Does it have nulls?

# In[61]:


data.isnull().sum()


# In[45]:


data.info()


# ### Does it have duplicate values?

# In[3]:


data.nunique()


# ### Does it have outliers?

# In[12]:


sns.boxplot(x="Sex",y="Age", data=data,palette="rainbow")


# In[13]:


sns.boxplot(x="Sex",y="Fare", data=data,palette="rainbow")


# ### EDA with visualizations to investigate the data

# In[28]:


sns.displot(data['Sex'],kde=False,bins=30)


# In[29]:


sns.displot(data['Age'],kde=False,bins=30)


# In[30]:


sns.jointplot(x='Sex',y='Age',data=data,kind='scatter')


# In[33]:


sns.jointplot(x='Age',y='Fare',data=data,kind='hex')


# In[34]:


sns.jointplot(x='Age',y='Fare',data=data,kind='reg')


# In[36]:


sns.pairplot(data)


# In[37]:


sns.pairplot(data,hue='Sex',palette='coolwarm')


# In[40]:


sns.barplot(x='Sex',y='Age',data=data)


# In[41]:


sns.barplot(x='Sex',y='Age',data=data,estimator=np.std)


# In[42]:


sns.countplot(x='Sex',data=data)


# In[43]:


sns.boxplot(x='Sex',y='Age',data=data,palette='rainbow')


# In[50]:


sns.boxplot(x='Age',y='Fare',hue="Sex",data=data,palette="coolwarm")


# In[52]:


sns.heatmap(data.corr())


# In[54]:


sns.heatmap(data.corr(),cmap='coolwarm',annot=True)


# In[55]:


pv=data.pivot_table(values="Age",index="Sex",columns="Survived")


# In[56]:


sns.heatmap(pv)


# In[57]:


sns.heatmap(pv,cmap="magma",linecolor='white',linewidth=1)


# In[58]:


sns.clustermap(pv)


# In[59]:


sns.clustermap(pv,cmap="coolwarm",standard_scale=1)


# In[ ]:




