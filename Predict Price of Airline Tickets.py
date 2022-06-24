#!/usr/bin/env python
# coding: utf-8

# In[42]:


import pandas as pd
import numpy as np


# In[43]:


data = pd.read_csv("Predict Price of Airline Tickets.csv")
data.info()
data.head()


# In[44]:


from sklearn.preprocessing import OneHotEncoder

#creating instance of one-hot-encoder
encoder = OneHotEncoder() # handle_unknown='ignore'

#perform one-hot encoding on 'team' column 
encode_column = pd.DataFrame(encoder.fit_transform(data[['Airline','Source','Destination','Route','Total_Stops']]).toarray())
encode_column


# In[45]:


from sklearn import preprocessing
from sklearn.model_selection import train_test_split

# data.info()

data.head(5)


le = preprocessing.LabelEncoder()
# data['Outlook'] = le.fit_transform(data.Outlook.values)

data = data.apply(le.fit_transform)

x = data.drop(['Price'],axis=1)
y = data['Price']

xx = np.array(x)
yy = np.array(y)
x_train, x_test, y_train, y_test = train_test_split(xx, yy, test_size=0.33, random_state=42)


# In[56]:


from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn import tree

regressor = RandomForestRegressor(n_estimators=50, max_depth=10) #max_depth=3

regressor.fit(x_train,y_train)


# tree_graph = tree(regressor)


# In[57]:


from sklearn import metrics

y_pred = regressor.predict(x_test)
y_test
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('Mean Absolute Error:', metrics.r2_score(y_test, y_pred))


# In[58]:


from sklearn.tree import export_text
from sklearn import tree
# x.columns
columns_name = ['Airline','Date_of_Journey','Source','Destination','Route','Dep_Time','Arrival_Time','Duration','Total_Stops','Price']

for tree_in_forest in regressor.estimators_:
  text_representation = export_text(tree_in_forest,feature_names=columns_name)
  print(text_representation)


# In[65]:


import matplotlib.pyplot as plt

import graphviz
# DOT data
i_tree =0
for tree_in_forest in regressor.estimators_:

    dot_data = tree.export_graphviz(tree_in_forest, out_file=None, 
                                    feature_names=x.columns,  
                      #  class_names=['Golf Players'],
                                    filled=True, rounded=True)

    # Draw graph
    graph = graphviz.Source(dot_data, format="png") 
    graph



graph.render('tree',format='png', view=True)


# In[66]:


features = columns_name
importances = regressor.feature_importances_
indices = np.argsort(importances)

plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()


# In[67]:


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
cv = KFold(n_splits=10, random_state=1, shuffle=True)

#build multiple linear regression model
# model = LinearRegression()

#use k-fold CV to evaluate model
scores = cross_val_score(regressor, x, y, scoring='neg_mean_squared_error',
                         cv=cv, n_jobs=-1)

print('mean_squared_error',np.mean(np.absolute(scores)))


# In[68]:


from sklearn.model_selection import GridSearchCV

param_grid = {  'bootstrap': [True], 'max_depth': [5, 10, None],
              'max_features': ['auto', 'log2'],
              'n_estimators': [5, 6, 7, 8, 9, 10, 11, 12, 13, 15]}

rfr = RandomForestRegressor(random_state = 1)

g_search = GridSearchCV(estimator = rfr, param_grid = param_grid, 

                          cv = 3, n_jobs = 1,
                           verbose = 0, return_train_score=True)

g_search.fit(x_train, y_train);

print(g_search.best_params_)


# In[ ]:




