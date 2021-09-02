#!/usr/bin/env python
# coding: utf-8

# 
# **A tumor is an abnormal lump or growth of cells. When the cells in the tumor are normal, it is benign. Something just went wrong, and they overgrew and produced a lump. When the cells are abnormal and can grow uncontrollably, they are cancerous cells, and the tumor is malignant.**
# 
# **In this note book i have tried to compare the 2 tumors- benign and malignant
#   i have added visualisatiion to show that how are these two tumors different
#   then i have applied various classification models for predictions for given features that whether the tumour is B or M.
#   i have also illustrated the importance of feature scaling.**

# # Loading Libraries and Data

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier
from sklearn import svm # for Support Vector Machine
from sklearn import metrics # for the check the error and accuracy
import matplotlib as mpl


# # LOADING THE DATA SET USING READ_CSV

# In[2]:


df = pd.read_csv('data.csv')


# 
# # BASIC PREPROCESSING OF THE DATA

# In[3]:


df.shape #569 rows and 33 columns


# In[4]:


df.columns


# ## cleaning
# 

# In[5]:


df.drop('id',axis=1,inplace=True)
df.drop('Unnamed: 32',axis=1,inplace=True)


# In[6]:


df.diagnosis.unique()


# In[7]:


df['diagnosis'] = df['diagnosis'].map({'M':1,'B':0})
df.head()


# ## VISUALIZATION

# In[8]:


# split dataframe into two based on diagnosis
dfM=df[df['diagnosis'] ==1]
dfB=df[df['diagnosis'] ==0]


# In[9]:


sns.pairplot(df.loc[:,'diagnosis':'area_mean'], hue="diagnosis");


# ### FOLLOWNG GRAPHS SHOW THE DIFFERENCE BETWEEN VARIOUS PARAMTERS OF BENIGN AND MALIGNANT TUMORS
# 

# In[10]:


dfB.plot(kind = "density",x= 'radius_mean', y = 'concavity_mean')
plt.xlabel("mean radius for benigm")
plt.ylabel("mean concavity for benigm")


# In[11]:


dfM.plot(kind = "density",x= 'radius_mean', y = 'concavity_mean')
plt.xlabel("mean radius for malignant")
plt.ylabel("mean concavity for malignant")


# # ABOVE TWO GRAPHS PROVE THE DIFFERENCE BETWEEN 2 TUMORS.

# In[12]:


dfB.plot(kind = "scatter",x= 'radius_mean', y = 'area_mean')
plt.xlabel("mean radius for benigm")
plt.ylabel("mean area for benigm")


# In[13]:


dfM.plot(kind = "scatter",x= 'radius_mean', y = 'area_mean')
plt.xlabel("mean radius for malignant")
plt.ylabel("mean area for malignant")


# ##### SIMILARLY , ABOVE TWO GRAPHS PROVE THAT M TUMORS ARE MUCH BIGGER THAN B TUMORS.

# In[14]:


g = sns.jointplot(x=dfM['radius_mean'], y=dfM['texture_mean'], data=df, kind="kde", color="m")
g.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker="+")
g.ax_joint.collections[0].set_alpha(0)
g.set_axis_labels("$mean radius for malignant$", "$mean texture for malignant$");


# In[15]:


g = sns.jointplot(x=dfB['radius_mean'], y=dfB['texture_mean'], data=dfB, kind="kde", color="m")
g.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker="+")
g.ax_joint.collections[0].set_alpha(0)
g.set_axis_labels("$mean radius$", "$mean texture$");


# In[16]:


avgB = {}
for i in range(2,dfB.shape[1]):
    m = np.mean(dfB.iloc[:,i])
    avgB.update({dfB.columns[i]:m})

avgB_df = pd.DataFrame(avgB,index = np.arange(1,31))
avgB_df = avgB_df.transpose()
avgB_df = avgB_df.iloc[:,:1]


# In[17]:


avgM = {}
for i in range(2,dfM.shape[1]):
    m = np.mean(dfM.iloc[:,i])
    avgM.update({dfM.columns[i]:m})

avgM_df = pd.DataFrame(avgM,index = np.arange(1,31))
avgM_df = avgM_df.transpose()
avgM_df = avgM_df.iloc[:,:1]


# In[18]:


#so now, i have 2 data frames and i want to have a combined barplot

avgB_df['hue']='B'
avgM_df['hue']='M'
res=pd.concat([avgB_df,avgM_df])
res = res.reset_index(level =0)
sns.barplot(x = res.iloc[:,0],y = res.iloc[:,1],data=res,hue='hue')
plt.xticks(rotation=90)
plt.ylabel('average of feature mentioned on X axis')
plt.show()


# In[19]:


g = sns.PairGrid(dfB.loc[:,'radius_mean':'smoothness_mean'])
g.map_diag(sns.kdeplot)
g.map_offdiag(sns.kdeplot, n_levels=6);


# In[20]:



g = sns.PairGrid(dfM.loc[:,'radius_mean':'smoothness_mean'])
g.map_diag(sns.kdeplot)
g.map_offdiag(sns.kdeplot, n_levels=6);


# **ANALYZING THE GRAPHS ABOVE, GIVE US A GOOD PICTORIAL IDEA FOR DIFFERENCES BETWEEN THE TUMORS**

# In[21]:


#lets see the amount of benigan and melignant tissues:
len(dfB)


# In[22]:


len(dfM)


# ### we can see that there are 357 B type and 212 M type cells

# ### #lets split the data now
# 

# In[25]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
for i in list(df.columns):
    if df[i].dtype=='object':
        df[i]=le.fit_transform(df[i])
y=df['diagnosis']
x=df.drop('diagnosis',axis=1)
#X = df.iloc[:,:]
#y = df.iloc[:,1]        
#lets split the data now
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size = 0.2,random_state = 1)


# # FITTING & TESTING THE CLASSIFICATION MODELS (without scaling the data)::

# In[26]:



model = svm.SVC()
model.fit(xtrain,ytrain)# now fit our model for traiing data
prediction=model.predict(xtest)# predict for the test data

metrics.accuracy_score(prediction,ytest)
print(metrics.accuracy_score(prediction,ytest))
metrics.confusion_matrix(ytest,prediction)


# In[28]:


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
#decision tree
classifier = DecisionTreeClassifier(criterion='gini', max_leaf_nodes= None,
                                    min_samples_leaf=14, min_samples_split=2 ,
                                    random_state = 12)
classifier.fit(xtrain,ytrain)
ypred = classifier.predict(xtest)
print(metrics.accuracy_score(ypred,ytest))
metrics.confusion_matrix(ytest,prediction)
y_pred = classifier.predict(xtest)
print(classification_report(ytest, y_pred))
pd.DataFrame(
    confusion_matrix(ytest, y_pred),
    columns=['Predicted No', 'Predicted Yes'],
    index=['Actual No', 'Actual Yes']
)

scores = cross_val_score(classifier, x, y, scoring='f1_macro', cv=10)
print('Macro-F1 average: {0}'.format(scores.mean()))


# 
# 
# ### but lets see if scaling the data gives different results

# 
# 

# # what if i scale the data now::

# In[30]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(x)


# In[31]:


#lets split the data now
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(X,y,test_size = 0.2,random_state = 1)

#decision tree
dt = DecisionTreeClassifier()
dt.fit(xtrain,ytrain)
ypred = dt.predict(xtest)
print('DECISION TREE CLASSIFIER:: ',metrics.accuracy_score(ypred,ytest))

#SVM
model = svm.SVC()
model.fit(xtrain,ytrain)# now fit our model for traiing data
prediction=model.predict(xtest)# predict for the test data
metrics.accuracy_score(prediction,ytest)
print('SUPPORT VECTOR MACHINE:: ',metrics.accuracy_score(prediction,ytest))




# In[ ]:




