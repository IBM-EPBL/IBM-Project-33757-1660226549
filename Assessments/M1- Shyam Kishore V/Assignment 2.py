#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")


# In[7]:


dt = pd.read_csv("C:\\Users\VSK\Downloads\Churn_Modelling.csv")


# In[8]:


dt.info()


# In[9]:


dt.describe()


# In[10]:


sns.displot(dt['Balance'], color='red')


# In[11]:


sns.displot(y="Balance", data= dt, color= 'red')


# In[12]:


sns.displot(x='Balance', data= dt, hue=dt['Age'])


# In[13]:


sns.distplot(dt['Balance'], color= 'orange')


# In[14]:


sns.distplot(dt["Balance"], hist= False, color= "orange")


# In[15]:


sns.boxplot(dt["Balance"],color='pink')


# In[16]:


sns.countplot(dt['Age'])


# In[17]:


sns.barplot(dt['Balance'], dt['EstimatedSalary'])


# In[18]:


sns.lineplot(dt["Age"],dt["EstimatedSalary"], color='purple')


# In[19]:


sns.scatterplot(x=dt.Balance, y=dt.RowNumber, color='green')


# In[20]:


sns.pointplot(x='Age', y='Tenure', data=dt, color='pink')


# In[21]:


sns.regplot(dt['Age'], dt['Tenure'], color='orange')


# In[22]:


sns.pairplot(data=dt[["RowNumber","Age","Tenure","Balance","NumOfProducts"]], kind="kde")


# In[23]:


sns.pairplot(data=dt[["RowNumber","Age","Tenure","Balance","NumOfProducts"]], hue="Age", diag_kind="hist")


# In[24]:


sns.pairplot(data=dt[["RowNumber","Age","Tenure","Balance","NumOfProducts"]], hue="Age")


# In[25]:


dt.describe()


# In[26]:


data=pd.DataFrame({"a":[1,2,np.nan],"b":[1,np.nan,np.nan],"c":[1,2,4]})
data


# In[27]:


data.isnull().any()


# In[28]:


data.isnull().sum()


# In[29]:


data.fillna(value = "S")


# In[30]:


data["a"].mean()


# In[31]:


data["a"].median()


# In[32]:


outlierss=dt.quantile(q=(0.25,0.75))
outlierss


# In[33]:


outlier_diff=outlierss.loc[0.75]-outlierss.loc[0.25]
outlier_diff


# In[34]:


low = outlierss.loc[0.25] - 1.5 * outlierss
low


# In[35]:


high = outlierss.loc[0.75] + 1.5 * outlierss
high


# In[36]:


sns.boxplot(dt["Age"],color='purple')


# In[37]:


dt["Age"]= np.where(dt["Age"]<25, 50, dt["Age"])
sns.boxplot(dt["Age"], color='pink')


# In[38]:


dt.head(4)


# In[39]:


dt["Gender"].replace({"Female":0,"Male":1},inplace = True)
dt["Geography"].replace({"France":1,"Spain":2,"Germany":3},inplace = True)
dt["Gender"].replace({"Female":0,"Male":1},inplace = True)
dt["Geography"].replace({"France":1,"Spain":2,"Germany":3},inplace = True)
dt.head(4)


# In[40]:


y = dt["Surname"]
x=dt.drop(columns=["Surname"], axis=1)
x.head()


# In[41]:


names=x.columns
names


# In[42]:


from sklearn.preprocessing import scale
X=scale(x)
X


# In[43]:


x = pd.DataFrame(X, columns = names )
x


# In[44]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.2, random_state=0)
x_train.head()


# In[45]:


x_train.shape, y_train.shape, x_test.shape, y_test.shape


# In[ ]:




