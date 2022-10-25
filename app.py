#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, plot_confusion_matrix

import warnings
warnings.filterwarnings("ignore")
plt.rcParams["figure.figsize"] = (12, 12)


# ##### Exploratory Data Analysis 

# In[2]:


train = pd.read_csv("train.csv", delimiter=";")
train.head()


# In[3]:


test = pd.read_csv("test.csv", delimiter=";")
test.head()


# In[4]:


train.info()


# In[5]:


train.describe().T


# In[6]:


test.info()


# In[7]:


test.describe().T


# We find that we have no missing values in any of the both datasets. Furthermore, it appears to be that the distribution of values are normally distributed, without any skewness. At least, that is our first hypothesis. Our main goal is to perform a Random Forest Classifier and find the best hyperparameters to obtain an accuracy score above 90% on the validation data. In the aftermath, we can proceed with making our final predictions on the training dataset.

# In[8]:


def correlation(dataframe=train):
    methods = ["pearson", "spearman", "kendall"]
    palette = ["magma", "viridis", "cubehelix"]
    for i in range(3):
        plt.style.use("ggplot")
        plt.figure(figsize=(12, 8))
        sns.heatmap(dataframe.corr(method=methods[i]), vmax=.8, square=True, annot=True, linewidths=.5, cmap=palette[i])
        plt.title(methods[i] + " correlation")
        plt.show()


# In[9]:


def boxplots(dataframe=train, y=train["target"]):
    for feature in dataframe.columns[:-1]:
        plt.figure(figsize=(9, 6))
        sns.boxplot(x=y,y=feature, data=dataframe)
        plt.grid(True)
        plt.show()


# In[10]:


def distplots(dataframe=train):
    colors = ["dimgrey", "lightcoral", "tan", "lightsteelblue", "teal", "indigo", "navy"]
    for feature in dataframe.columns[:-1]:
        plt.figure(figsize=(9, 6))
        sns.distplot(dataframe[feature], color= np.random.choice(colors))
        plt.show()


# In[11]:


def violinplots(dataframe=train):
    colors = ["dimgrey", "lightcoral", "tan", "lightsteelblue", "teal", "indigo", "navy"]
    for feature in dataframe.columns[:-1]:
        plt.figure(figsize=(9, 6))
        sns.violinplot(dataframe[feature], color= np.random.choice(colors),
                      orient="v")


# In[12]:


correlation(train)


# I want to draw the attention to the fact that all features except "feature4", "feature7" and "feature8" are terrifically correlated to the target variable. Hence, we might expect our model to not be polluted with noise. 

# In[13]:


boxplots(train)


# In[14]:


distplots(train)


# In[15]:


violinplots(train)


# ##### Random forest

# In[16]:


y = train["target"]
x = train.drop("target",axis=1)


# In[17]:


x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=34)


# In[18]:


rfc = RandomForestClassifier()


# In[19]:


param_grid = { 
    "n_estimators": [200, 500],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf" : [1, 2],
    "max_features": ["auto", "sqrt", "log2"],
    "max_depth" : [4, 6, 8, 10],
    "criterion" : ["gini", "entropy"],
    "bootstrap" : [True, False]
}


# In[ ]:


get_ipython().run_cell_magic('time', '', 'CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)\nCV_rfc.fit(x_train, y_train)')


# In[ ]:


CV_rfc.best_params_


# In[20]:


rfc_2 = RandomForestClassifier(random_state=42, bootstrap=False, max_features='log2',
                               n_estimators=500, max_depth=10, criterion='gini',
                              min_samples_leaf=1, min_samples_split=2)


# In[21]:


rfc_2.fit(x_train, y_train)


# In[22]:


y_pred = rfc_2.predict(x_val)


# ##### Model evaluation

# In[23]:


print("Accuracy for Random Forest on the validation data: ", round(accuracy_score(y_val, y_pred), 3))


# In[24]:


sns.set(font_scale=1.4)
sns.heatmap(confusion_matrix(y_val, y_pred), annot=True,
            fmt='g')


# In[25]:


clf_report = classification_report(y_val,
                                   y_pred,
                                   target_names=[0, 1, 2],
                                   output_dict=True)

ax = sns.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, annot=True, cmap="crest",
           linewidth=.5)
ax.xaxis.tick_top()


# ##### Final predictions

# In[26]:


y_final = rfc_2.predict(test)


# In[27]:


final_results = pd.DataFrame(y_final).set_axis(["final_status"], axis=1)


# In[28]:


final_results.to_csv("predictions.csv", index = False)


# In[29]:


final_results.to_json("predictions.json")

