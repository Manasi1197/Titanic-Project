# coding: utf-8

# In[1]:


import pandas as pd
from pandas import Series, DataFrame
import numpy as np
from numpy.random import randn
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


titanic_df = pd.read_csv('Downloads/train.csv')


# In[3]:


titanic_df.head()


# In[4]:


titanic_df.info()


# In[5]:


count = titanic_df.Sex.value_counts()


# In[6]:


sns.factorplot('Sex',count.values, data=titanic_df,kind='bar')


# In[7]:


sns.factorplot('Pclass',kind='count',data=titanic_df)


# In[8]:


sns.factorplot('Sex', data=titanic_df,kind='count')


# In[9]:


sns.factorplot('Sex',data=titanic_df,kind='count',hue='Pclass')


# In[10]:


sns.factorplot('Pclass',data=titanic_df,kind='count',hue='Sex')


# In[11]:


def male_female_child(passenger):
    age,sex = passenger
    if age < 16:
        return 'child'
    else:
        return sex


# In[12]:


titanic_df['person'] = titanic_df[['Age','Sex']].apply(male_female_child,axis=1)


# In[13]:


titanic_df.head(10)


# In[36]:


sns.factorplot('Pclass',data=titanic_df,kind='count',hue='person')


# In[15]:


titanic_df['Age'].min()


# In[16]:


titanic_df['Age'].hist(bins=70)


# In[28]:


fig = sns.FacetGrid(titanic_df,hue='person',aspect = 2)
fig.map(sns.kdeplot,'Age',shade= True)

oldest = titanic_df['Age'].max()

fig.set(xlim=(0,oldest))
fig.add_legend()


# In[37]:


fig = sns.FacetGrid(titanic_df,hue='Pclass',aspect=4)
fig.map(sns.kdeplot,'Age',shade=True)

oldest = titanic_df['Age'].max()

fig.set(xlim=(0,oldest))
fig.add_legend()


# In[31]:


titanic_df.head()


# In[43]:


deck = titanic_df['Cabin'].dropna()
deck


# In[140]:


level = []
for i in deck.values:
    level.append(i[0:1])
    


# In[56]:


level


# In[153]:


titanic_df['Survived'].describe()


# In[151]:


cabin_df = DataFrame(level)
cabin_df.columns = ['Cabin']
sns.factorplot('Cabin',data = cabin_df,palette = 'winter_d',kind='count')


# In[155]:


cabin_df.describe()


# In[61]:


cabin_df = cabin_df[cabin_df.Cabin!='T']


# In[62]:


cabin_df


# In[76]:


sns.factorplot('Cabin',data = cabin_df,palette = 'summer',kind='count')


# In[69]:


sns.factorplot('Embarked',data=titanic_df,kind='count',palette='winter')


# In[78]:


sns.factorplot('Pclass',data=titanic_df,hue='Embarked',kind='count')


# In[81]:


# Who was alone and who was with family
titanic_df.head(10)


# In[84]:


sns.factorplot('Parch',data = titanic_df,kind='count',palette='winter_d')


# In[85]:


titanic_df['alone']=titanic_df.SibSp+ titanic_df.Parch


# In[87]:


titanic_df.head(10)


# In[109]:


res=[]
for i in titanic_df['alone']:
    if i ==0:
        res.append('Alone')
    else:
        res.append('with family')    


# In[110]:


res


# In[111]:


family_or_not =DataFrame(res,columns=['alone'])


# In[112]:


family_or_not


# In[113]:


sns.factorplot('alone',data = family_or_not,kind='count',palette='winter_d')


# In[114]:


# Factors affecting survival of people on board
titanic_df['survivor'] = titanic_df.Survived.map({0:'No',1:'Yes'})


# In[138]:


titanic_df.head(10)


# In[119]:


sns.factorplot('survivor',data = titanic_df,kind='count',palette='Set1')


# In[120]:


sns.factorplot('survivor',data = titanic_df,kind='count',hue='Pclass',palette='winter_d')


# In[121]:


sns.factorplot('Pclass','Survived',data = titanic_df,palette='winter_d')


# In[123]:


sns.factorplot('Pclass','Survived',data = titanic_df,hue='person',palette='winter_d')


# In[124]:


sns.lmplot('Age','Survived',data=titanic_df)


# In[126]:


sns.lmplot('Age','Survived',data=titanic_df,hue='Pclass')


# In[127]:


generation = [10,20,40,60,80]

sns.lmplot('Age','Survived',data=titanic_df,hue='Pclass',x_bins=generation)


# In[136]:


sns.lmplot('Age','Survived',data=titanic_df,hue='Sex',x_bins=generation)


# In[139]:


sns.factorplot('Embarked','Survived',data=titanic_df,x_bins=generation)


# In[150]:


sns.factorplot('alone','Survived',hue='Sex',data=titanic_df)

