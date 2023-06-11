#!/usr/bin/env python
# coding: utf-8

# **Vedant Modak**
#   | BE(IT) undergrad @ PES Modern College of Engineering,Pune.

# **Customer Behavior Analysis**

# **Importing necessary libraries**

# In[6]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[9]:


df=pd.read_csv('F:\Data Analytics\Portfolio\Projects\Project - 2 (Customer behavior)\churn.csv')


# In[5]:


df.head()


# In[6]:


df.tail()


# In[7]:


df.shape


# In[8]:


df.info()


# **Cleaning the Dataset**

# In[9]:


df = df.drop_duplicates()


# Handling null values

# In[10]:


df.isnull().sum()


# In[11]:


df.describe()


# In[12]:


df['region_category'].mode()


# In[13]:


df['preferred_offer_types'].mode()


# In[14]:


df['points_in_wallet'].fillna(686.882199, inplace=True)


# In[15]:


df['preferred_offer_types'].fillna('Gift Vouchers/Coupons', inplace=True)


# In[16]:


df['region_category'].fillna('Town',inplace=True)


# In[17]:


df.isnull().sum()


# In[18]:


df['avg_frequency_login_days'] = df['avg_frequency_login_days']. replace(['Error'], [0])


# **Removing Outliers**

# In[19]:


sns.boxplot(df['avg_time_spent'])


# In[20]:


sns.boxplot(df['avg_transaction_value'])


# In[21]:


sns.boxplot(df['days_since_last_login'])


# In[22]:


sns.boxplot(df['age'])


# In[23]:


def remove_outliers(column):
    Q1 = column.quantile(0.25)
    Q3 = column.quantile(0.75)
    IQR = Q3 - Q1
    threshold = 1.5 * IQR
    outlier_mask = (column < Q1 - threshold) | (column > Q3 + threshold)
    return column[~outlier_mask]


# In[24]:


col_name = ['avg_time_spent', 'avg_transaction_value', 'days_since_last_login', 'age']
for col in col_name:
    df[col] = remove_outliers(df[col])


# In[25]:


plt.figure(figsize=(10, 6)) 

for col in col_name:
    sns.boxplot(data=df[col])
    plt.title(col)
    plt.show()


# **Customer Behavior Analysis**

# In[26]:


df.corr()


# In[27]:


plt.figure(figsize=(18, 12))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()


# Churn Risk Score has a good correlation with Points in wallet
# 

# Lesser points in wallet, greater the chur risk

# In[28]:


customer_totals = df.groupby('security_no')['avg_transaction_value'].sum()
sorted_totals = customer_totals.sort_values(ascending=False)
top_10_customers = sorted_totals.head(10)
print("Top 10 Customers by Transaction value:")
print(top_10_customers)


# In[29]:


df['preferred_offer_types'].value_counts()


# In[30]:


df.columns


# In[31]:


sort=df.sort_values(by = 'gender')
sort.tail(10)


# **Analyzing gender wise spending**

# In[32]:


gender_wise_spending =df.groupby('avg_time_spent')['points_in_wallet'].sum()
gender_wise_spending


# In[39]:


sns.barplot(x='gender', y='points_in_wallet',data=df)


# Both Males and Females have almost equal points in wallet

# In[10]:


sns.barplot(x='gender', y='avg_transaction_value',data=df)


# Both Males and Females spend almost equal amount of money

# In[12]:


sns.relplot(x = "gender", y = "avg_transaction_value", hue= "churn_risk_score", data =df);


# It is more likely that customers will stop buying products if the transaction value increases above 50,000 approx

# In[19]:


plt.figure(figsize=(20, 12))
sns.countplot(x = "complaint_status", hue = "churn_risk_score", palette = "ch:.25", data = df)


# **The Risk of loosing a customer is almost equal irrespective of whether the customer's complaint is solved or unsolved.
# Solving more complaints may  not reduce the churn risk.**

# In[23]:


df2=df[['points_in_wallet', 'churn_risk_score']]
df2


# In[24]:


plt.scatter(df2.points_in_wallet, df2.churn_risk_score, marker='+', color='red')


# In[101]:


plt.figure(figsize=(15, 6))
plt.scatter(df.membership_category, df.churn_risk_score, marker='+', color='red')


# **The Risk of loosing a customer is less if the customer has any sort of Membership**

# **Lets make a logistic regression model to predict the churn score on membership basis**

# In[38]:


df3=df[['membership_category', 'churn_risk_score']]
df3


# In[58]:


df3['membership_category'].unique()


# **Converting categorical data into numerical fromat using label encoding**

# In[59]:


from sklearn.preprocessing import LabelEncoder


# In[61]:


encode = LabelEncoder()
label=encode.fit_transform(df3["membership_category"])


# In[62]:


df4=df3.drop("membership_category",axis='columns')


# In[63]:


df4.head(2)


# In[64]:


df4["membership_category"]=label


# In[66]:


df4.head(2)


# In[89]:


from sklearn.model_selection import train_test_split


# In[90]:


X_train, X_test, y_train, y_test = train_test_split(df4[['membership_category']], df.churn_risk_score, test_size=0.2 )


# In[91]:


X_train.shape


# In[92]:


X_test.shape


# In[93]:


from sklearn.linear_model import LogisticRegression


# In[94]:


model = LogisticRegression()
model.fit(X_train,y_train)


# In[95]:


model.predict(X_test)


# **Lets check the accuracy of the model**

# In[96]:


model.score(X_test,y_test)


# **This model works almost 80% accurately !**

# Lets test this manually

# In[100]:


model.predict([[1]])

