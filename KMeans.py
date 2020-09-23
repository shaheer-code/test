#!/usr/bin/env python
# coding: utf-8

# In[7]:


from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

df = pd.read_csv("https://raw.githubusercontent.com/datawizardsai/Data-Science/master/Mall_Customers.csv")


label_encoder = LabelEncoder()
df['Gender'] = label_encoder.fit_transform(df['Gender'])



x = df[['Age','Annual Income','Spending Score','Gender' ]] #features


sse = []
k_range = range(1,20)
for k in k_range:
    km = KMeans(n_clusters=k)
    km.fit(x)
    sse.append(km.inertia_)
plt.xlabel('K')
plt.ylabel('Sum of squared error')
plt.plot(k_range,sse)
plt.show()
plt.clf()


km = KMeans(n_clusters=5)
y_predicted = km.fit_predict(x)
y_predicted



df['cluster']=y_predicted
df



df.cluster.unique()
df1 = df[df.cluster==0]
df2 = df[df.cluster==1]
df3 = df[df.cluster==2]
df4 = df[df.cluster==3]
df5 = df[df.cluster==4]
plt.scatter(df1['Annual Income'],df1['Spending Score'],color='blue')
plt.scatter(df2['Annual Income'],df2['Spending Score'],color='green')
plt.scatter(df3['Annual Income'],df3['Spending Score'],color='yellow')
plt.scatter(df4['Annual Income'],df4['Spending Score'],color='red')
plt.scatter(df5['Annual Income'],df5['Spending Score'],color='black')


# In[ ]:




