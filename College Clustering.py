# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 12:20:16 2020

@author: Dell
"""

# Importing all the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, classification_report

# Importing the data
data = pd.read_csv("College_data.csv",index_col=0)

# Exploring the data
data.head()
data.info()
data.describe()

# Working on K Means model
kmeans = KMeans(n_clusters=2)
kmeans.fit(data.drop('Private',axis=1))

# Finding the cluster's center vector
kmeans.cluster_centers_

# Evaluating the model
# There is no perfect way to evaluate if the clustering id done right if labels are not know
# but here we are having the labels in the dataset so we will take the advantage of it

# Defining a new column in data named "Cluster" with 1 for "Private" and 0 for "Public"
def converter(cluster):
    if cluster == 'Yes':
        return 1
    else:
        return 0

data['Cluster']=data['Private'].apply(converter)
data.head()

print(confusion_matrix(data['Cluster'],kmeans.labels_))
print(classification_report(data['Cluster'],kmeans.labels_))