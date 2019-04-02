# coding: utf-8
# In[ ]:
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.cluster import KMeans
from sklearn import preprocessing
import pylab as pl
from sklearn import decomposition
from sklearn import metrics
from sklearn.metrics import silhouette_samples, silhouette_score


myData = pd.read_csv('final_twitter_data.csv' , sep=',', encoding='latin1')
newData=pd.concat([myData['userFollowersNumber'], myData['character_length'], myData['total_count'], myData['score']],
                  axis=1,
                  keys=['userFollowersNumber', 'character_length', 'total_count', 'score'])

# print(newData)
x = newData.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
normalizedDataFrame = pd.DataFrame(x_scaled)
pprint(normalizedDataFrame[:10])

# #########################  K-means ############################
# Check K from 2 to 20 and print out the score
for i in range(2,20):
    k = i
    kmeans = KMeans(n_clusters=k)
    cluster_labels = kmeans.fit_predict(normalizedDataFrame)

    # Determine if the clustering is good
    silhouette_avg = silhouette_score(normalizedDataFrame, cluster_labels)
    print("For n_clusters =", k, "The average silhouette_score is :", silhouette_avg)
# #########################  K-means ############################
